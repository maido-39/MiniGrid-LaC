"""
Temperature vs Entropy Analysis
================================
Temperature 변화에 따른 Entropy 분포 및 답변 경향성 분석

실험 설계:
- Temperature: 0.2 ~ 1.5, 0.15 단위
- Language Prompt: 3가지 (불확실/확실/이상함)
- 반복: 각 조합당 5회
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Google Generative AI
from google import genai
from google.oauth2 import service_account

# GCP 인증 설정
KEY_PATH = "/home/syaro/stan_GCP_API.json"
PROJECT_ID = "composed-hash-485105-k5"
LOCATION = "us-central1"

# ============================================
# 설정
# ============================================
MODEL_ID = "gemini-2.0-flash"  # gemini-2.5-flash causes 404 due to library alias issue
IMAGE_PATH = "../minigrid_debug.png"

# Temperature 범위
TEMPERATURES = np.arange(0.2, 1.55, 0.15).round(2).tolist()  # [0.2, 0.35, ..., 1.5]

# Language Prompts
PROMPTS = {
    "uncertain": "move toward toilet, usually colored room is toilet.",
    "certain": "i'm hungry. grab some apple",
    "strange": "find and move toward desktop pc"
}

N_RUNS = 5  # 각 조합당 반복 횟수

# 결과 저장 경로
RESULTS_DIR = "./results"
LOG_FILE = "./analysis_log.json"

# ============================================
# System Prompt (노트북과 동일)
# ============================================
SYSTEM_PROMPT_STEPWISE = """
## ROLE
You are a well-calibrated robot controller that assigns probabilities to actions.

## ACTION SPACE
- "north": Move up (↑)
- "south": Move down (↓)
- "west": Move left (←)
- "east": Move right (→)

## IMAGE ANALYSIS
Analyze the image to identify:
- Robot position (blue agent)
- Target objects (fruits, goals, etc.)
- Obstacles (walls, barriers)
- Open paths

Use this information to inform your probability assignments.

## TASK (Verbalized Confidence)

For each of 3 steps, answer: "What is the probability that each direction is CORRECT?"

## HOW TO THINK

1. Analyze the IMAGE: Where is the robot? Where are targets/obstacles?

2. Interpret the COMMAND: What does the user want?
   - Specific direction ("move east") → prioritize that direction
   - Target-based ("go to apple") → find target in image, move toward it
   - Unknown target → use your knowledge, or spread if truly unidentifiable

3. Combine image + command to assign probabilities:
   - CONFIDENT: concentrate probability (0.7-0.95 on best direction)
   - PARTIAL INFO: moderate concentration (0.4-0.6 on likely directions)
   - UNCERTAIN: spread, but show any slight preferences you have

## EXECUTABILITY
Rate how well you can execute the command (0.0 to 1.0):
- 1.0: Clear target visible, clear path
- 0.5-0.8: Target visible but path unclear, or partial match
- 0.1-0.4: Target not visible but can make educated guess
- 0.0: Cannot determine anything

## OUTPUT FORMAT (STRICT JSON)
{
  "executability": 0.0-1.0,
  "step1": {"north": P, "south": P, "west": P, "east": P},
  "step2": {"north": P, "south": P, "west": P, "east": P},
  "step3": {"north": P, "south": P, "west": P, "east": P},
  "best_guesses": ["direction1", "direction2", "direction3"],
  "reasoning": "Brief explanation (≤50 chars)"
}
"""


# ============================================
# 유틸리티 함수
# ============================================
def calculate_entropy(probs: dict) -> float:
    """Shannon Entropy 계산"""
    values = np.array(list(probs.values()))
    values = values[values > 0]
    if len(values) == 0:
        return 0.0
    return -np.sum(values * np.log2(values))


def normalize_probs(probs: dict) -> dict:
    """확률 정규화"""
    total = sum(probs.values())
    if total == 0:
        return {k: 0.25 for k in probs}
    return {k: v / total for k, v in probs.items()}


def run_single_experiment(
    client,
    user_command: str,
    image: Image.Image,
    temperature: float
) -> dict:
    """단일 실험 실행"""
    
    formatted_prompt = f"""## USER COMMAND:
{user_command}

Assign probability distribution for each of 3 steps."""

    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[formatted_prompt, image],
            config=genai.types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT_STEPWISE,
                response_mime_type="application/json",
                temperature=temperature,
                thinking_config=genai.types.ThinkingConfig(thinking_budget=0)
            )
        )
        
        raw_output = response.candidates[0].content.parts[0].text.strip()
        result = json.loads(raw_output)
        
        # Entropy 계산
        steps = {}
        entropies = []
        
        for i in range(1, 4):
            step_key = f"step{i}"
            if step_key in result:
                step_probs = result[step_key]
                prob_sum = sum(step_probs.values())
                if abs(prob_sum - 1.0) > 0.01:
                    step_probs = normalize_probs(step_probs)
                steps[step_key] = step_probs
                entropies.append(calculate_entropy(step_probs))
            else:
                steps[step_key] = {"north": 0.25, "south": 0.25, "west": 0.25, "east": 0.25}
                entropies.append(2.0)
        
        # 가중 평균 (50/30/20)
        weights = [0.5, 0.3, 0.2]
        weighted_entropy = np.average(entropies, weights=weights)
        
        return {
            'success': True,
            'executability': result.get('executability', 0.5),
            'steps': steps,
            'entropies': entropies,
            'weighted_entropy': weighted_entropy,
            'best_guesses': result.get('best_guesses', []),
            'reasoning': result.get('reasoning', ''),
            'raw': raw_output
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'entropies': [2.0, 2.0, 2.0],
            'weighted_entropy': 2.0
        }


def run_experiment_batch(
    client,
    prompt_name: str,
    prompt_text: str,
    image: Image.Image,
    temperature: float,
    n_runs: int
) -> dict:
    """배치 실험 실행"""
    
    results = []
    entropies = []
    executabilities = []
    best_guesses_count = {"north": 0, "south": 0, "west": 0, "east": 0}
    
    for run_idx in range(n_runs):
        result = run_single_experiment(client, prompt_text, image, temperature)
        results.append(result)
        
        if result['success']:
            entropies.append(result['weighted_entropy'])
            executabilities.append(result['executability'])
            
            # Best guess 카운트
            for guess in result.get('best_guesses', []):
                if guess in best_guesses_count:
                    best_guesses_count[guess] += 1
    
    # 통계 계산
    if entropies:
        entropy_mean = np.mean(entropies)
        entropy_std = np.std(entropies)
        entropy_min = np.min(entropies)
        entropy_max = np.max(entropies)
    else:
        entropy_mean = entropy_std = entropy_min = entropy_max = 2.0
    
    if executabilities:
        exec_mean = np.mean(executabilities)
        exec_std = np.std(executabilities)
    else:
        exec_mean = exec_std = 0.5
    
    # Best guess 비율
    total_guesses = sum(best_guesses_count.values())
    if total_guesses > 0:
        best_guess_ratio = {k: v / total_guesses for k, v in best_guesses_count.items()}
    else:
        best_guess_ratio = {k: 0.25 for k in best_guesses_count}
    
    return {
        'prompt_name': prompt_name,
        'prompt_text': prompt_text,
        'temperature': temperature,
        'n_runs': n_runs,
        'success_count': len(entropies),
        'entropy_mean': entropy_mean,
        'entropy_std': entropy_std,
        'entropy_min': entropy_min,
        'entropy_max': entropy_max,
        'exec_mean': exec_mean,
        'exec_std': exec_std,
        'best_guess_ratio': best_guess_ratio,
        'raw_results': results
    }


def save_results(all_results: List[dict], output_path: str):
    """결과를 JSON으로 저장"""
    # numpy 타입 변환
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj
    
    converted = convert_numpy(all_results)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] 결과 저장: {output_path}")


def create_analysis_plots(all_results: List[dict], output_dir: str):
    """분석 플롯 생성"""
    
    # 데이터 정리 (pandas 없이)
    data_by_prompt = {p: {'temps': [], 'entropy_means': [], 'entropy_stds': [], 
                          'exec_means': [], 'best_east': [], 'best_north': [],
                          'best_south': [], 'best_west': []} 
                      for p in PROMPTS.keys()}
    
    for r in all_results:
        p = r['prompt_name']
        data_by_prompt[p]['temps'].append(r['temperature'])
        data_by_prompt[p]['entropy_means'].append(r['entropy_mean'])
        data_by_prompt[p]['entropy_stds'].append(r['entropy_std'])
        data_by_prompt[p]['exec_means'].append(r['exec_mean'])
        data_by_prompt[p]['best_east'].append(r['best_guess_ratio'].get('east', 0))
        data_by_prompt[p]['best_north'].append(r['best_guess_ratio'].get('north', 0))
        data_by_prompt[p]['best_south'].append(r['best_guess_ratio'].get('south', 0))
        data_by_prompt[p]['best_west'].append(r['best_guess_ratio'].get('west', 0))
    
    # ============================================
    # Plot 1: Temperature vs Entropy (by prompt)
    # ============================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Temperature vs Entropy Analysis', fontsize=16, fontweight='bold')
    
    # 1-1: Entropy Mean by Temperature
    ax1 = axes[0, 0]
    for prompt_name in PROMPTS.keys():
        d = data_by_prompt[prompt_name]
        ax1.errorbar(
            d['temps'], 
            d['entropy_means'],
            yerr=d['entropy_stds'],
            label=prompt_name,
            marker='o',
            capsize=3
        )
    ax1.set_xlabel('Temperature')
    ax1.set_ylabel('Entropy (bits)')
    ax1.set_title('Entropy Mean ± STD vs Temperature')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=2.0, color='gray', linestyle='--', alpha=0.5)
    
    # 1-2: Entropy STD by Temperature
    ax2 = axes[0, 1]
    for prompt_name in PROMPTS.keys():
        d = data_by_prompt[prompt_name]
        ax2.plot(d['temps'], d['entropy_stds'], label=prompt_name, marker='s')
    ax2.set_xlabel('Temperature')
    ax2.set_ylabel('Entropy STD')
    ax2.set_title('Entropy Variance vs Temperature')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 1-3: Executability by Temperature
    ax3 = axes[1, 0]
    for prompt_name in PROMPTS.keys():
        d = data_by_prompt[prompt_name]
        ax3.plot(d['temps'], d['exec_means'], label=prompt_name, marker='^')
    ax3.set_xlabel('Temperature')
    ax3.set_ylabel('Executability')
    ax3.set_title('Executability vs Temperature')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 1-4: East bias by Temperature
    ax4 = axes[1, 1]
    for prompt_name in PROMPTS.keys():
        d = data_by_prompt[prompt_name]
        ax4.plot(d['temps'], d['best_east'], label=prompt_name, marker='d')
    ax4.set_xlabel('Temperature')
    ax4.set_ylabel('East Selection Ratio')
    ax4.set_title('East Direction Bias vs Temperature')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temperature_entropy_analysis.png'), dpi=150)
    plt.close()
    
    # ============================================
    # Plot 2: Direction Distribution Heatmap
    # ============================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Direction Distribution by Temperature', fontsize=14, fontweight='bold')
    
    for idx, prompt_name in enumerate(PROMPTS.keys()):
        ax = axes[idx]
        d = data_by_prompt[prompt_name]
        
        # 온도순 정렬
        sorted_indices = np.argsort(d['temps'])
        sorted_temps = [d['temps'][i] for i in sorted_indices]
        
        heatmap_data = np.array([
            [d['best_north'][i] for i in sorted_indices],
            [d['best_south'][i] for i in sorted_indices],
            [d['best_west'][i] for i in sorted_indices],
            [d['best_east'][i] for i in sorted_indices]
        ])
        
        im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        ax.set_yticks(range(4))
        ax.set_yticklabels(['North', 'South', 'West', 'East'])
        ax.set_xticks(range(len(sorted_temps)))
        ax.set_xticklabels([f'{t:.2f}' for t in sorted_temps], rotation=45)
        ax.set_xlabel('Temperature')
        ax.set_title(f'{prompt_name}')
        
        # 값 표시
        for i in range(4):
            for j in range(len(sorted_temps)):
                val = heatmap_data[i, j]
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)
    
    plt.colorbar(im, ax=axes, label='Selection Ratio', shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'direction_distribution_heatmap.png'), dpi=150)
    plt.close()
    
    # ============================================
    # Plot 3: Entropy Distribution Box Plot
    # ============================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Entropy Distribution by Prompt Type', fontsize=14, fontweight='bold')
    
    for idx, prompt_name in enumerate(PROMPTS.keys()):
        ax = axes[idx]
        prompt_results = [r for r in all_results if r['prompt_name'] == prompt_name]
        
        # 각 temperature별 entropy 값 수집
        temp_entropies = {}
        for r in prompt_results:
            temp = r['temperature']
            raw_entropies = [rr['weighted_entropy'] for rr in r['raw_results'] if rr['success']]
            temp_entropies[temp] = raw_entropies
        
        # Box plot
        positions = list(range(len(TEMPERATURES)))
        bp = ax.boxplot(
            [temp_entropies.get(t, []) for t in TEMPERATURES],
            positions=positions,
            patch_artist=True
        )
        
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        ax.set_xticks(positions)
        ax.set_xticklabels([f'{t:.2f}' for t in TEMPERATURES], rotation=45)
        ax.set_xlabel('Temperature')
        ax.set_ylabel('Entropy (bits)')
        ax.set_title(f'{prompt_name}')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'entropy_boxplot_by_temperature.png'), dpi=150)
    plt.close()
    
    print(f"[INFO] 플롯 저장 완료: {output_dir}")


def main():
    """메인 실행 함수"""
    
    print("=" * 70)
    print("Temperature vs Entropy Analysis")
    print("=" * 70)
    print(f"Temperature range: {TEMPERATURES}")
    print(f"Prompts: {list(PROMPTS.keys())}")
    print(f"Runs per combination: {N_RUNS}")
    print(f"Total experiments: {len(TEMPERATURES) * len(PROMPTS) * N_RUNS}")
    print("=" * 70)
    
    # 결과 폴더 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(RESULTS_DIR, f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Gemini 클라이언트 초기화 (VertexAI)
    scopes = ['https://www.googleapis.com/auth/cloud-platform']
    credentials = service_account.Credentials.from_service_account_file(
        KEY_PATH, scopes=scopes
    )
    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
        credentials=credentials
    )
    
    # 이미지 로드
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, IMAGE_PATH)
    image = Image.open(image_path)
    print(f"[INFO] 이미지 로드: {image_path}")
    
    # 전체 결과 저장
    all_results = []
    
    # 실험 실행
    total_combinations = len(TEMPERATURES) * len(PROMPTS)
    current = 0
    
    for temp in TEMPERATURES:
        for prompt_name, prompt_text in PROMPTS.items():
            current += 1
            print(f"\n[{current}/{total_combinations}] Temperature={temp}, Prompt={prompt_name}")
            
            batch_result = run_experiment_batch(
                client=client,
                prompt_name=prompt_name,
                prompt_text=prompt_text,
                image=image,
                temperature=temp,
                n_runs=N_RUNS
            )
            
            all_results.append(batch_result)
            
            # 중간 결과 출력
            print(f"   Entropy: {batch_result['entropy_mean']:.4f} ± {batch_result['entropy_std']:.4f}")
            print(f"   Exec: {batch_result['exec_mean']:.2f}")
            print(f"   Best guesses: {batch_result['best_guess_ratio']}")
    
    # 결과 저장
    save_results(all_results, os.path.join(results_dir, 'raw_results.json'))
    
    # 플롯 생성
    create_analysis_plots(all_results, results_dir)
    
    # 요약 출력
    print("\n" + "=" * 70)
    print("실험 완료!")
    print("=" * 70)
    print(f"결과 저장 위치: {results_dir}")
    
    # 요약 테이블
    print("\n📊 요약 테이블")
    print("-" * 70)
    print(f"{'Prompt':<12} {'Temp':<6} {'Entropy':<20} {'Exec':<8} {'East Bias':<10}")
    print("-" * 70)
    
    for r in all_results:
        print(f"{r['prompt_name']:<12} {r['temperature']:<6.2f} "
              f"{r['entropy_mean']:.4f} ± {r['entropy_std']:.4f}   "
              f"{r['exec_mean']:<8.2f} {r['best_guess_ratio'].get('east', 0):<10.2f}")
    
    return all_results


if __name__ == "__main__":
    results = main()
