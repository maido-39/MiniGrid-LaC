#!/usr/bin/env python3
"""
Step별 Action Logprobs에서 Shannon Entropy를 계산하고 시각화하는 스크립트

사용법:
    python analyze_step_entropy.py <experiment_log.json 경로>
    
예시:
    python analyze_step_entropy.py "logs_good/EPISODE 1 - scenario2_absolute_example_map_20260123_173946/experiment_log.json"
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from typing import List, Dict, Tuple, Optional


def parse_logprob_string(logprob_str: str) -> float:
    """
    "east:-0.0000" 형식의 문자열에서 logprob 값을 추출
    
    Args:
        logprob_str: "token:logprob" 형식의 문자열
        
    Returns:
        logprob 값 (float)
    """
    # "token:logprob" 형식에서 logprob 추출
    parts = logprob_str.rsplit(':', 1)
    if len(parts) == 2:
        try:
            return float(parts[1])
        except ValueError:
            return None
    return None


def calculate_shannon_entropy(logprobs: List[float]) -> float:
    """
    Logprobs 리스트에서 Shannon Entropy를 계산
    
    Args:
        logprobs: log probability 값들의 리스트
        
    Returns:
        Shannon Entropy (bits)
    """
    if not logprobs or len(logprobs) == 0:
        return 0.0
    
    # logprob를 확률로 변환
    # p(x) = exp(logprob)
    probs = np.exp(logprobs)
    
    # 정규화 (확률의 합이 1이 되도록)
    probs = probs / probs.sum()
    
    # Shannon Entropy 계산: H(X) = -Σ p(x) * log2(p(x))
    # 0인 확률은 제외 (log2(0)은 정의되지 않음)
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * np.log2(p)
    
    return entropy


def extract_first_action_logprobs(step_data: Dict) -> Optional[List[float]]:
    """
    Step 데이터에서 첫 번째 action_logprobs의 logprobs 값을 추출
    
    Args:
        step_data: step의 JSON 데이터
        
    Returns:
        logprob 값들의 리스트, 없으면 None
    """
    try:
        # vlm_response.action_logprobs_info.action_logprobs[0] 접근
        vlm_response = step_data.get('vlm_response', {})
        action_logprobs_info = vlm_response.get('action_logprobs_info', {})
        action_logprobs = action_logprobs_info.get('action_logprobs', [])
        
        if not action_logprobs or len(action_logprobs) == 0:
            return None
        
        # 첫 번째 action_logprobs 항목 가져오기
        first_action = action_logprobs[0]
        
        # 구조: [token_str, top_logprobs_list, entropy, position]
        if len(first_action) < 2:
            return None
        
        top_logprobs_list = first_action[1]
        if not isinstance(top_logprobs_list, list):
            return None
        
        # 각 logprob 문자열에서 값 추출
        logprobs = []
        for logprob_str in top_logprobs_list:
            logprob_val = parse_logprob_string(logprob_str)
            if logprob_val is not None:
                logprobs.append(logprob_val)
        
        return logprobs if logprobs else None
        
    except (KeyError, IndexError, TypeError) as e:
        return None


def analyze_experiment_log(json_path: str) -> Tuple[List[int], List[float]]:
    """
    Experiment log JSON 파일을 분석하여 각 step의 entropy를 계산
    
    Args:
        json_path: experiment_log.json 파일 경로
        
    Returns:
        (steps, entropies) 튜플
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {json_path}")
    
    print(f"JSON 파일 로딩 중: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("JSON 데이터는 리스트 형식이어야 합니다.")
    
    steps = []
    entropies = []
    
    print(f"총 {len(data)}개의 step을 분석 중...")
    
    for step_data in data:
        step_num = step_data.get('step')
        if step_num is None:
            continue
        
        # 첫 번째 action_logprobs의 logprobs 추출
        logprobs = extract_first_action_logprobs(step_data)
        
        if logprobs is None:
            print(f"  Step {step_num}: logprobs 데이터 없음, 건너뜀")
            continue
        
        # Shannon Entropy 계산
        entropy = calculate_shannon_entropy(logprobs)
        
        steps.append(step_num)
        entropies.append(entropy)
        
        print(f"  Step {step_num}: H(Step{step_num}) = {entropy:.6f}")
    
    return steps, entropies


def visualize_entropy(steps: List[int], entropies: List[float], output_path: Optional[str] = None):
    """
    Step별 Entropy를 시각화
    
    Args:
        steps: step 번호 리스트
        entropies: entropy 값 리스트
        output_path: 출력 파일 경로 (None이면 표시만)
    """
    plt.figure(figsize=(12, 6))
    plt.plot(steps, entropies, marker='o', linestyle='-', markersize=3, linewidth=1)
    plt.xlabel('Step #', fontsize=12)
    plt.ylabel('Shannon Entropy H(Step#)', fontsize=12)
    plt.title('Step별 Action Logprobs Shannon Entropy', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n시각화 결과 저장: {output_path}")
    else:
        plt.show()


def main():
    if len(sys.argv) < 2:
        print("사용법: python analyze_step_entropy.py <experiment_log.json 경로>")
        sys.exit(1)
    
    json_path = sys.argv[1]
    
    try:
        # 분석 수행
        steps, entropies = analyze_experiment_log(json_path)
        
        if len(steps) == 0:
            print("분석할 데이터가 없습니다.")
            return
        
        print(f"\n총 {len(steps)}개의 step에서 entropy 계산 완료")
        print(f"Entropy 범위: {min(entropies):.6f} ~ {max(entropies):.6f}")
        print(f"평균 Entropy: {np.mean(entropies):.6f}")
        
        # 결과를 텍스트 파일로 저장
        json_file = Path(json_path)
        output_dir = json_file.parent
        output_txt = output_dir / f"step_entropy_{json_file.stem}.txt"
        
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write("Step #\tH(Step#)\n")
            for step, entropy in zip(steps, entropies):
                f.write(f"{step}\t{entropy:.6f}\n")
        
        print(f"\n결과 저장: {output_txt}")
        
        # 시각화
        output_png = output_dir / f"step_entropy_{json_file.stem}.png"
        visualize_entropy(steps, entropies, str(output_png))
        
    except Exception as e:
        print(f"오류 발생: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
