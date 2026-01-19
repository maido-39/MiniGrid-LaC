"""
SayCan-style Action Entropy Calculation for MiniGrid

This script implements SayCan-style action selection with probability distribution
and entropy calculation using few-shot examples and logprobs.

SayCan 방식:
- Few-shot 예시 기반 프롬프트 (시스템 프롬프트 없이)
- 각 action candidate에 대해 "I would 1. [action], 2. done." 형태로 평가
- logprobs를 사용하여 각 action의 확률 계산
- Usefulness: LLM의 logprobs로 계산
- Feasibility: 간단한 affordance 체크
- 최종 확률: P(action) = usefulness × feasibility

Based on:
- Original: VLM_interact_minigrid-absolute_emoji.py
- SayCan paper: https://say-can.github.io/

Usage:
    # Run from src/ directory
    cd src/
    python test_script/action_entropy/saycan_action_entropy.py [json_map_path]
"""

import sys
from pathlib import Path

# Add src directory to Python path
script_dir = Path(__file__).parent.resolve()
src_dir = script_dir.parent.parent  # Go up from test_script/action_entropy/ to src/
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from minigrid import register_minigrid_envs
from utils import MiniGridEmojiWrapper, load_emoji_map_from_json
import numpy as np
import cv2  # type: ignore
from typing import Dict, List, Tuple, Optional
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import csv
from datetime import datetime
import re

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("openai 라이브러리가 필요합니다: pip install openai")

# Register MiniGrid environments
register_minigrid_envs()

# VLM configuration
VLM_MODEL = "gpt-4o"
VLM_TEMPERATURE = 0.0
VLM_MAX_TOKENS = 2000

# Display configuration
DISPLAY_MAX_SIZE = 1000

load_dotenv()


class AbsoluteDirectionEmojiWrapper(MiniGridEmojiWrapper):
    """Emoji Wrapper supporting absolute direction movement"""
    
    def __init__(self, *args, **kwargs):
        kwargs['use_absolute_movement'] = True
        super().__init__(*args, **kwargs)


def get_system_prompt() -> str:
    """Generate System Prompt (absolute coordinate version - emoji environment)"""
    return """You are a robot operating on a grid map.

## Environment
Grid world with walls which must step on and detour (black, brick emoji 🧱)
robot is represented as emoji 🤖, which is your avatar.
you can move in ANY direction regardless of the robot's current heading.

## Action Space, Coordinate System
**CRITICAL**: All movements are in ABSOLUTE directions (UP/DOWN/LEFT/RIGHT).
- "move up": Move UP (upward on the image)
- "move down": Move DOWN (downward on the image)
- "move left": Move LEFT (leftward on the image)
- "move right": Move RIGHT (rightward on the image)

## Movement Rules
- You cannot step on walls (black/brick emoji 🧱), you should detour around them.
- When you step on an emoji object, the block will glow green
- Use absolute directions (up/down/left/right) regardless of robot's current heading
"""


def get_saycan_few_shot_examples() -> str:
    """Generate few-shot examples for SayCan-style prompting (based on example_map.json)"""
    return """User: How would you go to the kitchen area?
Robot: I would 1. move up, 2. move up, 3. move left, 4. done.

User: How would you go to the dining area?
Robot: I would 1. move up, 2. move up, 3. move right, 4. move right, 5. done.

User: How would you go to the restroom?
Robot: I would 1. move down, 2. move right, 3. move right, 4. move right, 5. done.

User: How would you go to the storage area?
Robot: I would 1. move up, 2. move up, 3. move up, 4. move right, 5. move right, 6. done.

User: How would you go to the preparation area?
Robot: I would 1. move up, 2. move up, 3. move right, 4. done.

User: How would you go around obstacles to reach a destination?
Robot: I would 1. move right, 2. move up, 3. move left, 4. move up, 5. done.
"""


def build_saycan_prompt(
    user_command: str,
    action_candidate: str,
    few_shot_examples: str
) -> str:
    """Build SayCan-style prompt for a single action candidate"""
    prompt = few_shot_examples
    prompt += f"\nUser: {user_command}\n"
    prompt += f"Robot: I would 1. {action_candidate}, 2. done."
    return prompt


def get_all_action_usefulness_with_logprobs(
    client: OpenAI,
    image: np.ndarray,
    user_command: str,
    action_candidates: List[str],
    few_shot_examples: str,
    model: str = VLM_MODEL,
    temperature: float = VLM_TEMPERATURE
) -> Tuple[Dict[str, float], Optional[Dict]]:
    """
    Get usefulness scores for all actions using single API call with top_logprobs
    
    프롬프트를 "I would 1. " 까지 생성하고, 다음 토큰의 top_logprobs에서
    각 action candidate의 확률을 추출합니다.
    
    Returns:
        (usefulness_scores_dict, logprobs_info)
        usefulness_scores_dict: {action: usefulness_score} 형태의 딕셔너리
        logprobs_info: logprobs 정보 (디버깅용)
    """
    from utils.vlm.handlers.openai_handler import OpenAIHandler
    
    # Encode image
    handler = OpenAIHandler()
    image_b64 = handler.encode_image(image)
    
    # Build prompt: few-shot examples + user command + "I would 1. " 까지
    prompt = few_shot_examples
    prompt += f"\nUser: {user_command}\n"
    prompt += "Robot: I would 1. "
    
    # Build messages with system prompt (environment understanding) + few-shot examples
    messages = [
        {
            "role": "system",
            "content": get_system_prompt()
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                }
            ]
        }
    ]
    
    try:
        # top_logprobs를 최대값(5)으로 설정하여 많은 후보 토큰 확인
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=1,  # 다음 토큰 하나만 생성
            logprobs=True,
            top_logprobs=5  # 최대 5개의 top 후보 확인
        )
        
        choice = response.choices[0]
        logprobs_info = None
        usefulness_scores = {}
        
        # Extract logprob for each action candidate
        if hasattr(choice, 'logprobs') and choice.logprobs and choice.logprobs.content:
            logprobs_info = {
                'content': choice.logprobs.content,
                'finish_reason': choice.finish_reason
            }
            
            # 첫 번째 토큰의 top_logprobs에서 action candidate 찾기
            if len(choice.logprobs.content) > 0:
                first_token_info = choice.logprobs.content[0]
                
                # Debug: Print all top logprobs
                print(f"\n[Debug] Top logprobs for first token:")
                print(f"  Main token: '{first_token_info.token}' (logprob: {first_token_info.logprob:.4f})")
                if hasattr(first_token_info, 'top_logprobs') and first_token_info.top_logprobs:
                    for idx, top_token_info in enumerate(first_token_info.top_logprobs):
                        print(f"  Top {idx+1}: '{top_token_info.token}' (logprob: {top_token_info.logprob:.4f})")
                
                # 각 action candidate에 대해 매칭되는 토큰 찾기
                for action_candidate in action_candidates:
                    action_words = action_candidate.lower().split()  # ['move', 'right']
                    max_logprob = -float('inf')
                    matched_token = None
                    
                    # Check main token - 정확한 매칭 우선
                    main_token = first_token_info.token.lower().strip()
                    main_token_clean = main_token.strip('.,!?;:')
                    
                    # 정확한 단어 매칭 시도
                    for word in action_words:
                        if word == main_token_clean or main_token_clean == word:
                            max_logprob = max(max_logprob, first_token_info.logprob)
                            matched_token = main_token
                            break
                    
                    # Check top_logprobs - 정확한 매칭 우선
                    if hasattr(first_token_info, 'top_logprobs') and first_token_info.top_logprobs:
                        for top_token_info in first_token_info.top_logprobs:
                            top_token = top_token_info.token.lower().strip()
                            top_token_clean = top_token.strip('.,!?;:')
                            
                            # 정확한 단어 매칭 시도
                            for word in action_words:
                                if word == top_token_clean:
                                    max_logprob = max(max_logprob, top_token_info.logprob)
                                    if matched_token is None or top_token_info.logprob > max_logprob:
                                        matched_token = top_token
                                    break
                    
                    # 정확한 매칭이 없으면 부분 매칭 시도 (하지만 낮은 가중치)
                    if max_logprob == -float('inf'):
                        for word in action_words:
                            if word in main_token_clean or main_token_clean in word:
                                max_logprob = max(max_logprob, first_token_info.logprob - 2.0)  # 패널티
                                matched_token = main_token
                                break
                        
                        if hasattr(first_token_info, 'top_logprobs') and first_token_info.top_logprobs:
                            for top_token_info in first_token_info.top_logprobs:
                                top_token_clean = top_token_info.token.lower().strip('.,!?;:')
                                for word in action_words:
                                    if word in top_token_clean or top_token_clean in word:
                                        max_logprob = max(max_logprob, top_token_info.logprob - 2.0)  # 패널티
                                        if matched_token is None:
                                            matched_token = top_token_clean
                                        break
                    
                    # Convert logprob to usefulness score
                    if max_logprob != -float('inf'):
                        usefulness_scores[action_candidate] = max_logprob
                        print(f"  Action '{action_candidate}': logprob={max_logprob:.4f} (matched: '{matched_token}')")
                    else:
                        # If not found, use a very low score
                        usefulness_scores[action_candidate] = -10.0
                        print(f"  Action '{action_candidate}': NOT FOUND (using -10.0)")
                
                # Normalize all scores using softmax
                if usefulness_scores:
                    # Convert logprobs to probabilities using softmax
                    logprobs_array = np.array(list(usefulness_scores.values()))
                    # Shift to avoid overflow
                    logprobs_array = logprobs_array - np.max(logprobs_array)
                    probs = np.exp(logprobs_array)
                    probs = probs / probs.sum()
                    
                    # Convert back to usefulness scores (0-1 range)
                    for i, action_candidate in enumerate(action_candidates):
                        usefulness_scores[action_candidate] = float(probs[i])
            else:
                # Fallback: uniform distribution
                for action_candidate in action_candidates:
                    usefulness_scores[action_candidate] = 1.0 / len(action_candidates)
        else:
            # Fallback: uniform distribution
            for action_candidate in action_candidates:
                usefulness_scores[action_candidate] = 1.0 / len(action_candidates)
        
        return usefulness_scores, logprobs_info
        
    except Exception as e:
        print(f"Error getting logprobs for actions: {e}")
        # Fallback: uniform distribution
        return {action: 1.0 / len(action_candidates) for action in action_candidates}, None


def get_action_feasibility(
    wrapper: AbsoluteDirectionEmojiWrapper,
    action_idx: int,
    state: Dict
) -> float:
    """
    Get feasibility score for an action (간단한 affordance 체크)
    
    Returns:
        feasibility_score: 0.0 to 1.0
    """
    action_space = wrapper.get_absolute_action_space()
    action_name = action_space['action_mapping'].get(action_idx, "")
    
    # 간단한 feasibility 체크
    agent_pos = state['agent_pos']
    if isinstance(agent_pos, np.ndarray):
        agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
    else:
        agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
    
    size = wrapper.size
    
    # Movement actions feasibility check
    if action_name == "move up":
        # Check if can move up (not at top boundary)
        if agent_y <= 0:
            return 0.0
        # Check if next cell is wall
        next_cell = wrapper.env.grid.get(agent_x, agent_y - 1)
        if next_cell is not None and next_cell.type == 'wall':
            return 0.0
        return 1.0
    elif action_name == "move down":
        if agent_y >= size - 1:
            return 0.0
        next_cell = wrapper.env.grid.get(agent_x, agent_y + 1)
        if next_cell is not None and next_cell.type == 'wall':
            return 0.0
        return 1.0
    elif action_name == "move left":
        if agent_x <= 0:
            return 0.0
        next_cell = wrapper.env.grid.get(agent_x - 1, agent_y)
        if next_cell is not None and next_cell.type == 'wall':
            return 0.0
        return 1.0
    elif action_name == "move right":
        if agent_x >= size - 1:
            return 0.0
        next_cell = wrapper.env.grid.get(agent_x + 1, agent_y)
        if next_cell is not None and next_cell.type == 'wall':
            return 0.0
        return 1.0
    elif action_name in ["pickup", "drop", "toggle"]:
        # For now, assume these are always feasible
        # Could add more sophisticated checks
        return 1.0
    else:
        return 0.5  # Unknown action


def create_scenario2_environment():
    """Create Scenario 2 environment (emoji version)"""
    size = 10
    
    # Create outer walls
    walls = []
    for i in range(size):
        walls.append((i, 0))
        walls.append((i, size-1))
        walls.append((0, i))
        walls.append((size-1, i))
    
    # Blue pillar: 2x2 Grid -> 🧱(brick) emoji
    blue_pillar_positions = [(3, 4), (4, 4), (3, 5), (4, 5)]
    
    # Table: Purple 1x2 Grid -> 🖥️📱
    # table_positions = [(5, 1), (6, 1)]  # Not used, kept for reference
    
    # Start and goal positions
    start_pos = (1, 8)
    goal_pos = (8, 1)
    
    # Create emoji objects
    objects = []
    
    # 🧱(brick) emoji: blue, can step on
    for pos in blue_pillar_positions:
        objects.append({
            'type': 'emoji',
            'pos': pos,
            'emoji_name': 'brick',
            'color': 'blue',
            'can_pickup': False,
            'can_overlap': False,
            'use_emoji_color': False
        })
    
    # 🖥️📱(desktop/workstation) emoji: purple, can step on
    objects.append({
        'type': 'emoji',
        'pos': (5, 1),
        'emoji_name': 'desktop',
        'color': 'purple',
        'can_pickup': False,
        'can_overlap': True,
        'use_emoji_color': True
    })
    
    objects.append({
        'type': 'emoji',
        'pos': (6, 1),
        'emoji_name': 'workstation',
        'color': 'purple',
        'can_pickup': False,
        'can_overlap': True,
        'use_emoji_color': False
    })
    
    room_config = {
        'start_pos': start_pos,
        'goal_pos': goal_pos,
        'walls': walls,
        'objects': objects,
        'use_robot_emoji': True,
        'robot_emoji_color': 'red',
        'use_robot_emoji_color': True
    }
    
    return AbsoluteDirectionEmojiWrapper(size=size, room_config=room_config)




def compute_probability_distribution(
    usefulness_scores: List[float],
    feasibility_scores: List[float],
    action_candidates: List[str]
) -> np.ndarray:
    """Compute final probability distribution: P(action) = usefulness * feasibility"""
    n_actions = len(action_candidates)
    probs = np.zeros(n_actions)
    
    for i in range(n_actions):
        usefulness = usefulness_scores[i] if i < len(usefulness_scores) else 0.0
        feasibility = feasibility_scores[i] if i < len(feasibility_scores) else 0.0
        probs[i] = usefulness * feasibility
    
    # Normalize to get probability distribution
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        # Uniform distribution if all zeros
        probs = np.ones(n_actions) / n_actions
    
    return probs


def calculate_entropy(probs: np.ndarray) -> float:
    """Calculate Shannon entropy of probability distribution"""
    # Remove zeros to avoid log(0)
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))


def visualize_action_distribution(
    action_candidates: List[str],
    usefulness_scores: List[float],
    feasibility_scores: List[float],
    probs: np.ndarray,
    entropy_value: float,
    step: int,
    save_dir: str = "test_script/action_entropy/plots"
) -> None:
    """Visualize action probability distribution - 간단하고 효율적인 단일 그래프"""
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Create single figure
    _, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data
    x = np.arange(len(action_candidates))
    action_labels = [a.replace('move ', '') for a in action_candidates]
    selected_idx = np.argmax(probs)
    
    # Create grouped bar chart
    width = 0.25
    x1 = x - width
    x2 = x
    x3 = x + width
    
    bars1 = ax.bar(x1, usefulness_scores, width, label='Usefulness', alpha=0.8, color='#3498db')
    bars2 = ax.bar(x2, probs, width, label='Final P(action)', alpha=0.8, color='#2ecc71')
    bars3 = ax.bar(x3, feasibility_scores, width, label='Feasibility', alpha=0.8, color='#e74c3c')
    
    # Highlight selected action
    bars2[selected_idx].set_edgecolor('gold')
    bars2[selected_idx].set_linewidth(3)
    bars2[selected_idx].set_alpha(1.0)
    
    # Add value labels
    for i, (bar1, bar2, bar3) in enumerate(zip(bars1, bars2, bars3)):
        # Usefulness
        if usefulness_scores[i] > 0.05:
            ax.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height(),
                   f'{usefulness_scores[i]:.2f}', ha='center', va='bottom', fontsize=8)
        # Final probability
        if probs[i] > 0.05:
            ax.text(bar2.get_x() + bar2.get_width()/2., bar2.get_height(),
                   f'{probs[i]:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        # Feasibility
        if feasibility_scores[i] > 0.05:
            ax.text(bar3.get_x() + bar3.get_width()/2., bar3.get_height(),
                   f'{feasibility_scores[i]:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Formatting
    ax.set_xlabel('Action', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score / Probability', fontsize=12, fontweight='bold')
    ax.set_title(f'Step {step} - Action Distribution (Entropy: {entropy_value:.3f} bits)', 
                fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(action_labels, fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.15])
    
    # Add selected action annotation
    ax.text(0.02, 0.98, f'Selected: {action_labels[selected_idx]} (P={probs[selected_idx]:.3f})',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    # Save figure (lower DPI for efficiency)
    filename = save_path / f'step_{step:03d}_entropy_{entropy_value:.3f}.png'
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    print(f"  Saved plot to: {filename}")
    
    plt.close()


def create_experiment_folder(instruction: str, base_dir: str = "test_script/action_entropy/experiments") -> Path:
    """Create experiment folder with timestamp and instruction"""
    # Create safe folder name from instruction
    safe_instruction = re.sub(r'[^\w\s-]', '', instruction[:50])  # Remove special chars, limit length
    safe_instruction = re.sub(r'[-\s]+', '_', safe_instruction)  # Replace spaces/hyphens with underscore
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{safe_instruction}"
    
    experiment_path = Path(base_dir) / folder_name
    experiment_path.mkdir(parents=True, exist_ok=True)
    
    return experiment_path


def save_step_data_to_csv(
    step_data: List[Dict],
    experiment_path: Path,
    action_candidates: List[str]
) -> None:
    """Save all step data to CSV file"""
    csv_path = experiment_path / "step_data.csv"
    
    # Prepare CSV headers
    headers = ['step', 'timestamp', 'entropy', 'selected_action', 'selected_prob']
    for action in action_candidates:
        headers.extend([
            f'{action}_usefulness',
            f'{action}_feasibility',
            f'{action}_probability'
        ])
    
    # Write CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(step_data)
    
    print(f"\n[CSV] Saved step data to: {csv_path}")


def visualize_temporal_distribution(
    step_data: List[Dict],
    action_candidates: List[str],
    experiment_path: Path,
    instruction: str
) -> None:
    """Visualize probability distribution and entropy over time"""
    steps = [d['step'] for d in step_data]
    entropy_values = [d['entropy'] for d in step_data]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f'Action Distribution Over Time\nInstruction: {instruction[:60]}...', 
                 fontsize=14, fontweight='bold')
    
    # 1. Entropy over time
    ax1 = axes[0]
    ax1.plot(steps, entropy_values, marker='o', linewidth=2, markersize=6, color='#e74c3c')
    ax1.fill_between(steps, entropy_values, alpha=0.3, color='#e74c3c')
    ax1.set_xlabel('Step', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Entropy (bits)', fontsize=12, fontweight='bold')
    ax1.set_title('Entropy Over Time', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.set_ylim([0, max(entropy_values) * 1.2 if entropy_values else 3.0])
    
    # Add max entropy line (log2 of number of actions)
    max_entropy = np.log2(len(action_candidates))
    ax1.axhline(y=max_entropy, color='gray', linestyle=':', linewidth=2, 
                label=f'Max Entropy (log2({len(action_candidates)}) = {max_entropy:.2f})')
    ax1.legend()
    
    # 2. Probability distribution over time (heatmap-like line plot)
    ax2 = axes[1]
    
    # Plot probability for each action over time
    colors = plt.cm.Set3(np.linspace(0, 1, len(action_candidates)))  # type: ignore
    for i, action in enumerate(action_candidates):
        probs = [d[f'{action}_probability'] for d in step_data]
        ax2.plot(steps, probs, marker='o', linewidth=2, markersize=4, 
                label=action.replace('move ', ''), color=colors[i], alpha=0.8)
    
    ax2.set_xlabel('Step', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax2.set_title('Action Probability Distribution Over Time', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.set_ylim([0, 1.1])
    
    plt.tight_layout()
    
    # Save figure
    plot_path = experiment_path / "temporal_distribution.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"[Plot] Saved temporal distribution to: {plot_path}")
    
    plt.close()
    
    # Also create a stacked area chart for better visualization
    _, ax3 = plt.subplots(figsize=(14, 6))
    
    # Prepare data for stacked area
    prob_matrix = np.array([[d[f'{action}_probability'] for action in action_candidates] 
                           for d in step_data])
    
    # Stacked area plot
    ax3.stackplot(steps, *prob_matrix.T, labels=[a.replace('move ', '') for a in action_candidates],
                  alpha=0.7, colors=colors)
    
    ax3.set_xlabel('Step', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax3.set_title('Stacked Action Probability Distribution Over Time', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(alpha=0.3, linestyle='--')
    ax3.set_ylim([0, 1.0])
    
    plt.tight_layout()
    
    plot_path2 = experiment_path / "temporal_distribution_stacked.png"
    plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
    print(f"[Plot] Saved stacked temporal distribution to: {plot_path2}")
    
    plt.close()


def display_image(img, window_name="SayCan Action Entropy"):
    """Display image using OpenCV"""
    if img is not None:
        try:
            img_bgr = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
            height, width = img_bgr.shape[:2]
            if height < DISPLAY_MAX_SIZE and width < DISPLAY_MAX_SIZE:
                scale = min(DISPLAY_MAX_SIZE // height, DISPLAY_MAX_SIZE // width)
                if scale > 1:
                    new_width = width * scale
                    new_height = height * scale
                    img_bgr = cv2.resize(img_bgr, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            cv2.imshow(window_name, img_bgr)
            cv2.waitKey(1)
        except Exception as e:
            print(f"Image display error: {e}")


def main():
    """Main function"""
    script_dir = Path(__file__).parent.resolve()
    
    # JSON map file path via command line argument
    json_map_path = None
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("Usage:")
            print("  python saycan_action_entropy.py [json_map_path]")
            print("  Example: python saycan_action_entropy.py ./config/example_map.json")
            print("\nIf no JSON path is provided, uses hardcoded scenario2 environment.")
            return
        else:
            user_path = Path(sys.argv[1])
            if user_path.is_absolute():
                json_map_path = str(user_path)
            else:
                script_relative = script_dir / user_path
                cwd_relative = Path.cwd() / user_path
                if script_relative.exists():
                    json_map_path = str(script_relative.resolve())
                elif cwd_relative.exists():
                    json_map_path = str(cwd_relative.resolve())
                else:
                    json_map_path = str(user_path)
    
    print("=" * 60)
    print("SayCan-style Action Entropy Calculation")
    print("=" * 60)
    
    # Create environment
    print("\n[1] Creating environment...")
    if json_map_path:
        print(f"  Loading map from: {json_map_path}")
        wrapper = load_emoji_map_from_json(json_map_path)
    else:
        # Default to example_map.json
        default_map_path = script_dir.parent / "config" / "example_map.json"
        if default_map_path.exists():
            print(f"  Using default map: {default_map_path}")
            wrapper = load_emoji_map_from_json(str(default_map_path))
        else:
            print("  Using hardcoded scenario2 environment (example_map.json not found)")
            wrapper = create_scenario2_environment()
    
    wrapper.reset()
    state = wrapper.get_state()
    print(f"Agent start position: {state['agent_pos']}")
    
    # Get action space (movement actions only: no pickup, drop, toggle)
    action_space = wrapper.get_absolute_action_space()
    all_actions = action_space['actions']
    
    # Filter to only movement actions
    movement_actions = ['move up', 'move down', 'move left', 'move right']
    action_candidates = [action for action in all_actions if action in movement_actions]
    
    # Create mapping from filtered action to original action index
    action_mapping = {}
    for i, action in enumerate(all_actions):
        if action in action_candidates:
            action_mapping[action] = i
    
    print(f"\nAll available actions: {all_actions}")
    print(f"Action candidates (movement only): {action_candidates}")
    print(f"Action mapping: {action_mapping}")
    
    # Initialize OpenAI client for logprobs
    print("\n[2] Initializing OpenAI client...")
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        client = OpenAI(api_key=api_key)
        print(f"OpenAI client initialization complete: {VLM_MODEL}")
    except Exception as e:
        print(f"OpenAI client initialization failed: {e}")
        return
    
    # Get mission/instruction from user
    print("\n" + "=" * 60)
    print("Mission/Instruction 입력")
    print("=" * 60)
    
    # Default mission based on example_map.json (goal is at [12, 1])
    default_mission = "Go to the goal location at the top right corner."
    print(f"\n기본 미션: {default_mission}")
    print("\n사용 가능한 지역 예시:")
    print("  - kitchen (🟩), dining (🟦), restroom (🟪), storage (🟣)")
    print("  - preparation (🟨), plating (🟧), water (🟫), broom (⚫)")
    print("\n미션을 입력하세요 (Enter만 누르면 기본 미션 사용):")
    user_input = input("> ").strip()
    
    if user_input:
        mission = user_input
        print(f"\n✓ 사용자 입력 미션: {mission}")
    else:
        mission = default_mission
        print(f"\n✓ 기본 미션 사용: {mission}")
    
    # Few-shot examples
    few_shot_examples = get_saycan_few_shot_examples()
    
    # Create experiment folder
    experiment_path = create_experiment_folder(mission)
    print(f"\n[Experiment] Created folder: {experiment_path}")
    
    # Create plots subdirectory in experiment folder
    plots_path = experiment_path / "plots"
    plots_path.mkdir(exist_ok=True)
    
    # Data collection for CSV
    step_data = []
    
    # Main loop
    step = 0
    done = False
    WINDOW_NAME = "SayCan Action Entropy"
    
    print("\n" + "=" * 60)
    print("Experiment started")
    print(f"Mission: {mission}")
    print(f"Experiment folder: {experiment_path}")
    print("=" * 60)
    
    while not done:
        step += 1
        print("\n" + "=" * 80)
        print(f"STEP {step}")
        print("=" * 80)
        
        # Current state
        image = wrapper.get_image()
        state = wrapper.get_state()
        print(f"Position: {state['agent_pos']}, Direction: {state['agent_dir']}")
        
        display_image(image, WINDOW_NAME)
        
        # Get usefulness scores from LLM using single API call with top_logprobs
        print("\n[3] Getting usefulness scores from LLM (single API call with top_logprobs)...")
        usefulness_scores_dict, _ = get_all_action_usefulness_with_logprobs(
            client, image, mission, action_candidates, few_shot_examples
        )
        
        # Convert dict to list (maintain order)
        usefulness_scores = [usefulness_scores_dict.get(action, 0.0) for action in action_candidates]
        
        print(f"Usefulness scores: {dict(enumerate(usefulness_scores))}")
        
        # Get feasibility scores (affordance)
        print("\n[4] Getting feasibility scores (affordance)...")
        feasibility_scores = []
        for action_name in action_candidates:
            # Get original action index from mapping
            original_idx = action_mapping[action_name]
            feasibility = get_action_feasibility(wrapper, original_idx, state)
            feasibility_scores.append(feasibility)
        
        print(f"Feasibility scores: {dict(enumerate(feasibility_scores))}")
        
        # Compute probability distribution
        print("\n[5] Computing probability distribution...")
        probs = compute_probability_distribution(
            usefulness_scores, feasibility_scores, action_candidates
        )
        
        # Calculate entropy
        entropy_value = calculate_entropy(probs)
        
        # Display results
        print("\n" + "-" * 80)
        print("Action Probability Distribution:")
        print("-" * 80)
        for i, (action, prob) in enumerate(zip(action_candidates, probs)):
            usefulness = usefulness_scores[i] if i < len(usefulness_scores) else 0.0
            feasibility = feasibility_scores[i] if i < len(feasibility_scores) else 0.0
            print(f"  {i}. {action:15s} | Usefulness: {usefulness:.3f} | Feasibility: {feasibility:.3f} | P(action): {prob:.3f}")
        print("-" * 80)
        print(f"Entropy: {entropy_value:.3f} bits")
        print("-" * 80)
        
        # Select action (sample from distribution or take argmax)
        filtered_action_idx = np.argmax(probs)
        
        # Collect step data for CSV (individual step plots removed for efficiency)
        step_record = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'entropy': entropy_value,
            'selected_action': action_candidates[filtered_action_idx],
            'selected_prob': float(probs[filtered_action_idx])
        }
        
        # Add data for each action
        for i, action in enumerate(action_candidates):
            step_record[f'{action}_usefulness'] = usefulness_scores[i]
            step_record[f'{action}_feasibility'] = feasibility_scores[i]
            step_record[f'{action}_probability'] = float(probs[i])
        
        step_data.append(step_record)
        action_name = action_candidates[filtered_action_idx]
        
        # Map filtered action index to original action space index
        original_action_idx = action_mapping[action_name]
        
        print(f"\n[7] Selected action: {action_name} (filtered index: {filtered_action_idx}, original index: {original_action_idx}, probability: {probs[filtered_action_idx]:.3f})")
        
        # Execute action using original action index
        print("[8] Executing action...")
        try:
            _, reward, terminated, truncated, _ = wrapper.step(original_action_idx)
            done = terminated or truncated
            print(f"Reward: {reward}, Done: {done}")
        except Exception as e:
            print(f"Action execution failed: {e}")
            import traceback
            traceback.print_exc()
            break
        
        # Display updated state
        updated_image = wrapper.get_image()
        display_image(updated_image, WINDOW_NAME)
        
        # Check termination
        if done:
            print("\n" + "=" * 80)
            print("Goal reached! Terminating")
            print("=" * 80)
            break
        
        # Maximum step limit
        if step >= 100:
            print("\nMaximum step count (100) reached.")
            break
    
    # Save data and create visualizations
    if step_data:
        print("\n" + "=" * 80)
        print("Saving experiment data...")
        print("=" * 80)
        
        # Save CSV
        save_step_data_to_csv(step_data, experiment_path, action_candidates)
        
        # Create temporal visualizations
        print("\n[Visualization] Creating temporal distribution plots...")
        visualize_temporal_distribution(step_data, action_candidates, experiment_path, mission)
        
        print(f"\n[Experiment] All data saved to: {experiment_path}")
        print(f"  - CSV: step_data.csv")
        print(f"  - Plots: temporal_distribution.png, temporal_distribution_stacked.png")
        print(f"  - Step plots: plots/step_*.png")
    
    # Clean up resources
    cv2.destroyAllWindows()
    wrapper.close()
    print("\nExperiment completed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()

