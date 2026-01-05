"""
시나리오 2 실험 환경 테스트 스크립트 (VLM 제어 버전)

시나리오 2: 파란 기둥으로 가서 오른쪽으로 돌고, 테이블 옆에 멈추시오

환경 구성:
- 벽: 검은색 (외벽)
- 파란 기둥: 파란색 2x2 Grid (통과불가)
- 테이블: 보라색 1x3 Grid (통과불가)
- 시작점: 빨강 1x1
- 종료점: 초록 1x1

레이아웃 (10x10):
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬜️⬜️⬜️⬜️🟪🟪🟪🟩⬛
⬛⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬛
⬛⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬛
⬛⬜️⬜️🟦🟦⬜️⬜️⬜️⬜️⬛ 
⬛⬜️⬜️🟦🟦⬜️⬜️⬜️⬜️⬛
⬛⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬛
⬛⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬛
⬛🟥⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
"""

from minigrid import register_minigrid_envs
from custom_environment import CustomRoomWrapper
from vlm_wrapper import ChatGPT4oVLMWrapper
from vlm_postprocessor import VLMResponsePostProcessor
import json
from datetime import datetime
from pathlib import Path
import csv
from PIL import Image
import numpy as np
import cv2
import hashlib
import time

# MiniGrid 환경 등록
register_minigrid_envs()


# ============================================================================
# VLM 설정
# ============================================================================
# VLM 설정을 여기서 변경할 수 있습니다.
# 
# OpenAI GPT-4 계열 비전 지원 모델:
# - "gpt-4o": 최신 멀티모달 모델 (가장 빠르고 정확, 권장)
# - "gpt-4o-mini": 경량 버전 (저렴하고 빠름)
# - "gpt-4-turbo": 이전 버전 (비전 지원)
# - "gpt-4-vision-preview": 구버전 (deprecated, 사용 비권장)
#
# OpenAI GPT-5 계열 비전 지원 모델 (2025년 출시):
# - "gpt-5": 최신 모델 (비전 지원, 멀티모달 추론 강화)
# - "gpt-5-mini": 경량 버전 (API 사용 가능 시)
#
# 참고: 
# - "gpt-4o-nano"는 존재하지 않는 모델입니다.
# - GPT-5는 2025년 8월 출시되었으나, API 모델명은 OpenAI 공식 문서를 확인하세요.
#   실제 API에서는 "gpt-5" 또는 다른 이름일 수 있습니다.
VLM_MODEL = "gpt-4o"  # 사용할 모델명 (GPT-5 사용 시 "gpt-5"로 변경)
VLM_TEMPERATURE = 0.0  # 생성 온도 (0.0 ~ 2.0)
VLM_MAX_TOKENS = 1000  # 최대 토큰 수

# 액션 예측 개수 설정
ACTION_PREDICTION_COUNT = 5  # VLM이 예측할 액션 개수 (첫 번째만 실행, 나머지는 로깅용)


# ============================================================================
# System Prompt 정의
# ============================================================================
# 이 부분은 환경 설명과 VLM 응답 포맷팅 가이드를 포함합니다.
# 필요에 따라 수정할 수 있습니다.

def get_system_prompt(action_count: int, memory_summary: str = "", grounding_section: str = "") -> str:
    """
    System Prompt 생성 함수 (동적 값 포함)
    
    Args:
        action_count: 예측할 액션 개수
        memory_summary: 이전 행동 요약 (영구 메모리)
        grounding_section: Grounding 지식 섹션 (실수 분석 및 교훈)
    
    Returns:
        System Prompt 문자열
    """
    memory_section = ""
    if memory_summary:
        memory_section = f"""
## Permanent Memory (Current Progress Summary)
{memory_summary}

**Important**: This memory contains a concise summary of what was done in the previous step and current progress toward the mission goal. Use this to understand where the robot is in the mission and plan the next actions accordingly. The memory will be updated after each step with a new summary (not accumulated).
"""
    
    grounding_section_text = ""
    if grounding_section:
        grounding_section_text = f"""
## Grounding Knowledge (Lessons Learned from Mistakes)
{grounding_section}

**Important**: This section contains knowledge learned from previous mistakes. When you made an error, the analysis of why it was wrong and how to avoid it in the future is recorded here. Always refer to this section to avoid repeating the same mistakes.
"""
    
    return f"""You are a robot action planner for object goal navigation.
{memory_section}
{grounding_section_text}

## Environment
Grid world with walls (black), blue pillar (impassable), purple table (impassable), robot (red arrow shows heading), and goal (green marker if present).

## Robot Orientation (CRITICAL)
**IMPORTANT**: The robot is represented as a RED ARROW. The robot's heading (orientation) is determined by the direction the ARROW POINT is pointing.
- The arrow's point (tip) indicates the robot's facing direction
- The arrow is drawn from the center of the robot's cell, pointing in the direction of movement
- When the arrow points RIGHT (→) → heading = 0 (East, facing right)
- When the arrow points DOWN (↓) → heading = 1 (South, facing down)
- When the arrow points LEFT (←) → heading = 2 (West, facing left)
- When the arrow points UP (↑) → heading = 3 (North, facing up)
- Always check the arrow's direction to determine the robot's current heading before planning actions
- The arrow is RED and clearly visible in the image

## Action Space
- 0 or "turn left": Rotate 90° counterclockwise
- 1 or "turn right": Rotate 90° clockwise
- 2 or "move forward": Move one cell forward in heading direction
- 3 or "pickup": Pick up object in front
- 4 or "drop": Drop carried object
- 5 or "toggle": Interact with objects (e.g., open doors)
- 6 or "done": Complete the task (terminate episode)

## Movement Rules
**CRITICAL**: All movements are RELATIVE to robot's current heading direction.
- "forward" = move one cell in facing direction
- "turn left/right" = rotate 90° from current heading
- Think in relative movements, NOT absolute coordinates
- Note: There is NO backward movement action. To move backward, turn 180° (turn left twice or turn right twice) then move forward.

## Response Format
You MUST predict a sequential trajectory of {action_count} actions. This is a continuous sequence of actions that the robot will take step by step. Respond in JSON format:
```json
{{
    "trajectory": [
        "<action_name_or_number>",
        "<action_name_or_number>",
        "<action_name_or_number>",
        ...
    ],
    "trajectory_reasoning": "<brief summary of the overall trajectory strategy>",
    "environment_info": "<description of current state with spatial relationships relative to robot>",
    "memory_update": "<concise summary updating the permanent memory: what was done in the previous step and current progress toward the mission goal. This will REPLACE the entire previous memory, not append to it. Keep it brief (2-3 sentences max).>",
    "grounding_update": "<ONLY if user feedback indicates a mistake: analyze why the previous action was wrong and provide knowledge to avoid this mistake in the future. Keep it brief (2-3 sentences). If no mistake feedback, leave empty or omit this field.>"
}}
```

**memory_update** (REQUIRED): You MUST provide a concise summary that updates the permanent memory. This should describe:
- What action was just taken in this step
- Current progress toward completing the mission
- This summary will REPLACE the entire previous memory (not append), so include all relevant context in a brief format (2-3 sentences max).
- This field is REQUIRED and must be included in every response.

**grounding_update** (REQUIRED when feedback detected): You MUST carefully analyze the user's prompt to determine if it contains ANY feedback indicating that the previous action was wrong, incorrect, or needs correction. Be SENSITIVE to feedback - even subtle corrections should be detected. Examples of feedback include:
- Explicit corrections: "wrong", "incorrect", "that's wrong", "no", "not that", "don't", "shouldn't", "error", "mistake"
- Questions about mistakes: "why did you...", "why didn't you...", "what are you doing", "where are you going"
- Negative feedback: "not feasible", "cannot", "should not", "avoid", "collided", "touching walls"
- Corrections with explanations: "you cannot turn to wall", "path should not collide", "you didn't even touch"
- ANY indication that the previous action was not correct or needs adjustment

**CRITICAL**: If the user prompt contains ANY of the above patterns or suggests the previous action was wrong, you MUST provide the "grounding_update" field. When provided, it should:
- Analyze why the previous action was wrong
- Explain what should have been done instead
- Provide knowledge/guidance to avoid this mistake in the future
- Keep it brief (2-3 sentences max)
- Be specific about the mistake and the correct approach

**Only omit this field** if the user prompt is clearly a normal instruction or continuation without any negative feedback or correction.

**Important**: 
- You MUST provide exactly {action_count} actions in the "trajectory" array as a sequential sequence
- The trajectory represents consecutive actions: action[0] is executed first, then action[1], then action[2], etc.
- Each action in the trajectory should consider the state after the previous action
- Only the first action will be executed in this step, but the full trajectory will be logged for analysis
- Think of this as planning the next {action_count} steps ahead

**environment_info**: Describe environment relative to robot's heading:
- Robot's heading and relative location to objects
- Obstacles and open paths relative to heading
- Traversability (blocked vs open paths)
- Spatial relationships affecting navigation

## Notes
- Valid JSON format required
- Actions must be from the list above
- Complete mission from user prompt
- Use relative movements based on heading, not coordinates
- Provide exactly {action_count} actions as a sequential trajectory
- Consider how each action affects the robot's position and heading for the next action
"""


def create_scenario2_environment():
    """
    시나리오 2 실험 환경 생성
    
    Returns:
        CustomRoomWrapper: 시나리오 2 환경 Wrapper 인스턴스
    """
    size = 10
    
    # 외벽 생성 (검은색 벽)
    walls = []
    for i in range(size):
        walls.append((i, 0))      # 상단 벽
        walls.append((i, size-1))  # 하단 벽
        walls.append((0, i))      # 좌측 벽
        walls.append((size-1, i))  # 우측 벽
    
    # 파란 기둥: 2x2 Grid (통과불가)
    # 위치: (3, 4), (4, 4), (3, 5), (4, 5)
    # MiniGrid 좌표계: (x, y) = (열, 행)
    blue_pillar_positions = [
        (3, 4),  # 왼쪽 위
        (4, 4),  # 오른쪽 위
        (3, 5),  # 왼쪽 아래
        (4, 5),  # 오른쪽 아래
    ]
    
    # 테이블: 보라색 1x3 Grid (통과불가)
    # 위치: (5, 1), (6, 1), (7, 1)
    table_positions = [
        (5, 1),  # 왼쪽
        (6, 1),  # 중앙
        (7, 1),  # 오른쪽
    ]
    
    # 객체 리스트 생성
    objects = []
    
    # 파란 기둥 배치 (파란색 Box로 구현)
    # 참고: Box는 통과 가능하지만, 시각적으로는 색상이 있는 객체로 표시됩니다.
    # 통과 불가능하게 만들려면 나중에 CustomRoomEnv를 확장해야 합니다.
    for pos in blue_pillar_positions:
        objects.append({
            'type': 'box',
            'pos': pos,
            'color': 'blue'
        })
    
    # 테이블 배치 (보라색 Box로 구현)
    for pos in table_positions:
        objects.append({
            'type': 'box',
            'pos': pos,
            'color': 'purple'
        })
    
    # 시작점: 빨강 1x1 (에이전트 시작 위치)
    # 위치: (1, 8) - 레이아웃에서 🟥 위치
    start_pos = (1, 8)
    
    # 종료점: 초록 1x1 (Goal)
    # 위치: (8, 1) - 레이아웃에서 🟩 위치
    goal_pos = (8, 1)
    
    # room_config 구성
    room_config = {
        'start_pos': start_pos,
        'goal_pos': goal_pos,
        'walls': walls,
        'objects': objects
    }
    
    # Wrapper 생성 및 반환
    return CustomRoomWrapper(size=size, room_config=room_config)


def calculate_predicted_path(
    start_pos: tuple,
    start_dir: int,
    predicted_actions: list,
    wrapper: CustomRoomWrapper
) -> list:
    """
    예측된 액션들을 기반으로 경로를 계산하는 함수
    
    Args:
        start_pos: 시작 위치 (x, y)
        start_dir: 시작 방향 (0=East, 1=South, 2=West, 3=North)
        predicted_actions: 예측된 액션 리스트 (각 항목은 {'action': str, ...})
        wrapper: CustomRoomWrapper 인스턴스 (액션 파싱용)
    
    Returns:
        경로 리스트: [(x, y, direction), ...] - 각 스텝의 위치와 방향
    """
    path = [(start_pos[0], start_pos[1], start_dir)]  # 시작 위치와 방향
    
    current_x, current_y = start_pos[0], start_pos[1]
    current_dir = start_dir
    
    # 방향 벡터: [dx, dy] for each direction
    # 0=East(→), 1=South(↓), 2=West(←), 3=North(↑)
    direction_vectors = {
        0: (1, 0),   # East: x+1
        1: (0, 1),   # South: y+1
        2: (-1, 0),  # West: x-1
        3: (0, -1)   # North: y-1
    }
    
    for action_item in predicted_actions:
        # action_item이 딕셔너리인 경우 'action' 키에서 추출, 아니면 직접 사용
        if isinstance(action_item, dict):
            action_str = str(action_item.get('action', '2'))
        else:
            action_str = str(action_item)
        
        try:
            action_idx = wrapper.parse_action(action_str)
        except (ValueError, AttributeError):
            # 파싱 실패 시 move forward로 간주
            action_idx = 2
        
        # 액션에 따라 위치/방향 업데이트
        if action_idx == 0:  # turn left (반시계 방향)
            current_dir = (current_dir - 1) % 4
        elif action_idx == 1:  # turn right (시계 방향)
            current_dir = (current_dir + 1) % 4
        elif action_idx == 2:  # move forward
            dx, dy = direction_vectors[current_dir]
            current_x += dx
            current_y += dy
        # else: pickup, drop, toggle, done은 위치 변경 없음
        
        # 경로에 추가
        path.append((current_x, current_y, current_dir))
    
    return path


def visualize_grid_cli(wrapper: CustomRoomWrapper, state: dict, predicted_path: list = None):
    """
    CLI에서 그리드를 텍스트로 시각화하는 함수
    
    Args:
        wrapper: CustomRoomWrapper 인스턴스
        state: 현재 환경 상태 딕셔너리
        predicted_path: 예측된 경로 리스트 [(x, y, dir), ...] (선택적)
    """
    env = wrapper.env
    size = wrapper.size
    
    # 에이전트 위치 및 방향
    agent_pos = state['agent_pos']
    if isinstance(agent_pos, np.ndarray):
        agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
    else:
        agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
    
    agent_dir = state['agent_dir']
    # 같은 폭의 문자 사용 (정렬 문제 해결)
    direction_symbols = {0: '>', 1: 'v', 2: '<', 3: '^'}
    direction_names = {0: 'East', 1: 'South', 2: 'West', 3: 'North'}
    agent_symbol = direction_symbols.get(agent_dir, 'A')
    agent_dir_name = direction_names.get(agent_dir, 'Unknown')
    
    # 예측 경로 위치 집합 생성 (시각화용)
    predicted_path_set = set()
    if predicted_path:
        for x, y, _ in predicted_path:
            # 그리드 범위 내에 있는 경우만 추가
            if 0 <= x < size and 0 <= y < size:
                predicted_path_set.add((x, y))
    
    # 그리드 생성
    grid_chars = []
    for y in range(size):
        row = []
        for x in range(size):
            # 그리드 셀 확인
            cell = env.grid.get(x, y)
            
            # 에이전트 위치인 경우
            if x == agent_x and y == agent_y:
                row.append(agent_symbol)
            # 예측 경로 위치인 경우 (에이전트 위치가 아닌 경우만)
            elif (x, y) in predicted_path_set:
                row.append('·')  # 예측 경로 표시
            # 벽인 경우
            elif cell is not None and cell.type == 'wall':
                row.append('⬛')
            # Goal인 경우
            elif cell is not None and cell.type == 'goal':
                row.append('🟩')
            # 객체인 경우 (색상에 따라)
            elif cell is not None:
                if hasattr(cell, 'color'):
                    if cell.color == 'blue':
                        row.append('🟦')
                    elif cell.color == 'purple':
                        row.append('🟪')
                    elif cell.color == 'red':
                        row.append('🟥')
                    elif cell.color == 'green':
                        row.append('🟩')
                    else:
                        row.append('🟨')  # 기타 객체
                else:
                    row.append('🟨')
            # 빈 공간
            else:
                row.append('⬜️')
        grid_chars.append(row)
    
    # 그리드 출력
    print("\n" + "=" * 60)
    print("Current Grid State:")
    print("=" * 60)
    legend = "Legend: ⬛=Wall, ⬜️=Empty, 🟦=Blue Pillar, 🟪=Purple Table, 🟩=Goal, >v<^=Agent Direction (R/D/L/U)"
    if predicted_path:
        legend += ", ·=Predicted Path"
    print(legend)
    print("=" * 60)
    for y in range(size):
        row_str = ''.join(grid_chars[y])
        print(row_str)
    print("=" * 60)
    print(f"Agent Position: ({agent_x}, {agent_y}), Direction: {agent_dir} ({agent_symbol} = {agent_dir_name})")
    print("=" * 60 + "\n")


def get_user_prompt(task_hint: str = None) -> str:
    """
    사용자로부터 프롬프트를 입력받는 함수 (CLI)
    
    Args:
        task_hint: 작업 힌트 (자동 실행되지 않음, 단지 힌트로만 표시)
    
    Returns:
        사용자가 입력한 프롬프트 문자열
    """
    print("\n" + "=" * 60)
    if task_hint:
        print(f"Task Hint: {task_hint}")
        print("=" * 60)
    print("Enter your instruction for the agent (or press Enter to use default):")
    user_input = input("> ").strip()
    
    # 기본 프롬프트 (사용자가 아무것도 입력하지 않은 경우)
    if not user_input:
        # Task 정보를 포함한 기본 프롬프트
        if task_hint:
            default_prompt = f"Task: {task_hint}\n\nBased on the current image, choose the next action to complete this task."
        else:
            default_prompt = "Based on the current image, choose the next action to complete the task."
        print(f"Using default prompt: {default_prompt}")
        return default_prompt
    
    return user_input


def save_experiment_data(
    step: int,
    image: np.ndarray,
    state: dict,
    action: int,
    action_name: str,
    user_prompt: str,
    vlm_response: dict,
    reward: float,
    done: bool,
    log_dir: Path,
    all_predicted_actions: list = None,
    vlm_input: dict = None,
    vlm_output: dict = None,
    memory_summary: str = None,
    grounding_section: str = None
):
    """
    실험 데이터를 로깅하는 함수
    
    Args:
        step: 현재 스텝 번호
        image: 환경 이미지 (numpy 배열)
        state: 환경 상태 정보
        action: 실행된 액션 (정수)
        action_name: 실행된 액션 이름 (문자열)
        user_prompt: 사용자 프롬프트
        vlm_response: VLM 응답 딕셔너리 (actions, environment_info)
        reward: 보상
        done: 에피소드 종료 여부
        log_dir: 로그 디렉토리 경로
        all_predicted_actions: VLM이 예측한 모든 액션 리스트 (로깅용)
        vlm_input: VLM 입력 정보 딕셔너리 (image_info, system_prompt, user_prompt)
        vlm_output: VLM 출력 정보 딕셔너리 (raw_response, parsed_response, tokens_used 등)
        memory_summary: 현재 영구 메모리 요약 (선택적)
        grounding_section: 현재 Grounding 지식 섹션 (선택적)
    """
    # 1. 이미지 저장 (PNG)
    image_path = log_dir / f"step_{step:04d}.png"
    img_pil = Image.fromarray(image)
    img_pil.save(image_path)
    
    # 2. JSON 로그 저장
    # numpy 타입을 Python 기본 타입으로 변환
    agent_pos = state['agent_pos']
    if isinstance(agent_pos, np.ndarray):
        agent_pos_list = [int(agent_pos[0]), int(agent_pos[1])]
    elif isinstance(agent_pos, (tuple, list)):
        agent_pos_list = [int(agent_pos[0]), int(agent_pos[1])]
    else:
        # numpy scalar나 다른 타입인 경우
        try:
            if hasattr(agent_pos, '__len__') and len(agent_pos) >= 2:
                agent_pos_list = [int(agent_pos[0]), int(agent_pos[1])]
            else:
                agent_pos_list = [0, 0]
        except (TypeError, IndexError):
            agent_pos_list = [0, 0]
    
    json_data = {
        "step": int(step),
        "timestamp": datetime.now().isoformat(),
        "state": {
            "agent_pos": agent_pos_list,
            "agent_dir": int(state['agent_dir']),
            "mission": str(state['mission']) if state['mission'] else ""
        },
        "action": {
            "index": int(action),
            "name": str(action_name)
        },
        "user_prompt": str(user_prompt),
        "vlm_response": {k: str(v) for k, v in vlm_response.items()},
        "all_predicted_actions": all_predicted_actions if all_predicted_actions else [],
        "memory_summary": str(memory_summary) if memory_summary else "",
        "grounding_section": str(grounding_section) if grounding_section else "",
        "reward": float(reward),
        "done": bool(done),
        "image_path": str(image_path.name),
        "vlm_input": vlm_input if vlm_input else {},
        "vlm_output": vlm_output if vlm_output else {}
    }
    
    # 2. JSON 로그 저장 (하나의 파일에 배열로 누적)
    json_path = log_dir / "experiment_log.json"
    
    # 기존 데이터 읽기 (파일이 존재하는 경우)
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                all_data = json.load(f)
                if not isinstance(all_data, list):
                    all_data = [all_data]  # 기존 데이터가 리스트가 아니면 리스트로 변환
            except json.JSONDecodeError:
                all_data = []
    else:
        all_data = []
    
    # 새 데이터 추가
    all_data.append(json_data)
    
    # 파일에 저장
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    
    # 2-1. VLM I/O 별도 로그 파일 저장 (텍스트 형식, 하나의 파일에 누적)
    if vlm_input or vlm_output:
        vlm_io_path = log_dir / "vlm_io_log.txt"
        
        # 추가 모드로 열기 (파일이 없으면 생성)
        with open(vlm_io_path, 'a', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"VLM I/O Log - Step {step}\n")
            f.write("=" * 80 + "\n\n")
            
            # VLM 입력 정보
            f.write("VLM INPUT:\n")
            f.write("-" * 80 + "\n")
            if vlm_input:
                f.write(f"Image Shape: {vlm_input.get('image_shape', 'N/A')}\n")
                f.write(f"Image Dtype: {vlm_input.get('image_dtype', 'N/A')}\n")
                f.write(f"Image Value Range: [{vlm_input.get('image_min', 'N/A')}, {vlm_input.get('image_max', 'N/A')}]\n")
                f.write(f"System Prompt: Used (Length: {vlm_input.get('system_prompt_length', 0)} characters, see {vlm_input.get('system_prompt_file', 'system_prompt.txt')})\n")
                f.write(f"User Prompt Length: {vlm_input.get('user_prompt_length', 0)} characters\n")
                f.write(f"\nUser Prompt:\n{vlm_input.get('user_prompt', 'N/A')}\n")
                # System Prompt는 VLM API 호출에 포함되지만, 전체 내용은 system_prompt.txt 파일 참조
            else:
                f.write("No input data\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
            
            # VLM 출력 정보
            f.write("VLM OUTPUT:\n")
            f.write("-" * 80 + "\n")
            if vlm_output:
                f.write(f"Raw Response Length: {vlm_output.get('raw_response_length', 0)} characters\n")
                f.write(f"Inference Time: {vlm_output.get('inference_time_seconds', 0):.2f} seconds\n")
                f.write(f"Tokens Used: {vlm_output.get('tokens_used', 0)}\n")
                f.write(f"Parsing Success: {vlm_output.get('parsing_success', 'N/A')}\n")
                if vlm_output.get('parsing_error'):
                    f.write(f"Parsing Error: {vlm_output.get('parsing_error')}\n")
                f.write(f"\nRaw Response:\n{vlm_output.get('raw_response', 'N/A')}\n")
                if vlm_output.get('parsed_response'):
                    f.write(f"\nParsed Response:\n")
                    for k, v in vlm_output.get('parsed_response', {}).items():
                        f.write(f"  {k}: {v}\n")
            else:
                f.write("No output data\n")
            
            # 영구 메모리 정보
            if memory_summary:
                f.write("\n" + "-" * 80 + "\n")
                f.write("PERMANENT MEMORY (Updated):\n")
                f.write("-" * 80 + "\n")
                f.write(f"{memory_summary}\n")
            
            # Grounding 지식 정보
            if grounding_section:
                f.write("\n" + "-" * 80 + "\n")
                f.write("GROUNDING KNOWLEDGE:\n")
                f.write("-" * 80 + "\n")
                f.write(f"{grounding_section}\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
    
    # 3. CSV 로그 저장 (추가 모드)
    csv_path = log_dir / "experiment_log.csv"
    file_exists = csv_path.exists()
    
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 헤더 작성 (첫 번째 실행 시에만)
        if not file_exists:
            writer.writerow([
                "step", "timestamp", "agent_x", "agent_y", "agent_dir",
                "action_index", "action_name", "user_prompt",
                "vlm_action_executed", "vlm_environment_info", "vlm_all_predicted_actions",
                "memory_summary",
                "reward", "done", "image_path",
                "vlm_image_shape", "vlm_image_dtype", "vlm_system_prompt_len",
                "vlm_user_prompt_len", "vlm_raw_response_len", "vlm_inference_time_seconds", "vlm_tokens_used"
            ])
        
        # 데이터 작성
        agent_pos = state['agent_pos']
        if isinstance(agent_pos, np.ndarray):
            agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
        elif isinstance(agent_pos, (tuple, list)):
            agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
        else:
            # numpy scalar인 경우
            try:
                if hasattr(agent_pos, '__len__') and len(agent_pos) >= 2:
                    agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
                else:
                    agent_x, agent_y = 0, 0
            except (TypeError, IndexError):
                agent_x, agent_y = 0, 0
        
        # VLM I/O 정보 추출
        vlm_img_shape = vlm_input.get('image_shape', '') if vlm_input else ''
        vlm_img_dtype = vlm_input.get('image_dtype', '') if vlm_input else ''
        vlm_sys_prompt_len = vlm_input.get('system_prompt_length', 0) if vlm_input else 0
        vlm_usr_prompt_len = vlm_input.get('user_prompt_length', 0) if vlm_input else 0
        vlm_raw_resp_len = vlm_output.get('raw_response_length', 0) if vlm_output else 0
        vlm_inference_time = vlm_output.get('inference_time_seconds', 0.0) if vlm_output else 0.0
        vlm_tokens = vlm_output.get('tokens_used', 0) if vlm_output else 0
        
        # 모든 예측된 액션을 JSON 문자열로 변환 (CSV용)
        all_actions_json = json.dumps(all_predicted_actions, ensure_ascii=False) if all_predicted_actions else "[]"
        # 첫 번째 액션 정보 (실행된 액션)
        first_action_str = all_predicted_actions[0].get('action', '') if all_predicted_actions else vlm_response.get('action', '')
        
        writer.writerow([
            step,
            datetime.now().isoformat(),
            agent_x,
            agent_y,
            int(state['agent_dir']),
            int(action),
            action_name,
            user_prompt,
            first_action_str,  # 실행된 액션 (첫 번째 예측)
            vlm_response.get('environment_info', ''),
            all_actions_json,  # 모든 예측된 액션 (JSON 문자열)
            str(memory_summary) if memory_summary else '',  # 영구 메모리 요약
            str(grounding_section) if grounding_section else '',  # Grounding 지식
            float(reward),
            bool(done),
            image_path.name,
            str(vlm_img_shape),
            str(vlm_img_dtype),
            int(vlm_sys_prompt_len),
            int(vlm_usr_prompt_len),
            int(vlm_raw_resp_len),
            float(vlm_inference_time),
            int(vlm_tokens)
        ])


def run_vlm_controlled_experiment():
    """
    VLM을 통한 환경 제어 실험 메인 함수
    """
    print("=" * 60)
    print("시나리오 2: VLM 제어 실험")
    print("=" * 60)
    print("\n환경 구성:")
    print("  - 파란 기둥: 2x2 Grid (통과불가)")
    print("  - 테이블: 보라색 1x3 Grid (통과불가)")
    print("  - 시작점: 빨강 (1, 8)")
    print("  - 종료점: 초록 (8, 1)")
    print("\nMission: 파란 기둥으로 가서 오른쪽으로 돌고, 테이블 옆에 멈추시오")
    
    # 로그 디렉토리 생성
    log_dir = Path("logs") / f"scenario2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n로그 디렉토리: {log_dir}")
    
    # 영구 메모리 파일 경로
    memory_file = log_dir / "permanent_memory.txt"
    
    # 영구 메모리 초기화 (파일이 없으면 빈 문자열)
    memory_summary = ""
    grounding_section = ""
    if memory_file.exists():
        with open(memory_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # 메모리 파일 구조: === MEMORY SUMMARY === 와 === GROUNDING === 구분
            if "=== MEMORY SUMMARY ===" in content:
                parts = content.split("=== GROUNDING ===")
                memory_summary = parts[0].replace("=== MEMORY SUMMARY ===", "").strip()
                if len(parts) > 1:
                    grounding_section = parts[1].strip()
            else:
                # 구버전 형식 (전체를 memory_summary로 처리)
                memory_summary = content
        print(f"영구 메모리 로드: memory_summary={len(memory_summary)} 문자, grounding={len(grounding_section)} 문자")
    else:
        print("영구 메모리 초기화: 빈 메모리")
    
    # System Prompt 생성 (동적 값 포함, 메모리 및 grounding 포함)
    SYSTEM_PROMPT = get_system_prompt(ACTION_PREDICTION_COUNT, memory_summary, grounding_section)
    
    # System Prompt를 처음에만 별도 파일로 저장
    system_prompt_path = log_dir / "system_prompt.txt"
    with open(system_prompt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("System Prompt\n")
        f.write("=" * 80 + "\n\n")
        f.write(SYSTEM_PROMPT)
        f.write("\n\n" + "=" * 80 + "\n")
    print(f"System Prompt 저장: {system_prompt_path}")
    
    # 1. 환경 생성
    print("\n[1] 환경 생성 중...")
    wrapper = create_scenario2_environment()
    
    # 2. 환경 초기화
    print("[2] 환경 초기화 중...")
    wrapper.reset()
    
    # 환경 상태 정보 출력
    state = wrapper.get_state()
    print(f"에이전트 시작 위치: {state['agent_pos']}")
    print(f"에이전트 방향: {state['agent_dir']} (0=오른쪽, 1=아래, 2=왼쪽, 3=위)")
    print(f"미션: {state['mission']}")
    
    # 3. VLM Wrapper 초기화
    print("\n[3] VLM Wrapper 초기화 중...")
    try:
        # 코드 상단의 VLM 설정 변수 사용
        vlm = ChatGPT4oVLMWrapper(
            model=VLM_MODEL,
            temperature=VLM_TEMPERATURE,
            max_tokens=VLM_MAX_TOKENS
        )
        
        print(f"VLM Wrapper 초기화 완료")
        print(f"  - 모델: {VLM_MODEL}")
        print(f"  - Temperature: {VLM_TEMPERATURE}")
        print(f"  - Max Tokens: {VLM_MAX_TOKENS}")
    except Exception as e:
        print(f"VLM Wrapper 초기화 실패: {e}")
        print("\nVLM 설정을 확인하세요:")
        print("  - 코드 상단의 VLM_MODEL, VLM_TEMPERATURE, VLM_MAX_TOKENS 변수 확인")
        print("  - OpenAI API 키 확인: export OPENAI_API_KEY=your-key")
        return
    
    # 4. VLM PostProcessor 초기화
    print("[4] VLM PostProcessor 초기화 중...")
    postprocessor = VLMResponsePostProcessor(
        required_fields=["trajectory", "environment_info", "memory_update"]  # memory_update는 필수 필드
    )
    print(f"PostProcessor 초기화 완료 (궤적 길이: {ACTION_PREDICTION_COUNT})")
    
    # 5. 액션 공간 정보 출력
    action_space = wrapper.get_action_space()
    print(f"\n액션 공간: {action_space['actions']}")
    
    # 메인 루프
    step = 0
    done = False
    task_hint = "Mission: Go to the blue pillar, turn right, then stop next to the table."
    
    # 이전 스텝의 예측 경로 저장 (시각화 유지용)
    previous_predicted_path = None
    
    print("\n" + "=" * 60)
    print("실험 시작")
    print("=" * 60)
    print("OpenCV 창이 열립니다. 환경을 확인할 수 있습니다.")
    print("=" * 60)
    
    # OpenCV 창 이름 고정 (하나의 창만 사용)
    WINDOW_NAME = "Scenario 2: VLM Control"
    
    def display_image(img, window_name=None, predicted_path=None, cell_size=32):
        """
        OpenCV를 사용하여 이미지를 표시하는 헬퍼 함수
        하나의 창만 사용하여 업데이트만 수행
        
        Args:
            img: 표시할 이미지 (RGB)
            window_name: 창 이름
            predicted_path: 예측된 경로 리스트 [(x, y, dir), ...] (선택적)
            cell_size: 그리드 셀 크기 (픽셀)
        """
        if window_name is None:
            window_name = WINDOW_NAME
        
        if img is not None:
            try:
                # RGB를 BGR로 변환 (OpenCV는 BGR 형식을 사용)
                img_bgr = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
                
                # 예측 경로 그리기
                if predicted_path and len(predicted_path) > 1:
                    # 경로를 그리드 좌표에서 픽셀 좌표로 변환
                    for i in range(len(predicted_path) - 1):
                        x1, y1, _ = predicted_path[i]
                        x2, y2, _ = predicted_path[i + 1]
                        
                        # 그리드 셀 중심 좌표 계산
                        px1 = int((x1 + 0.5) * cell_size)
                        py1 = int((y1 + 0.5) * cell_size)
                        px2 = int((x2 + 0.5) * cell_size)
                        py2 = int((y2 + 0.5) * cell_size)
                        
                        # 경로 선 그리기 (노란색, 두께 2)
                        cv2.line(img_bgr, (px1, py1), (px2, py2), (0, 255, 255), 2)
                    
                    # 경로 점 표시 (작은 원)
                    for x, y, _ in predicted_path[1:]:  # 시작점 제외
                        px = int((x + 0.5) * cell_size)
                        py = int((y + 0.5) * cell_size)
                        cv2.circle(img_bgr, (px, py), 3, (0, 255, 255), -1)
                
                # 이미지 크기 조정 (더 크게 표시)
                height, width = img_bgr.shape[:2]
                max_size = 1200
                scale = 1
                if height < max_size and width < max_size:
                    scale = min(max_size // height, max_size // width, 6)
                
                if scale > 1:
                    new_width = width * scale
                    new_height = height * scale
                    img_bgr = cv2.resize(img_bgr, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
                
                # 이미지 창에 표시 (같은 창 이름 사용하여 업데이트만 수행)
                cv2.imshow(window_name, img_bgr)
                cv2.waitKey(1)  # 창 업데이트를 위해 짧은 대기
            except Exception as e:
                print(f"이미지 표시 오류: {e}")
    
    while not done:
        step += 1
        print("\n" + "=" * 80)
        print(f"STEP {step} START")
        print("=" * 80)
        
        # 현재 환경 이미지 가져오기
        image = wrapper.get_image()
        state = wrapper.get_state()
        
        # 환경 정보 출력
        print(f"현재 위치: {state['agent_pos']}, 방향: {state['agent_dir']}")
        
        # 영구 메모리 읽기 (VLM 호출 전)
        if memory_file.exists():
            with open(memory_file, 'r', encoding='utf-8') as f:
                memory_summary = f.read().strip()
        else:
            memory_summary = ""
        
        # System Prompt 업데이트 (메모리 포함)
        SYSTEM_PROMPT = get_system_prompt(ACTION_PREDICTION_COUNT, memory_summary)
        
        # 초기 시각화 (이전 스텝의 예측 경로 표시)
        all_predicted_actions = []
        
        # CLI 텍스트 시각화 (이전 예측 경로 포함)
        visualize_grid_cli(wrapper, state, previous_predicted_path)
        
        # GUI에 현재 상태 표시 (이전 예측 경로 포함)
        display_image(image, WINDOW_NAME, previous_predicted_path)
        
        # 3. User prompt 입력 받기 (task hint 포함)
        user_prompt = get_user_prompt(task_hint=task_hint)
        
        # 5. VLM에 요청 전송
        print(f"\n[5] VLM에 요청 전송 중...")
        # VLM 호출 직전에 이미지를 다시 가져와서 최신 상태 보장
        image = wrapper.get_image()
        
        # 이미지 해시를 계산하여 변경 여부 확인 (디버깅용)
        image_hash = hashlib.md5(image.tobytes()).hexdigest()[:8] if image is not None else None
        print(f"  - 이미지 해시 (변경 확인용): {image_hash}")
        
        # VLM 입력 정보 수집
        # System Prompt는 VLM API 호출에 포함되지만, 로깅에서는 참조만 저장 (전체 내용은 system_prompt.txt에 저장됨)
        vlm_input_info = {
            'image_shape': str(image.shape) if image is not None else None,
            'image_dtype': str(image.dtype) if image is not None else None,
            'image_min': float(image.min()) if image is not None else None,
            'image_max': float(image.max()) if image is not None else None,
            'image_hash': image_hash,
            'system_prompt_length': len(SYSTEM_PROMPT),
            'system_prompt_file': 'system_prompt.txt',  # System Prompt는 별도 파일 참조
            'user_prompt_length': len(user_prompt),
            'user_prompt': user_prompt
        }
        
        # 이미지 정보 확인 및 출력
        if image is not None:
            print(f"  - 이미지 크기: {image.shape}, 타입: {image.dtype}")
            print(f"  - 이미지 값 범위: [{image.min()}, {image.max()}]")
            print(f"  - Agent 위치 (이미지 전송 시점): {state['agent_pos']}")
        else:
            print("  - 경고: 이미지가 None입니다!")
        
        try:
            # VLM 추론 시간 측정 시작
            vlm_start_time = time.time()
            vlm_response_raw = vlm.generate(
                image=image,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt
            )
            # VLM 추론 시간 측정 종료
            vlm_end_time = time.time()
            vlm_inference_time = vlm_end_time - vlm_start_time
            
            print("VLM 응답 수신 완료")
            print(f"  - 응답 길이: {len(vlm_response_raw)} 문자")
            print(f"  - 추론 시간: {vlm_inference_time:.2f}초")
            print(f"  - 응답 미리보기: {vlm_response_raw[:150]}...")
        except Exception as e:
            print(f"VLM API 호출 실패: {e}")
            vlm_response_raw = ""
            vlm_inference_time = 0.0
            vlm_input_info['error'] = str(e)
            break
        
        # 6. VLM 응답 후처리
        print(f"[6] VLM 응답 후처리 중...")
        # VLM 출력 정보 수집
        vlm_output_info = {
            'raw_response': vlm_response_raw,
            'raw_response_length': len(vlm_response_raw),
            'inference_time_seconds': vlm_inference_time,  # VLM 추론 시간 추가
            'tokens_used': 0  # vlm_wrapper에서 토큰 정보를 반환하지 않으므로 0으로 설정
        }
        
        # actions 배열 파싱 및 첫 번째 액션 추출
        all_predicted_actions = []  # 모든 예측된 액션 저장 (로깅용)
        first_action_str = '2'  # 기본값: move forward
        first_action_index = 2
        first_action_name = "move forward"
        
        try:
            vlm_response = postprocessor.process(vlm_response_raw, strict=True)
            vlm_output_info['parsed_response'] = vlm_response
            vlm_output_info['parsing_success'] = True
            
            # trajectory 배열에서 모든 액션 추출
            trajectory_list = vlm_response.get('trajectory', [])
            
            # trajectory가 이미 리스트인 경우 그대로 사용
            # (vlm_postprocessor가 리스트 타입을 유지하도록 수정됨)
            if isinstance(trajectory_list, str):
                # 문자열인 경우 JSON 파싱 시도 (하위 호환성)
                try:
                    trajectory_list = json.loads(trajectory_list)
                except (json.JSONDecodeError, ValueError):
                    trajectory_list = []
            
            if not isinstance(trajectory_list, list):
                trajectory_list = []
            
            print(f"파싱된 응답:")
            print(f"  - 예측된 궤적 길이: {len(trajectory_list)}")
            trajectory_reasoning = vlm_response.get('trajectory_reasoning', 'N/A')
            if trajectory_reasoning and trajectory_reasoning != 'N/A':
                print(f"  - 궤적 전략: {trajectory_reasoning[:150]}...")
            print(f"  - Environment Info: {vlm_response.get('environment_info', 'N/A')[:100]}...")
            
            # 모든 액션 정보 저장 및 출력 (순차적 궤적으로 처리)
            for idx, action_item in enumerate(trajectory_list):
                # action_item이 문자열이거나 숫자인 경우
                if isinstance(action_item, (str, int)):
                    action_str = str(action_item)
                elif isinstance(action_item, dict):
                    # 딕셔너리인 경우 'action' 키에서 추출
                    action_str = str(action_item.get('action', '2'))
                else:
                    action_str = str(action_item)
                
                all_predicted_actions.append({
                    'step': idx + 1,
                    'action': action_str
                })
                
                print(f"  - Step {idx + 1}: {action_str}")
                
                # 첫 번째 액션만 추출 (실행용)
                if idx == 0:
                    first_action_str = action_str
            
            # 첫 번째 액션 파싱 및 예측 경로 계산
            predicted_path = None
            if all_predicted_actions:
                try:
                    first_action_index = wrapper.parse_action(first_action_str)
                    first_action_name = wrapper.ACTION_NAMES.get(first_action_index, f"action_{first_action_index}")
                except ValueError as e:
                    print(f"첫 번째 액션 파싱 실패: {e}, 기본 액션 사용")
                    first_action_index = 2
                    first_action_name = "move forward"
                
                # 예측 경로 계산
                agent_pos = state['agent_pos']
                if isinstance(agent_pos, np.ndarray):
                    start_pos = (int(agent_pos[0]), int(agent_pos[1]))
                else:
                    start_pos = (int(agent_pos[0]), int(agent_pos[1]))
                start_dir = int(state['agent_dir'])
                
                predicted_path = calculate_predicted_path(
                    start_pos=start_pos,
                    start_dir=start_dir,
                    predicted_actions=all_predicted_actions,
                    wrapper=wrapper
                )
                print(f"  - 예측 경로 계산 완료: {len(predicted_path)}개 위치")
            else:
                print("경고: trajectory 배열이 비어있습니다. 기본 액션 사용")
                predicted_path = None
                
        except ValueError as e:
            print(f"응답 파싱 실패: {e}")
            print(f"원본 응답: {vlm_response_raw[:200]}...")
            vlm_output_info['parsing_success'] = False
            vlm_output_info['parsing_error'] = str(e)
            # 파싱 실패 시 기본 액션 사용
            vlm_response = {
                'trajectory': [],
                'trajectory_reasoning': 'Parsing failed',
                'environment_info': 'Parsing failed'
            }
        
        # 7. MiniGrid 액션 실행 (첫 번째 액션만)
        print(f"\n[7] 액션 실행 중...")
        print(f"실행할 액션 (첫 번째 예측): {first_action_name} (인덱스: {first_action_index})")
        
        try:
            # 첫 번째 액션 실행
            _, reward, terminated, truncated, _ = wrapper.step(first_action_index)
            done = terminated or truncated
            
            print(f"보상: {reward}, 종료: {done}")
            
        except Exception as e:
            print(f"액션 실행 실패: {e}")
            print("기본 액션(move forward)을 사용합니다.")
            first_action_index = 2
            first_action_name = "move forward"
            _, reward, terminated, truncated, _ = wrapper.step(first_action_index)
            done = terminated or truncated
        
        # 8. 환경 정보 출력
        print(f"[8] 환경 정보:")
        new_state = wrapper.get_state()
        print(f"  - 위치: {new_state['agent_pos']}")
        print(f"  - 방향: {new_state['agent_dir']}")
        print(f"  - 보상: {reward}")
        print(f"  - 종료: {done}")
        
        # 영구 메모리 업데이트 (VLM 응답 후)
        memory_update = vlm_response.get('memory_update', '')
        grounding_update = vlm_response.get('grounding_update', '')
        
        # Memory Summary 업데이트
        if memory_update and memory_update.strip():
            new_memory_summary = memory_update.strip()
            memory_summary = new_memory_summary
            print(f"[8-1] 영구 메모리 업데이트 완료: {len(memory_summary)} 문자")
            print(f"  - 메모리 내용: {memory_summary[:100]}...")
        else:
            # memory_update가 없거나 비어있는 경우 경고
            print(f"[8-1] 경고: 영구 메모리 업데이트 없음")
            if 'memory_update' not in vlm_response:
                print(f"  - memory_update 필드가 VLM 응답에 없습니다.")
            elif not memory_update.strip():
                print(f"  - memory_update 필드가 비어있습니다.")
        
        # Grounding 업데이트 (feedback이 있는 경우)
        if grounding_update and grounding_update.strip():
            print(f"\n[8-2] ⚠️  Feedback 인식됨: Grounding 지식 업데이트")
            print("=" * 80)
            
            # 업데이트된 grounding (새로 추가된 부분)
            new_grounding_text = grounding_update.strip()
            print(f"\n[새로 추가된 Grounding 지식]:")
            print("-" * 80)
            print(new_grounding_text)
            print("-" * 80)
            
            # Grounding 섹션에 새로운 지식 추가 (누적)
            if grounding_section:
                new_grounding = f"{grounding_section}\n\n{new_grounding_text}"
            else:
                new_grounding = new_grounding_text
            grounding_section = new_grounding
            
            # 전체 Grounding 출력
            print(f"\n[전체 Grounding 지식 (누적)]:")
            print("=" * 80)
            print(grounding_section)
            print("=" * 80)
            print(f"\nGrounding 지식 업데이트 완료: 총 {len(grounding_section)} 문자")
        
        # 메모리 파일에 저장 (구조화된 형식)
        with open(memory_file, 'w', encoding='utf-8') as f:
            f.write("=== MEMORY SUMMARY ===\n")
            f.write(memory_summary)
            f.write("\n\n=== GROUNDING ===\n")
            f.write(grounding_section)
        
        # 액션 실행 후 CLI 텍스트 시각화 (예측 경로 포함)
        visualize_grid_cli(wrapper, new_state, predicted_path)
        
        # 액션 실행 후 업데이트된 이미지 표시 (예측 경로 포함, 같은 창에 업데이트)
        updated_image = wrapper.get_image()
        display_image(updated_image, WINDOW_NAME, predicted_path)
        
        # 다음 스텝을 위해 예측 경로 저장 (시각화 유지용)
        previous_predicted_path = predicted_path
        
        # 9. 로깅
        print(f"[9] 실험 데이터 로깅 중...")
        try:
            save_experiment_data(
                step=step,
                image=image,
                state=state,
                action=first_action_index,
                action_name=first_action_name,
                user_prompt=user_prompt,
                vlm_response=vlm_response,
                reward=reward,
                done=done,
                log_dir=log_dir,
                all_predicted_actions=all_predicted_actions,
                vlm_input=vlm_input_info,
                vlm_output=vlm_output_info,
                memory_summary=memory_summary,
                grounding_section=grounding_section
            )
            print(f"  - 이미지: {log_dir / f'step_{step:04d}.png'}")
            print(f"  - JSON: {log_dir / 'experiment_log.json'} (누적)")
            print(f"  - VLM I/O: {log_dir / 'vlm_io_log.txt'} (누적)")
            print(f"  - CSV: {log_dir / 'experiment_log.csv'} (누적)")
        except Exception as e:
            print(f"로깅 오류: {e}")
        
        # Step 종료 표시
        print("\n" + "=" * 80)
        print(f"STEP {step} END")
        print("=" * 80)
        
        # Goal 도착 확인
        if done:
            print("\n" + "=" * 80)
            print("Goal 도착! 실험 종료")
            print("=" * 80)
            break
        
        # 최대 스텝 제한 (무한 루프 방지)
        if step >= 100:
            print("\n" + "=" * 80)
            print("최대 스텝 수(100)에 도달했습니다. 실험을 종료합니다.")
            print("=" * 80)
            break
    
    # 리소스 정리
    cv2.destroyAllWindows()
    wrapper.close()
    print(f"\n실험 완료. 로그는 {log_dir}에 저장되었습니다.")


def main():
    """
    메인 함수
    
    VLM 설정은 코드 상단의 변수에서 변경할 수 있습니다:
    - VLM_TYPE: VLM 타입 ("gpt4o", "chatgpt4o", "openai")
    - VLM_MODEL: 사용할 모델명 (예: "gpt-4o", "gpt-4o-mini")
    - VLM_TEMPERATURE: 생성 온도 (0.0 ~ 2.0)
    - VLM_MAX_TOKENS: 최대 토큰 수
    """
    try:
        run_vlm_controlled_experiment()
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
