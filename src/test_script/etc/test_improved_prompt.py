"""
성공한 솔루션 B 프롬프트 테스트 스크립트
GUI 없이 테스트
"""

from minigrid import register_minigrid_envs
# Actual path: legacy.relative_movement.custom_environment
from legacy import CustomRoomWrapper
# Actual paths: utils.vlm.vlm_wrapper, utils.vlm.vlm_postprocessor
from utils import VLMWrapper, VLMResponsePostProcessor
import numpy as np

# MiniGrid 환경 등록
register_minigrid_envs()

# VLM 설정
VLM_MODEL = "gpt-4o"
VLM_TEMPERATURE = 0.0
VLM_MAX_TOKENS = 1000


def get_system_prompt(wrapper: CustomRoomWrapper) -> str:
    """System Prompt 생성 (성공한 솔루션 B 프롬프트 적용)"""
    heading = wrapper.get_heading()
    heading_short = wrapper.get_heading_short()
    heading_info = f"{heading} ({heading_short})"
    
    return f"""You are a robot operating in a grid-based environment.

## Robot State (Authoritative)
- The robot's current heading is {heading_info}.
- Heading indicates the robot's forward-facing direction.
- This heading is ground-truth and MUST be used as-is.
- Do NOT infer or reinterpret the robot's heading from the image.

## CRITICAL DISTINCTION: Two Coordinate Systems

### 1. ALLOCENTRIC (Absolute/Global) Coordinates
- Used in the IMAGE: Top=North, Bottom=South, Left=West, Right=East
- This is FIXED and does NOT change with robot orientation
- The image shows objects in this coordinate system

### 2. EGOCENTRIC (Relative/Robot-centric) Coordinates
- Used for ACTIONS: Front=heading direction, Left/Right relative to heading
- This CHANGES when the robot rotates
- Actions must be chosen in this coordinate system

## Environment
Grid world with:
- Walls (black, impassable)
- Blue pillar (impassable)
- Purple table (impassable)
- Robot (red arrow marker)
- Goal (green marker, if present)

The image describes the environment layout ONLY.
Do NOT use the image to estimate robot orientation.

## Action Space
- "turn left": Rotate 90° counterclockwise
- "turn right": Rotate 90° clockwise
- "move forward": Move one cell forward in heading direction
- "pickup": Pick up object in front
- "drop": Drop carried object
- "toggle": Interact with objects (e.g., open doors)

## Movement Rules (CRITICAL: EXECUTE STEP-BY-STEP)

**STEP 1: Identify target in ALLOCENTRIC coordinates**
- Look at the image
- Find the blue pillar
- Note its position: Is it at the Top (North), Bottom (South), Left (West), or Right (East) of the image?

**STEP 2: Get robot heading (provided)**
- Robot heading: {heading_info}
- This tells you which direction the robot is facing in ALLOCENTRIC coordinates

**STEP 3: Transform from ALLOCENTRIC to EGOCENTRIC**
Use this EXACT lookup table:

| Robot Heading | Object at North | Object at South | Object at East | Object at West |
|---------------|------------------|------------------|----------------|----------------|
| East (→)      | LEFT             | RIGHT            | FRONT          | BACK           |
| West (←)      | RIGHT            | LEFT             | BACK           | FRONT          |
| North (↑)     | FRONT            | BACK             | RIGHT          | LEFT           |
| South (↓)     | BACK             | FRONT            | LEFT           | RIGHT          |

**STEP 4: Choose action based on EGOCENTRIC position**
- If EGOCENTRIC position is FRONT → "move forward"
- If EGOCENTRIC position is LEFT → "turn left"
- If EGOCENTRIC position is RIGHT → "turn right"
- If EGOCENTRIC position is BACK → "turn left" (to face the object)

## Response Format (STRICT)
Respond in valid JSON. You MUST fill strictly following the "reasoning_trace" logic.

```json
{{
  "reasoning_trace": {{
    "step1_allocentric_pos": "<e.g. The blue pillar is at the Top (North) of the image>",
    "step2_robot_heading": "<e.g. East>",
    "step3_transformation": "<e.g. Using lookup table: North when heading East = LEFT>",
    "step4_egocentric_pos": "<e.g. Therefore, the pillar is to my LEFT in egocentric coordinates>",
    "step5_selected_action": "<e.g. turn left>"
  }},
  "action": "<action_name>",
  "environment_info": "<description of current state with spatial relationships relative to robot heading orientation>",
  "reasoning": "<explanation of why you chose this action>"
}}
```

Important:
- Valid JSON format required
- Actions must be from the list above
- Complete ALL 5 steps in reasoning_trace before selecting actions
- Complete the mission specified by the user
"""


def create_scenario2_environment():
    """시나리오 2 환경 생성"""
    size = 10
    
    # 외벽 생성
    walls = []
    for i in range(size):
        walls.append((i, 0))
        walls.append((i, size-1))
        walls.append((0, i))
        walls.append((size-1, i))
    
    # 파란 기둥: 2x2 Grid
    blue_pillar_positions = [(3, 4), (4, 4), (3, 5), (4, 5)]
    for pos in blue_pillar_positions:
        walls.append((pos[0], pos[1], 'blue'))
    
    # 테이블: 보라색 1x3 Grid
    table_positions = [(5, 1), (6, 1), (7, 1)]
    for pos in table_positions:
        walls.append((pos[0], pos[1], 'purple'))
    
    # 시작점과 종료점
    start_pos = (1, 8)
    goal_pos = (8, 1)
    
    room_config = {
        'start_pos': start_pos,
        'goal_pos': goal_pos,
        'walls': walls,
        'objects': []
    }
    
    return CustomRoomWrapper(size=size, room_config=room_config)


def test_single_step():
    """단일 스텝 테스트"""
    print("=" * 80)
    print("성공한 솔루션 B 프롬프트 테스트")
    print("=" * 80)
    
    # 환경 생성
    print("\n[1] 환경 생성 중...")
    wrapper = create_scenario2_environment()
    wrapper.reset()
    
    state = wrapper.get_state()
    agent_pos = state['agent_pos']
    agent_dir = state['agent_dir']
    heading = wrapper.get_heading()
    
    print(f"에이전트 시작 위치: {agent_pos}")
    print(f"에이전트 방향: {agent_dir}")
    print(f"에이전트 Heading: {heading}")
    
    # VLM 초기화
    print("\n[2] VLM 초기화 중...")
    try:
        vlm = VLMWrapper(
            model=VLM_MODEL,
            temperature=VLM_TEMPERATURE,
            max_tokens=VLM_MAX_TOKENS
        )
        print(f"VLM 초기화 완료: {VLM_MODEL}")
    except Exception as e:
        print(f"VLM 초기화 실패: {e}")
        return
    
    # PostProcessor 초기화
    postprocessor = VLMResponsePostProcessor(required_fields=["action"])
    
    # System Prompt 생성
    system_prompt = get_system_prompt(wrapper)
    
    # 현재 이미지 가져오기
    image = wrapper.get_image()
    
    # User Prompt
    user_prompt = "Based on the current image, choose the next action to complete the mission: Go to the blue pillar, turn right, then stop next to the table."
    
    print("\n[3] VLM에 요청 전송 중...")
    print(f"Robot Heading: {heading}")
    print(f"User Prompt: {user_prompt}")
    
    try:
        vlm_response_raw = vlm.generate(
            image=image,
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        print("VLM 응답 수신 완료")
    except Exception as e:
        print(f"VLM API 호출 실패: {e}")
        return
    
    # 응답 파싱
    print("\n[4] 응답 파싱 중...")
    try:
        vlm_response = postprocessor.process(vlm_response_raw, strict=False)
        action_str = vlm_response.get('action', 'move forward')
        
        print(f"\n{'='*80}")
        print("VLM 응답 분석")
        print(f"{'='*80}")
        print(f"선택한 액션: {action_str}")
        
        # Reasoning trace 출력
        if 'reasoning_trace' in vlm_response:
            trace = vlm_response['reasoning_trace']
            print("\n[추론 과정 (Reasoning Trace)]")
            if isinstance(trace, dict):
                for step, content in trace.items():
                    print(f"  {step}: {content}")
            else:
                print(f"  {trace}")
        
        print(f"\nEnvironment Info: {vlm_response.get('environment_info', 'N/A')}")
        print(f"Reasoning: {vlm_response.get('reasoning', 'N/A')}")
        
        # GT 계산 (간단한 버전)
        print(f"\n{'='*80}")
        print("예상 정답 분석")
        print(f"{'='*80}")
        print(f"로봇 위치: {agent_pos}, 방향: {agent_dir} ({heading})")
        print(f"파란 기둥 위치: (3, 4), (4, 4), (3, 5), (4, 5)")
        print(f"로봇이 East를 향하고, 파란 기둥이 North에 있으므로")
        print(f"예상 정답: turn left (North when heading East = LEFT)")
        
        print(f"\n{'='*80}")
        if action_str.lower() in ['turn left', 'left']:
            print("✅ 정답! (turn left)")
        else:
            print(f"❌ 오답 (예상: turn left, 실제: {action_str})")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"응답 파싱 실패: {e}")
        print(f"원본 응답: {vlm_response_raw[:500]}...")
        import traceback
        traceback.print_exc()
    
    wrapper.close()
    print("\n테스트 완료.")


if __name__ == "__main__":
    try:
        test_single_step()
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()

