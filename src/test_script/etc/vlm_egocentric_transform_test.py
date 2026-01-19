"""
VLA Egocentric Transform Test

VLA 모델의 "Allocentric(절대 좌표) to Egocentric(상대 좌표) 변환 실패" 문제를 해결하기 위한 테스트 스크립트.

2가지 솔루션을 비교 테스트:
- 솔루션 B: CoT(Chain of Thought)를 통한 좌표 변환 강제
- 솔루션 C: Visual Prompting (이미지 전처리)

사용법:
    python vlm_egocentric_transform_test.py
"""

from minigrid import register_minigrid_envs
# Actual path: legacy.relative_movement.custom_environment
from legacy import CustomRoomWrapper
# Actual paths: utils.vlm.vlm_wrapper, utils.vlm.vlm_postprocessor
from utils import ChatGPT4oVLMWrapper, VLMResponsePostProcessor
import numpy as np
import cv2
import json
import random
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path
from PIL import Image
import os

# MiniGrid 환경 등록
register_minigrid_envs()

# VLM 설정
VLM_MODEL = "gpt-4o"
VLM_TEMPERATURE = 0.0
VLM_MAX_TOKENS = 1000

# Mission 설정
DEFAULT_MISSION = "Go to the blue pillar, turn right, then stop next to the table."


def create_scenario2_environment() -> CustomRoomWrapper:
    """시나리오 2 환경 생성"""
    size = 10
    
    walls = []
    for i in range(size):
        walls.append((i, 0))
        walls.append((i, size-1))
        walls.append((0, i))
        walls.append((size-1, i))
    
    blue_pillar_positions = [(3, 4), (4, 4), (3, 5), (4, 5)]
    for pos in blue_pillar_positions:
        walls.append((pos[0], pos[1], 'blue'))
    
    table_positions = [(5, 1), (6, 1), (7, 1)]
    for pos in table_positions:
        walls.append((pos[0], pos[1], 'purple'))
    
    start_pos = (1, 8)
    goal_pos = (8, 1)
    
    room_config = {
        'start_pos': start_pos,
        'goal_pos': goal_pos,
        'walls': walls,
        'objects': []
    }
    
    return CustomRoomWrapper(size=size, room_config=room_config)


def visualize_grid_cli(wrapper: CustomRoomWrapper, state: dict):
    """CLI에서 그리드를 텍스트로 시각화"""
    env = wrapper.env
    size = wrapper.size
    
    agent_pos = state['agent_pos']
    if isinstance(agent_pos, np.ndarray):
        agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
    else:
        agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
    
    agent_dir = state['agent_dir']
    direction_symbols = {0: '>', 1: 'v', 2: '<', 3: '^'}
    agent_symbol = direction_symbols.get(agent_dir, 'A')
    
    grid_chars = []
    for y in range(size):
        row = []
        for x in range(size):
            cell = env.grid.get(x, y)
            
            if x == agent_x and y == agent_y:
                row.append(agent_symbol)
            elif cell is not None and cell.type == 'wall':
                if hasattr(cell, 'color'):
                    if cell.color == 'blue':
                        row.append('🟦')
                    elif cell.color == 'purple':
                        row.append('🟪')
                    elif cell.color == 'red':
                        row.append('🟥')
                    elif cell.color == 'green':
                        row.append('🟩')
                    elif cell.color == 'yellow':
                        row.append('🟨')
                    else:
                        row.append('⬛')
                else:
                    row.append('⬛')
            elif cell is not None and cell.type == 'goal':
                row.append('🟩')
            elif cell is not None:
                if hasattr(cell, 'color'):
                    if cell.color == 'blue':
                        row.append('🟦')
                    elif cell.color == 'purple':
                        row.append('🟪')
                    else:
                        row.append('🟨')
                else:
                    row.append('🟨')
            else:
                row.append('⬜️')
        grid_chars.append(row)
    
    print("\n" + "=" * 60)
    print("Current Grid State:")
    print("=" * 60)
    for y in range(size):
        print(''.join(grid_chars[y]))
    print("=" * 60)
    print(f"Agent Position: ({agent_x}, {agent_y}), Direction: {agent_dir} ({agent_symbol})")
    print("=" * 60 + "\n")


def display_image(img: np.ndarray, window_name: str = "VLM Egocentric Transform Test"):
    """이미지 표시 (GUI 비활성화 - 서버 환경에서 실행 시 필요)"""
    # GUI 표시 비활성화 - 이미지는 파일로만 저장
    pass


class SolutionB_CoTReasoning:
    """솔루션 B: CoT(Chain of Thought)를 통한 좌표 변환 강제"""
    
    def __init__(self, vlm: ChatGPT4oVLMWrapper, postprocessor: VLMResponsePostProcessor):
        self.vlm = vlm
        self.postprocessor = postprocessor
    
    def get_heading_info(self, wrapper: CustomRoomWrapper) -> str:
        """Heading 정보 가져오기"""
        heading = wrapper.get_heading()
        heading_short = wrapper.get_heading_short()
        return f"{heading} ({heading_short})"
    
    def get_system_prompt(self, wrapper: CustomRoomWrapper) -> str:
        """CoT 강제 프롬프트 생성"""
        heading_info = self.get_heading_info(wrapper)
        
        return f"""You are a robot operating in a grid-based environment.

## Robot State (Authoritative)
- The robot's current heading is {heading_info}.
- Heading indicates the robot's forward-facing direction.
- This heading is ground-truth and MUST be used as-is.
- Do NOT infer or reinterpret the robot's heading from the image.

## Coordinate Convention
- Top of the image: North
- Bottom of the image: South
- Left of the image: West
- Right of the image: East

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
You must perform a mental coordinate transformation. Do NOT trust "Up" in the image as "Front".

1. **Identify Global Position**: Where is the target object in the image? (e.g., Top=North, Right=East)
2. **Confirm Robot Heading**: Which compass direction is the robot facing? (Provided in Robot State)
3. **Calculate Relative Position**:
   - IF Object is North AND Robot faces East -> Object is on the LEFT.
   - IF Object is North AND Robot faces West -> Object is on the RIGHT.
   - IF Object is East AND Robot faces North -> Object is on the RIGHT.
   - IF Object is West AND Robot faces North -> Object is on the LEFT.
   - (Derive strictly based on rotation)

Rules:
- All movements are RELATIVE to the robot's current heading.
- "move forward" moves one cell in the facing direction.
- "turn left/right" rotates 90° relative to current heading.
- Do NOT reason using absolute coordinates when choosing actions.

## Response Format (STRICT)
Respond in valid JSON. You MUST fill strictly following the "reasoning_trace" logic.

```json
{{
  "reasoning_trace": {{
    "target_global_pos": "<e.g. The blue pillar is at the Top (North) of the grid>",
    "robot_heading": "<e.g. East>",
    "calculation": "<e.g. North is 90 degrees counter-clockwise from East.>",
    "relative_pos": "<e.g. Therefore, the pillar is to my Left.>"
  }},
  "action": ["<action1>", "<action2>", "<action3>"]
}}
```

Important:
- EXACTLY 3 actions must be provided.
- Only the first action will be executed.
- Actions must come from the defined action space.
- Complete the reasoning_trace before selecting actions.
- Complete the mission specified by the user.
"""
    
    def test(self, image: np.ndarray, wrapper: CustomRoomWrapper, user_prompt: str) -> Dict:
        """솔루션 B 테스트 실행"""
        system_prompt = self.get_system_prompt(wrapper)
        
        print("\n[솔루션 B] CoT 강제 프롬프트로 VLM 호출 중...")
        try:
            raw_response = self.vlm.generate(
                image=image,
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            
            if not raw_response:
                print("VLM 응답이 비어있습니다.")
                return {}
            
            print("VLM 응답 수신 완료")
            parsed = self.postprocessor.process(raw_response, strict=False)
            return parsed
        except Exception as e:
            print(f"VLM API 호출 실패: {e}")
            return {}


class SolutionC_VisualPrompting:
    """솔루션 C: Visual Prompting (이미지 전처리)"""
    
    def __init__(self, vlm: ChatGPT4oVLMWrapper, postprocessor: VLMResponsePostProcessor):
        self.vlm = vlm
        self.postprocessor = postprocessor
    
    def get_heading_info(self, wrapper: CustomRoomWrapper) -> str:
        """Heading 정보 가져오기"""
        heading = wrapper.get_heading()
        heading_short = wrapper.get_heading_short()
        return f"{heading} ({heading_short})"
    
    def preprocess_image(self, image: np.ndarray, wrapper: CustomRoomWrapper) -> np.ndarray:
        """이미지에 Visual Prompting 추가 (로봇의 시야 방향 표시)"""
        # 이미지 복사
        processed_image = image.copy()
        
        # 로봇 위치 및 방향 가져오기
        state = wrapper.get_state()
        agent_pos = state['agent_pos']
        if isinstance(agent_pos, np.ndarray):
            agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
        else:
            agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
        
        agent_dir = state['agent_dir']
        heading = wrapper.get_heading()
        
        # 셀 크기 (MiniGrid는 일반적으로 32x32 픽셀)
        cell_size = 32
        
        # 로봇 위치를 픽셀 좌표로 변환
        robot_pixel_x = agent_x * cell_size + cell_size // 2
        robot_pixel_y = agent_y * cell_size + cell_size // 2
        
        # 방향 벡터 (0=오른쪽/East, 1=아래/South, 2=왼쪽/West, 3=위/North)
        dir_vectors = {
            0: (1, 0),   # East (오른쪽)
            1: (0, 1),   # South (아래)
            2: (-1, 0),  # West (왼쪽)
            3: (0, -1)   # North (위)
        }
        
        forward_dx, forward_dy = dir_vectors[agent_dir]
        
        # 화살표 그리기 (로봇의 앞 방향)
        arrow_length = cell_size * 2
        arrow_end_x = robot_pixel_x + forward_dx * arrow_length
        arrow_end_y = robot_pixel_y + forward_dy * arrow_length
        
        # 화살표 색상 (빨간색)
        arrow_color = (255, 0, 0)
        arrow_thickness = 3
        
        # 화살표 선 그리기
        cv2.arrowedLine(
            processed_image,
            (robot_pixel_x, robot_pixel_y),
            (arrow_end_x, arrow_end_y),
            arrow_color,
            arrow_thickness,
            tipLength=0.3
        )
        
        # "Front" 텍스트 추가
        text_x = arrow_end_x + forward_dx * 10
        text_y = arrow_end_y + forward_dy * 10
        cv2.putText(
            processed_image,
            "Front",
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            arrow_color,
            2
        )
        
        # Heading 정보 텍스트 추가 (좌상단)
        heading_text = f"Heading: {heading}"
        cv2.putText(
            processed_image,
            heading_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        return processed_image
    
    def get_system_prompt(self, wrapper: CustomRoomWrapper) -> str:
        """Visual Prompting 프롬프트 생성"""
        heading_info = self.get_heading_info(wrapper)
        
        return f"""You are a robot operating in a grid-based environment.

## Robot State (Authoritative)
- The robot's current heading is {heading_info}.
- Heading indicates the robot's forward-facing direction.
- This heading is ground-truth and MUST be used as-is.
- The image contains a RED ARROW pointing in the robot's forward direction.
- The arrow labeled "Front" shows where the robot is facing.

## Coordinate Convention
- Top of the image: North
- Bottom of the image: South
- Left of the image: West
- Right of the image: East

## Environment
Grid world with:
- Walls (black, impassable)
- Blue pillar (impassable)
- Purple table (impassable)
- Robot (red arrow marker)
- Goal (green marker, if present)

## Visual Cues in Image
- RED ARROW: Points in the robot's forward-facing direction (heading)
- "Front" label: Indicates the direction the robot is facing
- Use the arrow direction to determine relative positions, NOT the image orientation

## Action Space
- "turn left": Rotate 90° counterclockwise
- "turn right": Rotate 90° clockwise
- "move forward": Move one cell forward in heading direction
- "pickup": Pick up object in front
- "drop": Drop carried object
- "toggle": Interact with objects (e.g., open doors)

## Movement Rules (CRITICAL)
- All movements are RELATIVE to the robot's current heading (shown by the RED ARROW).
- The arrow direction is the robot's "forward" direction.
- Objects to the left/right of the arrow are on the robot's left/right.
- "move forward" moves one cell in the arrow direction.
- "turn left/right" rotates 90° relative to current heading.

## Response Format (STRICT)
Respond in valid JSON:

```json
{{
  "action": ["<action1>", "<action2>", "<action3>"],
  "reasoning": "<explanation of why you chose this action based on the arrow direction>"
}}
```

Important:
- EXACTLY 3 actions must be provided.
- Only the first action will be executed.
- Actions must come from the defined action space.
- Use the RED ARROW direction to determine relative positions.
- Complete the mission specified by the user.
"""
    
    def test(self, image: np.ndarray, wrapper: CustomRoomWrapper, user_prompt: str) -> Dict:
        """솔루션 C 테스트 실행"""
        # 이미지 전처리
        processed_image = self.preprocess_image(image, wrapper)
        
        system_prompt = self.get_system_prompt(wrapper)
        
        print("\n[솔루션 C] Visual Prompting으로 VLM 호출 중...")
        try:
            raw_response = self.vlm.generate(
                image=processed_image,
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            
            if not raw_response:
                print("VLM 응답이 비어있습니다.")
                return {}
            
            print("VLM 응답 수신 완료")
            parsed = self.postprocessor.process(raw_response, strict=False)
            return parsed
        except Exception as e:
            print(f"VLM API 호출 실패: {e}")
            return {}


class EgocentricTransformTest:
    """VLA Egocentric Transform 테스트 메인 클래스"""
    
    def __init__(self):
        self.wrapper = None
        self.vlm = ChatGPT4oVLMWrapper(
            model=VLM_MODEL,
            temperature=VLM_TEMPERATURE,
            max_tokens=VLM_MAX_TOKENS
        )
        self.postprocessor = VLMResponsePostProcessor(required_fields=["action"])
        
        self.solution_b = SolutionB_CoTReasoning(self.vlm, self.postprocessor)
        self.solution_c = SolutionC_VisualPrompting(self.vlm, self.postprocessor)
    
    def initialize(self):
        """테스트 초기화"""
        print("=" * 60)
        print("VLA Egocentric Transform Test")
        print("=" * 60)
        print("\n환경 구성:")
        print("  - 파란 기둥: 2x2 Grid (색상이 있는 벽)")
        print("  - 테이블: 보라색 1x3 Grid (색상이 있는 벽)")
        print("  - 시작점: (1, 8)")
        print("  - 종료점: (8, 1)")
        print(f"\nMission: {DEFAULT_MISSION}")
        
        print("\n[1] 환경 생성 중...")
        self.wrapper = create_scenario2_environment()
        self.wrapper.reset()
        
        state = self.wrapper.get_state()
        print(f"에이전트 시작 위치: {state['agent_pos']}")
        print(f"에이전트 방향: {state['agent_dir']}")
        heading = self.wrapper.get_heading()
        print(f"에이전트 Heading: {heading}")
        
        print("\n[2] VLM 초기화 완료")
        print("\n" + "=" * 60)
        print("테스트 시작")
        print("=" * 60)
    
    def run_comparison_test(self):
        """2가지 솔루션 비교 테스트"""
        # 환경 리셋하여 동일한 초기 상태 보장
        self.wrapper.reset()
        
        # 현재 상태 가져오기
        image = self.wrapper.get_image()
        state = self.wrapper.get_state()
        
        # Heading 정보 출력
        heading = self.wrapper.get_heading()
        heading_desc = self.wrapper.get_heading_description()
        print(f"\n위치: {state['agent_pos']}, 방향: {state['agent_dir']} ({heading})")
        print(f"현재 Heading: {heading_desc}")
        
        # 그리드 시각화
        visualize_grid_cli(self.wrapper, state)
        
        # 사용자 프롬프트
        user_prompt = f"Mission: {DEFAULT_MISSION}\n\nBased on the current image, choose the next action to complete this task."
        
        # 결과 저장
        results = {}
        
        # 솔루션 B 테스트
        print("\n" + "=" * 80)
        print("솔루션 B: CoT 강제 (Chain of Thought)")
        print("=" * 80)
        display_image(image, "Solution B: CoT Reasoning")
        
        result_b = self.solution_b.test(image, self.wrapper, user_prompt)
        results['solution_b'] = result_b
        
        if result_b:
            print("\n[솔루션 B 결과]")
            print("-" * 80)
            action = result_b.get('action', [])
            if isinstance(action, str):
                action = [action]
            if not isinstance(action, list):
                action = [str(action)]
            
            print(f"Action: {action[0] if action else 'N/A'}")
            
            reasoning_trace = result_b.get('reasoning_trace', {})
            if isinstance(reasoning_trace, dict):
                print(f"Target Global Pos: {reasoning_trace.get('target_global_pos', 'N/A')}")
                print(f"Robot Heading: {reasoning_trace.get('robot_heading', 'N/A')}")
                print(f"Calculation: {reasoning_trace.get('calculation', 'N/A')}")
                print(f"Relative Pos: {reasoning_trace.get('relative_pos', 'N/A')}")
            else:
                print(f"Reasoning Trace: {reasoning_trace}")
        
        # 솔루션 C 테스트 (환경 리셋하여 동일한 초기 상태 보장)
        self.wrapper.reset()
        image_c = self.wrapper.get_image()
        
        print("\n" + "=" * 80)
        print("솔루션 C: Visual Prompting (이미지 전처리)")
        print("=" * 80)
        
        result_c = self.solution_c.test(image_c, self.wrapper, user_prompt)
        results['solution_c'] = result_c
        
        # 전처리된 이미지 표시
        processed_image = self.solution_c.preprocess_image(image_c, self.wrapper)
        display_image(processed_image, "Solution C: Visual Prompting")
        
        if result_c:
            print("\n[솔루션 C 결과]")
            print("-" * 80)
            action = result_c.get('action', [])
            if isinstance(action, str):
                action = [action]
            if not isinstance(action, list):
                action = [str(action)]
            
            print(f"Action: {action[0] if action else 'N/A'}")
            print(f"Reasoning: {result_c.get('reasoning', 'N/A')}")
        
        # 결과 비교
        print("\n" + "=" * 80)
        print("결과 비교")
        print("=" * 80)
        
        action_b = results.get('solution_b', {}).get('action', [])
        if isinstance(action_b, str):
            action_b = [action_b]
        if not isinstance(action_b, list):
            action_b = [str(action_b)]
        action_b = action_b[0] if action_b else None
        
        action_c = results.get('solution_c', {}).get('action', [])
        if isinstance(action_c, str):
            action_c = [action_c]
        if not isinstance(action_c, list):
            action_c = [str(action_c)]
        action_c = action_c[0] if action_c else None
        
        print(f"솔루션 B (CoT) 선택한 액션: {action_b}")
        print(f"솔루션 C (Visual) 선택한 액션: {action_c}")
        
        # 예상 정답: 로봇이 East를 향하고, 파란 기둥이 North에 있으므로 "turn left"가 정답
        expected_action = "turn left"
        print(f"\n예상 정답: {expected_action} (로봇이 East를 향하고, 파란 기둥이 North에 있으므로 왼쪽으로 회전)")
        
        if action_b == expected_action:
            print("✓ 솔루션 B: 정답!")
        else:
            print(f"✗ 솔루션 B: 오답 (예상: {expected_action})")
        
        if action_c == expected_action:
            print("✓ 솔루션 C: 정답!")
        else:
            print(f"✗ 솔루션 C: 오답 (예상: {expected_action})")
        
        return results
    
    def cleanup(self):
        """리소스 정리"""
        # GUI 비활성화로 cv2.destroyAllWindows() 제거
        if self.wrapper:
            self.wrapper.close()
        print("\n테스트 완료.")


def main():
    """메인 함수"""
    try:
        test = EgocentricTransformTest()
        test.initialize()
        test.run_comparison_test()
        test.cleanup()
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

