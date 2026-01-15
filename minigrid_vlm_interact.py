"""
MiniGrid VLM 상호작용 스크립트 (간소화 버전)

VLM을 사용하여 MiniGrid 환경을 제어하고 시각화합니다.
로깅, 메모리, 그라운딩 등 복잡한 기능은 제거하고 핵심 기능만 포함합니다.

사용법:
    python minigrid_vlm_interact.py
"""

from minigrid import register_minigrid_envs
from legacy.custom_environment_relative_movement import CustomRoomWrapper
from vlm_wrapper import ChatGPT4oVLMWrapper
from vlm_postprocessor import VLMResponsePostProcessor
import numpy as np
import cv2

# MiniGrid 환경 등록
register_minigrid_envs()

# VLM 설정
VLM_MODEL = "gpt-4o"
VLM_TEMPERATURE = 0.0
VLM_MAX_TOKENS = 1000


def get_system_prompt() -> str:
    """System Prompt 생성"""
    return """You are a robot operating on a grid map.

## Environment
Grid world with walls (black), blue pillar (impassable), purple table (impassable), robot (red arrow shows heading), and goal (green marker if present).

## Robot Orientation
In the image, the red triangle represents the robot.
The robot's heading direction is defined as the direction pointed by the triangle's apex (sharp tip).
The top of the image is North, and the bottom is South.
The left is West, and the right is East.

## Action Space
- "turn left": Rotate 90° counterclockwise
- "turn right": Rotate 90° clockwise
- "move forward": Move one cell forward in heading direction
- "pickup": Pick up object in front
- "drop": Drop carried object
- "toggle": Interact with objects (e.g., open doors)

## Movement Rules
**CRITICAL**: All movements are RELATIVE to robot's current heading direction.
- "forward" = move one cell in facing direction
- "turn left/right" = rotate 90° from current heading
- Think in relative movements, NOT absolute coordinates



## Response Format
Respond in JSON format:
```json
{
    "action": "<action_name_or_number>",
    "environment_info": "<description of current state with spatial relationships relative to robot heading orientation>",
    "reasoning": "<explanation of why you chose this action>"
}
```

**Important**: 
- Valid JSON format required
- Actions must be from the list above
- Complete mission from user prompt
- Use relative movements based on heading, not coordinates
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
    
    # 파란 기둥: 2x2 Grid (색상이 있는 벽으로 변경)
    blue_pillar_positions = [(3, 4), (4, 4), (3, 5), (4, 5)]
    for pos in blue_pillar_positions:
        walls.append((pos[0], pos[1], 'blue'))
    
    # 테이블: 보라색 1x3 Grid (색상이 있는 벽으로 변경)
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
        'objects': []  # box 객체 제거
    }
    
    return CustomRoomWrapper(size=size, room_config=room_config)


def visualize_grid_cli(wrapper: CustomRoomWrapper, state: dict):
    """CLI에서 그리드를 텍스트로 시각화"""
    env = wrapper.env
    size = wrapper.size
    
    # 에이전트 위치 및 방향
    agent_pos = state['agent_pos']
    if isinstance(agent_pos, np.ndarray):
        agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
    else:
        agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
    
    agent_dir = state['agent_dir']
    direction_symbols = {0: '>', 1: 'v', 2: '<', 3: '^'}
    agent_symbol = direction_symbols.get(agent_dir, 'A')
    
    # 그리드 생성
    grid_chars = []
    for y in range(size):
        row = []
        for x in range(size):
            cell = env.grid.get(x, y)
            
            if x == agent_x and y == agent_y:
                row.append(agent_symbol)
            elif cell is not None and cell.type == 'wall':
                # 색상이 있는 벽 표시
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
                        row.append('⬛')  # 기본 색상 (grey)
                else:
                    row.append('⬛')  # 색상 없음
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
    
    # 그리드 출력
    print("\n" + "=" * 60)
    print("Current Grid State:")
    print("=" * 60)
    for y in range(size):
        print(''.join(grid_chars[y]))
    print("=" * 60)
    print(f"Agent Position: ({agent_x}, {agent_y}), Direction: {agent_dir} ({agent_symbol})")
    print("=" * 60 + "\n")


def display_image(img, window_name="MiniGrid VLM Control", cell_size=32):
    """OpenCV를 사용하여 이미지 표시"""
    if img is not None:
        try:
            img_bgr = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
            
            # 이미지 크기 조정
            height, width = img_bgr.shape[:2]
            max_size = 800
            if height < max_size and width < max_size:
                scale = min(max_size // height, max_size // width, 4)
                if scale > 1:
                    new_width = width * scale
                    new_height = height * scale
                    img_bgr = cv2.resize(img_bgr, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            
            cv2.imshow(window_name, img_bgr)
            cv2.waitKey(1)
        except Exception as e:
            print(f"이미지 표시 오류: {e}")


def main():
    """메인 함수"""
    print("=" * 60)
    print("MiniGrid VLM 상호작용")
    print("=" * 60)
    print("\n환경 구성:")
    print("  - 파란 기둥: 2x2 Grid")
    print("  - 테이블: 보라색 1x3 Grid")
    print("  - 시작점: (1, 8)")
    print("  - 종료점: (8, 1)")
    print("\nMission: 파란 기둥으로 가서 오른쪽으로 돌고, 테이블 옆에 멈추시오")
    
    # 환경 생성
    print("\n[1] 환경 생성 중...")
    wrapper = create_scenario2_environment()
    wrapper.reset()
    
    state = wrapper.get_state()
    print(f"에이전트 시작 위치: {state['agent_pos']}")
    print(f"에이전트 방향: {state['agent_dir']}")
    
    # VLM 초기화
    print("\n[2] VLM 초기화 중...")
    try:
        vlm = ChatGPT4oVLMWrapper(
            model=VLM_MODEL,
            temperature=VLM_TEMPERATURE,
            max_tokens=VLM_MAX_TOKENS
        )
        print(f"VLM 초기화 완료: {VLM_MODEL}")
    except Exception as e:
        print(f"VLM 초기화 실패: {e}")
        return
    
    # PostProcessor 초기화
    postprocessor = VLMResponsePostProcessor(required_fields=["action", "environment_info"])
    
    # System Prompt
    SYSTEM_PROMPT = get_system_prompt()
    
    # 메인 루프
    step = 0
    done = False
    WINDOW_NAME = "MiniGrid VLM Control"
    
    print("\n" + "=" * 60)
    print("실험 시작")
    print("=" * 60)
    
    while not done:
        step += 1
        print("\n" + "=" * 80)
        print(f"STEP {step}")
        print("=" * 80)
        
        # 현재 상태
        image = wrapper.get_image()
        state = wrapper.get_state()
        print(f"위치: {state['agent_pos']}, 방향: {state['agent_dir']}")
        
        # CLI 시각화
        visualize_grid_cli(wrapper, state)
        
        # GUI 시각화
        display_image(image, WINDOW_NAME)
        
        # 사용자 프롬프트 입력
        print("명령을 입력하세요 (Enter: 기본 프롬프트):")
        user_prompt = input("> ").strip()
        if not user_prompt:
            user_prompt = "Based on the current image, choose the next action to complete the mission: Go to the blue pillar, turn right, then stop next to the table."
        
        # VLM 호출
        print("\n[3] VLM에 요청 전송 중...")
        try:
            vlm_response_raw = vlm.generate(
                image=image,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt
            )
            print(f"VLM 응답 수신 완료")
        except Exception as e:
            print(f"VLM API 호출 실패: {e}")
            break
        
        # 응답 파싱
        print("[4] 응답 파싱 중...")
        try:
            vlm_response = postprocessor.process(vlm_response_raw, strict=True)
            action_str = vlm_response.get('action', '2')
            print(f"파싱된 액션: {action_str}")
            print(f"Environment Info: {vlm_response.get('environment_info', 'N/A')}")
            print(f"Reasoning: {vlm_response.get('reasoning', 'N/A')}")
        except ValueError as e:
            print(f"응답 파싱 실패: {e}")
            print(f"원본 응답: {vlm_response_raw[:200]}...")
            action_str = '2'  # 기본값: move forward
        
        # 액션 실행
        print(f"\n[5] 액션 실행 중...")
        try:
            action_index = wrapper.parse_action(action_str)
            action_name = wrapper.ACTION_NAMES.get(action_index, f"action_{action_index}")
            print(f"실행할 액션: {action_name} (인덱스: {action_index})")
            
            _, reward, terminated, truncated, _ = wrapper.step(action_index)
            done = terminated or truncated
            
            print(f"보상: {reward}, 종료: {done}")
        except Exception as e:
            print(f"액션 실행 실패: {e}")
            # 기본 액션 사용
            _, reward, terminated, truncated, _ = wrapper.step(2)
            done = terminated or truncated
        
        # 업데이트된 상태 표시
        new_state = wrapper.get_state()
        visualize_grid_cli(wrapper, new_state)
        updated_image = wrapper.get_image()
        display_image(updated_image, WINDOW_NAME)
        
        # 종료 확인
        if done:
            print("\n" + "=" * 80)
            print("Goal 도착! 종료")
            print("=" * 80)
            break
        
        # 최대 스텝 제한
        if step >= 100:
            print("\n최대 스텝 수(100)에 도달했습니다.")
            break
    
    # 리소스 정리
    cv2.destroyAllWindows()
    wrapper.close()
    print("\n실험 완료.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()

