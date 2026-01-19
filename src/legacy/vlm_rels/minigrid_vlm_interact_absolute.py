"""
⚠️ 레거시 코드 ⚠️

MiniGrid VLM 상호작용 스크립트 (레거시 - 상대 움직임 기반 절대 움직임)

이 파일은 레거시 코드입니다. 새로운 프로젝트에서는 다음을 사용하세요:
- minigrid_vlm_interact_absolute_emoji.py (이모지 환경 지원)
- minigrid_customenv_emoji.MiniGridEmojiWrapper (직접 사용)

이 스크립트는 CustomRoomWrapper(상대 움직임)를 기반으로 절대 움직임을 구현한 레거시 버전입니다.
로봇이 상/하/좌/우로 직접 이동할 수 있는 절대 좌표 기반 액션 공간을 제공합니다.

사용법:
    python legacy/minigrid_vlm_interact_absolute.py

레거시 코드: CustomRoomWrapper 기반 (상대 움직임을 절대 움직임으로 변환)
새 표준: minigrid_customenv_emoji.MiniGridEmojiWrapper (네이티브 절대 움직임 지원)

이 파일은 하위 호환성을 위해 유지되지만, 새로운 코드에서는 사용하지 않는 것을 권장합니다.
"""

from minigrid import register_minigrid_envs
# Actual path: legacy.relative_movement.custom_environment_relative_movement
from legacy import CustomRoomWrapperRelative as CustomRoomWrapper
# Actual paths: utils.vlm.vlm_wrapper, utils.vlm.vlm_postprocessor
from utils import ChatGPT4oVLMWrapper, VLMResponsePostProcessor
import numpy as np
import cv2
from typing import Union, Tuple, Dict, Optional

# MiniGrid 환경 등록
register_minigrid_envs()

# VLM 설정
VLM_MODEL = "gpt-4o"
VLM_TEMPERATURE = 0.0
VLM_MAX_TOKENS = 1000


class AbsoluteDirectionWrapper(CustomRoomWrapper):
    """
    절대 방향(상/하/좌/우) 이동을 지원하는 Wrapper
    
    기존 CustomRoomWrapper를 확장하여 상/하/좌/우로 직접 이동할 수 있는
    액션 공간을 제공합니다. 로봇의 현재 방향과 관계없이 절대 좌표계 기준으로
    이동할 수 있습니다.
    """
    
    # 절대 방향 액션 이름과 인덱스 매핑
    ABSOLUTE_ACTION_NAMES = {
        0: "move up",      # North (위)
        1: "move down",    # South (아래)
        2: "move left",    # West (왼쪽)
        3: "move right",   # East (오른쪽)
        4: "pickup",
        5: "drop",
        6: "toggle"
    }
    
    # 절대 방향 액션 별칭
    ABSOLUTE_ACTION_ALIASES = {
        # 위 (North)
        "move up": 0, "up": 0, "north": 0, "n": 0, "move north": 0,
        "go up": 0, "go north": 0,
        # 아래 (South)
        "move down": 1, "down": 1, "south": 1, "s": 1, "move south": 1,
        "go down": 1, "go south": 1,
        # 왼쪽 (West)
        "move left": 2, "left": 2, "west": 2, "w": 2, "move west": 2,
        "go left": 2, "go west": 2,
        # 오른쪽 (East)
        "move right": 3, "right": 3, "east": 3, "e": 3, "move east": 3,
        "go right": 3, "go east": 3,
        # 기타 액션
        "pickup": 4, "pick up": 4, "pick_up": 4, "grab": 4,
        "drop": 5, "put down": 5, "put_down": 5, "release": 5,
        "toggle": 6, "interact": 6, "use": 6, "activate": 6
    }
    
    # MiniGrid 방향 매핑 (0=East, 1=South, 2=West, 3=North)
    DIRECTION_TO_AGENT_DIR = {
        "north": 3,  # 위
        "south": 1,  # 아래
        "west": 2,   # 왼쪽
        "east": 0    # 오른쪽
    }
    
    def __init__(self, *args, **kwargs):
        """절대 방향 Wrapper 초기화"""
        super().__init__(*args, **kwargs)
    
    def _get_target_direction(self, absolute_action: int) -> int:
        """
        절대 액션을 MiniGrid 방향으로 변환
        
        Args:
            absolute_action: 절대 액션 인덱스 (0=up, 1=down, 2=left, 3=right)
        
        Returns:
            target_dir: MiniGrid 방향 (0=East, 1=South, 2=West, 3=North)
        """
        direction_map = {
            0: 3,  # up -> North
            1: 1,  # down -> South
            2: 2,  # left -> West
            3: 0   # right -> East
        }
        return direction_map.get(absolute_action, 0)
    
    def _calculate_rotation(self, current_dir: int, target_dir: int) -> list:
        """
        현재 방향에서 목표 방향으로 회전하기 위한 액션 시퀀스 계산
        
        Args:
            current_dir: 현재 방향 (0=East, 1=South, 2=West, 3=North)
            target_dir: 목표 방향 (0=East, 1=South, 2=West, 3=North)
        
        Returns:
            rotation_actions: 회전 액션 리스트 (0=turn left, 1=turn right)
        """
        if current_dir == target_dir:
            return []  # 이미 올바른 방향
        
        # 방향 차이 계산
        diff = (target_dir - current_dir) % 4
        
        if diff == 1:
            # 시계 방향 90도 (오른쪽으로 1번 회전)
            return [1]  # turn right
        elif diff == 2:
            # 180도 회전 (오른쪽으로 2번 회전 또는 왼쪽으로 2번 회전)
            return [1, 1]  # turn right twice (더 짧은 경로)
        elif diff == 3:
            # 반시계 방향 90도 (왼쪽으로 1번 회전)
            return [0]  # turn left
        
        return []
    
    def step_absolute(self, action: Union[int, str]) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        절대 방향 액션을 실행
        
        Args:
            action: 절대 방향 액션 (정수 인덱스 또는 액션 이름 문자열)
                - 0 또는 "move up": 위로 이동 (North)
                - 1 또는 "move down": 아래로 이동 (South)
                - 2 또는 "move left": 왼쪽으로 이동 (West)
                - 3 또는 "move right": 오른쪽으로 이동 (East)
                - 4 또는 "pickup": 물체 집기
                - 5 또는 "drop": 물체 놓기
                - 6 또는 "toggle": 상호작용
        
        Returns:
            observation: 새로운 관찰 (딕셔너리)
            reward: 보상 (float)
            terminated: 목표 달성 여부 (bool)
            truncated: 시간 초과 여부 (bool)
            info: 추가 정보 (딕셔너리)
        """
        # 액션이 문자열인 경우 정수로 변환
        if isinstance(action, str):
            action = self.parse_absolute_action(action)
        
        # 이동 액션이 아닌 경우 (pickup, drop, toggle) 직접 실행
        if action >= 4:
            # 기존 MiniGrid 액션으로 변환 (4=pickup, 5=drop, 6=toggle)
            return super().step(action)
        
        # 이동 액션인 경우: 현재 방향 확인 후 필요한 회전 수행
        current_dir = self.env.agent_dir
        target_dir = self._get_target_direction(action)
        
        # 회전 액션 계산
        rotation_actions = self._calculate_rotation(current_dir, target_dir)
        
        # 회전 실행
        for rot_action in rotation_actions:
            obs, reward, terminated, truncated, info = super().step(rot_action)
            if terminated or truncated:
                return obs, reward, terminated, truncated, info
        
        # 목표 방향으로 회전 완료 후 전진
        obs, reward, terminated, truncated, info = super().step(2)  # move forward
        
        return obs, reward, terminated, truncated, info
    
    def parse_absolute_action(self, action_str: str) -> int:
        """
        절대 방향 액션 문자열을 인덱스로 변환
        
        Args:
            action_str: 액션 텍스트 (예: "move up", "left", "north" 등)
        
        Returns:
            action: 액션 인덱스 (0-6)
        
        Raises:
            ValueError: 알 수 없는 액션인 경우
        """
        # 공백 제거
        action_str = action_str.strip()
        
        # 숫자 문자열인 경우 직접 변환
        try:
            action_int = int(action_str)
            if 0 <= action_int <= 6:
                return action_int
        except ValueError:
            pass
        
        # 소문자로 변환
        action_str_lower = action_str.lower()
        
        # 액션 별칭에서 찾기
        if action_str_lower in self.ABSOLUTE_ACTION_ALIASES:
            return self.ABSOLUTE_ACTION_ALIASES[action_str_lower]
        
        # 직접 매핑에서 찾기
        for idx, name in self.ABSOLUTE_ACTION_NAMES.items():
            if action_str_lower == name.lower():
                return idx
        
        # 찾지 못한 경우 에러 발생
        raise ValueError(
            f"Unknown absolute action: '{action_str}'. "
            f"Available actions: {list(self.ABSOLUTE_ACTION_ALIASES.keys())} or numbers 0-6"
        )
    
    def get_absolute_action_space(self) -> Dict:
        """
        절대 방향 액션 공간 정보 반환
        
        Returns:
            action_space_info: 액션 공간 정보 딕셔너리
        """
        return {
            'n': 7,
            'actions': list(self.ABSOLUTE_ACTION_NAMES.values()),
            'action_mapping': self.ABSOLUTE_ACTION_NAMES,
            'action_aliases': self.ABSOLUTE_ACTION_ALIASES
        }


def get_system_prompt() -> str:
    """System Prompt 생성 (절대 좌표 버전)"""
    return """You are a robot operating on a grid map.

## Environment
Grid world with walls (black), blue pillar (impassable), purple table (impassable), robot (red arrow shows heading), and goal (green marker if present).

## Coordinate System
The top of the image is North (up), and the bottom is South (down).
The left is West (left), and the right is East (right).

## Robot Orientation
In the image, the red triangle represents the robot.
The robot's heading direction is shown by the triangle's apex (sharp tip).
However, you can move in ANY direction regardless of the robot's current heading.

## Action Space (Absolute Directions)
You can move directly in absolute directions:
- "move up" or "up" or "north" or "n": Move one cell North (upward)
- "move down" or "down" or "south" or "s": Move one cell South (downward)
- "move left" or "left" or "west" or "w": Move one cell West (leftward)
- "move right" or "right" or "east" or "e": Move one cell East (rightward)
- "pickup": Pick up object at current location
- "drop": Drop carried object
- "toggle": Interact with objects (e.g., open doors)

## Movement Rules
**CRITICAL**: All movements are in ABSOLUTE directions (North/South/East/West).
- "up" = move North (upward on the image)
- "down" = move South (downward on the image)
- "left" = move West (leftward on the image)
- "right" = move East (rightward on the image)
- The robot will automatically rotate to face the correct direction before moving
- You don't need to think about the robot's current heading - just specify the direction you want to go

## Response Format
Respond in JSON format:
```json
{
    "action": "<action_name_or_number>",
    "environment_info": "<description of current state with spatial relationships in absolute coordinates (North/South/East/West)>",
    "reasoning": "<explanation of why you chose this action>"
}
```

**Important**: 
- Valid JSON format required
- Actions must be from the list above
- Complete mission from user prompt
- Use absolute directions (up/down/left/right), not relative to robot heading
- Think in terms of the image: up=North, down=South, left=West, right=East
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
    
    return AbsoluteDirectionWrapper(size=size, room_config=room_config)


def visualize_grid_cli(wrapper: AbsoluteDirectionWrapper, state: dict):
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


def display_image(img, window_name="MiniGrid VLM Control (Absolute)", cell_size=32):
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
            print(f"Image display error: {e}")


def main():
    """메인 함수"""
    print("=" * 60)
    print("MiniGrid VLM Interaction (Absolute Coordinate Movement Version)")
    print("=" * 60)
    print("\nEnvironment Configuration:")
    print("  - Blue Pillar: 2x2 Grid")
    print("  - Table: Purple 1x3 Grid")
    print("  - Start Point: (1, 8)")
    print("  - End Point: (8, 1)")
    print("\nMission: Go to the blue pillar, turn right, then stop next to the table")
    print("\nAction Space: Direct movement in up/down/left/right (absolute coordinates)")
    
    # 환경 생성
    print("\n[1] Creating environment...")
    wrapper = create_scenario2_environment()
    wrapper.reset()
    
    state = wrapper.get_state()
    print(f"Agent start position: {state['agent_pos']}")
    print(f"Agent direction: {state['agent_dir']}")
    
    # 액션 공간 정보 출력
    action_space = wrapper.get_absolute_action_space()
    print(f"\nAbsolute Direction Action Space:")
    print(f"  - Available actions: {action_space['actions']}")
    
    # VLM 초기화
    print("\n[2] Initializing VLM...")
    try:
        vlm = ChatGPT4oVLMWrapper(
            model=VLM_MODEL,
            temperature=VLM_TEMPERATURE,
            max_tokens=VLM_MAX_TOKENS
        )
        print(f"VLM initialization completed: {VLM_MODEL}")
    except Exception as e:
        print(f"VLM initialization failed: {e}")
        return
    
    # PostProcessor 초기화
    postprocessor = VLMResponsePostProcessor(required_fields=["action", "environment_info"])
    
    # System Prompt
    SYSTEM_PROMPT = get_system_prompt()
    
    # 메인 루프
    step = 0
    done = False
    WINDOW_NAME = "MiniGrid VLM Control (Absolute)"
    
    print("\n" + "=" * 60)
    print("Experiment Started")
    print("=" * 60)
    
    while not done:
        step += 1
        print("\n" + "=" * 80)
        print(f"STEP {step}")
        print("=" * 80)
        
        # 현재 상태
        image = wrapper.get_image()
        state = wrapper.get_state()
        print(f"Position: {state['agent_pos']}, Direction: {state['agent_dir']}")
        
        # CLI 시각화
        visualize_grid_cli(wrapper, state)
        
        # GUI 시각화
        display_image(image, WINDOW_NAME)
        
        # 사용자 프롬프트 입력
        print("Enter command (Enter: default prompt):")
        user_prompt = input("> ").strip()
        if not user_prompt:
            user_prompt = "Based on the current image, choose the next action to complete the mission: Go to the blue pillar, turn right, then stop next to the table. Use absolute directions (up/down/left/right)."
        
        # VLM 호출
        print("\n[3] Sending request to VLM...")
        try:
            vlm_response_raw = vlm.generate(
                image=image,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt
            )
            print(f"VLM response received")
        except Exception as e:
            print(f"VLM API call failed: {e}")
            break
        
        # 응답 파싱
        print("[4] Parsing response...")
        try:
            vlm_response = postprocessor.process(vlm_response_raw, strict=True)
            action_str = vlm_response.get('action', 'up')
            print(f"Parsed action: {action_str}")
            print(f"Environment Info: {vlm_response.get('environment_info', 'N/A')}")
            print(f"Reasoning: {vlm_response.get('reasoning', 'N/A')}")
        except ValueError as e:
            print(f"Response parsing failed: {e}")
            print(f"Original response: {vlm_response_raw[:200]}...")
            action_str = 'up'  # 기본값: move up
        
        # 액션 실행
        print(f"\n[5] Executing action...")
        try:
            action_index = wrapper.parse_absolute_action(action_str)
            action_name = wrapper.ABSOLUTE_ACTION_NAMES.get(action_index, f"action_{action_index}")
            print(f"Action to execute: {action_name} (index: {action_index})")
            
            _, reward, terminated, truncated, _ = wrapper.step_absolute(action_index)
            done = terminated or truncated
            
            print(f"Reward: {reward}, Done: {done}")
        except Exception as e:
            print(f"Action execution failed: {e}")
            import traceback
            traceback.print_exc()
            # 기본 액션 사용
            try:
                _, reward, terminated, truncated, _ = wrapper.step_absolute(0)  # move up
                done = terminated or truncated
            except:
                break
        
        # 업데이트된 상태 표시
        new_state = wrapper.get_state()
        visualize_grid_cli(wrapper, new_state)
        updated_image = wrapper.get_image()
        display_image(updated_image, WINDOW_NAME)
        
        # 종료 확인
        if done:
            print("\n" + "=" * 80)
            print("Goal reached! Terminating")
            print("=" * 80)
            break
        
        # 최대 스텝 제한
        if step >= 100:
            print("\nMaximum step count (100) reached.")
            break
    
    # 리소스 정리
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

