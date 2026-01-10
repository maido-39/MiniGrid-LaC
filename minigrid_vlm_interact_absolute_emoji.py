"""
MiniGrid VLM 상호작용 스크립트 (절대 좌표 이동 버전 - 이모지 환경)

VLM을 사용하여 MiniGrid 환경을 제어하고 시각화합니다.
로봇이 상/하/좌/우로 직접 이동할 수 있는 절대 좌표 기반 액션 공간을 제공합니다.

환경 구성:
- 🧱(brick) 이모지: 2x2 Grid, 파란색, 올라설 수 있음
- 🖥️📱(desktop/workstation) 이모지: 1x2 Grid, 보라색, 올라설 수 있음

사용법:
    python minigrid_vlm_interact_absolute_emoji.py
"""

from minigrid import register_minigrid_envs
from minigrid_customenv_emoji import MiniGridEmojiWrapper
from vlm_wrapper import ChatGPT4oVLMWrapper
from vlm_postprocessor import VLMResponsePostProcessor
import numpy as np
import cv2
from typing import Union, Tuple, Dict, Optional

# MiniGrid 환경 등록
register_minigrid_envs()

# VLM 설정
VLM_MODEL = "gpt-4o"
VLM_TEMPERATURE = 0.0
VLM_MAX_TOKENS = 1000


class AbsoluteDirectionEmojiWrapper(MiniGridEmojiWrapper):
    """
    절대 방향(상/하/좌/우) 이동을 지원하는 이모지 Wrapper
    
    기존 MiniGridEmojiWrapper를 확장하여 상/하/좌/우로 직접 이동할 수 있는
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
        "move up": 0, "up": 0, "north": 0, "n": 0, "move north": 0,
        "go up": 0, "go north": 0,
        "move down": 1, "down": 1, "south": 1, "s": 1, "move south": 1,
        "go down": 1, "go south": 1,
        "move left": 2, "left": 2, "west": 2, "w": 2, "move west": 2,
        "go left": 2, "go west": 2,
        "move right": 3, "right": 3, "east": 3, "e": 3, "move east": 3,
        "go right": 3, "go east": 3,
        "pickup": 4, "pick up": 4, "pick_up": 4, "grab": 4,
        "drop": 5, "put down": 5, "put_down": 5, "release": 5,
        "toggle": 6, "interact": 6, "use": 6, "activate": 6
    }
    
    def __init__(self, *args, **kwargs):
        """절대 방향 Wrapper 초기화"""
        super().__init__(*args, **kwargs)
    
    def _get_target_direction(self, absolute_action: int) -> int:
        """절대 액션을 MiniGrid 방향으로 변환"""
        direction_map = {
            0: 3,  # up -> North
            1: 1,  # down -> South
            2: 2,  # left -> West
            3: 0   # right -> East
        }
        return direction_map.get(absolute_action, 0)
    
    def _calculate_rotation(self, current_dir: int, target_dir: int) -> list:
        """현재 방향에서 목표 방향으로 회전하기 위한 액션 시퀀스 계산"""
        if current_dir == target_dir:
            return []
        
        diff = (target_dir - current_dir) % 4
        
        if diff == 1:
            return [1]  # turn right
        elif diff == 2:
            return [1, 1]  # turn right twice
        elif diff == 3:
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
        """
        if isinstance(action, str):
            action = self.parse_absolute_action(action)
        
        if action >= 4:
            return self.step(action)
        
        current_dir = self.env.agent_dir
        target_dir = self._get_target_direction(action)
        
        rotation_actions = self._calculate_rotation(current_dir, target_dir)
        
        for rot_action in rotation_actions:
            obs, reward, terminated, truncated, info = self.step(rot_action)
            if terminated or truncated:
                return obs, reward, terminated, truncated, info
        
        obs, reward, terminated, truncated, info = self.step(2)  # move forward
        return obs, reward, terminated, truncated, info
    
    def parse_absolute_action(self, action_str: str) -> int:
        """절대 방향 액션 문자열을 인덱스로 변환"""
        action_str = action_str.strip()
        
        try:
            action_int = int(action_str)
            if 0 <= action_int <= 6:
                return action_int
        except ValueError:
            pass
        
        action_str_lower = action_str.lower()
        
        if action_str_lower in self.ABSOLUTE_ACTION_ALIASES:
            return self.ABSOLUTE_ACTION_ALIASES[action_str_lower]
        
        for idx, name in self.ABSOLUTE_ACTION_NAMES.items():
            if action_str_lower == name.lower():
                return idx
        
        raise ValueError(
            f"Unknown absolute action: '{action_str}'. "
            f"Available actions: {list(self.ABSOLUTE_ACTION_ALIASES.keys())} or numbers 0-6"
        )
    
    def get_absolute_action_space(self) -> Dict:
        """절대 방향 액션 공간 정보 반환"""
        return {
            'n': 7,
            'actions': list(self.ABSOLUTE_ACTION_NAMES.values()),
            'action_mapping': self.ABSOLUTE_ACTION_NAMES,
            'action_aliases': self.ABSOLUTE_ACTION_ALIASES
        }


def get_system_prompt() -> str:
    """System Prompt 생성 (절대 좌표 버전 - 이모지 환경)"""
    return """You are a robot operating on a grid map.

## Environment
Grid world with walls (black), blue brick emoji 🧱 (passable, you can step on it), purple desktop/workstation emoji 🖥️📱 (passable, you can step on it), robot (red arrow shows heading), and goal (green marker if present).

## Coordinate System
The top of the image is North (up), and the bottom is South (down).
The left is West (left), and the right is East (right).

## Robot Orientation
In the image, the red triangle represents the robot.
The robot's heading direction is shown by the triangle's apex (sharp tip).
However, you can move in ANY direction regardless of the robot's current heading.

## Action Space (Absolute Directions)
You can move directly in absolute directions:
- "up": Move North
- "down": Move South
- "left": Move West
- "right": Move East
- "pickup": Pick up object
- "drop": Drop object
- "toggle": Interact with objects

## Movement Rules
**CRITICAL**: All movements are in ABSOLUTE directions (North/South/East/West).
- "up" = move North (upward on the image)
- "down" = move South (downward on the image)
- "left" = move West (leftward on the image)
- "right" = move East (rightward on the image)
- The robot will automatically rotate to face the correct direction before moving
- You don't need to think about the robot's current heading - just specify the direction you want to go
- You can step on emoji objects (🧱 brick, 🖥️ desktop, 📱 workstation)
- When you step on an emoji object, the block will glow green

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
    """시나리오 2 환경 생성 (이모지 버전)"""
    size = 10
    
    # 외벽 생성
    walls = []
    for i in range(size):
        walls.append((i, 0))
        walls.append((i, size-1))
        walls.append((0, i))
        walls.append((size-1, i))
    
    # 파란 기둥: 2x2 Grid -> 🧱(brick) 이모지로 변경, 올라설 수 있게
    blue_pillar_positions = [(3, 4), (4, 4), (3, 5), (4, 5)]
    
    # 테이블: 보라색 1x3 Grid -> 🖥️📱 (1x2로 수정), 올라설 수 있게
    # 1x2로 수정: (5, 1), (6, 1) -> desktop과 workstation
    table_positions = [(5, 1), (6, 1)]
    
    # 시작점과 종료점
    start_pos = (1, 8)
    goal_pos = (8, 1)
    
    # 이모지 객체 생성
    objects = []
    
    # 🧱(brick) 이모지: 파란색, 올라설 수 있음
    for pos in blue_pillar_positions:
        objects.append({
            'type': 'emoji',
            'pos': pos,
            'emoji_name': 'brick',
            'color': 'blue',
            'can_pickup': False,
            'can_overlap': True,  # 올라설 수 있음
            'use_emoji_color': True  # 원래 이모지 색상 사용
        })
    
    # 🖥️📱(desktop/workstation) 이모지: 보라색, 올라설 수 있음
    objects.append({
        'type': 'emoji',
        'pos': (5, 1),
        'emoji_name': 'desktop',
        'color': 'purple',
        'can_pickup': False,
        'can_overlap': True,  # 올라설 수 있음
        'use_emoji_color': True
    })
    
    objects.append({
        'type': 'emoji',
        'pos': (6, 1),
        'emoji_name': 'workstation',
        'color': 'purple',
        'can_pickup': False,
        'can_overlap': True,  # 올라설 수 있음
        'use_emoji_color': True
    })
    
    room_config = {
        'start_pos': start_pos,
        'goal_pos': goal_pos,
        'walls': walls,
        'objects': objects
    }
    
    return AbsoluteDirectionEmojiWrapper(size=size, room_config=room_config)


def visualize_grid_cli(wrapper: AbsoluteDirectionEmojiWrapper, state: dict):
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
                if hasattr(cell, 'color'):
                    color_map = {
                        'blue': '🟦',
                        'purple': '🟪',
                        'red': '🟥',
                        'green': '🟩',
                        'yellow': '🟨'
                    }
                    row.append(color_map.get(cell.color, '⬛'))
                else:
                    row.append('⬛')
            elif cell is not None and cell.type == 'goal':
                row.append('🟩')
            elif cell is not None and cell.type == 'emoji':
                # 이모지 객체 표시
                if hasattr(cell, 'emoji_name'):
                    emoji_map = {
                        'brick': '🧱',
                        'desktop': '🖥️',
                        'workstation': '📱',
                        'tree': '🌲',
                        'mushroom': '🍄',
                        'flower': '🌼',
                        'cat': '🐈',
                        'grass': '🌾',
                        'rock': '🗿',
                        'box': '📦',
                        'chair': '🪑',
                        'apple': '🍎'
                    }
                    emoji_char = emoji_map.get(cell.emoji_name, '❓')
                    # 로봇이 위에 있으면 초록색 테두리 표시를 위해 특별 표시
                    if hasattr(cell, 'agent_on_top') and cell.agent_on_top:
                        row.append(f'[{emoji_char}]')  # 테두리 표시
                    else:
                        row.append(emoji_char)
                else:
                    row.append('❓')
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


def display_image(img, window_name="MiniGrid VLM Control (Absolute Emoji)", cell_size=32):
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
    print("MiniGrid VLM 상호작용 (절대 좌표 이동 버전 - 이모지 환경)")
    print("=" * 60)
    print("\n환경 구성:")
    print("  - 🧱(brick) 이모지: 2x2 Grid, 파란색, 올라설 수 있음")
    print("  - 🖥️📱(desktop/workstation) 이모지: 1x2 Grid, 보라색, 올라설 수 있음")
    print("  - 시작점: (1, 8)")
    print("  - 종료점: (8, 1)")
    print("\nMission: 파란 기둥(🧱)으로 가서 오른쪽으로 돌고, 테이블(🖥️📱) 옆에 멈추시오")
    print("\n액션 공간: 상/하/좌/우로 직접 이동 가능 (절대 좌표)")
    
    # 환경 생성
    print("\n[1] 환경 생성 중...")
    wrapper = create_scenario2_environment()
    wrapper.reset()
    
    state = wrapper.get_state()
    print(f"에이전트 시작 위치: {state['agent_pos']}")
    print(f"에이전트 방향: {state['agent_dir']}")
    
    # 액션 공간 정보 출력
    action_space = wrapper.get_absolute_action_space()
    print(f"\n절대 방향 액션 공간:")
    print(f"  - 사용 가능한 액션: {action_space['actions']}")
    
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
    WINDOW_NAME = "MiniGrid VLM Control (Absolute Emoji)"
    
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
            user_prompt = "Based on the current image, choose the next action to complete the mission: Go to the blue brick emoji 🧱, turn right, then stop next to the desktop/workstation emoji 🖥️📱. Use absolute directions (up/down/left/right)."
        
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
            action_str = vlm_response.get('action', 'up')
            print(f"파싱된 액션: {action_str}")
            print(f"Environment Info: {vlm_response.get('environment_info', 'N/A')}")
            print(f"Reasoning: {vlm_response.get('reasoning', 'N/A')}")
        except ValueError as e:
            print(f"응답 파싱 실패: {e}")
            print(f"원본 응답: {vlm_response_raw[:200]}...")
            action_str = 'up'  # 기본값: move up
        
        # 액션 실행
        print(f"\n[5] 액션 실행 중...")
        try:
            action_index = wrapper.parse_absolute_action(action_str)
            action_name = wrapper.ABSOLUTE_ACTION_NAMES.get(action_index, f"action_{action_index}")
            print(f"실행할 액션: {action_name} (인덱스: {action_index})")
            
            _, reward, terminated, truncated, _ = wrapper.step_absolute(action_index)
            done = terminated or truncated
            
            print(f"보상: {reward}, 종료: {done}")
        except Exception as e:
            print(f"액션 실행 실패: {e}")
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
