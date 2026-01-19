"""
시나리오 2 키보드 제어 스크립트 (절대 좌표 이동 버전)

JSON 파일에서 맵을 로드하고 키보드로 절대 방향 이동을 테스트합니다.

사용법:
    python scenario2_keyboard_control.py [json_map_path]
    예: python scenario2_keyboard_control.py example_map.json

조작법:
    - 화살표 키 또는 w/a/s/d: 절대 방향 이동
      - ↑ 또는 'w': 위로 이동 (North)
      - ↓ 또는 's': 아래로 이동 (South)
      - ← 또는 'a': 왼쪽으로 이동 (West)
      - → 또는 'd': 오른쪽으로 이동 (East)
    - 'r': 환경 리셋
    - 'q': 종료
"""

from minigrid import register_minigrid_envs
# Actual paths: utils.map_manager.minigrid_customenv_emoji, utils.map_manager.emoji_map_loader
from utils import MiniGridEmojiWrapper, load_emoji_map_from_json
import numpy as np
import cv2
from pathlib import Path
import sys

# MiniGrid 환경 등록
register_minigrid_envs()


class Visualizer:
    """시각화 클래스"""
    
    def __init__(self, window_name: str = "Scenario 2: Keyboard Control (Absolute)"):
        self.window_name = window_name
    
    def visualize_grid_cli(self, wrapper: MiniGridEmojiWrapper, state: dict):
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
        
        print("\n" + "=" * 60)
        print("Current Grid State:")
        print("=" * 60)
        for y in range(size):
            print(''.join(grid_chars[y]))
        print("=" * 60)
        print(f"Agent Position: ({agent_x}, {agent_y}), Direction: {agent_dir} ({agent_symbol})")
        print("=" * 60 + "\n")
    
    def display_image(self, img: np.ndarray):
        """OpenCV를 사용하여 이미지 표시"""
        if img is None:
            return
        
        try:
            img_bgr = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
            
            height, width = img_bgr.shape[:2]
            max_size = 800
            if height < max_size and width < max_size:
                scale = min(max_size // height, max_size // width, 4)
                if scale > 1:
                    new_width = width * scale
                    new_height = height * scale
                    img_bgr = cv2.resize(img_bgr, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            
            cv2.imshow(self.window_name, img_bgr)
            cv2.waitKey(1)
        except Exception as e:
            print(f"이미지 표시 오류: {e}")
    
    def cleanup(self):
        """리소스 정리"""
        cv2.destroyAllWindows()


def get_keyboard_action():
    """
    키보드 입력을 받아서 절대 방향 액션으로 변환
    
    Returns:
        int 또는 str: 절대 방향 액션 인덱스 (0-6) 또는 특수 명령 ('quit', 'reset')
        None: 유효하지 않은 키 입력 또는 키 입력 없음
    """
    key = cv2.waitKey(30) & 0xFF
    
    if key == 0 or key == 255:
        return None
    
    # 특수 명령
    if key == ord('q'):
        return 'quit'
    elif key == ord('r'):
        return 'reset'
    
    # 절대 방향 이동 (화살표 키)
    # OpenCV는 특수 키를 직접 지원하지 않으므로 일반 키 사용
    # 위 (North)
    if key == ord('w') or key == ord('W'):
        return 0  # move up
    # 아래 (South)
    elif key == ord('s') or key == ord('S'):
        return 1  # move down
    # 왼쪽 (West)
    elif key == ord('a') or key == ord('A'):
        return 2  # move left
    # 오른쪽 (East)
    elif key == ord('d') or key == ord('D'):
        return 3  # move right
    
    # 기타 액션
    elif key == ord('p'):
        return 4  # pickup
    elif key == ord('x'):
        return 5  # drop
    elif key == ord('t'):
        return 6  # toggle
    
    return None


def main():
    """메인 함수"""
    # 명령줄 인자로 JSON 맵 파일 경로 지정
    json_map_path = "../../config/example_map.json"
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("사용법:")
            print("  python scenario2_keyboard_control.py [json_map_path]")
            print("  예: python scenario2_keyboard_control.py ../../config/example_map.json")
            print("\n조작법:")
            print("  - w/a/s/d: 절대 방향 이동 (w: 위, s: 아래, a: 왼쪽, d: 오른쪽)")
            print("  - p: pickup, x: drop, t: toggle")
            print("  - r: 환경 리셋")
            print("  - q: 종료")
            return
        else:
            json_map_path = sys.argv[1]
    
    print("=" * 60)
    print("시나리오 2: 키보드 제어 (절대 좌표 이동 버전)")
    print("=" * 60)
    print(f"\n맵 파일: {json_map_path}")
    print("\n조작법:")
    print("  - w: 위로 이동 (North)")
    print("  - s: 아래로 이동 (South)")
    print("  - a: 왼쪽으로 이동 (West)")
    print("  - d: 오른쪽으로 이동 (East)")
    print("  - p: pickup, x: drop, t: toggle")
    print("  - r: 환경 리셋")
    print("  - q: 종료")
    print("\n환경 생성 중...")
    
    # 환경 생성
    wrapper = load_emoji_map_from_json(json_map_path)
    wrapper.reset()
    
    state = wrapper.get_state()
    print(f"에이전트 시작 위치: {state['agent_pos']}")
    print(f"에이전트 방향: {state['agent_dir']}")
    
    # 액션 공간 정보 출력
    action_space = wrapper.get_absolute_action_space()
    print(f"\n절대 방향 액션 공간:")
    print(f"  - 사용 가능한 액션: {action_space['actions']}")
    print("\n" + "=" * 60)
    print("키보드 제어 시작")
    print("=" * 60)
    
    visualizer = Visualizer()
    step_count = 0
    done = False
    
    # 메인 루프
    while True:
        # 현재 상태 가져오기
        image = wrapper.get_image()
        state = wrapper.get_state()
        
        # CLI 시각화
        visualizer.visualize_grid_cli(wrapper, state)
        
        # GUI 시각화
        visualizer.display_image(image)
        
        # 키보드 입력 받기
        action = get_keyboard_action()
        
        if action is None:
            continue
        
        # 특수 명령 처리
        if action == 'quit':
            print("\n프로그램을 종료합니다.")
            break
        elif action == 'reset':
            print("\n환경을 리셋합니다...")
            wrapper.reset()
            state = wrapper.get_state()
            step_count = 0
            done = False
            print(f"에이전트 위치: {state['agent_pos']}, 방향: {state['agent_dir']}")
            continue
        
        # 액션 실행
        try:
            action_name = action_space['action_mapping'].get(action, f"action_{action}")
            print(f"\n[Step {step_count + 1}] 액션 실행: {action_name} (인덱스: {action})")
            
            _, reward, terminated, truncated, _ = wrapper.step(action)
            done = terminated or truncated
            step_count += 1
            
            # 업데이트된 상태
            new_state = wrapper.get_state()
            print(f"위치: {new_state['agent_pos']}, 방향: {new_state['agent_dir']}")
            print(f"보상: {reward}, 종료: {done}")
            
            if done:
                print("\n" + "=" * 60)
                print("Goal 도착! 에피소드 종료")
                print("=" * 60)
                print("'r'를 눌러 리셋하거나 'q'를 눌러 종료하세요.")
        
        except Exception as e:
            print(f"액션 실행 실패: {e}")
            import traceback
            traceback.print_exc()
    
    # 리소스 정리
    visualizer.cleanup()
    wrapper.close()
    print("\n프로그램 종료.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()

