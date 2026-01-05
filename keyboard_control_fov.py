"""
키보드로 MiniGrid 환경을 제어하는 예제 스크립트 (시야 제한 기능 포함)

사용법:
    python keyboard_control_fov.py

조작법:
    - 'w': 앞으로 이동 (move forward)
    - 'a': 왼쪽으로 회전 (turn left)
    - 'd': 오른쪽으로 회전 (turn right)
    - 's': 뒤로 이동 (move backward) - 일부 환경에서만 지원
    - 'r': 환경 리셋
    - 'q': 종료
    - 'f': 시야 제한 토글 (켜기/끄기)
    - '+': 시야 범위 증가
    - '-': 시야 범위 감소

시야 제한 기능:
    - 시야 밖의 칸들은 검은색으로 표시됩니다
    - 에이전트가 바라보는 방향 기준으로 앞쪽만 볼 수 있습니다
"""

from minigrid import register_minigrid_envs
import gymnasium as gym
import cv2

# MiniGrid 환경 등록 (필수: 환경을 사용하기 전에 등록해야 함)
register_minigrid_envs()


class _GymEnvWrapper:
    """Gymnasium 환경을 CustomRoomWrapper와 호환되도록 감싸는 클래스"""
    def __init__(self, gym_env):
        self.env = gym_env
        # unwrapped를 통해 실제 MiniGrid 환경에 접근
        unwrapped = gym_env.unwrapped
        if hasattr(unwrapped, 'width'):
            self.size = unwrapped.width  # MiniGrid 환경은 width와 height가 같음
        elif hasattr(unwrapped, 'grid_size'):
            self.size = unwrapped.grid_size
        else:
            # 기본값 사용 (이미지 크기로 추정)
            self.size = 10
        
    def reset(self):
        obs, info = self.env.reset()
        return obs, info
    
    def step(self, action):
        return self.env.step(action)
    
    def get_image(self, fov_range=None, fov_width=None):
        """이미지 반환 (시야 제한은 아직 미지원)"""
        # fov_range, fov_width는 호환성을 위해 받지만 사용하지 않음
        _ = fov_range, fov_width
        return self.env.render()
    
    def close(self):
        self.env.close()


def _wrap_gym_env(gym_env):
    """Gymnasium 환경을 래퍼로 감싸기"""
    return _GymEnvWrapper(gym_env)


def get_key():
    """
    키보드 입력을 받아서 액션으로 변환하는 함수
    
    OpenCV의 waitKey()를 사용하여 키보드 입력을 받고,
    입력된 키에 따라 액션 인덱스 또는 특수 명령을 반환합니다.
    
    Returns:
        int 또는 str: 액션 인덱스 (0-3) 또는 특수 명령 ('quit', 'reset', 'toggle_fov', 'fov_inc', 'fov_dec')
        None: 유효하지 않은 키 입력 또는 키 입력 없음
    """
    # 키보드 입력 확인 (1ms 대기, -1이면 입력 없음)
    key = cv2.waitKey(1) & 0xFF
    
    # 키 입력이 없으면 None 반환
    if key == 0 or key == 255:  # 0 또는 255는 키 입력 없음
        return None
    
    # 'q' 키: 프로그램 종료
    if key == ord('q'):
        return 'quit'
    # 'r' 키: 환경 리셋
    elif key == ord('r'):
        return 'reset'
    # 'f' 키: 시야 제한 토글
    elif key == ord('f'):
        return 'toggle_fov'
    # '+' 키: 시야 범위 증가
    elif key == ord('+') or key == ord('='):
        return 'fov_inc'
    # '-' 키: 시야 범위 감소
    elif key == ord('-') or key == ord('_'):
        return 'fov_dec'
    # 'w' 키: 앞으로 이동 (액션 2)
    elif key == ord('w'):
        return 2  # move forward
    # 's' 키: 뒤로 이동 (액션 3)
    elif key == ord('s'):
        return 3  # move backward
    # 'a' 키: 왼쪽으로 회전 (액션 0)
    elif key == ord('a'):
        return 0  # turn left
    # 'd' 키: 오른쪽으로 회전 (액션 1)
    elif key == ord('d'):
        return 1  # turn right
    # 그 외의 키: 무시
    else:
        return None


def main():
    """
    메인 함수: 키보드로 환경을 제어하는 메인 루프 (시야 제한 기능 포함)
    
    이 함수는 다음 순서로 동작합니다:
    1. CustomRoomWrapper를 사용하여 환경 생성
    2. 환경 초기화 (reset)
    3. 메인 루프:
       - 현재 환경 이미지 가져오기 (시야 제한 선택적 적용)
       - OpenCV로 이미지 표시
       - 키보드 입력 받기
       - 액션 실행
       - 상태 출력
    """
    # MiniGrid 내장 환경 선택
    print("=" * 50)
    print("MiniGrid 내장 환경 선택:")
    print("  1: FourRooms (4개의 방 구조 - 중간 복잡도)")
    print("  2: MultiRoom-N6 (6개의 방 - 복잡)")
    print("  3: DoorKey-16x16 (문과 열쇠 - 복잡)")
    print("  4: KeyCorridorS6R3 (복도와 열쇠 - 중간 복잡도)")
    print("  5: Playground (놀이터 - 다양한 객체)")
    print("  6: Empty-16x16 (빈 환경 - 간단)")
    print("=" * 50)
    
    # 기본값: FourRooms
    env_choice = input("환경을 선택하세요 (1-6, 기본값=1): ").strip()
    if not env_choice:
        env_choice = "1"
    
    env_map = {
        "1": ("MiniGrid-FourRooms-v0", "FourRooms (4개의 방)"),
        "2": ("MiniGrid-MultiRoom-N6-v0", "MultiRoom-N6 (6개의 방)"),
        "3": ("MiniGrid-DoorKey-16x16-v0", "DoorKey-16x16 (문과 열쇠)"),
        "4": ("MiniGrid-KeyCorridorS6R3-v0", "KeyCorridorS6R3 (복도와 열쇠)"),
        "5": ("MiniGrid-Playground-v0", "Playground (놀이터)"),
        "6": ("MiniGrid-Empty-16x16-v0", "Empty-16x16 (빈 환경)"),
    }
    
    if env_choice in env_map:
        env_id, env_name = env_map[env_choice]
        print(f"\n{env_name} 환경을 생성합니다...")
        # Gymnasium 환경 생성
        gym_env = gym.make(env_id, render_mode='rgb_array')
        # CustomRoomWrapper로 감싸서 시야 제한 기능 사용 가능하게 함
        # 내장 환경을 직접 사용하므로 wrapper는 gym_env를 감싸는 간단한 래퍼
        wrapper = _wrap_gym_env(gym_env)
    else:
        print("\n잘못된 선택입니다. FourRooms 환경을 사용합니다.")
        gym_env = gym.make("MiniGrid-FourRooms-v0", render_mode='rgb_array')
        wrapper = _wrap_gym_env(gym_env)
        env_name = "FourRooms (4개의 방)"
    
    # 시야 제한 설정
    fov_enabled = True  # 시야 제한 활성화 여부
    fov_range = 3  # 앞으로 볼 수 있는 거리 (칸 수)
    fov_width = 3  # 시야의 좌우 폭 (칸 수)
    
    print("=" * 50)
    print(f"MiniGrid 키보드 제어 예제 (시야 제한 기능) - {env_name}")
    print("=" * 50)
    print("\n조작법:")
    print("  - w/a/s/d: 에이전트 이동 (w: 앞으로, a: 왼쪽 회전, d: 오른쪽 회전, s: 뒤로)")
    print("  - 'r': 환경 리셋")
    print("  - 'f': 시야 제한 토글 (켜기/끄기)")
    print("  - '+': 시야 범위 증가")
    print("  - '-': 시야 범위 감소")
    print("  - 'q': 종료")
    print(f"\n현재 시야 설정: {'활성화' if fov_enabled else '비활성화'}")
    print(f"시야 범위: 앞으로 {fov_range}칸, 좌우 각 {fov_width//2}칸")
    print(f"\n환경: {env_name} (크기: {wrapper.size}x{wrapper.size})")
    print("\n환경을 시작합니다...\n")
    
    # 환경 초기화
    wrapper.reset()
    done = False
    step_count = 0
    
    # 메인 루프: 키보드 입력을 받아 환경을 제어
    while True:
        # 환경 이미지 가져오기 (시야 제한 선택적 적용)
        if fov_enabled:
            img = wrapper.get_image(fov_range=fov_range, fov_width=fov_width)
        else:
            img = wrapper.get_image()
        
        # OpenCV로 이미지 표시
        # MiniGrid는 (H, W, C) 형식이므로 BGR로 변환
        if img is not None:
            # RGB를 BGR로 변환 (OpenCV는 BGR 형식을 사용)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # 이미지 크기 조정 (적절한 크기로 확대, 최대 크기 제한)
            # 원본 이미지가 256x256이므로 적절한 크기로 확대
            height, width = img_bgr.shape[:2]
            # 최대 크기를 800x800으로 제한하여 적절한 크기로 조정
            max_size = 800
            if height > max_size or width > max_size:
                # 이미지가 이미 크면 그대로 사용
                scale = 1
            else:
                # 적절한 scale 계산 (최대 800x800 이하)
                scale = min(max_size // height, max_size // width, 4)  # 최대 4배까지만 확대
            
            if scale > 1:
                new_width = width * scale
                new_height = height * scale
                img_bgr = cv2.resize(img_bgr, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            
            # 이미지 창에 표시
            cv2.imshow('MiniGrid Environment (FOV)', img_bgr)
            
            # cv2.waitKey()는 imshow() 직후에 호출되어야 키 입력을 받을 수 있음
            key = cv2.waitKey(30) & 0xFF  # 30ms 대기
            
            # 키 입력 처리
            if key != 0 and key != 255:  # 키 입력이 있으면
                action = None
                if key == ord('q'):
                    action = 'quit'
                elif key == ord('r'):
                    action = 'reset'
                elif key == ord('f'):
                    action = 'toggle_fov'
                elif key == ord('+') or key == ord('='):
                    action = 'fov_inc'
                elif key == ord('-') or key == ord('_'):
                    action = 'fov_dec'
                elif key == ord('w'):
                    action = 2  # move forward
                elif key == ord('s'):
                    action = 3  # move backward
                elif key == ord('a'):
                    action = 0  # turn left
                elif key == ord('d'):
                    action = 1  # turn right
            else:
                action = None
        else:
            action = None
        
        # 키 입력이 없으면 계속 렌더링만 수행
        if action is None:
            continue
        
        # 'q' 키 입력: 프로그램 종료
        if action == 'quit':
            print("\n프로그램을 종료합니다.")
            break
        # 'r' 키 입력: 환경 리셋
        elif action == 'reset':
            print("\n환경을 리셋합니다...")
            wrapper.reset()  # obs, info는 사용하지 않음
            done = False
            step_count = 0
            continue
        # 'f' 키 입력: 시야 제한 토글
        elif action == 'toggle_fov':
            fov_enabled = not fov_enabled
            print(f"\n시야 제한: {'활성화' if fov_enabled else '비활성화'}")
            continue
        # '+' 키 입력: 시야 범위 증가
        elif action == 'fov_inc':
            fov_range = min(fov_range + 1, 10)  # 최대 10칸
            fov_width = min(fov_width + 1, 10)  # 최대 10칸
            print(f"\n시야 범위 증가: 앞으로 {fov_range}칸, 좌우 각 {fov_width//2}칸")
            continue
        # '-' 키 입력: 시야 범위 감소
        elif action == 'fov_dec':
            fov_range = max(fov_range - 1, 1)  # 최소 1칸
            fov_width = max(fov_width - 1, 1)  # 최소 1칸
            print(f"\n시야 범위 감소: 앞으로 {fov_range}칸, 좌우 각 {fov_width//2}칸")
            continue
        
        # 액션 실행
        obs, reward, terminated, truncated, info = wrapper.step(action)
        done = terminated or truncated
        step_count += 1
        
        # 상태 출력 (디버깅 및 모니터링용)
        print(f"Step {step_count}: Action={action}, Reward={reward}, Done={done}")
        
        # 에피소드 종료 시 메시지 출력
        if done:
            print(f"\n에피소드 종료! 총 {step_count} 스텝, 최종 보상: {reward}")
            print("'r'를 눌러 리셋하거나 'q'를 눌러 종료하세요.")
    
    # 리소스 정리
    cv2.destroyAllWindows()  # OpenCV 창 닫기
    wrapper.close()  # 환경 종료 및 리소스 정리


if __name__ == "__main__":
    main()

