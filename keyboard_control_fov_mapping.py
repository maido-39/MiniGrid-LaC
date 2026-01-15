"""
키보드로 MiniGrid 환경을 제어하는 예제 스크립트 (SLAM 스타일 FOV 맵핑 기능 포함)

사용법:
    python keyboard_control_fov_mapping.py

조작법:
    이동:
        - 'w': 앞으로 이동 (move forward)
        - 'a': 왼쪽으로 회전 (turn left)
        - 'd': 오른쪽으로 회전 (turn right)
    상호작용:
        - 'p' 또는 'e': 물체 집기 (pickup) - 열쇠 집기
        - 't' 또는 ' ' (스페이스): 상호작용 (toggle) - 문 열기/닫기
    기타:
        - 'r': 환경 리셋
        - 'f': 시야 제한 토글 (켜기/끄기)
        - '+': 시야 범위 증가
        - '-': 시야 범위 감소
        - 'q': 종료

SLAM 스타일 시야 제한 기능:
    - 현재 시야 범위 내: 밝게 표시
    - 탐색했던 곳 (시야 밖): 어둡게(반투명하게) 표시
    - 중요한 객체(열쇠, 문, 목표)가 있는 곳은 탐색했어도 밝게 유지
    - 아직 탐색하지 않은 곳: 검은색으로 표시
    - 에이전트가 바라보는 방향 기준으로 앞쪽만 볼 수 있습니다
"""

from minigrid import register_minigrid_envs
import gymnasium as gym
import cv2
import numpy as np
from typing import Optional

# MiniGrid 환경 등록 (필수: 환경을 사용하기 전에 등록해야 함)
register_minigrid_envs()


class SLAMFOVWrapper:
    """
    SLAM 스타일 FOV 맵핑을 지원하는 Gymnasium 환경 래퍼
    
    이 클래스는 탐색한 위치를 추적하고, 현재 시야 범위와 탐색한 영역을
    시각적으로 구분하여 표시합니다.
    """
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
        
        # 탐색한 위치를 추적하는 맵 (2D boolean 배열)
        self.visited_map = np.zeros((self.size, self.size), dtype=bool)
        
        # 현재 FOV 범위를 추적하는 맵 (매 프레임 업데이트)
        self.current_fov_map = np.zeros((self.size, self.size), dtype=bool)
        
    def reset(self):
        """환경 리셋 및 탐색 맵 초기화"""
        obs, info = self.env.reset()
        # 탐색 맵 초기화
        self.visited_map = np.zeros((self.size, self.size), dtype=bool)
        self.current_fov_map = np.zeros((self.size, self.size), dtype=bool)
        # 시작 위치는 이미 탐색한 것으로 표시
        self._update_visited_map()
        return obs, info
    
    def step(self, action):
        """액션 실행 및 탐색 맵 업데이트"""
        result = self.env.step(action)
        # 액션 실행 후 현재 위치를 탐색한 것으로 표시
        self._update_visited_map()
        return result
    
    def _update_visited_map(self):
        """현재 에이전트 위치를 탐색한 것으로 표시"""
        if not hasattr(self.env.unwrapped, 'agent_pos'):
            return
        
        agent_pos = self.env.unwrapped.agent_pos
        if isinstance(agent_pos, np.ndarray):
            agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
        else:
            agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
        
        # 현재 위치를 탐색한 것으로 표시
        if 0 <= agent_x < self.size and 0 <= agent_y < self.size:
            self.visited_map[agent_x, agent_y] = True
    
    def _get_fov_cells(self, fov_range: int, fov_width: int) -> np.ndarray:
        """
        현재 에이전트의 시야 범위 내 셀들을 반환
        
        Args:
            fov_range: 앞으로 볼 수 있는 거리 (칸 수)
            fov_width: 시야의 좌우 폭 (칸 수)
        
        Returns:
            fov_map: 현재 FOV 범위 내 셀들을 True로 표시한 2D boolean 배열
        """
        fov_map = np.zeros((self.size, self.size), dtype=bool)
        
        if not hasattr(self.env.unwrapped, 'agent_pos') or not hasattr(self.env.unwrapped, 'agent_dir'):
            return fov_map
        
        agent_pos = self.env.unwrapped.agent_pos
        if isinstance(agent_pos, np.ndarray):
            agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
        else:
            agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
        
        agent_dir = self.env.unwrapped.agent_dir
        
        # 각 셀에 대해 시야 범위 내인지 확인
        for grid_y in range(self.size):
            for grid_x in range(self.size):
                # 에이전트 위치에서 이 셀까지의 상대 위치
                dx = grid_x - agent_x
                dy = grid_y - agent_y
                
                # 에이전트 방향 기준으로 변환
                if agent_dir == 0:  # 오른쪽
                    rel_x, rel_y = dx, -dy  # y축 반전
                elif agent_dir == 1:  # 아래
                    rel_x, rel_y = dy, dx
                elif agent_dir == 2:  # 왼쪽
                    rel_x, rel_y = -dx, dy
                else:  # 위
                    rel_x, rel_y = -dy, -dx
                
                # 시야 범위 확인
                # 앞으로 fov_range 칸까지, 좌우로 각각 fov_width//2 칸까지
                in_fov = (
                    rel_x >= 0 and  # 앞쪽만
                    rel_x <= fov_range and  # 최대 거리
                    abs(rel_y) <= fov_width // 2  # 좌우 폭
                )
                
                if in_fov:
                    fov_map[grid_x, grid_y] = True
        
        return fov_map
    
    def get_image(self, fov_range: Optional[int] = None, fov_width: Optional[int] = None) -> np.ndarray:
        """
        SLAM 스타일 FOV 맵핑이 적용된 이미지 반환
        
        Args:
            fov_range: 앞으로 볼 수 있는 거리 (칸 수)
            fov_width: 시야의 좌우 폭 (칸 수)
        
        Returns:
            이미지: SLAM 스타일 시각화가 적용된 이미지
        """
        # 원본 이미지 가져오기
        image = self.env.render()
        if image is None:
            return np.zeros((self.size * 32, self.size * 32, 3), dtype=np.uint8)
        
        # FOV 파라미터가 제공되지 않으면 원본 이미지 반환
        if fov_range is None or fov_width is None:
            return image
        
        # SLAM 스타일 FOV 적용
        return self._apply_slam_fov(image, fov_range, fov_width)
    
    def _has_important_object(self, grid_x: int, grid_y: int) -> bool:
        """
        해당 셀에 중요한 객체(열쇠, 문, 목표 등)가 있는지 확인
        
        Args:
            grid_x: 그리드 X 좌표
            grid_y: 그리드 Y 좌표
        
        Returns:
            True: 중요한 객체가 있음, False: 없음
        """
        try:
            if not hasattr(self.env.unwrapped, 'grid'):
                return False
            
            cell = self.env.unwrapped.grid.get(grid_x, grid_y)
            if cell is None:
                return False
            
            # 객체 타입 확인
            obj_type = getattr(cell, 'type', None)
            if obj_type is None:
                return False
            
            # 중요한 객체 타입: key, door, goal
            important_types = ['key', 'door', 'goal']
            return obj_type in important_types
        except Exception:
            return False
    
    def _apply_slam_fov(self, image: np.ndarray, fov_range: int, fov_width: int) -> np.ndarray:
        """
        SLAM 스타일 FOV 맵핑 적용
        
        - 현재 FOV 내: 원본 이미지 그대로 (밝게)
        - 탐색했지만 FOV 밖: 어둡게(반투명하게) 표시 (단, 중요한 객체가 있으면 밝게 유지)
        - 탐색하지 않은 곳: 검은색
        
        Args:
            image: 원본 이미지 (H, W, 3)
            fov_range: 앞으로 볼 수 있는 거리
            fov_width: 시야의 좌우 폭
        
        Returns:
            masked_image: SLAM 스타일 시각화가 적용된 이미지
        """
        # 현재 FOV 범위 계산
        self.current_fov_map = self._get_fov_cells(fov_range, fov_width)
        
        # 이미지 복사 (원본 보존)
        masked_image = image.copy()
        h, w = image.shape[:2]
        
        # 각 셀의 크기 (MiniGrid는 일반적으로 32x32 픽셀)
        cell_size = 32
        
        # 각 셀에 대해 처리
        for grid_y in range(self.size):
            for grid_x in range(self.size):
                # 픽셀 좌표 계산
                pixel_x = grid_x * cell_size
                pixel_y = grid_y * cell_size
                end_x = min(pixel_x + cell_size, w)
                end_y = min(pixel_y + cell_size, h)
                
                in_fov = self.current_fov_map[grid_x, grid_y]
                visited = self.visited_map[grid_x, grid_y]
                has_important = self._has_important_object(grid_x, grid_y)
                
                # 셀 영역 가져오기
                cell_region = masked_image[pixel_y:end_y, pixel_x:end_x]
                
                if in_fov:
                    # 현재 FOV 내: 원본 이미지 그대로 (밝게)
                    # 탐색 맵에도 추가
                    self.visited_map[grid_x, grid_y] = True
                    # 이미 원본이므로 변경 없음
                elif visited:
                    # 탐색했지만 FOV 밖
                    if has_important:
                        # 중요한 객체(열쇠, 문, 목표)가 있으면 밝게 유지
                        # 원본 이미지 그대로 유지 (변경 없음)
                        pass
                    else:
                        # 일반 셀은 어둡게(반투명하게) 표시
                        # 어둡게 만들기 (밝기를 40%로 감소)
                        darkened = (cell_region * 0.4).astype(np.uint8)
                        masked_image[pixel_y:end_y, pixel_x:end_x] = darkened
                else:
                    # 탐색하지 않은 곳: 검은색
                    masked_image[pixel_y:end_y, pixel_x:end_x] = [0, 0, 0]
        
        return masked_image
    
    def close(self):
        """환경 종료 및 리소스 정리"""
        self.env.close()


def _wrap_gym_env(gym_env):
    """Gymnasium 환경을 SLAM FOV 래퍼로 감싸기"""
    return SLAMFOVWrapper(gym_env)


def get_key():
    """
    키보드 입력을 받아서 액션으로 변환하는 함수
    
    OpenCV의 waitKey()를 사용하여 키보드 입력을 받고,
    입력된 키에 따라 액션 인덱스 또는 특수 명령을 반환합니다.
    
    Returns:
        int 또는 str: 액션 인덱스 (0-5) 또는 특수 명령 ('quit', 'reset', 'toggle_fov', 'fov_inc', 'fov_dec')
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
    # 'a' 키: 왼쪽으로 회전 (액션 0)
    elif key == ord('a'):
        return 0  # turn left
    # 'd' 키: 오른쪽으로 회전 (액션 1)
    elif key == ord('d'):
        return 1  # turn right
    # 'p' 또는 'e' 키: 물체 집기 (액션 3) - 열쇠 집기
    elif key == ord('p') or key == ord('e'):
        return 3  # pickup
    # 't' 또는 ' ' (스페이스) 키: 상호작용 (액션 5) - 문 열기/닫기
    elif key == ord('t') or key == ord(' '):
        return 5  # toggle
    # 그 외의 키: 무시
    else:
        return None


def main():
    """
    메인 함수: 키보드로 환경을 제어하는 메인 루프 (SLAM 스타일 FOV 맵핑 기능 포함)
    
    이 함수는 다음 순서로 동작합니다:
    1. Gymnasium 환경 생성 및 SLAM FOV 래퍼로 감싸기
    2. 환경 초기화 (reset)
    3. 메인 루프:
       - 현재 환경 이미지 가져오기 (SLAM 스타일 FOV 맵핑 적용)
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
        # SLAM FOV 래퍼로 감싸기
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
    print(f"MiniGrid 키보드 제어 예제 (SLAM 스타일 FOV 맵핑) - {env_name}")
    print("=" * 50)
    print("\n조작법:")
    print("  이동:")
    print("    - 'w': 앞으로 이동 (move forward)")
    print("    - 'a': 왼쪽으로 회전 (turn left)")
    print("    - 'd': 오른쪽으로 회전 (turn right)")
    print("  상호작용:")
    print("    - 'p' 또는 'e': 물체 집기 (pickup) - 열쇠 집기")
    print("    - 't' 또는 ' ' (스페이스): 상호작용 (toggle) - 문 열기/닫기")
    print("  기타:")
    print("    - 'r': 환경 리셋")
    print("    - 'f': 시야 제한 토글 (켜기/끄기)")
    print("    - '+': 시야 범위 증가")
    print("    - '-': 시야 범위 감소")
    print("    - 'q': 종료")
    print(f"\n현재 시야 설정: {'활성화' if fov_enabled else '비활성화'}")
    print(f"시야 범위: 앞으로 {fov_range}칸, 좌우 각 {fov_width//2}칸")
    print("\nSLAM 스타일 시각화:")
    print("  - 현재 시야 범위 내: 밝게 표시")
    print("  - 탐색했던 곳 (시야 밖): 어둡게(반투명하게) 표시")
    print("  - 아직 탐색하지 않은 곳: 검은색으로 표시")
    print(f"\n환경: {env_name} (크기: {wrapper.size}x{wrapper.size})")
    print("\n환경을 시작합니다...\n")
    
    # 환경 초기화
    wrapper.reset()
    done = False
    step_count = 0
    
    # 메인 루프: 키보드 입력을 받아 환경을 제어
    while True:
        # 환경 이미지 가져오기 (SLAM 스타일 FOV 맵핑 선택적 적용)
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
            cv2.imshow('MiniGrid Environment (SLAM FOV Mapping)', img_bgr)
            
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
                elif key == ord('a'):
                    action = 0  # turn left
                elif key == ord('d'):
                    action = 1  # turn right
                elif key == ord('p') or key == ord('e'):
                    action = 3  # pickup
                elif key == ord('t') or key == ord(' '):
                    action = 5  # toggle
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
        _, reward, terminated, truncated, _ = wrapper.step(action)
        done = terminated or truncated
        step_count += 1
        
        # 상태 출력 (디버깅 및 모니터링용)
        visited_count = np.sum(wrapper.visited_map)
        total_cells = wrapper.size * wrapper.size
        action_names = {0: 'turn_left', 1: 'turn_right', 2: 'move_forward', 
                       3: 'pickup', 4: 'drop', 5: 'toggle', 6: 'done'}
        action_name = action_names.get(action, f'action_{action}')
        print(f"Step {step_count}: Action={action_name} ({action}), Reward={reward}, Done={done}, "
              f"탐색한 영역: {visited_count}/{total_cells} ({100*visited_count/total_cells:.1f}%)")
        
        # 에피소드 종료 시 메시지 출력
        if done:
            print(f"\n에피소드 종료! 총 {step_count} 스텝, 최종 보상: {reward}")
            print("'r'를 눌러 리셋하거나 'q'를 눌러 종료하세요.")
    
    # 리소스 정리
    cv2.destroyAllWindows()  # OpenCV 창 닫기
    wrapper.close()  # 환경 종료 및 리소스 정리


if __name__ == "__main__":
    main()

