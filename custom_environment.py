"""
MiniGrid 환경을 위한 Gym Wrapper

이 모듈은 MiniGrid 환경을 쉽게 생성하고 제어할 수 있는 Wrapper 클래스를 제공합니다.
VLM(Vision Language Model)과의 연동을 고려하여 설계되었습니다.

주요 기능:
- 환경 초기화 시 size, walls, room_config 등을 지정
- 현재 환경 이미지 반환 (VLM 입력용)
- 액션 공간 제어 API
- VLM이 반환한 텍스트를 액션으로 변환
"""

from minigrid import register_minigrid_envs
from minigrid.core.grid import Grid
from minigrid.core.world_object import Wall, Goal, Key, Ball, Box, Door
from minigrid.core.mission import MissionSpace
from minigrid.minigrid_env import MiniGridEnv
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt

# MiniGrid 환경 등록 (필수: 환경을 사용하기 전에 등록해야 함)
register_minigrid_envs()


class CustomRoomEnv(MiniGridEnv):
    """
    커스텀 방 구조를 가진 MiniGrid 환경 클래스
    
    이 클래스는 MiniGridEnv를 상속받아 커스텀 방 구조를 생성합니다.
    내부적으로 사용되며, 외부에서는 CustomRoomWrapper를 통해 사용하는 것을 권장합니다.
    """
    
    def __init__(self, size=10, room_config=None, **kwargs):
        """
        환경 초기화
        
        Args:
            size: 환경 크기 (기본값: 10)
            room_config: 방 구조 설정 딕셔너리
            **kwargs: MiniGridEnv의 추가 파라미터
        """
        self.size = size
        self.room_config = room_config or {}
        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=4 * size * size,
            **kwargs
        )
    
    @staticmethod
    def _gen_mission():
        """미션 텍스트 생성"""
        return "explore"
    
    def _gen_grid(self, width, height):
        """
        그리드를 생성하는 메서드
        
        이 메서드는 환경이 리셋될 때마다 호출되어 새로운 그리드를 생성합니다.
        
        Args:
            width: 그리드 너비
            height: 그리드 높이
        """
        # 1단계: 빈 그리드 생성
        self.grid = Grid(width, height)
        
        # 2단계: 외벽 생성 (전체 그리드를 둘러싸는 벽)
        self.grid.wall_rect(0, 0, width, height)
        
        # 3단계: 커스텀 설정이 있으면 적용
        if self.room_config:
            # 3-1: 벽 배치
            if 'walls' in self.room_config:
                for wall_x, wall_y in self.room_config['walls']:
                    # 좌표가 유효한 범위 내에 있는지 확인
                    if 0 <= wall_x < width and 0 <= wall_y < height:
                        self.grid.set(wall_x, wall_y, Wall())
            
            # 3-2: Goal 위치 설정 (공식 방법: put_obj 사용)
            if 'goal_pos' in self.room_config:
                goal_x, goal_y = self.room_config['goal_pos']
                if 0 <= goal_x < width and 0 <= goal_y < height:
                    # put_obj는 객체를 안전하게 배치하는 헬퍼 메서드
                    self.put_obj(Goal(), goal_x, goal_y)
            
            # 3-3: 객체 배치 (공식 방법: put_obj 사용)
            if 'objects' in self.room_config:
                for obj_info in self.room_config['objects']:
                    # 객체 정보 추출
                    obj_type = obj_info.get('type', 'key')
                    obj_pos = obj_info.get('pos', (1, 1))
                    obj_color = obj_info.get('color', 'yellow')
                    
                    obj_x, obj_y = obj_pos
                    if 0 <= obj_x < width and 0 <= obj_y < height:
                        # 객체 타입에 따라 적절한 객체 생성
                        if obj_type == 'key':
                            obj = Key(obj_color)
                        elif obj_type == 'ball':
                            obj = Ball(obj_color)
                        elif obj_type == 'box':
                            obj = Box(obj_color)
                        elif obj_type == 'door':
                            obj = Door(obj_color, is_locked=False, is_open=True)
                        else:
                            obj = Key(obj_color)  # 기본값
                        
                        # 객체를 그리드에 배치
                        self.put_obj(obj, obj_x, obj_y)
        
        # 4단계: 에이전트 시작 위치 설정
        if self.room_config and 'start_pos' in self.room_config:
            # 명시적으로 시작 위치가 지정된 경우
            start_x, start_y = self.room_config['start_pos']
            self.agent_pos = np.array([start_x, start_y])
            self.agent_dir = 0  # 0=오른쪽, 1=아래, 2=왼쪽, 3=위
        else:
            # 시작 위치가 지정되지 않은 경우 자동으로 빈 공간에 배치
            self.place_agent()
        
        # 5단계: Mission 설정 (공식 방법)
        self.mission = self._gen_mission()


class CustomRoomWrapper:
    """
    MiniGrid 환경을 제어하기 위한 Wrapper 클래스
    
    이 클래스는 CustomRoomEnv를 감싸서 더 편리한 API를 제공합니다.
    VLM과의 연동을 고려하여 설계되었습니다.
    
    사용 예시:
        # 환경 생성
        wrapper = CustomRoomWrapper(
            size=15,
            walls=[(5, 0), (5, 1), ...],
            room_config={'start_pos': (2, 2), 'goal_pos': (10, 10)}
        )
        
        # 이미지 가져오기 (VLM에 전달)
        image = wrapper.get_image()
        
        # VLM이 반환한 액션 실행
        action_str = "move forward"  # VLM이 반환한 텍스트
        action = wrapper.parse_action(action_str)
        obs, reward, done, info = wrapper.step(action)
    """
    
    # 액션 이름과 인덱스 매핑 (VLM이 텍스트로 액션을 반환할 수 있도록)
    ACTION_NAMES = {
        0: "turn left",
        1: "turn right", 
        2: "move forward",
        3: "move backward",
        4: "pickup",
        5: "drop",
        6: "toggle"
    }
    
    # 액션 이름의 다양한 표현 (VLM이 다양한 표현을 사용할 수 있도록)
    ACTION_ALIASES = {
        "turn left": 0, "left": 0, "rotate left": 0, "turn_left": 0,
        "turn right": 1, "right": 1, "rotate right": 1, "turn_right": 1,
        "move forward": 2, "forward": 2, "go forward": 2, "move_forward": 2, "w": 2,
        "move backward": 3, "backward": 3, "go backward": 3, "move_backward": 3, "s": 3,
        "pickup": 4, "pick up": 4, "pick_up": 4, "grab": 4,
        "drop": 5, "put down": 5, "put_down": 5, "release": 5,
        "toggle": 6, "interact": 6, "use": 6, "activate": 6
    }
    
    def __init__(
        self,
        size: int = 10,
        walls: Optional[List[Tuple[int, int]]] = None,
        room_config: Optional[Dict] = None,
        render_mode: str = 'rgb_array',
        **kwargs
    ):
        """
        Wrapper 초기화
        
        Args:
            size: 환경 크기 (기본값: 10)
            walls: 벽 위치 리스트 [(x1, y1), (x2, y2), ...] (기본값: None)
            room_config: 방 구조 설정 딕셔너리 (기본값: None)
                - start_pos: (x, y) 튜플 - 에이전트 시작 위치
                - goal_pos: (x, y) 튜플 - 목표 위치
                - objects: 객체 리스트 [{'type': 'key', 'pos': (x, y), 'color': 'yellow'}, ...]
            render_mode: 렌더링 모드 ('rgb_array' 또는 'human') (기본값: 'rgb_array')
            **kwargs: CustomRoomEnv의 추가 파라미터
        """
        # 입력 파라미터 저장
        self.size = size
        self.walls = walls or []
        self.render_mode = render_mode
        
        # room_config 구성 (walls가 별도로 제공된 경우 병합)
        if room_config is None:
            room_config = {}
        
        # walls가 별도로 제공된 경우 room_config에 추가
        if walls and 'walls' not in room_config:
            # 기존 walls가 있으면 병합, 없으면 새로 생성
            existing_walls = room_config.get('walls', [])
            room_config['walls'] = existing_walls + walls
        
        # 내부 환경 생성 (CustomRoomEnv 인스턴스)
        self.env = CustomRoomEnv(
            size=size,
            room_config=room_config,
            render_mode=render_mode,
            **kwargs
        )
        
        # 현재 관찰 상태 저장 (초기화 시 리셋)
        self.current_obs = None
        self.current_info = None
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        환경을 초기 상태로 리셋
        
        Args:
            seed: 랜덤 시드 (기본값: None)
        
        Returns:
            observation: 초기 관찰 (딕셔너리)
            info: 추가 정보 (딕셔너리)
        """
        # 환경 리셋
        self.current_obs, self.current_info = self.env.reset(seed=seed)
        return self.current_obs, self.current_info
    
    def step(self, action: Union[int, str]) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        액션을 실행하고 다음 상태로 전이
        
        Args:
            action: 액션 (정수 인덱스 또는 액션 이름 문자열)
                - 0 또는 "turn left": 왼쪽으로 회전
                - 1 또는 "turn right": 오른쪽으로 회전
                - 2 또는 "move forward": 앞으로 이동
                - 3 또는 "move backward": 뒤로 이동
                - 4 또는 "pickup": 물체 집기
                - 5 또는 "drop": 물체 놓기
                - 6 또는 "toggle": 상호작용 (문 열기 등)
        
        Returns:
            observation: 새로운 관찰 (딕셔너리)
            reward: 보상 (float)
            terminated: 목표 달성 여부 (bool)
            truncated: 시간 초과 여부 (bool)
            info: 추가 정보 (딕셔너리)
        """
        # 액션이 문자열인 경우 정수로 변환
        if isinstance(action, str):
            action = self.parse_action(action)
        
        # 액션 실행
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 현재 상태 업데이트
        self.current_obs = obs
        self.current_info = info
        
        return obs, reward, terminated, truncated, info
    
    def get_image(self) -> np.ndarray:
        """
        현재 환경의 이미지를 반환 (VLM 입력용)
        
        이 메서드는 VLM에 전달할 수 있는 RGB 이미지를 반환합니다.
        
        Returns:
            image: RGB 이미지 배열 (H, W, 3) 형태의 numpy 배열
        """
        # 환경 렌더링 (RGB 배열로 반환)
        image = self.env.render()
        
        # 이미지가 None인 경우 빈 배열 반환
        if image is None:
            return np.zeros((self.size * 32, self.size * 32, 3), dtype=np.uint8)
        
        return image
    
    def get_action_space(self) -> Dict:
        """
        액션 공간 정보 반환
        
        Returns:
            action_space_info: 액션 공간 정보 딕셔너리
                - n: 액션 개수
                - actions: 액션 이름 리스트
                - action_mapping: 액션 인덱스와 이름 매핑
        """
        return {
            'n': self.env.action_space.n,
            'actions': list(self.ACTION_NAMES.values()),
            'action_mapping': self.ACTION_NAMES,
            'action_aliases': self.ACTION_ALIASES
        }
    
    def get_action_names(self) -> List[str]:
        """
        액션 이름 리스트 반환 (VLM용)
        
        Returns:
            action_names: 액션 이름 리스트
        """
        return list(self.ACTION_NAMES.values())
    
    def parse_action(self, action_str: str) -> int:
        """
        VLM이 반환한 텍스트를 액션 인덱스로 변환
        
        이 메서드는 VLM이 반환한 텍스트 액션을 정수 인덱스로 변환합니다.
        다양한 표현을 지원합니다 (예: "move forward", "forward", "go forward" 등).
        
        Args:
            action_str: 액션 텍스트 (예: "move forward", "turn left")
        
        Returns:
            action: 액션 인덱스 (0-6)
        
        Raises:
            ValueError: 알 수 없는 액션인 경우
        """
        # 소문자로 변환하고 공백 제거
        action_str = action_str.lower().strip()
        
        # 액션 별칭에서 찾기
        if action_str in self.ACTION_ALIASES:
            return self.ACTION_ALIASES[action_str]
        
        # 직접 매핑에서 찾기
        for idx, name in self.ACTION_NAMES.items():
            if action_str == name.lower():
                return idx
        
        # 찾지 못한 경우 에러 발생
        raise ValueError(
            f"Unknown action: '{action_str}'. "
            f"Available actions: {list(self.ACTION_ALIASES.keys())}"
        )
    
    def get_state(self) -> Dict:
        """
        현재 환경 상태 정보 반환
        
        Returns:
            state: 환경 상태 딕셔너리
                - agent_pos: 에이전트 위치
                - agent_dir: 에이전트 방향
                - mission: 현재 미션
                - image: 현재 이미지
        """
        return {
            'agent_pos': self.env.agent_pos.copy() if hasattr(self.env, 'agent_pos') else None,
            'agent_dir': self.env.agent_dir if hasattr(self.env, 'agent_dir') else None,
            'mission': self.env.mission if hasattr(self.env, 'mission') else None,
            'image': self.get_image()
        }
    
    def close(self):
        """환경 종료 및 리소스 정리"""
        self.env.close()


# 편의 함수들 (기존 코드와의 호환성을 위해 유지)

def create_house_environment():
    """
    실내 집 환경 생성 (복도, 방, 차고 구조)
    
    Returns:
        CustomRoomWrapper: 실내 집 환경 Wrapper 인스턴스
    """
    size = 15
    
    # 벽 구조 정의 (복도와 방을 구분하는 벽)
    walls = []
    
    # 외벽 (상하좌우 경계)
    for i in range(size):
        walls.append((i, 0))  # 상단
        walls.append((i, size-1))  # 하단
        walls.append((0, i))  # 좌측
        walls.append((size-1, i))  # 우측
    
    # 내부 벽 (방 구분)
    # 복도 (중앙 세로)
    for i in range(5, 10):
        walls.append((7, i))  # 좌측 방과 복도 구분
    
    # 방 구분 벽 (가로)
    for i in range(1, 7):
        walls.append((i, 5))  # 상단 방 구분
        walls.append((i, 10))  # 하단 방 구분
    
    # 복도와 차고 구분
    for i in range(8, size-1):
        walls.append((i, 7))
    
    # 시작 위치 (거실)
    start_pos = (2, 2)
    
    # Goal 위치 (차고)
    goal_pos = (12, 12)
    
    # 객체 배치 (가위, 열쇠 등)
    objects = [
        {'type': 'key', 'pos': (3, 3), 'color': 'yellow'},  # 안방에 가위(키로 대체)
        {'type': 'key', 'pos': (12, 2), 'color': 'red'},   # 차고 열쇠
        {'type': 'ball', 'pos': (5, 8), 'color': 'blue'},  # 복도에 공
    ]
    
    room_config = {
        'start_pos': start_pos,
        'goal_pos': goal_pos,
        'walls': walls,
        'objects': objects
    }
    
    # Wrapper로 반환
    return CustomRoomWrapper(size=size, room_config=room_config)


def create_simple_room():
    """
    간단한 방 구조 예제
    
    Returns:
        CustomRoomWrapper: 간단한 방 환경 Wrapper 인스턴스
    """
    size = 8
    
    walls = [
        # 외벽
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0),
        (0, 7), (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7), (7, 7),
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
        (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6),
        # 내부 벽
        (3, 2), (3, 3), (3, 4),
    ]
    
    room_config = {
        'start_pos': (1, 1),
        'goal_pos': (6, 6),
        'walls': walls,
        'objects': [
            {'type': 'key', 'pos': (5, 2), 'color': 'green'},
        ]
    }
    
    # Wrapper로 반환
    return CustomRoomWrapper(size=size, room_config=room_config)


def visualize_environment(wrapper):
    """
    환경을 시각화
    
    Args:
        wrapper: CustomRoomWrapper 인스턴스
    """
    wrapper.reset()
    img = wrapper.get_image()
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.title("Custom MiniGrid Environment")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('custom_environment.png', dpi=150, bbox_inches='tight')
    print("환경 이미지가 'custom_environment.png'로 저장되었습니다.")
    plt.show()


def test_environment(wrapper):
    """
    환경 테스트 (랜덤 액션 실행)
    
    Args:
        wrapper: CustomRoomWrapper 인스턴스
    """
    obs, info = wrapper.reset()
    done = False
    step_count = 0
    max_steps = 100
    
    print("환경 테스트 시작...")
    state = wrapper.get_state()
    print(f"시작 위치: {state['agent_pos']}")
    print(f"미션: {state['mission']}")
    
    while not done and step_count < max_steps:
        # 랜덤 액션
        action = wrapper.env.action_space.sample()
        obs, reward, terminated, truncated, info = wrapper.step(action)
        done = terminated or truncated
        step_count += 1
        
        if step_count % 10 == 0:
            print(f"Step {step_count}: Reward={reward}, Done={done}")
    
    print(f"\n테스트 완료: 총 {step_count} 스텝, 최종 보상: {reward}")
    return obs, reward, done


def main():
    """
    메인 함수: 다양한 환경 생성 및 테스트
    """
    print("=" * 60)
    print("MiniGrid 커스텀 환경 생성 예제")
    print("=" * 60)
    
    # 예제 1: 간단한 방 구조
    print("\n[예제 1] 간단한 방 구조 생성")
    print("-" * 60)
    wrapper1 = create_simple_room()
    visualize_environment(wrapper1)
    test_environment(wrapper1)
    wrapper1.close()
    
    # 예제 2: 실내 집 환경
    print("\n[예제 2] 실내 집 환경 생성 (복도, 방, 차고)")
    print("-" * 60)
    wrapper2 = create_house_environment()
    visualize_environment(wrapper2)
    test_environment(wrapper2)
    wrapper2.close()
    
    # 예제 3: Wrapper 직접 사용
    print("\n[예제 3] Wrapper 직접 사용")
    print("-" * 60)
    custom_config = {
        'start_pos': (2, 2),
        'goal_pos': (10, 10),
        'walls': [
            # 외벽
            (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0), (11, 0), (12, 0),
            (0, 12), (1, 12), (2, 12), (3, 12), (4, 12), (5, 12), (6, 12), (7, 12), (8, 12), (9, 12), (10, 12), (11, 12), (12, 12),
            (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11),
            (12, 1), (12, 2), (12, 3), (12, 4), (12, 5), (12, 6), (12, 7), (12, 8), (12, 9), (12, 10), (12, 11),
            # 내부 벽
            (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9),
        ],
        'objects': [
            {'type': 'key', 'pos': (4, 4), 'color': 'yellow'},
            {'type': 'ball', 'pos': (8, 8), 'color': 'blue'},
            {'type': 'box', 'pos': (3, 9), 'color': 'green'},
        ]
    }
    
    wrapper3 = CustomRoomWrapper(size=13, room_config=custom_config)
    visualize_environment(wrapper3)
    test_environment(wrapper3)
    wrapper3.close()
    
    # 예제 4: VLM 연동 예시
    print("\n[예제 4] VLM 연동 예시")
    print("-" * 60)
    wrapper4 = create_simple_room()
    wrapper4.reset()
    
    # 이미지 가져오기 (VLM에 전달)
    image = wrapper4.get_image()
    print(f"이미지 크기: {image.shape}")
    
    # 액션 공간 정보 확인
    action_space = wrapper4.get_action_space()
    print(f"액션 개수: {action_space['n']}")
    print(f"액션 목록: {action_space['actions']}")
    
    # VLM이 반환한 텍스트 액션을 실행
    vlm_actions = ["move forward", "turn right", "move forward", "pickup"]
    for action_str in vlm_actions:
        try:
            action = wrapper4.parse_action(action_str)
            obs, reward, done, truncated, info = wrapper4.step(action)
            print(f"액션 '{action_str}' 실행: Reward={reward}, Done={done}")
            if done:
                break
        except ValueError as e:
            print(f"에러: {e}")
    
    wrapper4.close()
    
    print("\n" + "=" * 60)
    print("모든 예제 실행 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
