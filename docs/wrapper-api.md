# CustomRoomWrapper API 문서

이 문서는 `CustomRoomWrapper` 클래스의 API를 설명합니다. 이 Wrapper는 VLM(Vision Language Model)과의 연동을 고려하여 설계되었습니다.

## 개요

`CustomRoomWrapper`는 MiniGrid 환경을 쉽게 생성하고 제어할 수 있는 Wrapper 클래스입니다. 기존 `CustomRoomEnv`를 감싸서 더 편리한 API를 제공하며, VLM과의 연동을 지원합니다.

## 클래스 초기화

```python
wrapper = CustomRoomWrapper(
    size=10,
    walls=None,
    room_config=None,
    render_mode='rgb_array',
    **kwargs
)
```

### 파라미터

- `size` (int): 환경 크기 (기본값: 10)
- `walls` (List[Tuple[int, int]], optional): 벽 위치 리스트 [(x1, y1), (x2, y2), ...]
- `room_config` (Dict, optional): 방 구조 설정 딕셔너리
  - `start_pos`: (x, y) 튜플 - 에이전트 시작 위치
  - `goal_pos`: (x, y) 튜플 - 목표 위치
  - `walls`: 벽 위치 리스트 (walls 파라미터와 병합됨)
  - `objects`: 객체 리스트
- `render_mode` (str): 렌더링 모드 ('rgb_array' 또는 'human')
- `**kwargs`: CustomRoomEnv의 추가 파라미터

## 주요 메서드

### reset(seed=None)

환경을 초기 상태로 리셋합니다.

```python
obs, info = wrapper.reset(seed=42)
```

**Returns:**
- `obs`: 초기 관찰 (딕셔너리)
- `info`: 추가 정보 (딕셔너리)

### step(action)

액션을 실행하고 다음 상태로 전이합니다.

```python
obs, reward, terminated, truncated, info = wrapper.step(action)
```

**Parameters:**
- `action`: 액션 (정수 인덱스 또는 액션 이름 문자열)
  - **정수 인덱스 (0-6)**:
    - `0`: Turn left (왼쪽으로 90° 회전)
    - `1`: Turn right (오른쪽으로 90° 회전)
    - `2`: Move forward (현재 heading 방향으로 한 칸 전진)
    - `3`: Pickup (앞에 있는 객체 집기)
    - `4`: Drop (들고 있는 객체 놓기)
    - `5`: Toggle (문 열기/닫기 등 객체와 상호작용)
    - `6`: Done (작업 완료, 에피소드 종료)
  - **문자열 액션 이름** (대소문자 구분 없음, 다양한 동의어 지원):
    - 회전: `"turn left"`, `"left"`, `"rotate left"`, `"turn_left"` → 0
    - 회전: `"turn right"`, `"right"`, `"rotate right"`, `"turn_right"` → 1
    - 전진: `"move forward"`, `"forward"`, `"go forward"`, `"move_forward"`, `"w"` → 2
    - 집기: `"pickup"`, `"pick up"`, `"pick_up"`, `"grab"` → 3
    - 놓기: `"drop"`, `"put down"`, `"put_down"`, `"release"` → 4
    - 상호작용: `"toggle"`, `"interact"`, `"use"`, `"activate"` → 5
    - 완료: `"done"`, `"complete"`, `"finish"` → 6
  - **참고**: MiniGrid의 기본 액션 공간은 7개(`Discrete(7)`)이며, backward 액션은 지원되지 않습니다. 뒤로 이동하려면 180° 회전(왼쪽 또는 오른쪽으로 두 번 회전) 후 forward를 사용해야 합니다.

**Returns:**
- `obs`: 새로운 관찰 (딕셔너리)
- `reward`: 보상 (float)
- `terminated`: 목표 달성 여부 (bool)
- `truncated`: 시간 초과 여부 (bool)
- `info`: 추가 정보 (딕셔너리)

### get_image()

현재 환경의 이미지를 반환합니다 (VLM 입력용).

```python
image = wrapper.get_image()
```

**Returns:**
- `image`: RGB 이미지 배열 (H, W, 3) 형태의 numpy 배열

**사용 예시:**
```python
# VLM에 전달할 이미지 가져오기
image = wrapper.get_image()
# VLM에 이미지 전달
vlm_response = vlm_model.process_image(image)
```

### get_action_space()

액션 공간 정보를 반환합니다.

```python
action_space = wrapper.get_action_space()
```

**Returns:**
```python
{
    'n': 7,  # 액션 개수
    'actions': ['turn left', 'turn right', ...],  # 액션 이름 리스트
    'action_mapping': {0: 'turn left', 1: 'turn right', ...},  # 인덱스-이름 매핑
    'action_aliases': {'forward': 2, 'move forward': 2, ...}  # 별칭 매핑
}
```

### get_action_names()

액션 이름 리스트를 반환합니다 (VLM용).

```python
action_names = wrapper.get_action_names()
# ['turn left', 'turn right', 'move forward', 'move backward', 'pickup', 'drop', 'toggle']
```

### parse_action(action_str)

VLM이 반환한 텍스트를 액션 인덱스로 변환합니다.

```python
action = wrapper.parse_action("move forward")  # 2 반환
action = wrapper.parse_action("forward")      # 2 반환
action = wrapper.parse_action("w")            # 2 반환
```

**Parameters:**
- `action_str` (str): 액션 텍스트

**Returns:**
- `action` (int): 액션 인덱스 (0-6)

**지원하는 액션 표현:**
- "turn left", "left", "rotate left", "turn_left" → 0
- "turn right", "right", "rotate right", "turn_right" → 1
- "move forward", "forward", "go forward", "move_forward", "w" → 2
- "move backward", "backward", "go backward", "move_backward", "s" → 3
- "pickup", "pick up", "pick_up", "grab" → 4
- "drop", "put down", "put_down", "release" → 5
- "toggle", "interact", "use", "activate" → 6

### get_state()

현재 환경 상태 정보를 반환합니다.

```python
state = wrapper.get_state()
```

**Returns:**
```python
{
    'agent_pos': np.array([x, y]),  # 에이전트 위치
    'agent_dir': 0,                   # 에이전트 방향 (0=오른쪽, 1=아래, 2=왼쪽, 3=위)
    'mission': "explore",             # 현재 미션
    'image': np.array(...)            # 현재 이미지
}
```

### close()

환경을 종료하고 리소스를 정리합니다.

```python
wrapper.close()
```

## 사용 예시

### 기본 사용

```python
from custom_environment import CustomRoomWrapper

# 환경 생성
wrapper = CustomRoomWrapper(size=10)

# 환경 리셋
obs, info = wrapper.reset()

# 이미지 가져오기
image = wrapper.get_image()

# 액션 실행
obs, reward, done, truncated, info = wrapper.step(2)  # 앞으로 이동

# 종료
wrapper.close()
```

### VLM 연동 예시

```python
from custom_environment import CustomRoomWrapper

# 환경 생성
wrapper = CustomRoomWrapper(size=15)

# 환경 초기화
obs, info = wrapper.reset()

# VLM 루프
while not done:
    # 1. 현재 이미지 가져오기
    image = wrapper.get_image()
    
    # 2. VLM에 이미지 전달하고 액션 받기
    vlm_action = vlm_model.predict(image)  # 예: "move forward"
    
    # 3. VLM이 반환한 텍스트를 액션 인덱스로 변환
    action = wrapper.parse_action(vlm_action)
    
    # 4. 액션 실행
    obs, reward, terminated, truncated, info = wrapper.step(action)
    done = terminated or truncated

wrapper.close()
```

### 커스텀 환경 생성

```python
# 방법 1: walls와 room_config를 함께 사용
wrapper = CustomRoomWrapper(
    size=15,
    walls=[(5, 0), (5, 1), (5, 2)],
    room_config={
        'start_pos': (2, 2),
        'goal_pos': (10, 10),
        'objects': [
            {'type': 'key', 'pos': (3, 3), 'color': 'yellow'}
        ]
    }
)

# 방법 2: room_config에 모든 것을 포함
wrapper = CustomRoomWrapper(
    size=15,
    room_config={
        'start_pos': (2, 2),
        'goal_pos': (10, 10),
        'walls': [(5, 0), (5, 1), (5, 2)],
        'objects': [
            {'type': 'key', 'pos': (3, 3), 'color': 'yellow'}
        ]
    }
)
```

## 액션 인덱스

MiniGrid의 표준 액션 공간은 7개의 이산적(discrete) 액션으로 구성됩니다:

| 인덱스 | 액션 이름 | 설명 |
|--------|----------|------|
| 0 | turn left | 왼쪽으로 90° 회전 (counterclockwise) |
| 1 | turn right | 오른쪽으로 90° 회전 (clockwise) |
| 2 | move forward | 현재 heading 방향으로 한 칸 전진 |
| 3 | pickup | 앞에 있는 객체 집기 |
| 4 | drop | 들고 있는 객체 놓기 |
| 5 | toggle | 객체와 상호작용 (문 열기/닫기 등) |
| 6 | done | 작업 완료, 에피소드 종료 |

**참고사항:**
- MiniGrid에는 **backward 액션이 없습니다**. 뒤로 이동하려면 180° 회전(왼쪽 또는 오른쪽으로 두 번 회전) 후 `move forward`를 사용해야 합니다.
- 모든 액션은 로봇의 현재 heading 방향을 기준으로 작동합니다 (상대적 이동).
- 문자열 액션 이름은 대소문자를 구분하지 않으며, 다양한 동의어를 지원합니다 (예: "move forward", "forward", "go forward", "move_forward", "w" 모두 액션 2로 매핑됨).

## 참고

- [환경 생성 가이드](./environment-creation.md)
- [베스트 프랙티스](./best-practices.md)
- [VLM 연동 가이드](./vlm-integration.md) (추후 작성 예정)

