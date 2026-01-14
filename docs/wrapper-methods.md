# CustomRoomWrapper 메서드 가이드

`CustomRoomWrapper` 클래스의 모든 메서드에 대한 간단한 설명과 사용 예시입니다.

## 클래스 개요

`CustomRoomWrapper`는 `CustomRoomEnv`를 감싸서 VLM(Vision Language Model)과의 연동을 고려한 편리한 API를 제공하는 래퍼 클래스입니다.

## 초기화

### `__init__(size, walls, room_config, render_mode, **kwargs)`

환경을 초기화합니다.

**파라미터**:
- `size` (int): 환경 크기 (기본값: 10)
- `walls` (List[Tuple[int, int]], optional): 벽 위치 리스트
- `room_config` (Dict, optional): 방 구조 설정 딕셔너리
- `render_mode` (str): 렌더링 모드 (`'rgb_array'` 또는 `'human'`, 기본값: `'rgb_array'`)
- `**kwargs`: `CustomRoomEnv`의 추가 파라미터

**사용 예시**:
```python
wrapper = CustomRoomWrapper(
    size=10,
    room_config={
        'start_pos': (1, 1),
        'goal_pos': (8, 8),
        'walls': [(0, 0), (1, 0), ...],
        'objects': [...]
    }
)
```

---

## 환경 제어 메서드

### `reset(seed=None)`

환경을 초기 상태로 리셋합니다.

**파라미터**:
- `seed` (int, optional): 랜덤 시드

**반환값**:
- `observation`: 초기 관찰 (딕셔너리)
- `info`: 추가 정보 (딕셔너리)

**사용 예시**:
```python
obs, info = wrapper.reset()
obs, info = wrapper.reset(seed=42)  # 시드 지정
```

---

### `step(action)`

액션을 실행하고 다음 상태로 전이합니다.

**파라미터**:
- `action` (int 또는 str): 액션
  - 정수: `0` (turn left), `1` (turn right), `2` (move forward), `3` (pickup), `4` (drop), `5` (toggle), `6` (done)
  - 문자열: `"turn left"`, `"move forward"`, `"pickup"` 등

**반환값**:
- `observation`: 새로운 관찰 (딕셔너리)
- `reward`: 보상 (float)
- `terminated`: 목표 달성 여부 (bool)
- `truncated`: 시간 초과 여부 (bool)
- `info`: 추가 정보 (딕셔너리)

**사용 예시**:
```python
# 정수 액션
obs, reward, terminated, truncated, info = wrapper.step(2)  # move forward

# 문자열 액션
obs, reward, terminated, truncated, info = wrapper.step("move forward")
obs, reward, terminated, truncated, info = wrapper.step("turn left")
```

---

### `close()`

환경을 종료하고 리소스를 정리합니다.

**사용 예시**:
```python
wrapper.close()
```

---

## 이미지 및 시각화 메서드

### `get_image(fov_range=None, fov_width=None)`

현재 환경의 이미지를 반환합니다. VLM 입력으로 사용할 수 있습니다.

**파라미터**:
- `fov_range` (int, optional): 에이전트 앞으로 볼 수 있는 거리 (칸 수). `None`이면 시야 제한 없음
- `fov_width` (int, optional): 시야의 좌우 폭 (칸 수). `None`이면 시야 제한 없음

**반환값**:
- `image`: RGB 이미지 배열 (H, W, 3) 형태의 numpy 배열

**사용 예시**:
```python
# 전체 시야
image = wrapper.get_image()

# 시야 제한 적용 (앞으로 3칸, 좌우 각 1.5칸)
image = wrapper.get_image(fov_range=3, fov_width=3)
```

**참고**: `fov_range`와 `fov_width`가 모두 지정된 경우에만 시야 제한이 적용됩니다.

---

## 상태 정보 메서드

### `get_state()`

현재 환경 상태 정보를 반환합니다.

**반환값**:
- `state`: 환경 상태 딕셔너리
  - `agent_pos`: 에이전트 위치 (numpy array 또는 tuple)
  - `agent_dir`: 에이전트 방향 (0=오른쪽, 1=아래, 2=왼쪽, 3=위)
  - `mission`: 현재 미션 (문자열)
  - `image`: 현재 이미지 (numpy array)

**사용 예시**:
```python
state = wrapper.get_state()
print(f"위치: {state['agent_pos']}")
print(f"방향: {state['agent_dir']}")
print(f"미션: {state['mission']}")
```

---

## 액션 관련 메서드

### `get_action_space()`

액션 공간 정보를 반환합니다.

**반환값**:
- `action_space_info`: 액션 공간 정보 딕셔너리
  - `n`: 액션 개수
  - `actions`: 액션 이름 리스트
  - `action_mapping`: 액션 인덱스와 이름 매핑
  - `action_aliases`: 액션 별칭 딕셔너리

**사용 예시**:
```python
action_space = wrapper.get_action_space()
print(f"액션 개수: {action_space['n']}")
print(f"액션 목록: {action_space['actions']}")
print(f"액션 매핑: {action_space['action_mapping']}")
```

---

### `get_action_names()`

액션 이름 리스트를 반환합니다 (VLM용).

**반환값**:
- `action_names`: 액션 이름 리스트

**사용 예시**:
```python
action_names = wrapper.get_action_names()
# ['turn left', 'turn right', 'move forward', 'move backward', 'pickup', 'drop', 'toggle']
```

---

### `parse_action(action_str)`

VLM이 반환한 텍스트 액션을 정수 인덱스로 변환합니다.

**파라미터**:
- `action_str` (str): 액션 텍스트 (예: `"move forward"`, `"turn left"`, `"2"`)

**반환값**:
- `action` (int): 액션 인덱스 (0-6)

**예외**:
- `ValueError`: 알 수 없는 액션인 경우

**지원 형식**:
- 숫자 문자열: `"0"`, `"1"`, `"2"` 등
- 액션 이름: `"move forward"`, `"turn left"` 등
- 액션 별칭: `"forward"`, `"left"`, `"w"`, `"pick up"` 등

**사용 예시**:
```python
# 숫자 문자열
action = wrapper.parse_action("2")  # move forward

# 액션 이름
action = wrapper.parse_action("move forward")
action = wrapper.parse_action("turn left")

# 액션 별칭
action = wrapper.parse_action("forward")
action = wrapper.parse_action("left")
action = wrapper.parse_action("w")  # move forward
action = wrapper.parse_action("pick up")  # pickup
```

**지원하는 액션 별칭**:
- `turn left`: `"left"`, `"rotate left"`, `"turn_left"`
- `turn right`: `"right"`, `"rotate right"`, `"turn_right"`
- `move forward`: `"forward"`, `"go forward"`, `"move_forward"`, `"w"`
- `move backward`: `"backward"`, `"go backward"`, `"move_backward"`, `"s"`
- `pickup`: `"pick up"`, `"pick_up"`, `"grab"`
- `drop`: `"put down"`, `"put_down"`, `"release"`
- `toggle`: `"interact"`, `"use"`, `"activate"`

---

## 액션 인덱스 및 이름

### 액션 인덱스 매핑

| 인덱스 | 액션 이름 | 설명 |
|-------|---------|------|
| 0 | `turn left` | 90° 반시계 방향 회전 |
| 1 | `turn right` | 90° 시계 방향 회전 |
| 2 | `move forward` | 현재 heading 방향으로 한 칸 전진 |
| 3 | `move backward` | 현재 heading 방향의 반대로 한 칸 후진 (일부 환경에서만 지원) |
| 4 | `pickup` | 앞에 있는 객체 집기 |
| 5 | `drop` | 들고 있는 객체 놓기 |
| 6 | `toggle` | 앞에 있는 객체와 상호작용 (문 열기/닫기 등) |

**참고**: MiniGrid의 기본 액션 공간은 7가지(0-6)이며, `move backward`는 일부 환경에서만 지원됩니다. 대부분의 환경에서는 `turn left` 또는 `turn right`를 두 번 사용하여 180° 회전한 후 `move forward`를 사용해야 합니다.

---

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

# 상태 정보 가져오기
state = wrapper.get_state()
print(f"에이전트 위치: {state['agent_pos']}")

# 액션 실행
obs, reward, terminated, truncated, info = wrapper.step("move forward")

# 환경 종료
wrapper.close()
```

### VLM 연동 예시

```python
# VLM이 반환한 액션 문자열을 파싱하여 실행
vlm_action = "move forward"  # VLM이 반환한 텍스트
action_index = wrapper.parse_action(vlm_action)
obs, reward, done, truncated, info = wrapper.step(action_index)
```

### 시야 제한 사용

```python
# 시야 제한이 적용된 이미지 가져오기
image = wrapper.get_image(fov_range=3, fov_width=3)
# 에이전트 앞으로 3칸, 좌우 각 1.5칸만 보임
```

---

## 참고 자료

- [CustomRoomWrapper API 상세 문서](./wrapper-api.md)
- [CustomRoomEnv API 문서](./custom-environment-api.md)

