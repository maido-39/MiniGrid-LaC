# SLAM 스타일 FOV 맵핑 가이드

이 문서는 SLAM(Simultaneous Localization and Mapping) 스타일의 시야 제한(FOV, Field of View) 맵핑 기능을 설명합니다.

## 개요

SLAM 스타일 FOV 맵핑은 에이전트가 탐색한 영역을 추적하고, 현재 시야 범위와 탐색한 영역을 시각적으로 구분하여 표시하는 기능입니다. 이는 실제 로봇의 제한된 시야와 지도 구축 과정을 시뮬레이션합니다.

## 주요 특징

1. **탐색 영역 추적**: 에이전트가 방문한 모든 위치를 기록합니다.
2. **시야 제한**: 에이전트가 바라보는 방향 기준으로 앞쪽만 볼 수 있습니다.
3. **시각적 구분**:
   - 현재 시야 범위 내: 밝게 표시
   - 탐색했던 곳 (시야 밖): 어둡게(반투명하게) 표시
   - 중요한 객체(열쇠, 문, 목표)가 있는 곳: 탐색했어도 밝게 유지
   - 아직 탐색하지 않은 곳: 검은색으로 표시

## 사용법

### keyboard_control_fov_mapping.py

SLAM 스타일 FOV 맵핑 기능이 포함된 키보드 제어 스크립트입니다.

```bash
python keyboard_control_fov_mapping.py
```

### 조작법

**이동:**
- `w`: 앞으로 이동 (move forward)
- `a`: 왼쪽으로 회전 (turn left)
- `d`: 오른쪽으로 회전 (turn right)

**상호작용:**
- `p` 또는 `e`: 물체 집기 (pickup)
- `t` 또는 ` ` (스페이스): 상호작용 (toggle)

**FOV 제어:**
- `f`: 시야 제한 토글 (켜기/끄기)
- `+`: 시야 범위 증가
- `-`: 시야 범위 감소

**기타:**
- `r`: 환경 리셋
- `q`: 종료

### 환경 선택

스크립트 실행 시 다음 환경 중 하나를 선택할 수 있습니다:

1. FourRooms (4개의 방 구조)
2. MultiRoom-N6 (6개의 방)
3. DoorKey-16x16 (문과 열쇠)
4. KeyCorridorS6R3 (복도와 열쇠)
5. Playground (놀이터)
6. Empty-16x16 (빈 환경)

## SLAMFOVWrapper 클래스

### 초기화

```python
from keyboard_control_fov_mapping import SLAMFOVWrapper
import gymnasium as gym

# Gymnasium 환경 생성
gym_env = gym.make("MiniGrid-FourRooms-v0", render_mode='rgb_array')

# SLAM FOV 래퍼로 감싸기
wrapper = SLAMFOVWrapper(gym_env)
```

### 주요 메서드

#### reset()

환경을 리셋하고 탐색 맵을 초기화합니다.

```python
obs, info = wrapper.reset()
```

#### step(action)

액션을 실행하고 탐색 맵을 업데이트합니다.

```python
obs, reward, terminated, truncated, info = wrapper.step(action)
```

#### get_image(fov_range, fov_width)

SLAM 스타일 FOV 맵핑이 적용된 이미지를 반환합니다.

```python
image = wrapper.get_image(fov_range=3, fov_width=5)
```

**Parameters:**
- `fov_range` (int): 앞으로 볼 수 있는 거리 (칸 수)
- `fov_width` (int): 시야의 좌우 폭 (칸 수)

**Returns:**
- `image` (np.ndarray): SLAM 스타일 시각화가 적용된 이미지

## 시야 범위 계산

시야 범위는 에이전트의 현재 위치와 방향을 기준으로 계산됩니다:

- **앞쪽**: 에이전트가 바라보는 방향으로 `fov_range` 칸까지
- **좌우 폭**: 각각 `fov_width // 2` 칸까지

### 예시

```python
# 시야 범위: 앞으로 3칸, 좌우 각각 2칸
fov_range = 3
fov_width = 5  # 좌우 각각 2칸 (5 // 2 = 2)

image = wrapper.get_image(fov_range=fov_range, fov_width=fov_width)
```

## 시각화 규칙

### 1. 현재 시야 범위 내

- 원본 이미지 그대로 밝게 표시
- 탐색 맵에 자동으로 추가됨

### 2. 탐색했던 곳 (시야 밖)

- 일반 셀: 어둡게(밝기를 40%로 감소) 표시
- 중요한 객체가 있는 셀: 밝게 유지
  - 열쇠 (Key)
  - 문 (Door)
  - 목표 (Goal)

### 3. 아직 탐색하지 않은 곳

- 검은색으로 표시

## 사용 예시

### 기본 사용

```python
from keyboard_control_fov_mapping import SLAMFOVWrapper
import gymnasium as gym

# 환경 생성
gym_env = gym.make("MiniGrid-FourRooms-v0", render_mode='rgb_array')
wrapper = SLAMFOVWrapper(gym_env)

# 환경 초기화
obs, info = wrapper.reset()

# FOV 설정
fov_range = 3
fov_width = 5

# 메인 루프
done = False
while not done:
    # SLAM 스타일 이미지 가져오기
    image = wrapper.get_image(fov_range=fov_range, fov_width=fov_width)
    
    # 이미지 표시 (OpenCV 등)
    # cv2.imshow('SLAM FOV', image)
    
    # 액션 실행 (예: 키보드 입력 등)
    action = get_action_from_user()
    obs, reward, terminated, truncated, info = wrapper.step(action)
    done = terminated or truncated
```

### FOV 범위 동적 조정

```python
fov_range = 3
fov_width = 5

while True:
    # 사용자 입력에 따라 FOV 조정
    key = get_key()
    if key == '+':
        fov_range += 1
        fov_width += 2
    elif key == '-':
        fov_range = max(1, fov_range - 1)
        fov_width = max(1, fov_width - 2)
    
    # 이미지 가져오기
    image = wrapper.get_image(fov_range=fov_range, fov_width=fov_width)
```

## 내부 동작 원리

### 탐색 맵 (visited_map)

2D boolean 배열로 에이전트가 방문한 위치를 추적합니다.

```python
visited_map[grid_x, grid_y] = True  # 방문한 위치
```

### FOV 맵 (current_fov_map)

현재 시야 범위 내 셀들을 표시하는 2D boolean 배열입니다.

```python
current_fov_map[grid_x, grid_y] = True  # 현재 시야 범위 내
```

### 시야 범위 계산 알고리즘

1. 에이전트의 현재 위치와 방향 확인
2. 각 셀에 대해 에이전트 기준 상대 위치 계산
3. 에이전트 방향 기준으로 변환
4. 앞쪽 `fov_range` 칸, 좌우 각각 `fov_width // 2` 칸 내에 있는지 확인

## 주의사항

1. **성능**: 큰 맵에서 FOV 계산은 계산 비용이 있을 수 있습니다. 필요시 최적화를 고려하세요.

2. **중요 객체 감지**: 중요한 객체(열쇠, 문, 목표)는 MiniGrid의 내부 그리드 구조를 통해 감지됩니다. 커스텀 객체의 경우 추가 구현이 필요할 수 있습니다.

3. **시야 범위**: `fov_range`와 `fov_width`는 정수 값이어야 하며, 너무 작으면 시야가 매우 제한적일 수 있습니다.

## 참고 자료

- [키보드 제어 가이드](./keyboard-control.md)
- [Wrapper API](./wrapper-api.md)
- [MiniGrid 환경 목록](./minigrid-environments.md)

