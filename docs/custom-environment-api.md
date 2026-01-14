# CustomRoomEnv API 문서

## 클래스 개요

`CustomRoomEnv`는 커스텀 방 구조를 가진 MiniGrid 환경을 생성하는 클래스입니다.

## 생성자

```python
CustomRoomEnv(size=10, room_config=None, **kwargs)
```

### 파라미터

- `size` (int): 환경 크기 (기본값: 10)
- `room_config` (dict, optional): 방 구조 설정 딕셔너리
- `**kwargs`: MiniGridEnv의 추가 파라미터

### room_config 구조

```python
room_config = {
    'start_pos': (x, y),           # 시작 위치
    'goal_pos': (x, y),             # 목표 위치
    'walls': [(x1, y1), ...],       # 벽 위치 리스트
    'objects': [                    # 객체 리스트
        {
            'type': 'key',
            'pos': (x, y),
            'color': 'yellow'
        },
        ...
    ]
}
```

## room_config 상세 설명

### start_pos

에이전트의 시작 위치를 지정합니다.

```python
'start_pos': (2, 2)  # x=2, y=2 위치에서 시작
```

### goal_pos

목표(Goal)의 위치를 지정합니다.

```python
'goal_pos': (10, 10)  # x=10, y=10 위치에 Goal 배치
```

### walls

벽의 위치를 리스트로 지정합니다.

```python
'walls': [
    (0, 0), (1, 0), (2, 0),  # 상단 벽
    (0, 0), (0, 1), (0, 2),  # 좌측 벽
    (5, 3), (5, 4), (5, 5),  # 내부 벽
]
```

### objects

환경에 배치할 객체들의 리스트입니다.

#### 객체 타입

- `key`: 열쇠
- `ball`: 공
- `box`: 상자
- `door`: 문

#### 객체 속성

```python
{
    'type': 'key',              # 객체 타입
    'pos': (3, 3),              # 위치 (x, y)
    'color': 'yellow',          # 색상
    # Door 전용 속성:
    'is_locked': True,          # 잠금 여부 (door만)
    'is_open': False            # 열림 여부 (door만)
}
```

#### 지원 색상

- `red`
- `green`
- `blue`
- `purple`
- `yellow`
- `grey`

## 사용 예제

### 기본 사용

```python
from custom_environment import CustomRoomEnv

env = CustomRoomEnv(size=10)
obs, info = env.reset()
```

### 커스텀 설정 사용

```python
room_config = {
    'start_pos': (2, 2),
    'goal_pos': (8, 8),
    'walls': [
        (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)
    ],
    'objects': [
        {'type': 'key', 'pos': (3, 3), 'color': 'yellow'},
        {'type': 'door', 'pos': (5, 3), 'color': 'red', 'is_locked': True}
    ]
}

env = CustomRoomEnv(size=10, room_config=room_config)
```

### 사전 정의된 환경 사용

```python
from custom_environment import create_house_environment, create_simple_room

# 실내 집 환경
house_env = create_house_environment()

# 간단한 방 환경
simple_env = create_simple_room()
```

## 메서드

### reset()

환경을 초기 상태로 리셋합니다.

```python
obs, info = env.reset()
```

### step(action)

액션을 실행하고 다음 상태로 전이합니다.

```python
obs, reward, terminated, truncated, info = env.step(action)
```

### render()

환경을 렌더링합니다.

```python
img = env.render()  # render_mode='rgb_array'일 때
```

## 속성

- `grid`: Grid 객체
- `agent_pos`: 에이전트 위치 (numpy array)
- `agent_dir`: 에이전트 방향 (0=오른쪽, 1=아래, 2=왼쪽, 3=위)
- `mission`: 현재 미션 텍스트
- `size`: 환경 크기

## 참고

- [환경 생성 가이드](./environment-creation.md)
- [베스트 프랙티스](./best-practices.md)

