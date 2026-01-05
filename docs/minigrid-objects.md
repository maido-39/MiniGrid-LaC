# MiniGrid 오브젝트 및 속성

MiniGrid 환경에서 사용할 수 있는 모든 오브젝트 타입과 그 속성에 대한 가이드입니다.

## 오브젝트 타입

### 1. Wall (벽)

통과할 수 없는 장애물입니다.

```python
from minigrid.core.world_object import Wall

wall = Wall()
```

**속성**:
- `type`: `'wall'`
- `color`: 없음 (항상 검은색)
- 통과 불가능
- 에이전트가 벽을 통과할 수 없음

**사용 예시**:
```python
# 그리드에 벽 배치
self.grid.set(x, y, Wall())

# 또는 외벽 생성
self.grid.wall_rect(0, 0, width, height)
```

---

### 2. Goal (목표)

에이전트가 도달해야 하는 목표 지점입니다.

```python
from minigrid.core.world_object import Goal

goal = Goal()
```

**속성**:
- `type`: `'goal'`
- `color`: 없음 (항상 초록색)
- 도달 시 에피소드 종료 및 보상 획득
- 에이전트가 Goal에 도달하면 `terminated=True` 반환

**사용 예시**:
```python
# Goal 배치 (권장 방법)
self.put_obj(Goal(), x, y)

# 또는 직접 배치
self.grid.set(x, y, Goal())
```

---

### 3. Key (열쇠)

에이전트가 집을 수 있는 열쇠 오브젝트입니다. 문을 열 때 사용됩니다.

```python
from minigrid.core.world_object import Key

key = Key(color='yellow')
```

**속성**:
- `type`: `'key'`
- `color`: 색상 (문의 색상과 일치해야 함)
- `can_pickup`: `True` (에이전트가 집을 수 있음)
- `carryable`: `True` (에이전트가 들고 다닐 수 있음)

**지원 색상**:
- `'red'`
- `'green'`
- `'blue'`
- `'purple'`
- `'yellow'`
- `'grey'`

**사용 예시**:
```python
# 노란색 열쇠 배치
self.put_obj(Key('yellow'), x, y)

# room_config에서 사용
objects = [
    {'type': 'key', 'pos': (3, 3), 'color': 'yellow'}
]
```

---

### 4. Ball (공)

에이전트가 집을 수 있는 공 오브젝트입니다.

```python
from minigrid.core.world_object import Ball

ball = Ball(color='blue')
```

**속성**:
- `type`: `'ball'`
- `color`: 색상
- `can_pickup`: `True`
- `carryable`: `True`

**지원 색상**:
- `'red'`
- `'green'`
- `'blue'`
- `'purple'`
- `'yellow'`
- `'grey'`

**사용 예시**:
```python
# 파란색 공 배치
self.put_obj(Ball('blue'), x, y)

# room_config에서 사용
objects = [
    {'type': 'ball', 'pos': (5, 5), 'color': 'blue'}
]
```

---

### 5. Box (상자)

에이전트가 집을 수 있는 상자 오브젝트입니다. 통과 가능하지만 시각적으로는 장애물처럼 보일 수 있습니다.

```python
from minigrid.core.world_object import Box

box = Box(color='purple')
```

**속성**:
- `type`: `'box'`
- `color`: 색상
- `can_pickup`: `True`
- `carryable`: `True`
- **주의**: Box는 통과 가능하지만, 시각적으로는 장애물처럼 표시됨

**지원 색상**:
- `'red'`
- `'green'`
- `'blue'`
- `'purple'`
- `'yellow'`
- `'grey'`

**사용 예시**:
```python
# 보라색 상자 배치
self.put_obj(Box('purple'), x, y)

# room_config에서 사용 (통과 불가능한 객체처럼 사용)
objects = [
    {'type': 'box', 'pos': (3, 4), 'color': 'blue'}  # 파란 기둥처럼 사용
]
```

**참고**: Box는 통과 가능하므로, 통과 불가능한 장애물을 만들려면 다른 방법을 사용해야 합니다.

---

### 6. Door (문)

열고 닫을 수 있는 문 오브젝트입니다. 잠글 수 있습니다.

```python
from minigrid.core.world_object import Door

# 열린 문
door_open = Door(color='red', is_locked=False, is_open=True)

# 닫힌 문
door_closed = Door(color='red', is_locked=False, is_open=False)

# 잠긴 문
door_locked = Door(color='red', is_locked=True, is_open=False)
```

**속성**:
- `type`: `'door'`
- `color`: 색상 (열쇠의 색상과 일치해야 잠금 해제 가능)
- `is_locked`: 잠금 여부 (`True`/`False`)
- `is_open`: 열림 여부 (`True`/`False`)
- `can_pickup`: `False` (집을 수 없음)
- `carryable`: `False` (들고 다닐 수 없음)

**지원 색상**:
- `'red'`
- `'green'`
- `'blue'`
- `'purple'`
- `'yellow'`
- `'grey'`

**상태 조합**:
- `is_locked=True, is_open=False`: 잠긴 문 (열쇠 필요)
- `is_locked=False, is_open=False`: 닫힌 문 (toggle 액션으로 열 수 있음)
- `is_locked=False, is_open=True`: 열린 문 (통과 가능)

**사용 예시**:
```python
# 잠긴 빨간 문 배치
self.put_obj(Door('red', is_locked=True, is_open=False), x, y)

# 열린 문 배치
self.put_obj(Door('blue', is_locked=False, is_open=True), x, y)

# room_config에서 사용
objects = [
    {
        'type': 'door',
        'pos': (5, 5),
        'color': 'red',
        'is_locked': True,
        'is_open': False
    }
]
```

**문 열기/닫기**:
- 에이전트가 문 앞에서 `toggle` 액션(5)을 사용하면 문을 열거나 닫을 수 있습니다.
- 잠긴 문은 같은 색상의 열쇠를 들고 있어야 열 수 있습니다.

---

## 오브젝트 속성 요약

### 공통 속성

모든 오브젝트는 다음 속성을 가집니다:

- `type`: 오브젝트 타입 (`'wall'`, `'goal'`, `'key'`, `'ball'`, `'box'`, `'door'`)
- `color`: 색상 (일부 오브젝트만, Wall과 Goal은 색상 없음)

### 픽업 가능 여부

| 오브젝트 | can_pickup | carryable | 설명 |
|---------|-----------|-----------|------|
| Wall | `False` | `False` | 통과 불가능 |
| Goal | `False` | `False` | 목표 지점 |
| Key | `True` | `True` | 집을 수 있음 |
| Ball | `True` | `True` | 집을 수 있음 |
| Box | `True` | `True` | 집을 수 있음 (통과 가능) |
| Door | `False` | `False` | 상호작용만 가능 |

### 색상 지원

다음 오브젝트는 색상을 가질 수 있습니다:
- Key
- Ball
- Box
- Door

**지원 색상**:
- `'red'` (빨강)
- `'green'` (초록)
- `'blue'` (파랑)
- `'purple'` (보라)
- `'yellow'` (노랑)
- `'grey'` (회색)

---

## 오브젝트 배치 방법

### 권장 방법: `put_obj()` 사용

```python
# Goal 배치
self.put_obj(Goal(), x, y)

# 색상 오브젝트 배치
self.put_obj(Key('yellow'), x, y)
self.put_obj(Ball('blue'), x, y)
self.put_obj(Box('purple'), x, y)
self.put_obj(Door('red', is_locked=True, is_open=False), x, y)
```

**이유**: `put_obj()`는 객체를 안전하게 배치하는 헬퍼 메서드로, 충돌 검사 등을 자동으로 수행합니다.

### 직접 배치: `grid.set()` 사용

```python
# 직접 배치 (비권장)
self.grid.set(x, y, Wall())
self.grid.set(x, y, Goal())
```

**주의**: 직접 배치 시 충돌 검사 등을 수동으로 처리해야 합니다.

---

## CustomRoomWrapper에서 사용

`CustomRoomWrapper`를 사용할 때는 `room_config`의 `objects` 리스트에 오브젝트 정보를 지정합니다:

```python
room_config = {
    'start_pos': (1, 1),
    'goal_pos': (8, 8),
    'walls': [(0, 0), (1, 0), ...],
    'objects': [
        {'type': 'key', 'pos': (3, 3), 'color': 'yellow'},
        {'type': 'ball', 'pos': (5, 5), 'color': 'blue'},
        {'type': 'box', 'pos': (4, 4), 'color': 'purple'},
        {
            'type': 'door',
            'pos': (6, 6),
            'color': 'red',
            'is_locked': True,
            'is_open': False
        }
    ]
}

wrapper = CustomRoomWrapper(size=10, room_config=room_config)
```

---

## 오브젝트 상호작용

### Pickup (액션 3)

에이전트가 앞에 있는 오브젝트를 집을 수 있습니다:
- Key, Ball, Box는 집을 수 있음
- Wall, Goal, Door는 집을 수 없음

### Drop (액션 4)

에이전트가 들고 있는 오브젝트를 놓을 수 있습니다.

### Toggle (액션 5)

에이전트가 앞에 있는 오브젝트와 상호작용할 수 있습니다:
- Door: 문 열기/닫기
- 잠긴 문: 같은 색상의 열쇠를 들고 있어야 열 수 있음

---

## 참고 자료

- [MiniGrid 공식 문서](https://minigrid.farama.org/)
- [MiniGrid 환경 생성 튜토리얼](https://minigrid.farama.org/content/create_env_tutorial/)

