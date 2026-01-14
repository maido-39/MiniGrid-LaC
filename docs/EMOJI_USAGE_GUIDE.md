# 이모지 사용 가이드

이 문서는 MiniGrid 환경에서 이모지 객체를 사용하는 방법을 설명합니다.

## 목차

1. [이모지 객체 기본 사용법](#이모지-객체-기본-사용법)
2. [이모지 색상 설정](#이모지-색상-설정)
3. [원래 색상으로 이모지 렌더링](#원래-색상으로-이모지-렌더링)
4. [로봇 이모지 설정](#로봇-이모지-설정)
5. [새로운 이모지 추가하기](#새로운-이모지-추가하기)
6. [이모지 객체 속성](#이모지-객체-속성)
7. [예제 코드](#예제-코드)

---

## 이모지 객체 기본 사용법

`room_config`의 `objects` 리스트에 이모지 객체를 추가하여 맵에 배치할 수 있습니다.

### 기본 구조

```python
{
    'type': 'emoji',
    'pos': (x, y),              # 위치 (튜플)
    'emoji_name': 'tree',        # 이모지 이름
    'color': 'yellow',          # 색상 (선택사항, 기본값: 'yellow')
    'can_pickup': False,        # 집기 가능 여부 (기본값: False)
    'can_overlap': False,       # 겹칠 수 있는지 여부 (기본값: False)
    'use_emoji_color': True     # 원래 이모지 색상 사용 여부 (기본값: True)
}
```

### 예제

```python
from minigrid_customenv_emoji import MiniGridEmojiWrapper

room_config = {
    'start_pos': (1, 1),
    'goal_pos': (8, 8),
    'objects': [
        {
            'type': 'emoji',
            'pos': (3, 3),
            'emoji_name': 'tree',
            'color': 'green',
            'can_pickup': False,
            'use_emoji_color': True
        }
    ]
}

env = MiniGridEmojiWrapper(size=10, room_config=room_config)
```

---

## 이모지 색상 설정

이모지 객체는 두 가지 색상 모드를 지원합니다:

### 1. 단색 모드 (`use_emoji_color=False`)

지정한 `color` 파라미터로 이모지를 단색으로 렌더링합니다.

```python
{
    'type': 'emoji',
    'pos': (5, 5),
    'emoji_name': 'tree',
    'color': 'red',              # 빨간색으로 렌더링
    'use_emoji_color': False     # 단색 모드
}
```

**지원하는 색상:**
- `'red'`: 빨간색
- `'green'`: 초록색
- `'blue'`: 파란색
- `'purple'`: 보라색
- `'yellow'`: 노란색
- `'grey'`: 회색

### 2. 원래 색상 모드 (`use_emoji_color=True`)

이모지의 원래 컬러를 사용하여 렌더링합니다. 이 모드는 `imagetext_py` 라이브러리를 사용합니다.

```python
{
    'type': 'emoji',
    'pos': (5, 5),
    'emoji_name': 'tree',
    'color': 'yellow',           # 이 파라미터는 무시됨 (원래 색상 사용)
    'use_emoji_color': True      # 원래 색상 모드
}
```

**주의사항:**
- `use_emoji_color=True`일 때는 `imagetext_py` 라이브러리가 필요합니다.
- `imagetext_py`가 설치되어 있지 않으면 `ImportError`가 발생합니다.
- `color` 파라미터는 원래 색상 모드에서는 무시됩니다.

---

## 원래 색상으로 이모지 렌더링

원래 색상으로 이모지를 렌더링하려면 `use_emoji_color=True`를 설정합니다.

### 예제: 컬러 이모지 배치

```python
room_config = {
    'objects': [
        # 원래 색상으로 렌더링되는 이모지들
        {
            'type': 'emoji',
            'pos': (2, 2),
            'emoji_name': 'tree',        # 🌲 (초록색)
            'use_emoji_color': True
        },
        {
            'type': 'emoji',
            'pos': (4, 4),
            'emoji_name': 'apple',       # 🍎 (빨간색)
            'use_emoji_color': True
        },
        {
            'type': 'emoji',
            'pos': (6, 6),
            'emoji_name': 'flower',      # 🌼 (노란색)
            'use_emoji_color': True
        }
    ]
}
```

### 예제: 올라설 수 있는 컬러 이모지

```python
room_config = {
    'objects': [
        {
            'type': 'emoji',
            'pos': (3, 3),
            'emoji_name': 'brick',       # 🧱
            'color': 'blue',
            'can_overlap': True,          # 로봇이 올라설 수 있음
            'use_emoji_color': True       # 원래 색상 사용
        }
    ]
}
```

**참고:** `can_overlap=True`로 설정하면 로봇이 이모지 위를 지나갈 수 있습니다. 로봇이 이모지 위에 있을 때는 초록색 테두리가 표시됩니다.

---

## 로봇 이모지 설정

로봇(에이전트)을 화살표 대신 🤖 이모지로 표시할 수 있습니다. 이모지 오브젝트와 동일하게 색상과 원본 컬러 사용 여부를 설정할 수 있습니다.

### 기본 사용법

`room_config`에 로봇 이모지 설정을 추가합니다:

```python
room_config = {
    'start_pos': (1, 1),
    'goal_pos': (8, 8),
    'objects': [...],
    'use_robot_emoji': True,  # 로봇을 🤖 이모지로 표시
    'robot_emoji_color': 'red',  # 로봇 이모지 색상 (use_robot_emoji_color=False일 때만 사용)
    'use_robot_emoji_color': True  # 원본 이모지 컬러 사용
}
```

### 로봇 이모지 색상 모드

로봇 이모지도 이모지 오브젝트와 동일하게 두 가지 색상 모드를 지원합니다:

#### 1. 단색 모드 (`use_robot_emoji_color=False`)

지정한 `robot_emoji_color`로 로봇 이모지를 단색으로 렌더링합니다.

```python
room_config = {
    'use_robot_emoji': True,
    'robot_emoji_color': 'red',  # 빨간색으로 렌더링
    'use_robot_emoji_color': False  # 단색 모드
}
```

**지원하는 색상:**
- `'red'`: 빨간색
- `'green'`: 초록색
- `'blue'`: 파란색
- `'purple'`: 보라색
- `'yellow'`: 노란색 (기본값)
- `'grey'`: 회색

#### 2. 원래 색상 모드 (`use_robot_emoji_color=True`)

로봇 이모지의 원래 컬러를 사용하여 렌더링합니다. 이 모드는 `imagetext_py` 라이브러리를 사용합니다.

```python
room_config = {
    'use_robot_emoji': True,
    'robot_emoji_color': 'red',  # 이 파라미터는 무시됨 (원래 색상 사용)
    'use_robot_emoji_color': True  # 원래 색상 모드
}
```

**주의사항:**
- `use_robot_emoji_color=True`일 때는 `imagetext_py` 라이브러리가 필요합니다.
- `imagetext_py`가 설치되어 있지 않으면 PIL로 폴백되어 단색 모드로 렌더링됩니다.
- `robot_emoji_color` 파라미터는 원래 색상 모드에서는 무시됩니다.

### 예제: 원본 컬러 로봇 이모지

```python
room_config = {
    'start_pos': (1, 1),
    'goal_pos': (8, 8),
    'objects': [
        {'type': 'emoji', 'pos': (3, 3), 'emoji_name': 'tree'},
    ],
    'use_robot_emoji': True,  # 로봇을 🤖 이모지로 표시
    'use_robot_emoji_color': True  # 원본 컬러 사용
}
```

### 예제: 단색 로봇 이모지

```python
room_config = {
    'start_pos': (1, 1),
    'goal_pos': (8, 8),
    'objects': [
        {'type': 'emoji', 'pos': (3, 3), 'emoji_name': 'tree'},
    ],
    'use_robot_emoji': True,  # 로봇을 🤖 이모지로 표시
    'robot_emoji_color': 'blue',  # 파란색으로 렌더링
    'use_robot_emoji_color': False  # 단색 모드
}
```

### 예제: 화살표 사용 (기본)

```python
room_config = {
    'start_pos': (1, 1),
    'goal_pos': (8, 8),
    'objects': [...],
    # use_robot_emoji를 설정하지 않으면 기본 화살표 사용
}
```

**참고:**
- `use_robot_emoji`를 설정하지 않거나 `False`로 설정하면 기본 화살표 이미지(`arrow.png`)가 사용됩니다.
- 로봇 이모지는 방향에 따라 회전하지 않습니다 (항상 같은 방향으로 표시됩니다).

---

## 새로운 이모지 추가하기

새로운 이모지를 추가하려면 `EMOJI_MAP`에 이모지 이름과 실제 이모지 문자를 매핑해야 합니다.

### 1. EMOJI_MAP 수정

`minigrid_customenv_emoji.py` 파일의 `EMOJI_MAP` 딕셔너리에 새 항목을 추가합니다:

```python
EMOJI_MAP = {
    'tree': '🌲',
    'mushroom': '🍄',
    'flower': '🌼',
    'cat': '🐈',
    'grass': '🌾',
    'rock': '🗿',
    'box': '📦',
    'chair': '🪑',
    'apple': '🍎',
    'desktop': '🖥️',
    'workstation': '📱',
    'brick': '🧱',
    # 새로운 이모지 추가
    'dog': '🐕',           # 개
    'car': '🚗',           # 자동차
    'house': '🏠',         # 집
    'star': '⭐',          # 별
}
```

### 2. 이모지 사용

추가한 이모지는 `emoji_name`으로 사용할 수 있습니다:

```python
room_config = {
    'objects': [
        {
            'type': 'emoji',
            'pos': (5, 5),
            'emoji_name': 'dog',          # 새로 추가한 이모지
            'use_emoji_color': True
        }
    ]
}
```

### 3. 이모지 문자 찾기

이모지 문자를 찾는 방법:
- [Emoji List](https://emojipedia.org/)에서 원하는 이모지를 검색
- 운영체제의 이모지 키보드 사용
- 유니코드 이모지 코드 사용

**예시:**
- 🌲 (Tree): U+1F332
- 🍎 (Apple): U+1F34E
- 🐕 (Dog): U+1F415

---

## 이모지 객체 속성

### 필수 속성

- `type`: 반드시 `'emoji'`로 설정
- `pos`: 이모지 위치 `(x, y)` 튜플
- `emoji_name`: `EMOJI_MAP`에 등록된 이모지 이름

### 선택 속성

- `color` (기본값: `'yellow'`): 색상 이름
  - `use_emoji_color=False`일 때만 사용됨
  - 지원 색상: `'red'`, `'green'`, `'blue'`, `'purple'`, `'yellow'`, `'grey'`

- `can_pickup` (기본값: `False`): 집기 가능 여부
  - `True`: 에이전트가 앞에서 바라보면 집을 수 있음
  - `False`: 집을 수 없음 (장애물)

- `can_overlap` (기본값: `False`): 겹칠 수 있는지 여부
  - `True`: 로봇이 이모지 위를 지나갈 수 있음
  - `False`: 로봇이 이모지 위를 지나갈 수 없음 (장애물)

- `use_emoji_color` (기본값: `True`): 원래 이모지 색상 사용 여부
  - `True`: `imagetext_py`를 사용하여 원래 컬러로 렌더링
  - `False`: 지정한 `color`로 단색 렌더링

---

## 예제 코드

### 예제 1: 기본 이모지 배치

```python
from minigrid_customenv_emoji import MiniGridEmojiWrapper

room_config = {
    'start_pos': (1, 1),
    'goal_pos': (8, 8),
    'objects': [
        {'type': 'emoji', 'pos': (3, 3), 'emoji_name': 'tree'},
        {'type': 'emoji', 'pos': (5, 5), 'emoji_name': 'rock'},
        {'type': 'emoji', 'pos': (7, 7), 'emoji_name': 'flower'},
    ]
}

env = MiniGridEmojiWrapper(size=10, room_config=room_config)
obs, info = env.reset()
```

### 예제 2: 컬러 이모지와 단색 이모지 혼합

```python
room_config = {
    'objects': [
        # 원래 색상으로 렌더링
        {
            'type': 'emoji',
            'pos': (2, 2),
            'emoji_name': 'apple',
            'use_emoji_color': True
        },
        # 빨간색으로 렌더링
        {
            'type': 'emoji',
            'pos': (4, 4),
            'emoji_name': 'apple',
            'color': 'red',
            'use_emoji_color': False
        }
    ]
}
```

### 예제 3: 올라설 수 있는 이모지 플랫폼

```python
room_config = {
    'objects': [
        # 파란 벽돌 플랫폼 (올라설 수 있음)
        {
            'type': 'emoji',
            'pos': (3, 3),
            'emoji_name': 'brick',
            'color': 'blue',
            'can_overlap': True,
            'use_emoji_color': True
        },
        {
            'type': 'emoji',
            'pos': (4, 3),
            'emoji_name': 'brick',
            'color': 'blue',
            'can_overlap': True,
            'use_emoji_color': True
        }
    ]
}
```

### 예제 4: 집을 수 있는 이모지 아이템

```python
room_config = {
    'objects': [
        # 집을 수 있는 꽃
        {
            'type': 'emoji',
            'pos': (5, 5),
            'emoji_name': 'flower',
            'can_pickup': True,
            'use_emoji_color': True
        },
        # 집을 수 있는 사과
        {
            'type': 'emoji',
            'pos': (7, 7),
            'emoji_name': 'apple',
            'can_pickup': True,
            'use_emoji_color': True
        }
    ]
}
```

### 예제 5: 복잡한 환경 구성

```python
room_config = {
    'start_pos': (1, 1),
    'goal_pos': (8, 8),
    'walls': [
        (5, 2), (5, 3), (5, 4)  # 내부 벽
    ],
    'objects': [
        # 장애물 (올라설 수 없음)
        {'type': 'emoji', 'pos': (3, 3), 'emoji_name': 'tree', 'can_overlap': False},
        {'type': 'emoji', 'pos': (6, 6), 'emoji_name': 'rock', 'can_overlap': False},
        
        # 플랫폼 (올라설 수 있음)
        {'type': 'emoji', 'pos': (4, 4), 'emoji_name': 'brick', 'can_overlap': True},
        {'type': 'emoji', 'pos': (4, 5), 'emoji_name': 'brick', 'can_overlap': True},
        
        # 아이템 (집을 수 있음)
        {'type': 'emoji', 'pos': (7, 3), 'emoji_name': 'apple', 'can_pickup': True},
        {'type': 'emoji', 'pos': (2, 7), 'emoji_name': 'flower', 'can_pickup': True},
    ]
}

env = MiniGridEmojiWrapper(size=10, room_config=room_config)
```

### 예제 6: 로봇 이모지 사용

```python
room_config = {
    'start_pos': (1, 1),
    'goal_pos': (8, 8),
    'objects': [
        {'type': 'emoji', 'pos': (3, 3), 'emoji_name': 'tree'},
    ],
    # 원본 컬러 로봇 이모지
    'use_robot_emoji': True,
    'use_robot_emoji_color': True
}

env = MiniGridEmojiWrapper(size=10, room_config=room_config)
```

### 예제 7: 단색 로봇 이모지

```python
room_config = {
    'start_pos': (1, 1),
    'goal_pos': (8, 8),
    'objects': [
        {'type': 'emoji', 'pos': (3, 3), 'emoji_name': 'tree'},
    ],
    # 파란색 로봇 이모지
    'use_robot_emoji': True,
    'robot_emoji_color': 'blue',
    'use_robot_emoji_color': False
}

env = MiniGridEmojiWrapper(size=10, room_config=room_config)
```

---

## 주의사항

1. **imagetext_py 설치 필요**
   - `use_emoji_color=True`를 사용하려면 `imagetext_py` 라이브러리가 필요합니다.
   - 설치되지 않은 경우 `ImportError`가 발생합니다.

2. **이모지 이름 확인**
   - `emoji_name`은 반드시 `EMOJI_MAP`에 등록된 이름이어야 합니다.
   - 등록되지 않은 이름을 사용하면 `❓` (물음표)로 표시됩니다.

3. **위치 범위**
   - `pos`는 맵 크기 내의 유효한 좌표여야 합니다.
   - `(0, 0)`은 외벽이므로 사용할 수 없습니다.

4. **색상 모드**
   - `use_emoji_color=True`일 때는 `color` 파라미터가 무시됩니다.
   - 단색 모드를 원하면 `use_emoji_color=False`로 설정하세요.

5. **로봇 이모지 설정**
   - `use_robot_emoji=True`로 설정하면 로봇이 🤖 이모지로 표시됩니다.
   - `use_robot_emoji_color=True`일 때는 `robot_emoji_color` 파라미터가 무시됩니다.
   - 로봇 이모지는 방향에 따라 회전하지 않습니다 (항상 같은 방향으로 표시).

---

## 참고 자료

- [이모지 렌더링 분석 문서](./EMOJI_RENDERING_ANALYSIS.md)
- [커스텀 환경 API 문서](./custom-environment-api.md)
- [Emoji List](https://emojipedia.org/) - 이모지 검색 및 유니코드 확인

