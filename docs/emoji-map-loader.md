# 이모지 맵 JSON 로더 가이드

이 문서는 JSON 파일에서 이모지 맵을 로드하여 MiniGrid 환경을 생성하는 방법을 설명합니다.

## 개요

`emoji_map_loader.py` 모듈은 JSON 파일 형식으로 정의된 이모지 맵을 읽어서 `MiniGridEmojiWrapper` 환경으로 변환합니다. 이를 통해 코드 수정 없이 JSON 파일만 변경하여 다양한 맵을 쉽게 생성할 수 있습니다.

## JSON 파일 구조

### 기본 구조

```json
{
  "map": {
    "emoji_render": [...],
    "emoji_objects": {...},
    "robot_config": {...},
    "start_pos": [x, y],
    "goal_pos": [x, y]
  }
}
```

### emoji_render

맵의 레이아웃을 이모지로 표현합니다. 세 가지 형식을 지원합니다:

#### 1. 텍스트 형식 (줄바꿈으로 구분)

```json
{
  "map": {
    "emoji_render": "⬛⬛⬛⬛⬛\n⬛⬜️⬜️⬜️⬛\n⬛⬜️🟩⬜️⬛\n⬛⬛⬛⬛⬛"
  }
}
```

#### 2. 문자열 배열 형식

```json
{
  "map": {
    "emoji_render": [
      "⬛⬛⬛⬛⬛",
      "⬛⬜️⬜️⬜️⬛",
      "⬛⬜️🟩⬜️⬛",
      "⬛⬛⬛⬛⬛"
    ]
  }
}
```

#### 3. 2D 배열 형식

```json
{
  "map": {
    "emoji_render": [
      ["⬛", "⬛", "⬛", "⬛", "⬛"],
      ["⬛", "⬜️", "⬜️", "⬜️", "⬛"],
      ["⬛", "⬜️", "🟩", "⬜️", "⬛"],
      ["⬛", "⬛", "⬛", "⬛", "⬛"]
    ]
  }
}
```

### emoji_objects

각 이모지가 어떤 객체를 나타내는지 정의합니다.

```json
{
  "map": {
    "emoji_objects": {
      "⬛": {
        "type": "wall",
        "color": "grey",
        "can_pickup": false,
        "can_overlap": false
      },
      "⬜️": {
        "type": "empty",
        "can_pickup": false,
        "can_overlap": true
      },
      "🟩": {
        "type": "emoji",
        "emoji_name": "brick",
        "color": "green",
        "can_pickup": false,
        "can_overlap": true,
        "use_emoji_color": true
      }
    }
  }
}
```

#### 객체 타입

- **`wall`**: 벽 (통과 불가)
  - `color`: 벽 색상 (grey, red, green, blue, purple, yellow)
  - `can_pickup`: 항상 `false`
  - `can_overlap`: 항상 `false`

- **`empty`**: 빈 공간 (통과 가능)
  - `can_pickup`: 항상 `false`
  - `can_overlap`: 항상 `true`

- **`floor`**: 바닥 타일 (색상 있는 바닥)
  - `color`: 바닥 색상

- **`emoji`**: 이모지 객체
  - `emoji_name`: 이모지 이름 (EMOJI_MAP에 등록된 이름)
  - `color`: 색상 (선택사항)
  - `can_pickup`: 집기 가능 여부
  - `can_overlap`: 올라설 수 있는지 여부
  - `use_emoji_color`: 원래 이모지 색상 사용 여부

### robot_config

로봇(에이전트)의 이모지 표시 설정입니다.

```json
{
  "map": {
    "robot_config": {
      "use_robot_emoji": true,
      "robot_emoji_color": "red",
      "use_robot_emoji_color": true
    }
  }
}
```

- `use_robot_emoji`: 로봇을 이모지로 표시할지 여부 (기본값: `false`)
- `robot_emoji_color`: 로봇 이모지 색상 (단색 모드에서만 사용)
- `use_robot_emoji_color`: 원래 이모지 색상 사용 여부

### start_pos와 goal_pos

에이전트의 시작 위치와 목표 위치를 지정합니다.

```json
{
  "map": {
    "start_pos": [1, 1],
    "goal_pos": [12, 1]
  }
}
```

**참고**: `🟥` 이모지를 사용하면 해당 위치가 시작 위치로 자동 설정됩니다.

## 사용법

### 기본 사용법

```python
# Actual path: lib.map_manager.emoji_map_loader
from lib import load_emoji_map_from_json

# JSON 파일에서 맵 로드
wrapper = load_emoji_map_from_json("config/example_map.json")

# 환경 초기화
obs, info = wrapper.reset()

# 상태 확인
state = wrapper.get_state()
print(f"에이전트 위치: {state['agent_pos']}")
print(f"에이전트 방향: {state['agent_dir']}")
```

### EmojiMapLoader 클래스 직접 사용

```python
# Actual path: lib.map_manager.emoji_map_loader
from lib.map_manager.emoji_map_loader import EmojiMapLoader

# 로더 생성
loader = EmojiMapLoader("config/example_map.json")

# 맵 정보 확인
print(f"맵 크기: {loader.size}")
print(f"시작 위치: {loader.start_pos}")
print(f"목표 위치: {loader.goal_pos}")

# 환경 생성
wrapper = loader.create_wrapper()
```

## 완전한 예제

### example_map.json

```json
{
  "map": {
    "emoji_render": [
      "⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛",
      "⬛⬜️⬜️⬜️🟪🟪🟪⬜️🟦🟦🟦⬜️⬜️⬛",
      "⬛⬜️⬜️⬜️🟪🟪🟪⬜️🟦🟦🟦⬜️⬜️⬛",
      "⬛⬜️⬜️⬜️🟪🟪🟪⬜🟦🟦🟦⬜️⬜️⬛",
      "⬛⬜️⬜️⬜️⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛",
      "⬛🟩🟩🟩⬜️⬜️🟩🟩🟩⬜️⬜️⬜️⬜️⬛",
      "⬛🟩🟩🟩⬜️⬜️🟩🟩🟩⬜️🟩🟩🟩⬛",
      "⬛🟩🟩🟩⬜️⬜️🟩🟩🟩⬜️🟩🟩🟩⬛",
      "⬛⬜️⬜️⬜️⬜️⬛⬜️⬜️⬜️⬜️🟩🟩🟩⬛",
      "⬛⬛⬛⬛⬛⬛⬜️⬜️⬜️⬛⬛⬛⬛⬛",
      "⬛⬜️⬜️⬜️⬜️🟪🟪🟪⬜️⬜️⬜️⬜️⬜️⬛",
      "⬛⬜️🟥⬜️⬜️🟪🟪🟪⬜️⬜️⬜️⬜️⬜️⬛",
      "⬛⬜️⬜️⬜️⬜️🟪🟪🟪⬜️⬜️⬜️⬜️⬜️⬛",
      "⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛"
    ],
    "emoji_objects": {
      "⬛": {
        "type": "wall",
        "color": "grey",
        "can_pickup": false,
        "can_overlap": false
      },
      "⬜️": {
        "type": "empty",
        "can_pickup": false,
        "can_overlap": true
      },
      "🟦": {
        "type": "wall",
        "color": "blue",
        "can_pickup": false,
        "can_overlap": false
      },
      "🟥": {
        "type": "wall",
        "color": "red",
        "can_pickup": false,
        "can_overlap": false
      },
      "🟪": {
        "type": "floor",
        "color": "purple"
      },
      "🟩": {
        "type": "emoji",
        "emoji_name": "brick",
        "color": "green",
        "can_pickup": false,
        "can_overlap": true,
        "use_emoji_color": true
      }
    },
    "robot_config": {
      "use_robot_emoji": true,
      "robot_emoji_color": "red",
      "use_robot_emoji_color": true
    },
    "start_pos": [1, 1],
    "goal_pos": [12, 1]
  }
}
```

### Python 코드

```python
# Actual path: lib.map_manager.emoji_map_loader
from lib import load_emoji_map_from_json

# 맵 로드
wrapper = load_emoji_map_from_json("config/example_map.json")

# 환경 초기화
obs, info = wrapper.reset()

# 절대 좌표 이동 사용 (기본적으로 활성화됨)
obs, reward, terminated, truncated, info = wrapper.step("up")  # 위로 이동
obs, reward, terminated, truncated, info = wrapper.step("right")  # 오른쪽으로 이동
```

## 주요 특징

1. **절대 좌표 이동 자동 활성화**: JSON에서 로드한 맵은 자동으로 `use_absolute_movement=True`로 설정됩니다.

2. **이모지 맵 시각화**: JSON 파일의 `emoji_render`를 보면 맵의 전체 구조를 한눈에 파악할 수 있습니다.

3. **유연한 형식 지원**: 텍스트, 문자열 배열, 2D 배열 세 가지 형식을 모두 지원합니다.

4. **로봇 위치 마커**: `🟥` 이모지를 사용하면 해당 위치가 자동으로 시작 위치로 설정됩니다.

## 주의사항

1. **이모지 이름 확인**: `emoji_name`은 반드시 `EMOJI_MAP`에 등록된 이름이어야 합니다. 등록되지 않은 이름을 사용하면 오류가 발생할 수 있습니다.

2. **좌표 시스템**: JSON의 좌표는 `[x, y]` 형식이며, `emoji_render` 배열의 인덱스와 일치합니다.

3. **외벽 자동 생성**: 외벽은 자동으로 생성되므로 `emoji_render`의 가장자리는 항상 벽(`⬛`)이어야 합니다.

4. **절대 좌표 이동**: JSON에서 로드한 환경은 항상 절대 좌표 이동 모드로 동작합니다. 상대 이동이 필요한 경우 `use_absolute_movement=False`로 설정할 수 있습니다.

## 참고 자료

- [이모지 사용 가이드](./EMOJI_USAGE_GUIDE.md)
- [커스텀 환경 API](./custom-environment-api.md)
- [Wrapper API](./wrapper-api.md)

