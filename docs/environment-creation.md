# MiniGrid 환경 생성 가이드

이 문서는 MiniGrid 환경을 생성하는 방법을 설명합니다.

## 공식 튜토리얼 기반 방법

[공식 튜토리얼](https://minigrid.farama.org/content/create_env_tutorial/)을 따라 환경을 생성하는 것이 권장됩니다.

## 기본 구조

### 1. 클래스 정의

```python
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace
from minigrid.core.grid import Grid

class CustomRoomEnv(MiniGridEnv):
    def __init__(self, size=10, **kwargs):
        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=4 * size * size,
            **kwargs
        )
    
    @staticmethod
    def _gen_mission():
        return "explore"
```

### 2. 그리드 생성

`_gen_grid(width, height)` 메서드를 오버라이드하여 환경을 생성합니다.

## 핵심 메서드

### 그리드 생성

```python
self.grid = Grid(width, height)
```

### 외벽 생성

```python
self.grid.wall_rect(0, 0, width, height)
```

### 벽 배치

```python
# 단일 벽
self.grid.set(x, y, Wall())

# 연속된 벽
for i in range(height):
    self.grid.set(5, i, Wall())
```

### 객체 배치 (권장)

```python
# Goal 배치
self.put_obj(Goal(), x, y)

# 다른 객체 배치
self.put_obj(Key('yellow'), x, y)
self.put_obj(Door('red', is_locked=True), x, y)
```

**참고**: `self.put_obj()`는 객체를 안전하게 배치하는 헬퍼 메서드입니다. `self.grid.set()`보다 권장됩니다.

### 에이전트 배치

```python
# 명시적 위치 지정
self.agent_pos = np.array([x, y])
self.agent_dir = 0  # 0=오른쪽, 1=아래, 2=왼쪽, 3=위

# 자동 배치
self.place_agent()
```

### Mission 설정

```python
self.mission = self._gen_mission()
```

## 완전한 예제

```python
def _gen_grid(self, width, height):
    # 그리드 생성
    self.grid = Grid(width, height)
    
    # 외벽 생성
    self.grid.wall_rect(0, 0, width, height)
    
    # 내부 벽 생성
    for i in range(height):
        self.grid.set(5, i, Wall())
    
    # 문과 열쇠 배치
    self.put_obj(Door('red', is_locked=True), 5, 6)
    self.put_obj(Key('red'), 3, 6)
    
    # Goal 배치
    self.put_obj(Goal(), width - 2, height - 2)
    
    # 에이전트 배치
    self.agent_pos = np.array([1, 1])
    self.agent_dir = 0
    
    # Mission 설정
    self.mission = self._gen_mission()
```

## 참고

- [CustomRoomEnv 구현](../custom_environment.py) 참고
- [베스트 프랙티스](./best-practices.md) 참고

