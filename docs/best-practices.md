# MiniGrid 환경 생성 베스트 프랙티스

이 문서는 [MiniGrid 공식 튜토리얼](https://minigrid.farama.org/content/create_env_tutorial/)을 기반으로 한 권장사항을 정리합니다.

## 1. 객체 배치 방법

### ❌ 비권장: 직접 grid.set() 사용

```python
self.grid.set(x, y, Goal())
self.grid.set(x, y, Key('yellow'))
```

### ✅ 권장: put_obj() 사용

```python
self.put_obj(Goal(), x, y)
self.put_obj(Key('yellow'), x, y)
```

**이유**: `put_obj()`는 객체를 안전하게 배치하는 헬퍼 메서드로, 충돌 검사 등을 자동으로 수행합니다.

## 2. Mission 설정

`_gen_grid()` 메서드의 마지막에 반드시 Mission을 설정해야 합니다.

```python
def _gen_grid(self, width, height):
    # ... 그리드 생성 코드 ...
    
    # Mission 설정 (필수)
    self.mission = self._gen_mission()
```

**이유**: Mission은 환경의 목표를 나타내며, 일부 래퍼나 평가 도구에서 사용됩니다.

## 3. 에이전트 배치

명시적 위치가 있으면 직접 설정하고, 없으면 `place_agent()`를 사용합니다.

```python
if self.agent_start_pos is not None:
    self.agent_pos = self.agent_start_pos
    self.agent_dir = self.agent_start_dir
else:
    self.place_agent()
```

**이유**: `place_agent()`는 빈 공간을 자동으로 찾아 에이전트를 배치합니다.

## 4. 그리드 생성 순서

권장되는 순서:

1. 그리드 생성: `self.grid = Grid(width, height)`
2. 외벽 생성: `self.grid.wall_rect(0, 0, width, height)`
3. 내부 벽 배치: `self.grid.set(x, y, Wall())`
4. 객체 배치: `self.put_obj(obj, x, y)`
5. Goal 배치: `self.put_obj(Goal(), x, y)`
6. 에이전트 배치: `self.agent_pos = ...` 또는 `self.place_agent()`
7. Mission 설정: `self.mission = self._gen_mission()`

## 5. MissionSpace 정의

정적 메서드로 Mission을 생성하는 것이 권장됩니다.

```python
@staticmethod
def _gen_mission():
    return "explore"

# __init__에서
mission_space = MissionSpace(mission_func=self._gen_mission)
```

## 6. max_steps 설정

환경 크기에 비례하여 설정하는 것이 좋습니다.

```python
max_steps = 4 * size * size
```

또는 명시적으로 지정:

```python
max_steps = 256  # 고정값
```

## 7. Door 사용 시 주의사항

Door는 색상과 잠금 상태를 명시적으로 설정해야 합니다.

```python
# 잠긴 문
door = Door('red', is_locked=True, is_open=False)

# 열린 문
door = Door('red', is_locked=False, is_open=True)

# 열쇠와 함께 사용
key = Key('red')  # 문과 같은 색상
door = Door('red', is_locked=True)
```

## 8. 벽 배치 최적화

연속된 벽은 반복문으로 효율적으로 배치할 수 있습니다.

```python
# 세로 벽
for i in range(height):
    self.grid.set(x, i, Wall())

# 가로 벽
for i in range(width):
    self.grid.set(i, y, Wall())
```

## 9. 객체 배치 검증

배치 전에 좌표가 유효한지 확인하는 것이 좋습니다.

```python
if 0 <= x < width and 0 <= y < height:
    self.put_obj(obj, x, y)
```

## 10. 코드 구조

환경 클래스는 다음과 같은 구조를 권장합니다:

```python
class CustomEnv(MiniGridEnv):
    def __init__(self, size=10, **kwargs):
        # 초기화
        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(...)
    
    @staticmethod
    def _gen_mission():
        return "mission text"
    
    def _gen_grid(self, width, height):
        # 그리드 생성
        # 객체 배치
        # 에이전트 배치
        # Mission 설정
```

## 공식 튜토리얼 준수 체크리스트

- [ ] `MiniGridEnv`를 상속
- [ ] `MissionSpace`를 `mission_func`로 정의
- [ ] `_gen_grid(width, height)` 메서드 오버라이드
- [ ] `self.grid = Grid(width, height)` 사용
- [ ] `self.grid.wall_rect(0, 0, width, height)` 외벽 생성
- [ ] `self.put_obj()`로 객체 배치 (grid.set 대신)
- [ ] `self.mission = self._gen_mission()` 설정
- [ ] 에이전트 배치 로직 구현

## 참고 자료

- [MiniGrid 공식 튜토리얼](https://minigrid.farama.org/content/create_env_tutorial/)
- [환경 생성 가이드](./environment-creation.md)
- [CustomRoomEnv API](./custom-environment-api.md)

