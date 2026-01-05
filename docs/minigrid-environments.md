# MiniGrid 예제 환경 목록

MiniGrid 라이브러리에서 제공하는 내장 환경 목록입니다. `gymnasium.make()`를 사용하여 환경을 생성할 수 있습니다.

## 사용 방법

```python
from minigrid import register_minigrid_envs
import gymnasium as gym

register_minigrid_envs()
env = gym.make("MiniGrid-FourRooms-v0", render_mode='rgb_array')
```

## 환경 카테고리

### 1. 빈 환경 (Empty)

가장 기본적인 빈 그리드 환경입니다. 벽으로 둘러싸인 빈 공간만 있습니다.

| 환경 이름 | 크기 | 설명 |
|---------|------|------|
| `MiniGrid-Empty-5x5-v0` | 5x5 | 작은 빈 환경 |
| `MiniGrid-Empty-6x6-v0` | 6x6 | 중간 빈 환경 |
| `MiniGrid-Empty-8x8-v0` | 8x8 | 기본 빈 환경 |
| `MiniGrid-Empty-16x16-v0` | 16x16 | 큰 빈 환경 |
| `MiniGrid-Empty-Random-5x5-v0` | 5x5 | 랜덤 요소 포함 |
| `MiniGrid-Empty-Random-6x6-v0` | 6x6 | 랜덤 요소 포함 |

**특징**: 
- 벽으로 둘러싸인 빈 공간
- Goal 위치는 랜덤 또는 고정
- 기본적인 탐색 및 내비게이션 테스트에 적합

---

### 2. 문과 열쇠 (DoorKey)

열쇠를 찾아 문을 열고 Goal에 도달하는 환경입니다.

| 환경 이름 | 크기 | 설명 |
|---------|------|------|
| `MiniGrid-DoorKey-5x5-v0` | 5x5 | 작은 문-열쇠 환경 |
| `MiniGrid-DoorKey-6x6-v0` | 6x6 | 중간 문-열쇠 환경 |
| `MiniGrid-DoorKey-8x8-v0` | 8x8 | 기본 문-열쇠 환경 |
| `MiniGrid-DoorKey-16x16-v0` | 16x16 | 큰 문-열쇠 환경 |

**특징**:
- 잠긴 문을 열기 위해 열쇠를 찾아야 함
- 순차적 작업 수행 능력 테스트
- 메모리 및 계획 능력 필요

---

### 3. 복도와 열쇠 (KeyCorridor)

여러 방이 연결된 복도 구조에서 열쇠를 찾아 문을 여는 환경입니다.

| 환경 이름 | 크기 | 설명 |
|---------|------|------|
| `MiniGrid-KeyCorridorS3R1-v0` | - | 3개 방, 1개 복도 |
| `MiniGrid-KeyCorridorS3R2-v0` | - | 3개 방, 2개 복도 |
| `MiniGrid-KeyCorridorS3R3-v0` | - | 3개 방, 3개 복도 |
| `MiniGrid-KeyCorridorS4R3-v0` | - | 4개 방, 3개 복도 |
| `MiniGrid-KeyCorridorS5R3-v0` | - | 5개 방, 3개 복도 |
| `MiniGrid-KeyCorridorS6R3-v0` | - | 6개 방, 3개 복도 |

**특징**:
- 여러 방을 탐색해야 함
- 복잡한 경로 계획 필요
- 장거리 탐색 및 메모리 테스트

---

### 4. 여러 방 (MultiRoom)

여러 개의 방이 연결된 환경입니다.

| 환경 이름 | 크기 | 설명 |
|---------|------|------|
| `MiniGrid-MultiRoom-N2-S4-v0` | - | 2개 방, 크기 4 |
| `MiniGrid-MultiRoom-N4-S5-v0` | - | 4개 방, 크기 5 |
| `MiniGrid-MultiRoom-N6-v0` | - | 6개 방 |

**특징**:
- 복잡한 공간 구조
- 여러 방을 통과해야 Goal 도달
- 경로 계획 및 탐색 능력 테스트

---

### 5. 네 개의 방 (FourRooms)

고전적인 4개의 방이 연결된 환경입니다.

| 환경 이름 | 크기 | 설명 |
|---------|------|------|
| `MiniGrid-FourRooms-v0` | - | 4개의 방이 연결된 환경 |

**특징**:
- 간단하면서도 복잡한 구조
- 내비게이션 및 경로 계획 테스트에 적합
- 벤치마크 환경으로 자주 사용

---

### 6. 용암 건너기 (LavaCrossing)

용암을 피해 건너가야 하는 환경입니다.

| 환경 이름 | 크기 | 설명 |
|---------|------|------|
| `MiniGrid-LavaCrossingS9N1-v0` | - | 크기 9, 1개 용암 |
| `MiniGrid-LavaCrossingS9N2-v0` | - | 크기 9, 2개 용암 |
| `MiniGrid-LavaCrossingS9N3-v0` | - | 크기 9, 3개 용암 |
| `MiniGrid-LavaCrossingS11N5-v0` | - | 크기 11, 5개 용암 |

**특징**:
- 용암 타일을 밟으면 실패
- 안전한 경로 찾기 능력 테스트
- 위험 회피 학습에 적합

---

### 7. 용암 간격 (LavaGap)

용암으로 가로막힌 간격을 건너뛰어야 하는 환경입니다.

| 환경 이름 | 크기 | 설명 |
|---------|------|------|
| `MiniGrid-LavaGapS5-v0` | - | 크기 5 |
| `MiniGrid-LavaGapS6-v0` | - | 크기 6 |
| `MiniGrid-LavaGapS7-v0` | - | 크기 7 |

**특징**:
- 용암 간격을 건너뛰어야 함
- 점프 또는 우회 경로 필요
- 공간 추론 능력 테스트

---

### 8. 메모리 (Memory)

에이전트가 이전에 본 정보를 기억해야 하는 환경입니다.

| 환경 이름 | 크기 | 설명 |
|---------|------|------|
| `MiniGrid-MemoryS7-v0` | - | 크기 7 |
| `MiniGrid-MemoryS9-v0` | - | 크기 9 |
| `MiniGrid-MemoryS11-v0` | - | 크기 11 |
| `MiniGrid-MemoryS13-v0` | - | 크기 13 |
| `MiniGrid-MemoryS13Random-v0` | - | 크기 13, 랜덤 |
| `MiniGrid-MemoryS17Random-v0` | - | 크기 17, 랜덤 |

**특징**:
- 장기 메모리 능력 테스트
- 이전 관찰 정보를 기억해야 함
- 순차적 의사결정 필요

---

### 9. 동적 장애물 (Dynamic Obstacles)

움직이는 장애물이 있는 환경입니다.

| 환경 이름 | 크기 | 설명 |
|---------|------|------|
| `MiniGrid-Dynamic-Obstacles-5x5-v0` | 5x5 | 작은 동적 장애물 환경 |
| `MiniGrid-Dynamic-Obstacles-6x6-v0` | 6x6 | 중간 동적 장애물 환경 |
| `MiniGrid-Dynamic-Obstacles-8x8-v0` | 8x8 | 기본 동적 장애물 환경 |
| `MiniGrid-Dynamic-Obstacles-16x16-v0` | 16x16 | 큰 동적 장애물 환경 |
| `MiniGrid-Dynamic-Obstacles-Random-5x5-v0` | 5x5 | 랜덤 동적 장애물 |
| `MiniGrid-Dynamic-Obstacles-Random-6x6-v0` | 6x6 | 랜덤 동적 장애물 |

**특징**:
- 움직이는 장애물 회피 필요
- 동적 환경에서의 경로 계획
- 실시간 의사결정 능력 테스트

---

### 10. 미로 (ObstructedMaze)

복잡한 미로 구조의 환경입니다.

| 환경 이름 | 크기 | 설명 |
|---------|------|------|
| `MiniGrid-ObstructedMaze-1Dl-v0` | - | 1개 문, 잠금 |
| `MiniGrid-ObstructedMaze-1Dlh-v0` | - | 1개 문, 잠금, 숨김 |
| `MiniGrid-ObstructedMaze-1Dlhb-v0` | - | 1개 문, 잠금, 숨김, 블록 |
| `MiniGrid-ObstructedMaze-1Q-v0` | - | 1개 사분면 |
| `MiniGrid-ObstructedMaze-1Q-v1` | - | 1개 사분면 (v1) |
| `MiniGrid-ObstructedMaze-2Dl-v0` | - | 2개 문, 잠금 |
| `MiniGrid-ObstructedMaze-2Dlh-v0` | - | 2개 문, 잠금, 숨김 |
| `MiniGrid-ObstructedMaze-2Dlhb-v0` | - | 2개 문, 잠금, 숨김, 블록 |
| `MiniGrid-ObstructedMaze-2Dlhb-v1` | - | 2개 문, 잠금, 숨김, 블록 (v1) |
| `MiniGrid-ObstructedMaze-2Q-v0` | - | 2개 사분면 |
| `MiniGrid-ObstructedMaze-2Q-v1` | - | 2개 사분면 (v1) |
| `MiniGrid-ObstructedMaze-Full-v0` | - | 전체 미로 |
| `MiniGrid-ObstructedMaze-Full-v1` | - | 전체 미로 (v1) |

**특징**:
- 매우 복잡한 미로 구조
- 여러 열쇠와 문
- 장거리 탐색 및 계획 능력 테스트

---

### 11. 놀이터 (Playground)

다양한 객체와 구조가 있는 복합 환경입니다.

| 환경 이름 | 크기 | 설명 |
|---------|------|------|
| `MiniGrid-Playground-v0` | - | 다양한 객체와 구조 |

**특징**:
- 여러 종류의 객체 포함
- 복잡한 상호작용 가능
- 종합적인 능력 테스트

---

### 12. 가져오기 (Fetch)

객체를 가져와야 하는 환경입니다.

| 환경 이름 | 크기 | 설명 |
|---------|------|------|
| `MiniGrid-Fetch-5x5-N2-v0` | 5x5 | 2개 객체 |
| `MiniGrid-Fetch-6x6-N2-v0` | 6x6 | 2개 객체 |
| `MiniGrid-Fetch-8x8-N3-v0` | 8x8 | 3개 객체 |

**특징**:
- 여러 객체를 찾아야 함
- 객체 조작 능력 필요
- 순차적 작업 수행

---

### 13. 이동 (GoTo)

특정 위치나 객체로 이동해야 하는 환경입니다.

| 환경 이름 | 크기 | 설명 |
|---------|------|------|
| `MiniGrid-GoToDoor-5x5-v0` | 5x5 | 문으로 이동 |
| `MiniGrid-GoToDoor-6x6-v0` | 6x6 | 문으로 이동 |
| `MiniGrid-GoToDoor-8x8-v0` | 8x8 | 문으로 이동 |
| `MiniGrid-GoToObject-6x6-N2-v0` | 6x6 | 객체로 이동 (2개) |
| `MiniGrid-GoToObject-8x8-N2-v0` | 8x8 | 객체로 이동 (2개) |

**특징**:
- 특정 목표물로 이동
- 객체 인식 및 내비게이션
- 목표 지향 행동 테스트

---

### 14. 배치 (PutNear)

객체를 특정 위치 근처에 배치해야 하는 환경입니다.

| 환경 이름 | 크기 | 설명 |
|---------|------|------|
| `MiniGrid-PutNear-6x6-N2-v0` | 6x6 | 2개 객체 배치 |
| `MiniGrid-PutNear-8x8-N3-v0` | 8x8 | 3개 객체 배치 |

**특징**:
- 객체를 들고 이동
- 특정 위치에 배치
- 조작 및 배치 능력 테스트

---

### 15. 잠금 해제 (Unlock)

문을 잠금 해제해야 하는 환경입니다.

| 환경 이름 | 크기 | 설명 |
|---------|------|------|
| `MiniGrid-Unlock-v0` | - | 기본 잠금 해제 |
| `MiniGrid-UnlockPickup-v0` | - | 잠금 해제 및 픽업 |
| `MiniGrid-BlockedUnlockPickup-v0` | - | 차단된 잠금 해제 및 픽업 |
| `MiniGrid-LockedRoom-v0` | - | 잠긴 방 |

**특징**:
- 열쇠를 찾아 문 열기
- 순차적 작업 수행
- 메모리 및 계획 필요

---

### 16. 문 (Doors)

다양한 색상의 문이 있는 환경입니다.

| 환경 이름 | 크기 | 설명 |
|---------|------|------|
| `MiniGrid-RedBlueDoors-6x6-v0` | 6x6 | 빨강/파랑 문 |
| `MiniGrid-RedBlueDoors-8x8-v0` | 8x8 | 빨강/파랑 문 |

**특징**:
- 색상 구분 문
- 색상 인식 능력 테스트
- 조건부 행동 필요

---

### 17. 건너기 (Crossing)

장애물을 건너야 하는 환경입니다.

| 환경 이름 | 크기 | 설명 |
|---------|------|------|
| `MiniGrid-SimpleCrossingS9N1-v0` | - | 크기 9, 1개 장애물 |
| `MiniGrid-SimpleCrossingS9N2-v0` | - | 크기 9, 2개 장애물 |
| `MiniGrid-SimpleCrossingS9N3-v0` | - | 크기 9, 3개 장애물 |
| `MiniGrid-SimpleCrossingS11N5-v0` | - | 크기 11, 5개 장애물 |

**특징**:
- 장애물 회피 필요
- 안전한 경로 찾기
- 경로 계획 능력 테스트

---

### 18. WFC (Wave Function Collapse)

절차적 생성 환경입니다.

| 환경 이름 | 크기 | 설명 |
|---------|------|------|
| `MiniGrid-WFC-MazeSimple-v0` | - | 간단한 미로 |
| `MiniGrid-WFC-DungeonMazeScaled-v0` | - | 던전 미로 |
| `MiniGrid-WFC-ObstaclesAngular-v0` | - | 각진 장애물 |
| `MiniGrid-WFC-ObstaclesBlackdots-v0` | - | 검은 점 장애물 |
| `MiniGrid-WFC-ObstaclesHogs3-v0` | - | Hogs3 장애물 |
| `MiniGrid-WFC-RoomsFabric-v0` | - | 방 구조 |

**특징**:
- 절차적 생성 환경
- 매번 다른 구조
- 일반화 능력 테스트

---

### 19. 기타

| 환경 이름 | 크기 | 설명 |
|---------|------|------|
| `MiniGrid-DistShift1-v0` | - | 거리 이동 1 |
| `MiniGrid-DistShift2-v0` | - | 거리 이동 2 |

---

## 환경 선택 가이드

### 초보자용
- `MiniGrid-Empty-8x8-v0`: 가장 간단한 환경
- `MiniGrid-FourRooms-v0`: 구조가 명확한 환경

### 중급자용
- `MiniGrid-DoorKey-8x8-v0`: 순차적 작업 학습
- `MiniGrid-KeyCorridorS6R3-v0`: 복잡한 탐색

### 고급자용
- `MiniGrid-ObstructedMaze-Full-v0`: 매우 복잡한 미로
- `MiniGrid-MemoryS17Random-v0`: 장기 메모리 테스트

## 참고 자료

- [MiniGrid 공식 문서](https://minigrid.farama.org/)
- [MiniGrid GitHub](https://github.com/Farama-Foundation/MiniGrid)

