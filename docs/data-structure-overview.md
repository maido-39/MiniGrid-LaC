# 데이터 구조 개요: Episode, Grounding, Trust

이 문서는 실험에서 생성되는 주요 데이터 구조(Episode, Grounding, Trust)의 위치와 구조를 간단히 설명합니다.

## Episode (에피소드)

### 위치
```
logs/{experiment_name}/
└── episode_{id}_{timestamp}_{script}/
    ├── episode_{id}.json          # Episode 전체 데이터
    ├── grounding_episode_{id}.json # Episode별 Grounding
    ├── grounding_episode_{id}.txt  # Episode별 Grounding (TXT)
    └── images/
        ├── initial_state.png
        └── step_XXXX.png
```

### 구조
**`episode_{id}.json`**:
```json
{
  "episode_id": 1,
  "start_time": "2026-01-23T17:05:52.123456",
  "end_time": "2026-01-23T17:10:15.654321",
  "total_steps": 15,
  "termination_reason": "done" | "max_steps" | "user_command",
  "initial_state_image_path": "images/initial_state.png",
  "steps": [
    {
      "step_id": 1,
      "instruction": "Go to the kitchen",
      "status": "Success" | "Failure" | "WiP",
      "feedback": {
        "user_preference": "...",
        "spatial": "...",
        "procedural": "...",
        "general": "..."
      },
      "action": {"index": 0, "name": "move up"},
      "state": {"agent_pos": [5, 3], "agent_dir": 0},
      "image_path": "images/step_0001.png",
      "timestamp": "..."
    }
  ],
  "reflexion": {
    "trajectory_summary": "...",
    "error_diagnosis": "...",
    "correction_plan": "..."
  },
  "final_grounding": {...}
}
```

### 설명
- **하나의 실험 실행 단위**: 사용자가 지정한 episode_id로 구분
- **Step별 상세 기록**: 각 step의 instruction, status, feedback, action, state 저장
- **Reflexion 포함**: 에피소드 종료 후 VLM이 생성한 분석 결과
- **이미지 저장**: 초기 상태 및 각 step의 환경 이미지

---

## Grounding (그라운딩)

### 위치
```
logs/{experiment_name}/
├── episode_{id}_{timestamp}_{script}/
│   ├── grounding_episode_{id}.json  # Episode별 Grounding
│   └── grounding_episode_{id}.txt  # Episode별 Grounding (TXT)
└── grounding/
    ├── grounding_latest.json        # 최신 Grounding (전역)
    └── grounding_latest.txt         # 최신 Grounding TXT (전역)
```

### 구조
**`grounding_episode_{id}.json`**:
```json
{
  "expr_info": {
    "episode_id": 1
  },
  "grounding_per_step": [
    {
      "step_id": 1,
      "instruction": "Go to the kitchen",
      "status": "Success" | "Failure" | "WiP",
      "feedback": {
        "user_preference": "...",
        "spatial": "...",
        "procedural": "...",
        "general": "..."
      }
    }
  ],
  "stacked_grounding": {
    "user_preference": [
      "[ Step1 - Success ] : User prefers spicy food",
      "[ Step3 - Failure ] : Add more pepper next time"
    ],
    "spatial": [
      "[ Step2 - Success ] : Kitchen is at (5, 3)"
    ],
    "procedural": [],
    "general": []
  },
  "final_grounding": {
    "user_preference_grounding": {
      "content": "The user prefers food containing Pepper and Tomato."
    },
    "spatial_grounding": {
      "content": "Kitchen is located at grid coordinates (5, 3)."
    },
    "procedural_grounding": {
      "content": "To make pasta: First go to Kitchen, then Storage..."
    },
    "general_grounding": {
      "content": "You mustn't hit the wall."
    }
  }
}
```

### 설명
- **4가지 타입**: User Preference, Spatial, Procedural, General
- **Step별 누적**: 각 step의 feedback이 `stacked_grounding`에 누적
- **VLM 생성**: 에피소드 종료 시 **현재 에피소드 피드백만** 사용하여 VLM이 `final_grounding` 생성
- **전역 최신 버전**: `grounding/grounding_latest.json`에 최신 Grounding 저장 (다음 에피소드에서 사용)

### 여러 파일 병합 ⭐ **신규**

`GROUNDING_FILE_PATH`에 여러 파일을 지정하면 자동으로 병합됩니다:

**JSON 파일 병합**:
- 여러 JSON 파일의 `stacked_grounding` 카테고리별로 중복 제거하며 병합
- 여러 JSON 파일의 `final_grounding`에서 `generation_timestamp` 제외하고 병합
- Markdown 형식으로 렌더링 (H3→H4 헤더 레벨 조정)

**TXT 파일 병합**:
- 텍스트 내용을 그대로 병합

**혼합 파일**:
- JSON과 TXT 파일이 함께 있으면 각각 처리 후 병합

**설정**: `GROUNDING_MERGE_FORMAT = "txt"` (기본값: "txt")

---

## Trust (신뢰도)

### 위치
```
logs/{experiment_name}/
└── experiment_log.csv    # CSV 파일의 "trust_T" 컬럼
    └── experiment_log.json  # JSON 파일의 각 step 객체의 "trust_T" 필드
```

### 구조
**CSV (`experiment_log.csv`)**:
```csv
step,timestamp,...,entropy_H_X,entropy_H_X_given_S,entropy_H_X_given_LS,trust_T
1,2026-01-23T17:05:52,...,2.45,1.89,1.23,0.46
2,2026-01-23T17:05:55,...,2.38,1.92,1.18,0.38
...
```

**JSON (`experiment_log.json`)**:
```json
[
  {
    "step": 1,
    "timestamp": "2026-01-23T17:05:52",
    "entropy_H_X": 2.45,
    "entropy_H_X_given_S": 1.89,
    "entropy_H_X_given_LS": 1.23,
    "trust_T": 0.46
  }
]
```

### 계산 공식
```
T = (H(X) - H(X|S)) / (H(X) - H(X|L,S))
```

- **H(X)**: Language Instruction 없음, Grounding 없음
- **H(X|S)**: Language Instruction 없음, Grounding 있음
- **H(X|L,S)**: Language Instruction 있음, Grounding 있음

### 설명
- **Step별 계산**: 각 step마다 3개의 Entropy를 계산하여 Trust 값 도출
- **범위**: 0 ~ 1 (일반적으로), 음수나 NaN 가능
- **의미**: Grounding이 Language Instruction 대비 얼마나 효과적인지 측정
- **저장 위치**: `experiment_log.csv`와 `experiment_log.json`에만 저장 (episode JSON에는 없음)

---

## 데이터 접근 방법

### Episode 데이터 읽기
```python
import json
from pathlib import Path

episode_file = Path("logs/.../episode_1_.../episode_1.json")
with open(episode_file, 'r', encoding='utf-8') as f:
    episode_data = json.load(f)
    
print(f"Total steps: {episode_data['total_steps']}")
print(f"Termination: {episode_data['termination_reason']}")
```

### Grounding 데이터 읽기
```python
# Episode별 Grounding
grounding_file = Path("logs/.../episode_1_.../grounding_episode_1.json")
with open(grounding_file, 'r', encoding='utf-8') as f:
    grounding_data = json.load(f)

# 최신 Grounding (전역)
latest_grounding = Path("logs/.../grounding/grounding_latest.json")
with open(latest_grounding, 'r', encoding='utf-8') as f:
    latest_data = json.load(f)
```

### Trust 데이터 읽기
```python
import pandas as pd

# CSV에서 읽기
df = pd.read_csv("logs/.../experiment_log.csv")
trust_values = df['trust_T'].dropna()

# 특정 step의 Trust 값
step_5_trust = df[df['step'] == 5]['trust_T'].values[0]
```

---

## 요약

| 데이터 | 위치 | 형식 | 설명 |
|--------|------|------|------|
| **Episode** | `episode_{id}_{timestamp}_{script}/episode_{id}.json` | JSON | 전체 에피소드 데이터 (steps, reflexion, final_grounding) |
| **Grounding** | `episode_{id}_.../grounding_episode_{id}.json`<br>`grounding/grounding_latest.json` | JSON, TXT | Step별 feedback 누적 및 VLM 생성 final_grounding |
| **Trust** | `experiment_log.csv`<br>`experiment_log.json` | CSV, JSON | Step별 Trust 값 (3개 Entropy로 계산) |
