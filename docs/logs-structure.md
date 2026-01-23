# Logs 폴더 구조 및 파일 생성 방식

이 문서는 실험 실행 시 `logs/` 폴더에 생성되는 파일들의 구조와 생성 시점에 대해 설명합니다.

## 전체 구조

```
logs/
├── scenario2_absolute_{map_name}_{timestamp}/     # 메인 실험 로그 디렉토리
│   ├── episode_{episode_id}_{timestamp}_{script}/ # 각 에피소드별 폴더
│   │   ├── images/                                 # 이미지 폴더
│   │   │   ├── initial_state.png                   # 초기 상태 이미지
│   │   │   ├── step_0001.png                      # Step 1 이미지
│   │   │   ├── step_0002.png                      # Step 2 이미지
│   │   │   └── ...                                # 각 step의 이미지
│   │   ├── episode_{episode_id}.json              # 에피소드 JSON 로그
│   │   ├── grounding_episode_{episode_id}.json    # Grounding JSON 파일
│   │   └── grounding_episode_{episode_id}.txt     # Grounding TXT 파일
│   ├── grounding/                                  # 전역 Grounding 폴더
│   │   ├── grounding_latest.json                   # 최신 Grounding (JSON)
│   │   └── grounding_latest.txt                   # 최신 Grounding (TXT)
│   ├── experiment_log.json                        # 전체 실험 JSON 로그 (누적)
│   └── experiment_log.csv                         # 실험 데이터 CSV (누적)
```

## 디렉토리 및 파일 설명

### 1. 메인 실험 로그 디렉토리

**경로**: `logs/scenario2_absolute_{map_name}_{timestamp}/`

**생성 시점**: 실험 시작 시 (`ScenarioExperiment.__init__()`)

**설명**:
- `{map_name}`: 사용된 맵 파일 이름 (확장자 제외)
- `{timestamp}`: 실험 시작 시간 (`YYYYMMDD_HHMMSS` 형식)

**예시**: `logs/scenario2_absolute_example_map_20260123_143058/`

---

### 2. 에피소드별 폴더

**경로**: `logs/.../episode_{episode_id}_{timestamp}_{script_name}/`

**생성 시점**: 에피소드 시작 시 (`EpisodeManager._create_episode_directory()`)

**설명**:
- `{episode_id}`: 에피소드 번호 (사용자 입력)
- `{timestamp}`: 에피소드 시작 시간 (`YYYYMMDD_HHMMSS` 형식)
- `{script_name}`: 실행한 스크립트 이름 (확장자 제외)

**예시**: `episode_1_20260123_143104_scenario2_test_absolutemove_modularized/`

#### 2.1. images/ 폴더

**경로**: `episode_{id}_{timestamp}_{script}/images/`

**생성 시점**: 에피소드 폴더 생성 시 자동 생성

**파일들**:
- **`initial_state.png`**: 에피소드 시작 시 초기 상태 이미지
  - 생성 시점: 에피소드 첫 step (`EpisodeManager.save_initial_state_image()`)
  
- **`step_XXXX.png`**: 각 step의 환경 이미지
  - 생성 시점: 각 step 실행 후 (`ScenarioExperiment._log_step()`)
  - 파일명 형식: `step_0001.png`, `step_0002.png`, ... (4자리 숫자, 0 패딩)

#### 2.2. episode_{episode_id}.json

**경로**: `episode_{id}_{timestamp}_{script}/episode_{episode_id}.json`

**생성 시점**: 에피소드 종료 시 (`EpisodeManager.save()`)

**내용 구조**:
```json
{
  "episode_id": 1,
  "start_time": "2026-01-23T14:31:04.123456",
  "end_time": "2026-01-23T14:35:20.654321",
  "total_steps": 15,
  "termination_reason": "done" | "max_steps" | "user_command",
  "initial_state_image_path": "images/initial_state.png",
  "steps": [
    {
      "step_id": 1,
      "instruction": "...",
      "status": "Success" | "Failure" | "WiP",
      "feedback": {
        "user_preference": "...",
        "spatial": "...",
        "procedural": "...",
        "general": "..."
      },
      "action": {...},
      "state": {...},
      "image_path": "images/step_0001.png",
      "timestamp": "..."
    },
    ...
  ],
  "step_grounding": {
    "user_preference": [...],
    "spatial": [...],
    "procedural": [...],
    "general": [...]
  },
  "reflexion": {
    "trajectory_summary": "...",
    "error_diagnosis": "...",
    "correction_plan": "..."
  },
  "final_grounding": {
    "user_preference_grounding": {"content": "..."},
    "spatial_grounding": {"content": "..."},
    "procedural_grounding": {"content": "..."},
    "general_grounding": {"content": "..."}
  }
}
```

#### 2.3. grounding_episode_{episode_id}.json

**경로**: `episode_{id}_{timestamp}_{script}/grounding_episode_{episode_id}.json`

**생성 시점**: 
- 초기화: 에피소드 시작 시 (`GroundingFileManager.__init__()`)
- 업데이트: 각 step마다 feedback 추가 시 (`GroundingFileManager.append_step_feedback()`)
- 최종 저장: 에피소드 종료 시 final_grounding 생성 후 (`GroundingFileManager.save_final_grounding()`)

**내용 구조**:
```json
{
  "expr_info": {
    "episode_id": 1
  },
  "grounding_per_step": [
    {
      "step_id": 1,
      "instruction": "...",
      "status": "Success" | "Failure" | "WiP",
      "feedback": {
        "user_preference": "...",
        "spatial": "...",
        "procedural": "...",
        "general": "..."
      }
    },
    ...
  ],
  "stacked_grounding": {
    "user_preference": [
      "[ Step1 - Success ] : ...",
      "[ Step2 - Failure ] : ..."
    ],
    "spatial": [...],
    "procedural": [...],
    "general": [...]
  },
  "final_grounding": {
    "user_preference_grounding": {"content": "..."},
    "spatial_grounding": {"content": "..."},
    "procedural_grounding": {"content": "..."},
    "general_grounding": {"content": "..."}
  }
}
```

#### 2.4. grounding_episode_{episode_id}.txt

**경로**: `episode_{id}_{timestamp}_{script}/grounding_episode_{episode_id}.txt`

**생성 시점**: 
- 초기화: 에피소드 시작 시 (`GroundingFileManager._initialize_txt_file()`)
- 업데이트: 각 step마다 feedback 추가 시 (`GroundingFileManager._append_to_txt_file()`)
- 최종 저장: 에피소드 종료 시 final_grounding 추가 후

**내용 형식**:
```
## Grounding Knowledge (Experience from Past Failures, Successes)

### User Preference Grounding
[ Step1 - Success ] : ...
[ Step2 - Failure ] : ...

### Spatial Grounding
[ Step3 - Success ] : ...

### Procedural Grounding
[ Step5 - Failure ] : ...

### General Grounding Rules
[ Step2 - Failure ] : ...

---

## Final Grounding (Generated by VLM)

### User Preference Grounding
...

### Spatial Grounding
...

### Procedural Grounding
...

### General Grounding Rules
...
```

---

### 3. 전역 Grounding 폴더

**경로**: `logs/.../grounding/`

**생성 시점**: 첫 번째 에피소드 시작 시 (`GroundingFileManager.__init__()`)

**설명**: 모든 에피소드에서 공유하는 최신 Grounding 파일을 저장하는 폴더

#### 3.1. grounding_latest.json

**경로**: `logs/.../grounding/grounding_latest.json`

**생성 시점**: 각 에피소드 종료 시 final_grounding 생성 후 업데이트

**설명**: 가장 최근에 생성된 에피소드의 final_grounding을 저장하는 파일. 다음 에피소드에서 VLM에 전달할 수 있음.

#### 3.2. grounding_latest.txt

**경로**: `logs/.../grounding/grounding_latest.txt`

**생성 시점**: 각 에피소드 종료 시 final_grounding 생성 후 업데이트

**설명**: `grounding_latest.json`의 사람이 읽기 쉬운 TXT 형식 버전

---

### 4. 실험 전체 로그 파일

#### 4.1. experiment_log.json

**경로**: `logs/.../experiment_log.json`

**생성 시점**: 
- 초기화: 실험 시작 시 (`ScenarioExperiment._init_csv_logging()`)
- 업데이트: 각 step마다 (`ScenarioExperiment._log_step()`)

**설명**: 모든 에피소드와 step의 데이터를 누적하여 저장하는 JSON 파일. 매우 큰 파일이 될 수 있음.

**내용**: 각 step의 모든 정보 (state, action, VLM response, entropy, trust 등)를 배열로 저장

#### 4.2. experiment_log.csv

**경로**: `logs/.../experiment_log.csv`

**생성 시점**: 
- 초기화: 실험 시작 시 (`ScenarioExperiment._init_csv_logging()`)
- 업데이트: 각 step마다 (`ScenarioExperiment._log_step()`)

**설명**: 실험 데이터를 CSV 형식으로 저장. Excel 등에서 분석하기 용이함.

**CSV 헤더**:
- `step`, `episode_id`, `timestamp`
- `agent_pos_x`, `agent_pos_y`, `agent_dir`, `heading`
- `action_index`, `action_name`, `reward`, `done`
- `user_prompt`, `vlm_response`, `reasoning`
- `entropy_H_X`, `entropy_H_X_given_S`, `entropy_H_X_given_LS`, `trust_T`
- `carrying_object`, `is_pickup`, `is_drop`
- 기타 메타데이터

**특징**: 
- 모든 필드는 `csv.QUOTE_ALL`로 인용되어 있음 (다중 줄, 쉼표 포함 문자열 처리)
- `None` 또는 `NaN` 값은 빈 문자열로 저장

---

## 파일 생성 순서

### 에피소드 시작 시
1. 에피소드 폴더 생성: `episode_{id}_{timestamp}_{script}/`
2. `images/` 폴더 생성
3. `grounding_episode_{id}.json` 초기화
4. `grounding_episode_{id}.txt` 초기화 (헤더만)

### 각 Step 실행 시
1. 환경 이미지 저장: `images/step_XXXX.png`
2. `grounding_episode_{id}.json`에 step feedback 추가
3. `grounding_episode_{id}.txt`에 step feedback 추가
4. `experiment_log.json`에 step 데이터 추가
5. `experiment_log.csv`에 step 데이터 추가

### 에피소드 종료 시
1. VLM으로 final_grounding 생성 (`_generate_grounding_from_episode()`)
2. `grounding_episode_{id}.json`에 final_grounding 저장
3. `grounding_episode_{id}.txt`에 final_grounding 추가
4. `grounding_latest.json` 업데이트
5. `grounding_latest.txt` 업데이트
6. `episode_{id}.json` 저장
7. Reflexion 생성 및 `episode_{id}.json`에 저장 (선택적)

---

## 파일 크기 및 관리

### 파일 크기 예상치
- **이미지 파일**: 각 약 50-200 KB (PNG 형식)
- **episode_{id}.json**: 에피소드당 약 10-500 KB (step 수에 따라)
- **grounding_episode_{id}.json**: 에피소드당 약 5-50 KB
- **experiment_log.json**: 실험 전체에 따라 수 MB ~ 수십 MB 가능
- **experiment_log.csv**: 실험 전체에 따라 수 MB 가능

### 권장 사항
- 오래된 실험 로그는 주기적으로 백업 후 삭제
- `experiment_log.json`이 너무 커지면 주기적으로 분할 저장 고려
- 필요한 에피소드만 보관하고 나머지는 삭제

---

## 파일 읽기 및 분석

### JSON 파일 읽기
```python
import json
from pathlib import Path

# Episode JSON 읽기
with open("logs/.../episode_1_.../episode_1.json", 'r', encoding='utf-8') as f:
    episode_data = json.load(f)

# Grounding JSON 읽기
with open("logs/.../episode_1_.../grounding_episode_1.json", 'r', encoding='utf-8') as f:
    grounding_data = json.load(f)
```

### CSV 파일 읽기
```python
import pandas as pd

# CSV 읽기
df = pd.read_csv("logs/.../experiment_log.csv")

# 특정 에피소드만 필터링
episode_1_data = df[df['episode_id'] == 1]
```

### CSV 변환 스크립트
`src/utils/scripts/json_to_csv_converter.py`를 사용하여 episode JSON을 CSV로 변환할 수 있습니다.

---

## 참고사항

- 모든 파일은 UTF-8 인코딩으로 저장됩니다.
- 타임스탬프는 ISO 8601 형식 (`YYYY-MM-DDTHH:MM:SS.ffffff`)을 사용합니다.
- 이미지 파일명의 step 번호는 4자리 숫자로 0 패딩됩니다 (예: `step_0001.png`).
- Grounding 파일은 thread-safe하게 작성됩니다 (Lock 사용).
- 현재 에피소드에서 생성된 Grounding은 다음 에피소드부터 사용 가능합니다 (즉시 적용되지 않음).
