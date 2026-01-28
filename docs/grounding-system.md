# Grounding 지식 시스템

이 문서는 Grounding 지식 시스템의 동작 방식, 파일 구조, 병합 기능을 설명합니다.

## 개요

Grounding 지식 시스템은 사용자 피드백을 통해 실수를 학습하고 누적하여 다음 에피소드에서 활용할 수 있도록 하는 시스템입니다.

## 주요 특징

- **에피소드별 피드백 수집**: 각 step에서 사용자가 제공한 피드백을 타입별로 분류하여 저장
- **자동 Grounding 생성**: 에피소드 종료 시 VLM이 모든 피드백을 분석하여 Grounding 생성
- **System Prompt 자동 포함**: 생성된 Grounding이 다음 에피소드의 System Prompt에 자동 포함
- **여러 파일 병합**: 여러 Grounding 파일(JSON/TXT)을 자동으로 병합하여 사용

## 동작 흐름

### 1. 실험 시작 시

```
1. 사용자가 Episode 번호 입력
2. EpisodeManager 및 GroundingFileManager 초기화
3. GROUNDING_FILE_PATH에서 이전 Grounding 파일 읽기
4. System Prompt 생성 시 Grounding 내용 포함
```

### 2. Step별 피드백 수집

```
1. 사용자가 step에서 피드백 입력 (예: "feedback : spatial: kitchen is green")
2. GroundingFileManager.append_step_feedback() 호출
3. 피드백을 타입별로 분류하여 저장:
   - grounding_per_step: Step별 상세 기록
   - stacked_grounding: 타입별 누적 리스트
```

### 3. 에피소드 종료 시

```
1. _generate_grounding_from_episode() 호출
2. 현재 에피소드의 stacked_grounding만 사용 (이전 Grounding 무시)
3. VLM 호출하여 final_grounding 생성
4. 파일 저장:
   - episode_X/grounding_episode_X.json
   - episode_X/grounding_episode_X.txt
   - grounding/grounding_latest.json (전역 최신)
   - grounding/grounding_latest.txt (전역 최신)
```

### 4. 다음 에피소드에서 사용

```
1. GROUNDING_FILE_PATH에서 Grounding 파일 읽기
2. 여러 파일이면 자동 병합:
   - JSON 파일: 병합 후 Markdown 렌더링
   - TXT 파일: 텍스트 병합
3. System Prompt의 $grounding_content에 포함
```

## 파일 구조

### Episode별 Grounding 파일

**위치**: `logs/{experiment_name}/episode_{id}_{timestamp}_{script}/grounding_episode_{id}.json`

**구조**:
```json
{
  "expr_info": {
    "episode_id": 1
  },
  "grounding_per_step": [
    {
      "step_id": 1,
      "instruction": "...",
      "status": "Success",
      "feedback": {
        "user_preference": null,
        "spatial": "kitchen is green",
        "procedural": null,
        "general": null
      }
    }
  ],
  "stacked_grounding": {
    "user_preference": ["[ Step1 - Success ] : ..."],
    "spatial": ["[ Step1 - Success ] : kitchen is green"],
    "procedural": [],
    "general": []
  },
  "final_grounding": {
    "generation_timestamp": "2026-01-27T...",
    "user_preference_grounding": {"content": "..."},
    "spatial_grounding": {"content": "The green room is the kitchen."},
    "procedural_grounding": {"content": ""},
    "general_grounding_rules": {"content": ""}
  }
}
```

### 전역 최신 Grounding 파일

**위치**: `logs/{experiment_name}/grounding/grounding_latest.json` (또는 `.txt`)

**용도**: 다음 에피소드에서 자동으로 사용되는 최신 Grounding

## 여러 파일 병합 기능 ⭐ **신규**

### 설정

`src/utils/miscellaneous/global_variables.py`:

```python
GROUNDING_FILE_PATH = "file1.json,file2.json,file3.txt"  # 여러 파일 지원
GROUNDING_MERGE_FORMAT = "txt"  # "txt" | "json" | "both"
```

### JSON 파일 병합

여러 JSON 파일이 지정되면:

1. **stacked_grounding 병합**:
   - 각 카테고리(user_preference, spatial, procedural, general)별로 리스트 합치기
   - 중복 항목 제거

2. **final_grounding 병합**:
   - `generation_timestamp` 제외
   - 각 카테고리의 `content`를 줄바꿈으로 병합: `"- <content>\n\n- <content>"`

3. **Markdown 렌더링**:
   - 헤더 레벨 조정: H3 → H4 (System Prompt에 삽입되므로)
   - System Prompt에 포함

### TXT 파일 병합

여러 TXT 파일이 지정되면:
- 각 파일의 텍스트 내용을 `"\n\n---\n\n"`로 병합

### 혼합 파일 처리

JSON과 TXT 파일이 함께 있으면:
- JSON 파일: 병합 후 Markdown 렌더링
- TXT 파일: 텍스트 병합
- 최종: JSON Markdown + TXT 텍스트를 `"\n\n---\n\n"`로 병합

## API

### PromptOrganizer

#### `get_system_prompt(grounding_file_path=None)`

System Prompt 생성 시 Grounding 파일을 읽어서 포함합니다.

**파라미터**:
- `grounding_file_path`: Grounding 파일 경로 (str, List[str], 또는 None)

**동작**:
- JSON 파일: `merge_grounding_json_files()` → `render_grounding_to_markdown()`
- TXT 파일: 텍스트 읽기
- 혼합: 각각 처리 후 병합

#### `get_verbalized_entropy_system_prompt(grounding_file_path=None)`

Verbalized Entropy 모드용 System Prompt 생성 (동일한 Grounding 처리)

### GroundingFileManager

#### `append_step_feedback(step_id, instruction, status, feedback)`

Step별 피드백을 추가합니다.

**파라미터**:
- `step_id`: Step 번호
- `instruction`: 사용자 명령
- `status`: "Success" | "Failure" | "WiP"
- `feedback`: `{"user_preference": "...", "spatial": "...", ...}`

#### `get_stacked_grounding()`

누적된 Grounding 데이터를 반환합니다.

**반환값**:
```python
{
  "user_preference": ["[ Step1 - Success ] : ..."],
  "spatial": ["[ Step2 - Success ] : ..."],
  "procedural": [],
  "general": []
}
```

#### `save_final_grounding(final_grounding)`

VLM이 생성한 final_grounding을 저장합니다.

## 설정 옵션

### global_variables.py

```python
# Grounding 시스템 사용 여부
USE_NEW_GROUNDING_SYSTEM = True

# Grounding 생성 모드
GROUNDING_GENERATION_MODE = "episode"  # "episode"만 지원

# Grounding 저장 형식
GROUNDING_SAVE_FORMAT = "both"  # "json" | "txt" | "both"

# Grounding 파일 경로 (여러 파일 지원)
GROUNDING_FILE_PATH = "logs/grounding/grounding_latest.json,logs/grounding/episode1.json"

# Grounding 병합 형식
GROUNDING_MERGE_FORMAT = "txt"  # "txt" | "json" | "both"
```

## 사용 예시

### 단일 파일 사용

```python
# global_variables.py
GROUNDING_FILE_PATH = "logs/grounding/grounding_latest.txt"
```

### 여러 JSON 파일 병합

```python
# global_variables.py
GROUNDING_FILE_PATH = "logs/grounding/grounding_latest.json,logs/grounding/episode1.json,logs/grounding/episode2.json"
```

### 혼합 파일 사용

```python
# global_variables.py
GROUNDING_FILE_PATH = "logs/grounding/grounding_latest.json,logs/grounding/custom_grounding.txt"
```

## 주의사항

1. **Grounding 생성 시**: 현재 에피소드 피드백만 사용 (이전 Grounding 무시)
2. **Action 생성 시**: `GROUNDING_FILE_PATH`에 지정된 모든 Grounding 파일 사용
3. **파일 경로**: 상대 경로는 프로젝트 루트 또는 `src/` 기준으로 해석
4. **JSON 파싱**: JSON 파일이 손상되면 해당 파일은 건너뜀

---

**작성일**: 2026-01-27
