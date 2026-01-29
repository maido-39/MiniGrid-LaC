# Memory Prompt & Render 가이드

프롬프트 템플릿에서 메모리 객체를 참조하는 문법과, 메모리 값이 어떻게 문자열로 렌더되는지 정리한 문서입니다.

---

## 1. 개요

- **저장**: VLM 응답의 `memory` 블록 전체가 하나의 Dict `memory_dict`로 저장됩니다.
- **참조**: 프롬프트 템플릿에서는 **대괄호 문법** `$memory[속성이름]`, `$memory[속성][중첩속성]` 으로 참조합니다.
- **렌더**: 참조된 값(str / list / dict)은 **MemoryRenderer**에 의해 읽기 쉬운 텍스트로 변환된 뒤 치환됩니다.
- **없는 키**: 템플릿에 있는데 `memory_dict`에 없는 키는 기본값 `"None"`으로 치환되며, **강한 경고**와 **로그(WARNING)** 가 남습니다.

---

## 2. 프롬프트 문법

### 2.1 일반 변수 (기존)

- `$변수이름` 또는 `${변수이름}`  
- 예: `$last_action_str`, `$grounding_content`

### 2.2 메모리 전용 문법 (대괄호)

- **1단계**: `$memory[속성이름]` → `memory_dict["속성이름"]` 값을 렌더한 문자열로 치환
- **중첩**: `$memory[속성][중첩속성]` → `memory_dict["속성"]["중첩속성"]` 값을 렌더 후 치환
- **키 문자**: 대괄호 안 키는 **영문·숫자·밑줄·하이픈**만 허용 (`[a-zA-Z0-9_-]+`)

**프롬프트 예시 (system_prompt_start.txt 등)**

```text
## Memory (State Continuity)
- Previous Action: $memory[previous_action]
- Task Process: $memory[task_process]
- Goal only: $memory[task_process][goal]
- Scene: $memory[spatial_description]
- Plan: $memory[high-level_planning]
```

---

## 3. Render 규칙

| 타입 | 동작 |
|------|------|
| **str** | 그대로 반환. 빈 문자열은 `"None"` (또는 설정값). |
| **list** | 항목마다 `- 항목` 한 줄. 항목이 dict면 먼저 dict 규칙으로 렌더한 뒤 한 블록으로 취급. |
| **dict** | `key: value` 형태로 나열. value는 str/list/dict에 대해 재귀 렌더, **재귀 깊이 상한**(기본 8) 적용. |
| **숫자 / bool / None** | 예외 없이 `str(value)` 또는 `"None"` 처리. |
| **키 누락** | `$memory[typo_key]` 등 → 기본값 `"None"` 치환 + **강한 경고 출력** + **logging.warning** 기록. |

---

## 4. Render 예시

### 4.1 reasoning context 스타일 Memory

**JSON 예시**

```json
{
  "memory": {
    "spatial_description": "Left room has wall at E3. Corridor connects to restroom at G4.",
    "high-level_planning": [
      "Move to corridor",
      "Turn toward restroom",
      "Enter restroom",
      "Locate target object",
      "Return to start"
    ],
    "historical_summrization": "Reached corridor. Currently facing restroom door.",
    "immidate_action_instruction": "Take one step east into the restroom."
  }
}
```

**프롬프트에서 호출 예시**

```text
## Memory (State Continuity)
- Scene: $memory[spatial_description]
- Plan: $memory[high-level_planning]
- History: $memory[historical_summrization]
- Next: $memory[immidate_action_instruction]
```

**렌더 결과**

- `$memory[spatial_description]`  
  → `Left room has wall at E3. Corridor connects to restroom at G4.`

- `$memory[high-level_planning]`  
  → (list → 불릿 목록)

  ```text
  - Move to corridor
  - Turn toward restroom
  - Enter restroom
  - Locate target object
  - Return to start
  ```

- `$memory[historical_summrization]`  
  → `Reached corridor. Currently facing restroom door.`

- `$memory[immidate_action_instruction]`  
  → `Take one step east into the restroom.`

---

### 4.2 task_process + subgoals 메모리 구조

**JSON 예시**

```json
{
  "memory": {
    "task_process": {
      "status": "in_progress",
      "current_subgoal_id": 1,
      "subgoals": [
        {
          "subgoal_id": 1,
          "subgoal_type": "navigation",
          "target": "restroom",
          "explicit_completion_condition": "AGENT occupies any cell within E1-G3",
          "subgoal_status": "in_progress"
        },
        {
          "subgoal_id": 2,
          "subgoal_type": "navigation",
          "target": "storage",
          "explicit_completion_condition": "AGENT occupies any cell within J6-K6",
          "subgoal_status": "pending"
        }
      ]
    },
    "previous_action": "move right"
  }
}
```

**프롬프트에서 호출 예시**

```text
## Memory
- Last action: $memory[previous_action]
- Task state: $memory[task_process]
- Status only: $memory[task_process][status]
- Current subgoal id: $memory[task_process][current_subgoal_id]
- Subgoals list: $memory[task_process][subgoals]
```

**렌더 결과**

- `$memory[previous_action]`  
  → `move right`

- `$memory[task_process]`  
  → (dict → key: value, 값은 재귀 렌더, list는 여러 줄+들여쓰기)

  ```text
  status: in_progress
  current_subgoal_id: 1
  subgoals: - subgoal_id: 1
    subgoal_type: navigation
    target: restroom
    explicit_completion_condition: AGENT occupies any cell within E1-G3
    subgoal_status: in_progress
  - subgoal_id: 2
    subgoal_type: navigation
    target: storage
    explicit_completion_condition: AGENT occupies any cell within J6-K6
    subgoal_status: pending
  ```

- `$memory[task_process][status]`  
  → `in_progress`

- `$memory[task_process][current_subgoal_id]`  
  → 숫자 → `str(value)` → `1`

- `$memory[task_process][subgoals]`  
  → list of dict → 각 dict를 렌더한 뒤 리스트 규칙으로 `- ` 접두사 + 들여쓰기

  ```text
  - subgoal_id: 1
    subgoal_type: navigation
    target: restroom
    explicit_completion_condition: AGENT occupies any cell within E1-G3
    subgoal_status: in_progress
  - subgoal_id: 2
    subgoal_type: navigation
    target: storage
    ...
  ```

---

### 4.3 그밖에 가능한 구조와 동작

| 구조 | JSON 예시 | 프롬프트 호출 | 렌더 결과 |
|------|-----------|----------------|-----------|
| **빈/누락** | `{}` 또는 해당 키 없음 | `$memory[previous_action]` | 키 없음 → 기본값 `"None"` + **경고·로그** |
| **문자열만** | `{"previous_action": "north"}` | `$memory[previous_action]` | `north` |
| **빈 문자열** | `{"reason": ""}` | `$memory[reason]` | `"None"` |
| **숫자/불리언** | `{"count": 3}`, `{"done": true}` | `$memory[count]`, `$memory[done]` | `3`, `True` |
| **리스트(문자열)** | `{"steps": ["a","b"]}` | `$memory[steps]` | `- a\n- b` |
| **리스트(숫자)** | `{"ids": [1,2,3]}` | `$memory[ids]` | `- 1\n- 2\n- 3` |
| **중첩 dict** | `{"task_process": {"goal": "restroom", "status": "in_progress"}}` | `$memory[task_process]`, `$memory[task_process][goal]` | 전체: `goal: restroom\nstatus: in_progress` / 단일: `restroom`, `in_progress` |
| **하이픈 키** | `{"high-level_planning": ["step1"]}` | `$memory[high-level_planning]` | `- step1` |
| **3단계 중첩** | `{"a": {"b": {"c": "leaf"}}}` | `$memory[a][b][c]` | `leaf` (재귀 깊이 상한 내) |

---

## 5. 구현 위치

- **저장**: `PromptOrganizer.memory_dict`, `set_memory_dict(d)`
- **렌더**: `utils/prompt_manager/memory_renderer.py` — `render_memory_value(value, ...)`
- **치환**: `utils/prompt_manager/prompt_interp.py` — `_substitute_memory_brackets()`, `system_prompt_interp(..., memory=memory_dict)`
- **없는 키 경고**: `prompt_interp`에서 값이 `None`일 때 `tfu.cprint`(강한 경고) + `logger.warning` 기록

---

## 6. 프롬프트 개발 시 참고

- 메모리 스키마를 바꿀 때: `memory_dict` 구조와 기본값(vlm_postprocessor 등)만 맞추면 되며, 렌더/치환 로직은 수정하지 않아도 됩니다.
- 새 필드 추가: 프롬프트 템플릿에 `$memory[새필드이름]` 한 줄 추가, VLM 출력 스키마에 해당 필드 문서화.
- **dev-memory 테스트 스크립트**: 프롬프트 파일 + 이미지로 VLM을 돌려 **반환 JSON**, **memory 내용**, **memory로 렌더된 프롬프트**를 한 번에 확인할 수 있습니다.  
  → [dev-memory 테스트 스크립트 가이드](./dev-memory-test-guide.md) 참고. 스크립트 위치: `src/dev-memory/run_memory_dev.py`.
