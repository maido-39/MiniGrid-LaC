# MiniGrid-LaC 프로젝트 타임라인

이 문서는 MiniGrid-LaC 프로젝트의 시작부터 현재까지 주요 기능들이 추가된 타임라인을 담고 있습니다.

## 📅 타임라인 개요

프로젝트는 2025년 1월부터 시작되어 현재까지 지속적으로 발전하고 있습니다. 각 기능은 시간 순서대로 정리되어 있으며, 중요한 기능에는 사용법도 포함되어 있습니다.

---

## 🗓️ 2025년 1월 초반: 프로젝트 기초 구축

### 2025-01-07 ~ 2025-01-09: 기본 환경 래퍼 및 프롬프트 시스템

**기능**: MiniGrid 환경 래퍼 클래스 및 기본 프롬프트 시스템 구축

**주요 내용**:
- CustomRoomWrapper 클래스 구현
- 기본 프롬프트 템플릿 작성
- 환경 생성 및 제어 기본 기능

**사용법**:
```python
from utils.map_manager.minigrid_customenv_emoji import MiniGridEmojiWrapper

# 환경 생성
wrapper = MiniGridEmojiWrapper(size=10)
obs, info = wrapper.reset()
```

---

### 2025-01-11 ~ 2025-01-13: 이모지 맵 시스템

**기능**: 이모지 기반 맵 생성 및 JSON 맵 로더

**주요 내용**:
- 이모지 객체 렌더링 지원
- JSON 파일에서 맵 로드 기능
- 18가지 색상 팔레트 확장 (기존 6개 → 18개)
- 이모지 맵 생성기 통합

**사용법**:
```python
from utils.map_manager.emoji_map_loader import load_emoji_map_from_json

# JSON 파일에서 맵 로드
wrapper = load_emoji_map_from_json('config/example_map.json')
obs, info = wrapper.reset()
```

**JSON 맵 파일 형식**:
```json
{
  "size": 10,
  "map": [
    "🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦",
    "🟦🟫🟫🟫🟫🟫🟫🟫🟫🟦",
    ...
  ],
  "objects": {
    "🟦": {"type": "wall", "color": "blue"},
    "🟫": {"type": "floor", "color": "brown"}
  }
}
```

**관련 문서**: [이모지 맵 JSON 로더 가이드](./emoji-map-loader.md)

---

### 2025-01-12: 절대 좌표 이동 시스템

**기능**: 로봇 방향과 무관하게 상/하/좌/우 직접 이동

**주요 내용**:
- `step_absolute()` 메서드 구현
- 방향별 액션 파싱 (north/south/west/east)
- 인덱스 기반 액션 지원

**사용법**:
```python
# 절대 좌표 이동
obs, reward, done, truncated, info = wrapper.step_absolute('move up')
obs, reward, done, truncated, info = wrapper.step_absolute('move right')
obs, reward, done, truncated, info = wrapper.step_absolute(0)  # 인덱스
obs, reward, done, truncated, info = wrapper.step_absolute('north')  # 별칭
```

**지원하는 액션**:
- `'move up'`, `'north'`, `0` → 위로 이동
- `'move down'`, `'south'`, `1` → 아래로 이동
- `'move left'`, `'west'`, `2` → 왼쪽으로 이동
- `'move right'`, `'east'`, `3` → 오른쪽으로 이동

**관련 문서**: [Wrapper API](./wrapper-api.md#절대-좌표-이동-absolute-movement)

---

## 🗓️ 2025년 1월 중반: VLM 통합 및 모듈화

### 2025-01-14: 프로젝트 구조 모듈화

**기능**: 프로젝트를 utils 기반 모듈 구조로 재구성

**주요 내용**:
- `utils/` 디렉토리 구조 정립
- `map_manager/`, `vlm/`, `miscellaneous/`, `prompt_manager/` 모듈 분리
- `minigrid_lac.py` 메인 엔트리 포인트 생성
- `PromptOrganizer` 클래스 구현

**사용법**:
```python
from utils.miscellaneous.scenario_runner import ScenarioExperiment
from utils.miscellaneous.safe_minigrid_registration import safe_minigrid_reg

safe_minigrid_reg()
experiment = ScenarioExperiment(json_map_path="config/example_map.json")
experiment.run()
```

---

### 2025-01-19 ~ 2025-01-20: VLM 핸들러 시스템

**기능**: 다양한 VLM 모델 지원을 위한 통일된 인터페이스

**주요 내용**:
- OpenAI GPT-4o 핸들러
- Gemini API 핸들러 추가
- Qwen VLM 핸들러
- Gemma 핸들러
- VLMWrapper 통일된 인터페이스

**사용법**:
```python
from utils.vlm.vlm_wrapper import VLMWrapper

# VLM 래퍼 생성
vlm = VLMWrapper(
    model="gpt-4o",  # 또는 "gemini-2.5-flash", "qwen2.5-vl-32b-instruct" 등
    temperature=0.5,
    max_tokens=3000
)

# VLM 호출
response = vlm.generate(
    image=image,
    system_prompt=system_prompt,
    user_prompt=user_prompt
)
```

**지원하는 모델**:
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`
- **Gemini**: `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-3.0-flash`
- **Qwen**: `qwen2.5-vl-32b-instruct`, `qwen2.5-vl-7b-instruct`
- **Gemma**: `google/gemma-2-9b-it`

**관련 문서**: [VLM 핸들러 시스템 가이드](./vlm-handlers.md)

---

### 2025-01-20: Gemini Thinking 기능 지원

**기능**: Gemini 2.5/3 시리즈의 Thinking 기능 통합

**주요 내용**:
- Thinking 모드 활성화
- 중간 추론 과정 추출
- 최종 응답과 Thinking 분리

**사용법**:
```python
# global_variables.py에서 설정
GEMINI_THINKING_ENABLED = True

# VLM 호출 시 자동으로 Thinking 포함
response = vlm.generate(...)
thinking = response.get('thinking', '')
final_response = response.get('content', '')
```

**관련 문서**: [Gemini Thinking 기능 가이드](./LLM-API/gemini-thinking.md)

---

## 🗓️ 2025년 1월 후반: 고급 기능 추가

### 2025-01-20: Entropy 분석 시스템 (Logprobs 기반)

**기능**: VLM의 action 불확실성을 정량화하는 Entropy 계산

**주요 내용**:
- Vertex AI Gemini logprobs 기능 활용
- 3가지 조건으로 VLM 호출 (H(X), H(X|S), H(X|L,S))
- Shannon Entropy 계산
- Trust 값 계산

**사용법**:
```bash
cd src
python scenario2_test_entropy_comparison.py config/example_map.json
```

**설정**:
```python
# global_variables.py
LOGPROBS_ENABLED = True
VLM_MODEL = "gemini-2.5-flash-vertex"  # Vertex AI 모델 필요
```

**Entropy 계산 공식**:
```
H(X) ≥ H(X|S) ≥ H(X|L,S)
T = (H(X) - H(X|S)) / (H(X) - H(X|L,S))
```

**관련 문서**: [Entropy 및 Trust 계산 가이드](./entropy-trust-calculation.md)

---

### 2025-01-22: 다중 객체 운반 및 액션 실패 감지

**기능**: 여러 객체를 동시에 운반하고 액션 실패를 감지

**주요 내용**:
- 다중 객체 pickup/drop 지원
- 액션 실패 자동 감지
- 체스판 스타일 좌표 레이블 추가

**사용법**:
```python
# 여러 객체 운반
obs, reward, done, truncated, info = wrapper.step_absolute('pickup')
# info['carrying']에 운반 중인 객체 리스트 포함
```

---

### 2025-01-23: Grounding 지식 시스템

**기능**: 사용자 피드백을 통한 실수 학습 및 누적 시스템

**주요 내용**:
- 에피소드별 피드백 수집
- VLM 기반 Grounding 자동 생성
- System Prompt에 자동 포함
- 여러 Grounding 파일 병합 지원

**사용법**:
```python
# Step별 피드백 입력
# 실험 중에 "feedback : spatial: kitchen is green" 형식으로 입력

# 에피소드 종료 시 자동으로 Grounding 생성
# 다음 에피소드부터 자동으로 System Prompt에 포함됨
```

**설정**:
```python
# global_variables.py
USE_NEW_GROUNDING_SYSTEM = True
GROUNDING_FILE_PATH = "logs/grounding/grounding_latest.json"
GROUNDING_MERGE_FORMAT = "txt"  # "txt" | "json" | "both"
```

**피드백 형식**:
```
feedback : spatial: kitchen is green
feedback : procedural: always check door before entering
feedback : user_preference: prefer shortest path
```

**관련 문서**: [Grounding 지식 시스템 가이드](./grounding-system.md)

---

### 2025-01-23: Entropy VLM 호출 병렬화

**기능**: 3가지 조건의 VLM 호출을 병렬로 실행하여 속도 개선

**주요 내용**:
- `concurrent.futures`를 사용한 병렬 처리
- 피드백 UX 개선
- 여러 Grounding 파일 지원 강화

---

### 2025-01-25: 에피소드 분석 스크립트

**기능**: 에피소드 데이터 분석 및 시각화 도구

**주요 내용**:
- 영어 캡션 지원
- Box plot 가이드
- 통계 분석 스크립트

---

### 2025-01-27: Verbalized Entropy 시스템

**기능**: VLM이 직접 출력하는 확률 분포를 사용한 Entropy 계산 (Tian et al. 2023 기반)

**주요 내용**:
- Verbalized Confidence 방식
- Step-wise 확률 분포 추출
- 가중 평균 Entropy 계산 (50/30/20)
- JSON 파싱 자동 재시도

**사용법**:
```bash
cd src
python scenario2_test_entropy_comparison_refined_entropy.py config/example_map.json
```

**설정**:
```python
# global_variables.py
USE_VERBALIZED_ENTROPY = True
LOGPROBS_ENABLED = False  # 자동으로 False로 설정됨
VLM_MODEL = "gemini-2.5-flash"  # RLHF 모델 권장
```

**VLM 출력 형식**:
```json
{
  "executability": 0.95,
  "step1": {"north": 0.65, "south": 0.15, "west": 0.12, "east": 0.08},
  "step2": {"north": 0.45, "south": 0.30, "west": 0.15, "east": 0.10},
  "step3": {"north": 0.40, "south": 0.35, "west": 0.15, "east": 0.10},
  "reasoning": "Brief explanation"
}
```

**Entropy 계산**:
```python
# Step별 Entropy
H_step = -Σ p_i × log₂(p_i)

# 가중 평균 Entropy
H_weighted = 0.5 × H_step1 + 0.3 × H_step2 + 0.2 × H_step3
```

**장점**:
- RLHF 모델의 교정된 확률 사용
- logprobs 기능이 없는 모델에서도 사용 가능
- 명시적 확률 출력으로 해석 용이

**관련 문서**: [Entropy 및 Trust 계산 가이드](./entropy-trust-calculation.md#verbalized-entropy-방식-tian-et-al-2023-기반-신규)

---

### 2025-01-27: Grounding 파일 병합 개선

**기능**: 여러 Grounding 파일(JSON/TXT) 자동 병합 및 System Prompt 통합

**주요 내용**:
- JSON 파일 자동 병합 (stacked_grounding + final_grounding)
- TXT 파일 텍스트 병합
- 혼합 파일 지원 (JSON + TXT)
- Markdown 렌더링 최적화

**사용법**:
```python
# global_variables.py
GROUNDING_FILE_PATH = "file1.json,file2.json,file3.txt"  # 여러 파일 지원
GROUNDING_MERGE_FORMAT = "txt"  # "txt" | "json" | "both"
```

**병합 로직**:
- **JSON 파일**: stacked_grounding 리스트 합치기 + final_grounding content 병합
- **TXT 파일**: 텍스트를 `"\n\n---\n\n"`로 병합
- **혼합**: JSON Markdown + TXT 텍스트 병합

**관련 문서**: [Grounding 지식 시스템 가이드](./grounding-system.md#여러-파일-병합-기능-신규)

---

## 📊 기능별 요약

### 핵심 기능

1. **이모지 맵 시스템** (2025-01-11)
   - JSON 파일로 맵 정의
   - 18가지 색상 팔레트
   - 이모지 객체 렌더링

2. **절대 좌표 이동** (2025-01-12)
   - 방향 무관 이동
   - 다양한 액션 표현 지원

3. **VLM 통합** (2025-01-19)
   - 다중 VLM 모델 지원
   - 통일된 인터페이스
   - Gemini Thinking 기능

4. **Grounding 시스템** (2025-01-23)
   - 피드백 기반 학습
   - 자동 Grounding 생성
   - 파일 병합 지원

5. **Entropy 분석** (2025-01-20, 2025-01-27)
   - Logprobs 기반 (2025-01-20)
   - Verbalized Entropy (2025-01-27)
   - Trust 값 계산

### 지원 기능

- **Episode 관리**: 에피소드별 로깅 및 Grounding 생성
- **다중 객체 운반**: 여러 객체 동시 운반
- **액션 실패 감지**: 자동 실패 감지 및 피드백
- **병렬 처리**: Entropy VLM 호출 병렬화
- **분석 도구**: 에피소드 분석 및 시각화

---

## 🚀 빠른 시작 가이드

### 1. 기본 실험 실행

```bash
cd src
python minigrid_lac.py config/example_map.json
```

### 2. Entropy 분석 실험

```bash
# Logprobs 기반
python scenario2_test_entropy_comparison.py config/example_map.json

# Verbalized Entropy 기반
python scenario2_test_entropy_comparison_refined_entropy.py config/example_map.json
```

### 3. 설정 변경

`src/utils/miscellaneous/global_variables.py`에서 설정 변경:

```python
# VLM 설정
VLM_MODEL = "gemini-2.5-flash"
VLM_TEMPERATURE = 0.5
VLM_MAX_TOKENS = 3000

# Entropy 설정
LOGPROBS_ENABLED = True
USE_VERBALIZED_ENTROPY = True

# Grounding 설정
USE_NEW_GROUNDING_SYSTEM = True
GROUNDING_FILE_PATH = "logs/grounding/grounding_latest.json"
```

---

## 📚 관련 문서

- [README.md](../README.md) - 프로젝트 전체 개요
- [Wrapper API](./wrapper-api.md) - 환경 래퍼 API
- [VLM 핸들러 가이드](./vlm-handlers.md) - VLM 모델 사용법
- [Grounding 시스템](./grounding-system.md) - Grounding 지식 시스템
- [Entropy 계산 가이드](./entropy-trust-calculation.md) - Entropy 및 Trust 계산

---

**작성일**: 2026-01-27  
**최종 업데이트**: 2026-01-27
