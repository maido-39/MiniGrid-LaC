# 엔트로피 로깅 구조

## 개요

현재 시스템에서는 두 가지 방식으로 엔트로피를 계산하고 로깅합니다:

1. **Logprobs 기반** (`scenario2_test_entropy_comparison.py`)
2. **Verbalized Confidence 기반** (`scenario2_test_entropy_comparison_refined_entropy.py`) ⭐ 신규

---

## 로그 파일 위치

모든 로그는 다음 경로에 저장됩니다:

```
logs/<experiment_name>/
├── experiment_log.json          # 전체 실험 JSON 로그 (누적)
├── experiment_log.csv           # 전체 실험 CSV 로그 (누적)
└── episode_<N>_<timestamp>_<script>/
    ├── episode_<N>.json        # 에피소드별 JSON 로그
    └── images/
        └── step_XXXX.png       # 각 스텝의 환경 이미지
```

---

## 1. Logprobs 기반 엔트로피 로깅

### 스크립트
- `src/scenario2_test_entropy_comparison.py`

### CSV 필드 (`experiment_log.csv`)

| 필드명 | 설명 | 예시 |
|--------|------|------|
| `entropy_H_X` | 조건 없음 엔트로피 | `0.8476` |
| `entropy_H_X_given_S` | Spatial context 조건 엔트로피 | `0.5234` |
| `entropy_H_X_given_LS` | Language + Spatial 조건 엔트로피 | `0.3121` |
| `trust_T` | Trust 값 (H(X) - H(X\|L,S)) | `0.5355` |
| `vlm_action_logprobs_info` | Action logprobs 정보 (JSON) | `{"action_positions": [...], "action_logprobs": [...]}` |

### JSON 필드 (`experiment_log.json`)

각 step의 JSON 객체에 다음 필드가 포함됩니다:

```json
{
  "step": 1,
  "entropy_H_X": 0.8476,
  "entropy_H_X_given_S": 0.5234,
  "entropy_H_X_given_LS": 0.3121,
  "trust_T": 0.5355,
  "vlm_response": {
    "action_logprobs_info": {
      "action_positions": [10, 15, 20],
      "action_logprobs": [
        ["east", ["east:-0.0000", "north:-13.9026", ...], 0.0001, 10],
        ...
      ],
      "action_entropies": [0.0001, 0.0271, 0.2680]
    }
  }
}
```

### 특징
- VLM의 내부 log-probabilities를 사용
- 3가지 조건(H(X), H(X\|S), H(X\|L,S))으로 동시 호출
- `LOGPROBS_ENABLED = True` 필요 (Vertex AI Gemini 모델)

---

## 2. Verbalized Confidence 기반 엔트로피 로깅 ⭐

### 스크립트
- `src/scenario2_test_entropy_comparison_refined_entropy.py`

### CSV 필드 (`experiment_log.csv`)

| 필드명 | 설명 | 예시 |
|--------|------|------|
| `executability` | 명령 수행 가능성 (0.0-1.0) | `0.7` |
| `step1_probs` | Step 1 확률 분포 (JSON) | `{"north": 0.05, "south": 0.05, "west": 0.05, "east": 0.85}` |
| `step2_probs` | Step 2 확률 분포 (JSON) | `{"north": 0.05, "south": 0.05, "west": 0.05, "east": 0.85}` |
| `step3_probs` | Step 3 확률 분포 (JSON) | `{"north": 0.05, "south": 0.05, "west": 0.05, "east": 0.85}` |
| `step1_entropy` | Step 1 Shannon Entropy | `0.8476` |
| `step2_entropy` | Step 2 Shannon Entropy | `0.8476` |
| `step3_entropy` | Step 3 Shannon Entropy | `0.8476` |
| `weighted_entropy_H_X` | 가중 평균 엔트로피 (Step1×0.5 + Step2×0.3 + Step3×0.2) | `0.8476` |
| `weighted_entropy_H_X_given_S` | Spatial context 조건 가중 엔트로피 | `0.5234` |
| `weighted_entropy_H_X_given_LS` | Language + Spatial 조건 가중 엔트로피 | `0.3121` |
| `trust_T` | Trust 값 | `0.5355` |

### JSON 필드 (`experiment_log.json`)

각 step의 JSON 객체에 다음 필드가 포함됩니다:

```json
{
  "step": 1,
  "vlm_response": {
    "executability": 0.7,
    "step1": {"north": 0.05, "south": 0.05, "west": 0.05, "east": 0.85},
    "step2": {"north": 0.05, "south": 0.05, "west": 0.05, "east": 0.85},
    "step3": {"north": 0.05, "south": 0.05, "west": 0.05, "east": 0.85},
    "best_guesses": ["east", "east", "east"],
    "reasoning": "Moving east toward target"
  },
  "step_entropies": [0.8476, 0.8476, 0.8476],
  "weighted_entropy_H_X": 0.8476,
  "weighted_entropy_H_X_given_S": 0.5234,
  "weighted_entropy_H_X_given_LS": 0.3121,
  "trust_T": 0.5355
}
```

### 특징
- VLM이 직접 출력하는 확률값 사용 (Verbalized Confidence)
- 각 Step별 확률 분포를 명시적으로 기록
- 가중 평균 엔트로피 계산 (Step1: 50%, Step2: 30%, Step3: 20%)
- `USE_VERBALIZED_ENTROPY = True` 필요 (global_variables.py)

---

## 엔트로피 계산 방식 비교

| 항목 | Logprobs 기반 | Verbalized 기반 |
|------|---------------|----------------|
| **데이터 소스** | VLM 내부 log-probabilities | VLM 직접 출력 확률값 |
| **정확도** | RLHF 모델에서 교정 부족 | 더 정확하게 교정됨 |
| **Step별 분포** | ❌ 없음 | ✅ 각 Step별 확률 분포 |
| **Executability** | ❌ 없음 | ✅ 0.0-1.0 평가 |
| **필요 조건** | `LOGPROBS_ENABLED = True` | `USE_VERBALIZED_ENTROPY = True` |
| **모델 요구사항** | Vertex AI (logprobs 지원) | 모든 Gemini 모델 |

---

## 로그 파일 분석 스크립트

### 1. `analyze_entropy_trust.py`
- **용도**: `experiment_log.json`에서 엔트로피 및 Trust 값 추출 및 분석
- **출력**: 
  - `entropy_trust_experiment_log.png` - 시각화 플롯
  - `entropy_trust_experiment_log.txt` - 텍스트 요약

### 2. `analyze_step_entropy.py`
- **용도**: Step별 엔트로피 분석 (Verbalized 기반)
- **출력**: Step별 엔트로피 분포 플롯

---

## 실제 로그 예시

### Logprobs 기반 로그 (CSV)

```csv
"step","timestamp","agent_x","agent_y",...,"entropy_H_X","entropy_H_X_given_S","entropy_H_X_given_LS","trust_T"
"1","2026-01-23T20:09:27.835312","3","11",...,"0.8476","0.5234","0.3121","0.5355"
```

### Verbalized 기반 로그 (CSV)

```csv
"step","timestamp","agent_x","agent_y",...,"executability","step1_probs","step2_probs","step3_probs","step1_entropy","step2_entropy","step3_entropy","weighted_entropy_H_X","weighted_entropy_H_X_given_S","weighted_entropy_H_X_given_LS","trust_T"
"1","2026-01-23T20:09:27.835312","3","11",...,"0.7","{\"north\":0.05,\"south\":0.05,\"west\":0.05,\"east\":0.85}","{\"north\":0.05,\"south\":0.05,\"west\":0.05,\"east\":0.85}","{\"north\":0.05,\"south\":0.05,\"west\":0.05,\"east\":0.85}","0.8476","0.8476","0.8476","0.8476","0.5234","0.3121","0.5355"
```

---

## 로그 파일 접근 방법

### Python으로 CSV 읽기

```python
import csv
import json

# CSV 읽기
with open('experiment_log.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        step1_entropy = float(row['step1_entropy']) if row['step1_entropy'] else None
        step1_probs = json.loads(row['step1_probs']) if row['step1_probs'] else {}
        print(f"Step 1 Entropy: {step1_entropy}")
        print(f"Step 1 Probs: {step1_probs}")
```

### Python으로 JSON 읽기

```python
import json

# JSON 읽기
with open('experiment_log.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    for step in data:
        if 'step_entropies' in step:
            print(f"Step {step['step']}: Entropies = {step['step_entropies']}")
        if 'weighted_entropy_H_X' in step:
            print(f"Step {step['step']}: Weighted Entropy = {step['weighted_entropy_H_X']}")
```

---

## 관련 문서

- [Action Entropy 분석 보고서](./action-entropy-analysis.md)
- [Entropy 및 Trust 계산 가이드](../entropy-trust-calculation.md)
- [로그 구조 문서](../logs-structure.md)

---

**최종 업데이트**: 2026-01-27
