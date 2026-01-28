# Entropy and Trust Calculation in Minigrid VLM Experiments

이 문서는 Minigrid 환경에서 VLM(Vision Language Model)의 action uncertainty를 분석하기 위한 Entropy(엔트로피) 및 Trust(신뢰도) 계산 방법과 전체 logical flow를 설명합니다.

## 목차

1. [개요](#개요)
2. [Entropy 계산](#entropy-계산)
3. [Trust 계산](#trust-계산)
4. [Logical Flow](#logical-flow)
5. [구현 세부사항](#구현-세부사항)

---

## 개요

### 목적

VLM이 생성하는 action의 불확실성(uncertainty)을 정량화하기 위해 정보 이론의 엔트로피 개념을 사용합니다. 또한, Grounding과 Language Instruction이 VLM의 결정에 미치는 영향을 분석하기 위해 Trust 값을 계산합니다.

### 두 가지 Entropy 계산 방식

프로젝트는 두 가지 방식으로 Entropy를 계산할 수 있습니다:

1. **Logprobs 기반** (기존 방식)
   - VLM API의 내부 확률 분포 사용
   - Vertex AI Gemini의 logprobs 기능 필요
   - 스크립트: `scenario2_test_entropy_comparison.py`

2. **Verbalized Entropy** (Tian et al. 2023 기반) ⭐ **신규**
   - VLM이 직접 출력하는 확률 분포 사용
   - RLHF 모델(Gemini-2.5-flash 등)에서 더 정확한 교정된 확률
   - 스크립트: `scenario2_test_entropy_comparison_refined_entropy.py`

### 기본 개념

- **H(X)**: Language Instruction과 Grounding 없이 VLM이 action을 생성할 때의 엔트로피 (최대 불확실성)
- **H(X|S)**: Grounding만 제공하고 Language Instruction 없이 VLM이 action을 생성할 때의 엔트로피
- **H(X|L,S)**: Grounding과 Language Instruction 모두 제공했을 때의 엔트로피 (최소 불확실성)
- **Trust T**: Grounding이 Language Instruction에 비해 얼마나 효과적인지를 나타내는 지표

### 수식 요약

```
H(X) ≥ H(X|S) ≥ H(X|L,S)

T = (H(X) - H(X|S)) / (H(X) - H(X|L,S))
```

---

## Entropy 계산

### 1. Shannon Entropy 공식

각 토큰 위치에서의 Shannon Entropy는 다음과 같이 계산됩니다:

```
H = -Σ p_i × log₂(p_i)
```

여기서:
- `p_i`: i번째 토큰의 확률
- 합은 모든 가능한 토큰에 대해 수행

### 2. VLM API에서의 Logprobs 추출

VLM API(특히 Vertex AI Gemini)는 각 토큰 위치에 대해 다음 정보를 제공합니다:

- **tokens**: 생성된 토큰 리스트
- **token_logprobs**: 각 토큰의 log probability
- **top_logprobs**: 각 토큰 위치에서 상위 k개 후보의 log probability
- **entropies**: 각 토큰 위치에서 계산된 Shannon entropy

### 3. 전체 Vocabulary 고려

**문제점**: API는 top-k 후보만 제공하므로, 전체 vocabulary를 고려하지 않으면 엔트로피가 과소평가됩니다.

**해결 방법**: 나머지 확률 질량을 "기타" 카테고리로 추가하여 근사합니다.

#### 구현 로직 (`vlm_wrapper.py`)

```python
# 1. top-k 후보들의 확률 계산
probs = [exp(log_prob) for log_prob in top_k_logprobs]
probs_sum = sum(probs)

# 2. 나머지 확률 질량 추정
remaining_prob = max(0.0, 1.0 - probs_sum)

# 3. 전체 확률 분포 구성 (top-k + "기타")
all_probs = probs + [remaining_prob] if remaining_prob > 0 else probs

# 4. 정규화 (합이 1이 되도록)
total_prob = sum(all_probs)
normalized_probs = [p / total_prob for p in all_probs]

# 5. Shannon Entropy 계산
entropy = -sum([p * log2(p) if p > 0 else 0 for p in normalized_probs])
```

**예시**:
- top-5 확률 합 = 0.95
- remaining_prob = 0.05
- 전체 분포: [p1, p2, p3, p4, p5, 0.05]
- 정규화 후 엔트로피 계산

### 4. Action 토큰에서 Entropy 추출

#### 4.1 Action 필드 위치 찾기

1. VLM 응답을 JSON으로 파싱
2. `action` 필드의 토큰 위치를 찾음
3. Action 값(예: `["move up", "turn left"]`)의 각 토큰 위치를 식별

#### 4.2 Action Entropy 추출

`get_action_logprobs()` 메서드가 다음을 수행:

```python
# 1. action 필드의 토큰 위치 찾기
action_positions = find_action_token_positions(logprobs_metadata, "action")

# 2. 각 위치의 entropy 추출
entropies = logprobs_metadata.get('entropies', [])
action_entropies = [entropies[pos] for pos in action_positions]

# 3. 반환
return {
    'action_positions': action_positions,
    'action_logprobs': action_logprobs_info,
    'action_entropies': action_entropies
}
```

#### 4.3 최종 Entropy 값 선택

현재 구현에서는 **첫 번째 action 토큰의 entropy**를 사용합니다:

```python
def _calculate_entropy_from_logprobs(action_logprobs_info):
    action_entropies = action_logprobs_info.get('action_entropies', [])
    if not action_entropies:
        return None
    return action_entropies[0]  # 첫 번째 action의 entropy
```

**이유**: Action의 첫 번째 토큰이 가장 중요한 결정 지점이기 때문입니다.

---

## Trust 계산

### 1. Trust 공식

```
T = (H(X) - H(X|S)) / (H(X) - H(X|L,S))
```

### 2. 의미 해석

- **분자 (H(X) - H(X|S))**: Grounding만으로 줄어든 불확실성
  - Grounding이 없을 때와 있을 때의 엔트로피 차이
  - Grounding의 정보량

- **분모 (H(X) - H(X|L,S))**: Grounding + Language Instruction으로 줄어든 불확실성
  - 최대 불확실성에서 최소 불확실성까지의 차이
  - 전체 정보량

- **Trust T**: Grounding이 전체 정보 중 차지하는 비율
  - T = 1: Grounding만으로도 Language Instruction과 동일한 효과
  - T = 0: Grounding이 전혀 도움이 되지 않음
  - T > 1: Grounding이 Language Instruction보다 더 효과적 (이론적으로 가능하지만 드뭄)

### 3. Edge Case 처리

#### 3.1 분모가 0인 경우

```python
denominator = H(X) - H(X|L,S)
if abs(denominator) < 1e-10:  # 매우 작은 값은 0으로 처리
    return float('nan')
```

**의미**: H(X) = H(X|L,S)인 경우, Language Instruction이 전혀 도움이 되지 않음
- 이 경우 Trust는 정의되지 않음 (NaN 반환)

#### 3.2 Entropy 값이 None인 경우

```python
if H_X is None or H_X_given_S is None or H_X_given_LS is None:
    return None
```

**의미**: 하나라도 계산 실패 시 Trust 계산 불가

### 4. Trust 값의 범위

- **T ∈ [0, 1]**: 일반적인 경우
  - 0에 가까울수록: Grounding의 효과가 작음
  - 1에 가까울수록: Grounding의 효과가 큼

- **T > 1**: 이론적으로 가능하지만 드뭄
  - Grounding이 Language Instruction보다 더 효과적

- **T < 0**: 이론적으로 불가능 (H(X|S) > H(X)인 경우)
  - Grounding이 오히려 불확실성을 증가시킴 (비정상적인 경우)

---

## Logical Flow

### 전체 실행 흐름

```
┌─────────────────────────────────────────────────────────────┐
│                    EntropyComparisonExperiment.run_step()     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  1. 환경 상태 가져오기               │
        │     - 이미지, 상태, 사용자 프롬프트  │
        └─────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  2. System Prompt 생성               │
        │     - Grounding 포함                 │
        └─────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  3. 3가지 조건으로 VLM 호출          │
        │     (동시에 순차 실행)              │
        └─────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  H(X) 호출   │    │ H(X|S) 호출  │    │H(X|L,S) 호출 │
│              │    │              │    │              │
│ - No Lang    │    │ - No Lang    │    │ - With Lang   │
│ - No Ground  │    │ - With Ground│    │ - With Ground │
└──────────────┘    └──────────────┘    └──────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  4. 각 응답에서 Entropy 추출         │
        │     - get_action_logprobs()         │
        │     - _calculate_entropy_from_...() │
        └─────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  5. Trust 계산                       │
        │     - _calculate_trust()            │
        └─────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  6. Action 실행                      │
        │     - H(X|L,S) 결과 사용            │
        └─────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  7. CSV 로깅                         │
        │     - Entropy 3개 + Trust 저장      │
        └─────────────────────────────────────┘
```

### 상세 단계별 설명

#### Step 1: 환경 상태 가져오기

```python
self.image = self.wrapper.get_image()
self.state = self.wrapper.get_state()
self.user_prompt = self.prompt_organizer.get_user_prompt(...)
```

#### Step 2: System Prompt 생성

```python
system_prompt = self.prompt_organizer.get_system_prompt(
    self.wrapper, 
    self.last_action_result
)
```

**포함 내용**:
- Grounding 정보 (이전 피드백에서 축적된 지식)
- Last action result (이전 액션의 성공/실패 정보)
- Task process (목표, 상태 등)
- Previous action

#### Step 3: 3가지 조건으로 VLM 호출

##### 3.1 H(X) 호출

```python
H_X_result = self.vlm_gen_action_H_X(
    image=self.image,
    system_prompt=system_prompt,
    user_prompt=self.user_prompt
)
```

**내부 동작**:
1. `_get_system_prompt_without_grounding()` 호출 → Grounding 제거
2. `user_prompt`를 빈 문자열("")로 설정
3. `requester_with_logprobs()` 호출
4. `parser_action_with_logprobs()` 호출
5. `get_action_logprobs()` 호출하여 action entropy 추출

##### 3.2 H(X|S) 호출

```python
H_X_given_S_result = self.vlm_gen_action_H_X_given_S(
    image=self.image,
    system_prompt=system_prompt,  # Grounding 포함
    user_prompt=self.user_prompt
)
```

**내부 동작**:
1. `system_prompt` 그대로 사용 (Grounding 포함)
2. `user_prompt`를 빈 문자열("")로 설정
3. 나머지는 H(X)와 동일

##### 3.3 H(X|L,S) 호출

```python
H_X_given_LS_result = self.vlm_gen_action_H_X_given_LS(
    image=self.image,
    system_prompt=system_prompt,  # Grounding 포함
    user_prompt=self.user_prompt   # Language Instruction 포함
)
```

**내부 동작**:
1. 기존 `vlm_gen_action()` 메서드 래핑
2. 모든 정보 포함 (Grounding + Language Instruction)

#### Step 4: Entropy 추출

```python
# 각 결과에서 action_logprobs_info 추출
H_X_info = H_X_result.get('action_logprobs_info', {})
H_X_given_S_info = H_X_given_S_result.get('action_logprobs_info', {})
H_X_given_LS_info = H_X_given_LS_result.get('action_logprobs_info', {})

# Entropy 계산
self.entropy_H_X = self._calculate_entropy_from_logprobs(H_X_info)
self.entropy_H_X_given_S = self._calculate_entropy_from_logprobs(H_X_given_S_info)
self.entropy_H_X_given_LS = self._calculate_entropy_from_logprobs(H_X_given_LS_info)
```

#### Step 5: Trust 계산

```python
self.trust_T = self._calculate_trust(
    self.entropy_H_X,
    self.entropy_H_X_given_S,
    self.entropy_H_X_given_LS
)
```

#### Step 6: Action 실행

```python
# H(X|L,S) 결과를 사용하여 실제 action 실행
self.vlm_response_parsed = H_X_given_LS_result.get('parsed', {})
action_chunk = self.vlm_response_parsed.get('action', [])
action_str = str(action_chunk[0])  # 첫 번째 action 사용

# Action 실행
self.action_index = self.wrapper.parse_absolute_action(action_str)
_, self.reward, terminated, truncated, _ = self.wrapper.step(self.action_index)
```

#### Step 7: CSV 로깅

```python
self.csv_writer.writerow([
    # ... 기존 필드들 ...
    entropy_H_X_str,
    entropy_H_X_given_S_str,
    entropy_H_X_given_LS_str,
    trust_T_str
])
```

---

## 구현 세부사항

### 1. 파일 구조

```
src/
├── utils/
│   ├── miscellaneous/
│   │   └── scenario_runner.py          # ScenarioExperiment 클래스
│   │       ├── vlm_gen_action_H_X()
│   │       ├── vlm_gen_action_H_X_given_S()
│   │       ├── vlm_gen_action_H_X_given_LS()
│   │       ├── _get_system_prompt_without_grounding()
│   │       └── _calculate_entropy_from_logprobs()
│   └── vlm/
│       ├── vlm_wrapper.py              # Entropy 계산 로직
│       └── vlm_postprocessor.py        # get_action_logprobs()
└── scenario2_test_entropy_comparison.py  # EntropyComparisonExperiment
```

### 2. 주요 메서드

#### 2.1 `vlm_gen_action_H_X()`

**위치**: `scenario_runner.py:344`

**기능**: Language Instruction과 Grounding 없이 VLM 호출

**입력**:
- `image`: 환경 이미지
- `system_prompt`: 원본 system prompt (Grounding 제거됨)
- `user_prompt`: 원본 user prompt (빈 문자열로 대체됨)

**출력**:
```python
{
    'parsed': dict,                    # 파싱된 VLM 응답
    'logprobs_metadata': dict,         # 전체 logprobs 메타데이터
    'action_logprobs_info': dict       # Action logprobs 정보
}
```

#### 2.2 `vlm_gen_action_H_X_given_S()`

**위치**: `scenario_runner.py:431`

**기능**: Grounding만 포함하고 Language Instruction 없이 VLM 호출

**차이점**: `system_prompt`를 그대로 사용 (Grounding 포함)

#### 2.3 `vlm_gen_action_H_X_given_LS()`

**위치**: `scenario_runner.py:514`

**기능**: Grounding과 Language Instruction 모두 포함하여 VLM 호출

**구현**: 기존 `vlm_gen_action()` 메서드를 래핑

#### 2.4 `_calculate_entropy_from_logprobs()`

**위치**: `scenario_runner.py:321`

**기능**: `action_logprobs_info`에서 첫 번째 action의 entropy 추출

**입력**:
```python
{
    'action_entropies': [float, ...]  # 각 action 토큰의 entropy 리스트
}
```

**출력**: `float | None` (첫 번째 action의 entropy)

#### 2.5 `_calculate_trust()`

**위치**: `scenario2_test_entropy_comparison.py:60`

**기능**: Trust 값 계산

**공식**: `T = (H(X) - H(X|S)) / (H(X) - H(X|L,S))`

**Edge Cases**:
- Entropy 값이 None인 경우 → None 반환
- 분모가 0인 경우 → NaN 반환

### 3. 데이터 흐름

```
VLM API Response
    │
    ├─> logprobs_metadata
    │   ├─> tokens: List[str]
    │   ├─> token_logprobs: List[float]
    │   ├─> top_logprobs: List[List[Dict]]
    │   └─> entropies: List[float]  ← vlm_wrapper.py에서 계산
    │
    └─> raw_response (JSON 문자열)
        │
        └─> parser_action_with_logprobs()
            │
            └─> get_action_logprobs(logprobs_metadata)
                │
                ├─> action_positions: List[int]
                ├─> action_logprobs: List[List[...]]
                └─> action_entropies: List[float]
                    │
                    └─> _calculate_entropy_from_logprobs()
                        │
                        └─> entropy_H_X (첫 번째 값)
```

### 4. CSV 로깅 필드

**추가된 필드**:
- `entropy_H_X`: H(X) 값
- `entropy_H_X_given_S`: H(X|S) 값
- `entropy_H_X_given_LS`: H(X|L,S) 값
- `trust_T`: Trust 값

**형식**:
- Entropy 값이 None인 경우: 빈 문자열("")
- Trust 값이 NaN인 경우: 빈 문자열("")
- 그 외: 문자열로 변환된 숫자 값

### 5. 디버그 출력

**활성화 조건**: `DEBUG=True` (global_variables.py)

**출력 내용**:
- 각 VLM 호출의 action 결과
- Action logprobs 정보 (토큰, 위치, entropy, top logprobs)
- Entropy 비교 결과 (H(X), H(X|S), H(X|L,S), Trust)

---

## 사용 예시

### 실행 방법

```bash
cd src/
python scenario2_test_entropy_comparison.py config/scenario135_example_map.json
```

### 출력 예시

```
[Entropy Comparison] Performing 3 VLM calls...

[H(X)] Sending VLM request (no Language Instruction, no Grounding)...
[H(X)] Action: ["move up"]
[H(X)] Action logprobs info:
  - Action 1 token: 'move' (pos 15)
    entropy: 2.3456

[H(X|S)] Sending VLM request (no Language Instruction, with Grounding)...
[H(X|S)] Action: ["move up"]
[H(X|S)] Action logprobs info:
  - Action 1 token: 'move' (pos 15)
    entropy: 1.8765

[H(X|L,S)] Sending VLM request (with Language Instruction, with Grounding)...
[H(X|L,S)] Action: ["move up"]
[H(X|L,S)] Action logprobs info:
  - Action 1 token: 'move' (pos 15)
    entropy: 0.5432

[Entropy Comparison Results]
H(X):           2.3456
H(X|S):         1.8765
H(X|L,S):       0.5432
Trust T:        0.2604
```

### CSV 로그 예시

```csv
step,timestamp,agent_x,agent_y,...,entropy_H_X,entropy_H_X_given_S,entropy_H_X_given_LS,trust_T
1,2025-01-21T16:51:19,5,5,...,2.3456,1.8765,0.5432,0.2604
2,2025-01-21T16:51:25,5,6,...,2.1234,1.6543,0.4321,0.2789
```

---

## 참고사항

### 1. Entropy 값의 범위

- **이론적 범위**: [0, log₂(vocab_size)]
- **실제 범위**: 보통 [0, 5] 정도 (top-k=5 사용 시)
- **의미**:
  - 0에 가까울수록: 확실한 결정 (낮은 불확실성)
  - 값이 클수록: 불확실한 결정 (높은 불확실성)

### 2. Trust 값 해석

- **T ≈ 0**: Grounding이 거의 도움이 되지 않음
- **T ≈ 0.5**: Grounding과 Language Instruction이 비슷한 효과
- **T ≈ 1**: Grounding만으로도 충분한 정보 제공

### 3. 제한사항

1. **Top-k 근사**: 전체 vocabulary를 고려하지 않고 top-k만 사용하여 근사
2. **첫 번째 Action만 사용**: Action이 여러 토큰으로 구성되어도 첫 번째만 사용
3. **동시 호출**: 3개의 VLM 호출이 순차적으로 실행됨 (진짜 동시는 아님)

### 4. 향후 개선 방향

1. **전체 Action 토큰 고려**: 평균 entropy 또는 가중 평균 사용
2. **병렬 처리**: 3개 VLM 호출을 실제로 병렬로 실행
3. **더 정확한 Entropy**: 전체 vocabulary 확률 분포 추정 방법 개선

---

## Verbalized Entropy 방식 (Tian et al. 2023 기반) ⭐ **신규**

### 개요

Verbalized Entropy는 VLM이 직접 출력하는 확률 분포를 사용하여 Entropy를 계산하는 방식입니다. RLHF(Reinforcement Learning from Human Feedback) 모델에서는 내부 logprobs보다 텍스트로 출력하는 확률이 더 정확하게 교정(calibrated)되어 있습니다.

### 주요 특징

1. **Verbalized Confidence**: VLM이 JSON 응답에 직접 확률 값을 출력
2. **Step-wise 확률 분포**: 각 step(step1, step2, step3)별로 4방향(north/south/west/east)에 대한 확률 분포
3. **Action 추출**: 각 step의 확률 분포에서 argmax로 action 추출
4. **가중 평균 Entropy**: Step별 Entropy를 가중 평균 (50/30/20)

### JSON 출력 형식

```json
{
  "executability": 0.95,
  "step1": {"north": 0.65, "south": 0.15, "west": 0.12, "east": 0.08},
  "step2": {"north": 0.45, "south": 0.30, "west": 0.15, "east": 0.10},
  "step3": {"north": 0.40, "south": 0.35, "west": 0.15, "east": 0.10},
  "reasoning": "Brief explanation",
  "memory": {
    "task_process": {"status": "in_progress"},
    "previous_action": "move up"
  }
}
```

### Entropy 계산

#### Step별 Entropy

각 step의 확률 분포로 Shannon Entropy 계산:

```python
H_step = -Σ p_i × log₂(p_i)
```

여기서 `p_i`는 각 방향(north/south/west/east)의 확률입니다.

#### 가중 평균 Entropy

```python
H_weighted = 0.5 × H_step1 + 0.3 × H_step2 + 0.2 × H_step3
```

### Trust 계산

Logprobs 기반과 동일한 공식 사용:

```
T = (H(X) - H(X|S)) / (H(X) - H(X|L,S))
```

### 실행 방법

```bash
cd src
python scenario2_test_entropy_comparison_refined_entropy.py config/example_map.json
```

### 설정

`src/utils/miscellaneous/global_variables.py`:

```python
USE_VERBALIZED_ENTROPY = True  # Verbalized Entropy 방식 사용
LOGPROBS_ENABLED = False       # 자동으로 False로 설정됨
VLM_MODEL = "gemini-2.5-flash" # RLHF 모델 권장
```

### 장점

1. **교정된 확률**: RLHF 모델의 텍스트 출력 확률이 더 정확
2. **모델 독립적**: logprobs 기능이 없는 모델에서도 사용 가능
3. **명시적 확률**: VLM이 직접 확률을 출력하므로 해석이 용이

### 제한사항

1. **JSON 파싱 의존**: VLM이 올바른 JSON 형식으로 출력해야 함
2. **재시도 필요**: JSON 파싱 실패 시 자동 재시도 (최대 3회)
3. **확률 정규화**: 확률 합이 1.0이 아니면 자동 정규화

---

## 관련 파일

- `src/scenario2_test_entropy_comparison.py`: Logprobs 기반 Entropy 비교 실험
- `src/scenario2_test_entropy_comparison_refined_entropy.py`: Verbalized Entropy 비교 실험 ⭐ **신규**
- `src/utils/miscellaneous/scenario_runner.py`: ScenarioExperiment 클래스
- `src/utils/vlm/vlm_wrapper.py`: Entropy 계산 로직 (Logprobs 기반)
- `src/utils/vlm/vlm_postprocessor.py`: Action logprobs 추출 및 Verbalized Entropy 파싱
- `src/utils/miscellaneous/global_variables.py`: 전역 설정 (USE_VERBALIZED_ENTROPY, LOGPROBS_ENABLED, DEBUG 등)

---

**작성일**: 2025-01-21  
**최종 수정일**: 2026-01-27
