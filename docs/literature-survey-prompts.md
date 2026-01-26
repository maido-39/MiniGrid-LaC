# Literature Survey Prompts: LLM Output Entropy Estimation

## 연구 배경 및 목적

현재 VLM(Vision-Language Model)의 action 생성에서 불확실성을 정량화하기 위해 Shannon Entropy와 Trust 지표를 사용하고 있습니다. 그러나 LLM API가 제공하는 logprobs의 한계로 인해 정확한 entropy 계산이 어렵고, Trust 값이 불안정한 문제가 발생하고 있습니다.

### 현재 시스템

```
Trust T = (H(X) - H(X|S)) / (H(X) - H(X|L,S))

- H(X): 이미지만 주어졌을 때의 action entropy (최대 불확실성)
- H(X|S): 이미지 + Spatial Grounding (지도/위치 정보)
- H(X|L,S): 이미지 + Language Instruction + Grounding (최소 불확실성)
```

### 발견된 문제점

1. **Top-k logprobs 한계**: OpenAI/Gemini API는 top-5~20개의 토큰 확률만 제공
2. **Entropy 추정 부정확**: 나머지 vocabulary의 확률 질량 처리 문제
3. **수치적 불안정성**: 분모(H(X) - H(X|L,S)) ≈ 0일 때 Trust 발산
4. **간헐적 이상 값**: Logprobs에서 entropy > 1.0 같은 비정상 값 발생
5. **단일 토큰 사용**: 첫 번째 action 토큰만 사용하여 전체 sequence entropy 미반영

---

## Prompt 1: LLM Entropy 추정 방법론 조사

### English (Academic Search)

```
I'm researching methods to accurately estimate the output entropy of Large Language Models (LLMs) when only top-k logprobs are available from the API.

Context:
- We're using VLM (Vision-Language Model) for robot action generation in a grid-world environment
- The API (Gemini/GPT) provides only top-5 to top-20 token logprobs per position
- We calculate Shannon entropy H(X) = -Σ p(x) log₂ p(x) from these limited logprobs
- Current approach: Add remaining probability mass (1 - Σtop-k probs) as a single "other" category

Questions:
1. What are established methods for estimating full vocabulary entropy from limited top-k samples?
2. How do researchers handle the "long tail" of token probabilities in entropy estimation?
3. Are there correction factors or bounds for top-k entropy approximation?
4. What are alternative uncertainty quantification methods for LLM outputs that don't rely on full logprobs?

Relevant papers I should look for:
- Entropy estimation in neural language models
- Uncertainty quantification in LLMs
- Calibration of language model outputs
- Information-theoretic analysis of transformer outputs
```

### Korean (Search Keywords)

```
검색 키워드:
- "LLM entropy estimation top-k logprobs"
- "language model uncertainty quantification"
- "transformer output calibration"
- "neural network predictive entropy"
- "sequence model uncertainty bounds"

한국어 검색:
- "대규모 언어모델 불확실성 정량화"
- "언어모델 출력 엔트로피 추정"
- "자연어처리 확률 보정"
```

---

## Prompt 2: Trust/Mutual Information 계산 안정성

### English (Academic Search)

```
I'm working on a trust metric for VLM-based robot control that measures how much "grounding information" reduces action uncertainty compared to "language instruction".

Current formula:
T = (H(X) - H(X|S)) / (H(X) - H(X|L,S))

Where:
- H(X): Entropy without any context
- H(X|S): Entropy with spatial grounding (scene understanding)
- H(X|L,S): Entropy with both language instruction and grounding

Problems encountered:
1. Denominator approaches zero when language instruction provides minimal benefit
2. Trust values range from -170 to +170 instead of expected [0,1]
3. Negative Trust occurs when H(X|S) > H(X), violating entropy chain rule assumptions

Questions:
1. How is this type of "information contribution ratio" handled in information theory literature?
2. Are there numerically stable alternatives to this ratio formulation?
3. Should we use mutual information I(X;S) and I(X;L,S) instead?
4. What regularization or thresholding techniques prevent division-by-near-zero?

Related concepts to search:
- Mutual information estimation
- Conditional entropy in neural networks
- Information bottleneck theory
- Normalized information gain
```

### Alternative Formulations to Consider

```
1. Mutual Information approach:
   T = I(X; S) / I(X; L, S)
   where I(X; S) = H(X) - H(X|S)

2. Log-ratio approach:
   T = log(H(X) / H(X|S)) / log(H(X) / H(X|L,S))

3. Clipped/bounded approach:
   T = clip((H(X) - H(X|S)) / max(H(X) - H(X|L,S), ε), -10, 10)

4. KL-Divergence based:
   Compare distributions instead of entropies
```

---

## Prompt 3: 선행 연구 - LLM 기반 로봇 제어의 불확실성

### English (Robotics + LLM Literature)

```
Survey request: How do prior works measure and utilize LLM/VLM uncertainty in robot decision-making?

Context:
- Using VLM (GPT-4V, Gemini) for high-level robot action planning
- Need to quantify "how confident" the model is in its action choice
- Want to compare effect of different context (grounding vs. language instruction)

Topics to investigate:
1. How do "SayCan", "PaLM-E", "RT-2" and similar works handle action uncertainty?
2. Do they use entropy, confidence scores, or other uncertainty metrics?
3. How is uncertainty used for human intervention or replanning?
4. Are there established benchmarks for VLM uncertainty in embodied AI?

Specific questions:
1. What uncertainty metrics have been validated for VLM action generation?
2. How do multi-modal models (image + text) affect entropy estimation?
3. Are there domain-specific calibration methods for robot control LLMs?
```

### Key Papers to Search

```
1. "Language Models as Zero-Shot Planners" (Huang et al., 2022)
2. "SayCan" - Do As I Can, Not As I Say (Google, 2022)
3. "PaLM-E" - An Embodied Multimodal Language Model (Google, 2023)
4. "RT-2" - Vision-Language-Action Models (Google DeepMind, 2023)
5. "Code as Policies" (Liang et al., 2022)
6. "VoxPoser" (Huang et al., 2023)
7. Uncertainty quantification in LLM planning (various 2023-2024)

Search queries:
- "VLM robot uncertainty quantification"
- "LLM planning confidence calibration"
- "embodied AI action entropy"
- "language grounded robot decision uncertainty"
```

---

## Prompt 4: Entropy Estimation 기법 비교

### English (Technical Deep-Dive)

```
I need to compare entropy estimation techniques for language model outputs when only partial probability information is available.

Setup:
- Vocabulary size V ≈ 32,000 (typical LLM tokenizer)
- API provides top-k (k=5 to 20) logprobs at each position
- Ground truth full distribution is unavailable
- Need per-token entropy estimates for action uncertainty

Methods to investigate:

1. **Naive top-k**: H = -Σ_{i=1}^k p_i log(p_i)
   - Underestimates entropy (ignores tail)

2. **Renormalized top-k**: Normalize top-k to sum to 1
   - Biased but common practice

3. **Residual mass**: Add (1 - Σtop-k) as single pseudo-token
   - Our current approach
   - Underestimates when tail is spread across many tokens

4. **Maximum entropy principle**: Assume uniform over remaining V-k tokens
   - p_other = (1 - Σtop-k) / (V - k)
   - Conservative upper bound

5. **Zipf/power-law tail**: Model tail distribution
   - Based on empirical token frequency distributions

6. **Temperature scaling**: Calibrate logits before softmax
   - Can improve probability estimates

Questions:
1. Which method is most appropriate for action selection entropy?
2. Are there theoretical guarantees for any of these estimators?
3. How does choice of k affect estimation accuracy?
4. Should we average entropy over multiple tokens or use joint entropy?
```

---

## Prompt 5: 실용적 해결책 탐색

### English (Engineering-Focused)

```
I need practical solutions for the following LLM entropy estimation problems in a robotics application:

Problem 1: Numerical instability in Trust calculation
- Trust T = (H(X) - H(X|S)) / (H(X) - H(X|L,S))
- When denominator ≈ 0, T explodes to ±infinity
- Need stable alternative that maintains interpretability

Problem 2: Inconsistent logprobs from API
- Sometimes get entropy > 1.0 bit (normally expect < 0.01 for confident predictions)
- Possible causes: API bugs, tokenization issues, or genuinely uncertain outputs
- Need robust filtering or validation

Problem 3: High null rate in entropy values
- ~40% of steps have missing entropy due to failed VLM calls
- Even with retry logic, some fail
- Need to handle incomplete data for analysis

Problem 4: Single-token vs multi-token actions
- Actions like "move east" are 2+ tokens
- Currently only use first token's entropy
- Should we use: max, mean, product, or joint entropy?

Practical solutions I'm considering:
1. Epsilon-clamping: T = ... / max(denom, 0.001)
2. Softplus denominator: T = ... / log(1 + exp(denom))
3. Sigmoid transformation: T = sigmoid(raw_T) to bound [0,1]
4. Anomaly detection: Flag entropy > threshold as invalid
5. Imputation: Fill missing values with episode mean/median

What do practitioners recommend?
```

---

## Prompt 6: 종합 문헌 조사 요청

### Korean (Claude/GPT 대화용)

```
LLM/VLM의 출력 불확실성(entropy)을 정확하게 추정하는 방법에 대해 문헌 조사를 도와주세요.

## 연구 맥락

저는 VLM(Vision-Language Model)을 사용하여 grid-world 환경에서 로봇 액션을 생성하는 연구를 진행 중입니다. 
VLM이 얼마나 "확신"을 가지고 액션을 선택하는지 정량화하기 위해 Shannon Entropy를 사용하고 있습니다.

## 현재 문제점

1. **API 제한**: Gemini/GPT API는 각 토큰 위치에서 상위 5~20개의 logprobs만 제공합니다. 
   전체 vocabulary (32,000+ 토큰)의 확률 분포를 알 수 없어 정확한 entropy 계산이 불가능합니다.

2. **Trust 지표 불안정**: 
   - 공식: T = (H(X) - H(X|S)) / (H(X) - H(X|L,S))
   - 분모가 0에 가까워지면 Trust가 ±170까지 발산
   - 예상 범위 [0,1]을 크게 벗어남

3. **비정상 값 발생**:
   - 가끔 entropy > 1.0 같은 비정상적으로 높은 값 발생
   - 원인 불명 (API 버그? 토큰화 문제? 실제 불확실성?)

## 조사가 필요한 주제

1. **Top-k logprobs에서 전체 entropy 추정 방법**
   - Naive top-k, renormalization, residual mass, maximum entropy principle 등
   - 각 방법의 bias와 variance 특성

2. **LLM 출력 불확실성 정량화 선행 연구**
   - Predictive entropy, mutual information
   - Calibration 기법 (temperature scaling 등)
   - Uncertainty-aware LLM 논문들

3. **비율 기반 지표의 수치적 안정성**
   - Information gain ratio의 안정화 기법
   - 분모가 0에 가까울 때 처리 방법
   - 대안적 공식 (log-ratio, KL-divergence 등)

4. **로봇/Embodied AI에서의 LLM 불확실성 활용**
   - SayCan, PaLM-E, RT-2 등에서 불확실성 처리 방법
   - Human-in-the-loop 결정을 위한 confidence threshold

## 찾고 싶은 논문/자료

- "Semantic Uncertainty" (Kuhn et al., 2023) - LLM uncertainty 측정
- "Language Model Calibration" 관련 연구
- "Conformal Prediction for LLM" 최신 연구
- "Information-theoretic analysis of transformers"
- Robotics + LLM uncertainty 관련 workshop/survey 논문

## 질문

1. 위 주제들에 대한 핵심 논문 추천
2. 각 방법의 장단점 비교
3. 우리 상황(로봇 제어, 실시간 결정)에 가장 적합한 접근법
4. 구현 시 주의사항 및 best practice
```

---

## 추천 검색 전략

### Google Scholar 검색어

```
1. "LLM uncertainty quantification"
2. "language model entropy estimation"
3. "top-k logprobs entropy"
4. "transformer output calibration"
5. "VLM robot planning uncertainty"
6. "predictive entropy neural networks"
7. "information theoretic deep learning"
8. "conformal prediction language models"
```

### arXiv 검색어

```
cs.CL + cs.LG:
- "uncertainty quantification large language models"
- "entropy estimation autoregressive"
- "calibration language generation"

cs.RO + cs.AI:
- "VLM robot control uncertainty"
- "language grounded planning confidence"
```

### 주요 Conference/Workshop

```
- NeurIPS: Uncertainty & Robustness in Deep Learning Workshop
- ICML: Reliable Machine Learning Workshop
- ICLR: LLM Reasoning Workshop
- CoRL: Robot Learning Conference
- RSS: Robotics Science and Systems
- IROS/ICRA: Robotics conferences
```

---

## 예상 핵심 논문 (검색 시작점)

1. **Semantic Uncertainty** (Kuhn et al., NeurIPS 2023)
   - LLM의 의미론적 불확실성 측정

2. **Calibrating Language Models** (Jiang et al., 2021)
   - LM 출력의 확률 보정

3. **Teaching Models to Express Uncertainty** (Lin et al., 2022)
   - 불확실성 표현 학습

4. **Conformal Language Modeling** (Quach et al., 2023)
   - Conformal prediction을 LLM에 적용

5. **SayCan** (Ahn et al., 2022)
   - Affordance와 LLM 결합 (불확실성 처리 확인)

6. **Inner Monologue** (Huang et al., 2022)
   - Embodied reasoning과 feedback

---

**문서 작성일**: 2026-01-25
**목적**: Trust 계산 개선을 위한 문헌조사 가이드
