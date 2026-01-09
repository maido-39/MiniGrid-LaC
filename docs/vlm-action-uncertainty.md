# VLM Action Uncertainty Estimation

OpenAI API의 `logprobs` 기능을 활용하여 VLM이 예측한 action의 불확실도를 측정하고 시각화하는 모듈입니다.

## 개요

`vlm_action_uncertainty_estim.py`는 VLM(Vision Language Model)이 MiniGrid 환경에서 action을 예측할 때의 불확실도를 정량적으로 측정합니다. 이를 통해:

- **Action 예측의 신뢰도 평가**: VLM이 선택한 action이 얼마나 확실한지 측정
- **프롬프트 효과 분석**: 다양한 언어 명령이 action 불확실성에 미치는 영향 분석
- **대안 action 후보 확인**: VLM이 고려한 다른 action 후보들과 그 확률 확인

## 주요 클래스

### UncertaintyVLMWrapper

기존 `ChatGPT4oVLMWrapper`를 확장하여 `logprobs` 정보를 얻을 수 있게 하는 클래스입니다.

```python
from vlm_action_uncertainty_estim import UncertaintyVLMWrapper

wrapper = UncertaintyVLMWrapper(
    model="gpt-4o",
    logprobs=True,
    top_logprobs=5  # 상위 5개 토큰의 확률 반환
)

response_text, logprobs_info = wrapper.generate_with_logprobs(
    image=image,
    system_prompt="...",
    user_prompt="..."
)
```

**주요 메서드:**
- `generate_with_logprobs()`: logprobs 정보와 함께 응답 생성

### UncertaintyController

`MiniGridVLMController`를 확장하여 불확실도 측정 기능을 추가한 클래스입니다.

```python
from vlm_action_uncertainty_estim import UncertaintyController, create_scenario2_environment

env = create_scenario2_environment()
env.reset()

controller = UncertaintyController(
    env=env,
    model="gpt-4o",
    logprobs=True,
    top_logprobs=5
)
```

**주요 메서드:**

#### `generate_action_with_uncertainty()`

불확실도 정보와 함께 action을 생성합니다.

```python
response = controller.generate_action_with_uncertainty(
    user_prompt="Go to the blue pillar",
    mission="파란 기둥으로 가시오"
)

print(f"Action: {response['action']}")
print(f"Entropy: {response['uncertainty']['action_entropy']}")
print(f"Action Probability: {response['uncertainty']['action_prob']}")
print(f"Top Candidates: {response['uncertainty']['top_candidates']}")
```

**반환값:**
```python
{
    'action': str,                    # 선택된 action
    'environment_info': str,          # 환경 정보
    'reasoning': str,                 # 추론 과정
    'uncertainty': {
        'action_entropy': float,      # 엔트로피 (불확실도, 높을수록 불확실)
        'action_prob': float,         # 선택된 action의 확률
        'top_candidates': [           # 상위 후보 action들
            {'action': str, 'probability': float},
            ...
        ],
        'token_logprobs': [           # action 토큰들의 logprob
            {
                'token': str,
                'logprob': float,
                'top_alternatives': [...]
            },
            ...
        ]
    }
}
```

#### `analyze_prompt_uncertainty()`

여러 프롬프트에 대해 불확실도를 분석합니다.

```python
test_prompts = [
    "Go to the blue pillar",
    "Move towards the blue pillar",
    "Navigate to the blue pillar located in the center"
]

results = controller.analyze_prompt_uncertainty(
    prompts=test_prompts,
    mission="파란 기둥으로 가시오",
    save_dir="uncertainty_plots"  # 시각화 저장 디렉토리
)

for result in results:
    print(f"Prompt: {result['prompt']}")
    print(f"  Entropy: {result['entropy']}")
    print(f"  Action: {result['action']}")
    print(f"  Action Prob: {result['action_prob']}")
```

#### `visualize_uncertainty()`

단일 프롬프트에 대한 불확실도 정보를 시각화합니다.

```python
response = controller.generate_action_with_uncertainty(...)
controller.visualize_uncertainty(
    uncertainty_info=response['uncertainty'],
    save_path="uncertainty_plot.png"
)
```

**시각화 내용:**
- 상위 후보 action들의 확률 (막대 그래프)
- 토큰별 log probability (막대 그래프)

#### `visualize_prompt_comparison()`

여러 프롬프트에 대한 불확실도를 비교 시각화합니다.

```python
results = controller.analyze_prompt_uncertainty(...)
controller.visualize_prompt_comparison(
    analysis_results=results,
    save_path="prompt_comparison.png"
)
```

**시각화 내용:**
- 프롬프트별 엔트로피 (불확실도) 비교
- 프롬프트별 선택된 action의 확률 비교

## 사용 예제

### 기본 사용법

```python
from vlm_action_uncertainty_estim import (
    UncertaintyController,
    create_scenario2_environment
)

# 환경 생성
env = create_scenario2_environment()
env.reset()

# 컨트롤러 생성
controller = UncertaintyController(
    env=env,
    model="gpt-4o",
    logprobs=True,
    top_logprobs=5
)

# 단일 프롬프트 분석
response = controller.generate_action_with_uncertainty(
    user_prompt="Go to the blue pillar"
)

print(f"Selected Action: {response['action']}")
print(f"Uncertainty (Entropy): {response['uncertainty']['action_entropy']:.4f}")
print(f"Action Probability: {response['uncertainty']['action_prob']:.4f}")

# 상위 후보 확인
print("\nTop 5 Action Candidates:")
for i, cand in enumerate(response['uncertainty']['top_candidates'][:5]):
    print(f"  {i+1}. {cand['action']}: {cand['probability']:.4f}")

# 시각화
controller.visualize_uncertainty(
    response['uncertainty'],
    save_path="uncertainty_single.png"
)
```

### 프롬프트 비교 분석

```python
# 다양한 프롬프트 테스트
test_prompts = [
    "Go to the blue pillar",
    "Move towards the blue pillar",
    "Head to the blue pillar and turn right",
    "Navigate to the blue pillar located in the center",
    "Move up and right to reach the blue pillar"
]

mission = "파란 기둥으로 가서 오른쪽으로 돌고, 테이블 옆에 멈추시오"

# 분석 실행
results = controller.analyze_prompt_uncertainty(
    prompts=test_prompts,
    mission=mission,
    save_dir="uncertainty_plots"
)

# 결과 출력
print("\n" + "=" * 60)
print("프롬프트별 불확실도 분석 결과")
print("=" * 60)
for i, result in enumerate(results):
    print(f"\n프롬프트 {i+1}: {result['prompt']}")
    print(f"  Action: {result['action']}")
    if result['entropy'] is not None:
        print(f"  Entropy: {result['entropy']:.4f}")
        print(f"  Action Probability: {result['action_prob']:.4f}")
    else:
        print(f"  Entropy: N/A (logprobs not available)")

# 비교 시각화
controller.visualize_prompt_comparison(
    results,
    save_path="uncertainty_plots/prompt_comparison.png"
)
```

## 불확실도 지표 설명

### Entropy (엔트로피)

엔트로피는 action 예측의 불확실도를 나타내는 지표입니다.

- **낮은 엔트로피 (0에 가까움)**: VLM이 특정 action을 매우 확실하게 예측
- **높은 엔트로피**: 여러 action 후보들이 비슷한 확률을 가져 불확실함

수식:
```
Entropy = -Σ P(action_i) * log(P(action_i))
```

### Action Probability

선택된 action의 확률입니다. 1에 가까울수록 해당 action이 확실합니다.

### Top Candidates

VLM이 고려한 상위 action 후보들과 각각의 확률입니다.

## 주의사항

### OpenAI API logprobs 지원

OpenAI API의 `chat.completions`에서 `logprobs` 지원 여부는 모델과 API 버전에 따라 다를 수 있습니다.

- **지원되는 경우**: `logprobs_info`가 반환되어 불확실도 계산 가능
- **지원되지 않는 경우**: `logprobs_info`가 `None`으로 반환되고, 불확실도 계산은 건너뜀

현재 확인된 지원 모델:
- `gpt-4o` (일부 버전)
- `gpt-4-turbo` (일부 버전)

### 성능 고려사항

- `logprobs=True`로 설정하면 API 응답 시간이 증가할 수 있습니다
- `top_logprobs` 값을 크게 설정하면 더 많은 후보를 확인할 수 있지만, API 비용이 증가합니다

## 파일 구조

```
vlm_action_uncertainty_estim.py
├── UncertaintyVLMWrapper          # logprobs 지원 VLM Wrapper
├── UncertaintyController          # 불확실도 측정 Controller
│   ├── generate_action_with_uncertainty()
│   ├── analyze_prompt_uncertainty()
│   ├── visualize_uncertainty()
│   └── visualize_prompt_comparison()
└── create_scenario2_environment() # 환경 생성 헬퍼 함수
```

## 의존성

```python
# 필수
openai              # OpenAI API 클라이언트
numpy               # 수치 계산
matplotlib          # 시각화
minigrid_vlm_controller  # 기본 Controller
minigrid_customenv_emoji # 환경 Wrapper
vlm_wrapper         # VLM Wrapper
vlm_postprocessor   # 응답 후처리

# 선택적
python-dotenv       # 환경 변수 관리
```

## 향후 개선 방향

1. **더 정확한 토큰 매칭**: action 부분의 토큰을 더 정확하게 추출하는 알고리즘 개선
2. **다양한 불확실도 지표**: 엔트로피 외에 다른 불확실도 지표 추가 (예: Gini coefficient)
3. **실시간 모니터링**: 대화형 모드에서 실시간으로 불확실도 표시
4. **프롬프트 최적화**: 불확실도를 최소화하는 프롬프트 자동 생성

## 참고 자료

- [OpenAI API Documentation - Logprobs](https://platform.openai.com/docs/api-reference/chat/create#chat-create-logprobs)
- [Information Theory - Entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory))

