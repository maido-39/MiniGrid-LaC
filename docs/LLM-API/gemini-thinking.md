# Gemini Thinking 기능 가이드

Gemini 3 및 2.5 시리즈 모델은 추론 및 다단계 계획 능력을 크게 향상시키는 내부 '사고 과정'을 사용하므로 코딩, 고급 수학, 데이터 분석과 같은 복잡한 작업에 매우 효과적입니다.

이 문서는 Gemini API를 사용하여 Gemini의 사고 능력을 활용하는 방법을 설명합니다.

**참고**: 이 문서는 [Gemini API 공식 문서](https://ai.google.dev/gemini-api/docs/thinking?hl=ko)를 기반으로 작성되었습니다.

## 목차

- [Thinking을 통한 콘텐츠 생성](#thinking을-통한-콘텐츠-생성)
- [Thinking Budget 설정](#thinking-budget-설정)
- [Thinking 토큰 및 비용](#thinking-토큰-및-비용)
- [권장사항](#권장사항)
- [지원되는 모델](#지원되는-모델)
- [프로젝트에서 사용하기](#프로젝트에서-사용하기)

## Thinking을 통한 콘텐츠 생성

사고 모델을 사용한 요청 시작은 다른 콘텐츠 생성 요청과 유사합니다. 주요 차이점은 `model` 필드에서 사고 지원 모델 중 하나를 지정하는 것입니다.

### Python 예제

```python
from google import genai

client = genai.Client()
prompt = "Explain the concept of Occam's Razor and provide a simple, everyday example."
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=prompt
)

print(response.text)
```

## Thinking Budget 설정

Thinking 기능을 제어하기 위해 `thinking_budget` 파라미터를 사용할 수 있습니다.

### Thinking Budget 옵션

- **None (기본값)**: 기본 thinking 사용 (gemini-2.5-flash의 경우 기본적으로 활성화됨)
- **0**: Thinking 기능 비활성화 (빠르고 저렴)
- **양의 정수**: Thinking에 사용할 최대 토큰 수 설정

### 프로젝트에서 사용하기

이 프로젝트의 `VLMWrapper`와 `GeminiHandler`는 `thinking_budget` 파라미터를 지원합니다:

```python
from utils import VLMWrapper

# Thinking 기능 비활성화 (빠르고 저렴)
wrapper = VLMWrapper(
    model="gemini-2.5-flash",
    thinking_budget=0  # thinking 기능 끄기
)

# Thinking 기능 활성화 (기본값)
wrapper = VLMWrapper(
    model="gemini-2.5-flash",
    thinking_budget=None  # 기본 thinking 사용
)

# Thinking budget 설정 (특정 토큰 수로 제한)
wrapper = VLMWrapper(
    model="gemini-2.5-flash",
    thinking_budget=1000  # thinking에 최대 1000 토큰 사용
)
```

### 직접 Handler 사용하기

```python
from utils.vlm.handlers.gemini_handler import GeminiHandler

handler = GeminiHandler(
    model="gemini-2.5-flash",
    thinking_budget=0  # thinking 비활성화
)

response = handler.generate(
    user_prompt="복잡한 수학 문제를 풀어주세요"
)
```

### 검증 및 제한사항

- `thinking_budget`은 **gemini-2.5-flash 모델에서만** 지원됩니다
- 다른 모델에서 `thinking_budget`을 설정하면 `ValueError`가 발생합니다
- `thinking_budget`은 음수가 될 수 없습니다

```python
# ❌ 에러 발생: gemini-1.5-flash는 thinking_budget을 지원하지 않음
try:
    wrapper = VLMWrapper(
        model="gemini-1.5-flash",
        thinking_budget=0
    )
except ValueError as e:
    print(e)  # "thinking_budget is only supported for gemini-2.5-flash model."
```

## Thinking 토큰 및 비용

### 토큰 사용량 확인

사고가 사용 설정된 경우 대답 가격은 출력 토큰과 사고 토큰의 합계입니다. 생성된 사고 토큰의 총수는 `usage_metadata`에서 확인할 수 있습니다.

```python
from utils import VLMWrapper

wrapper = VLMWrapper(model="gemini-2.5-flash")
response = wrapper.generate(
    user_prompt="복잡한 문제를 풀어주세요",
    debug=True  # 디버그 모드에서 토큰 정보 출력
)

# 디버그 출력에서 다음 정보를 확인할 수 있습니다:
# - Input Tokens: 입력 토큰 수
# - Output Tokens: 출력 토큰 수
# - Thinking Tokens: 사고 토큰 수 (사용 가능한 경우)
# - Thinking Content: 사고 내용 (사용 가능한 경우)
```

### 비용 고려사항

- 사고 모델은 최종 대답의 품질을 개선하기 위해 전체 사고를 생성한 다음 요약을 출력합니다
- API에서 요약만 출력되더라도 가격은 모델이 요약을 생성하기 위해 생성해야 하는 전체 사고 토큰을 기반으로 합니다
- Thinking을 비활성화(`thinking_budget=0`)하면 비용을 절감할 수 있습니다

## 권장사항

### 디버깅 및 조향

- **추론 검토**: 사고 모델에서 예상한 대답을 얻지 못하는 경우 Gemini의 사고 요약을 주의 깊게 분석하는 것이 도움이 될 수 있습니다. 작업을 어떻게 분류하고 결론에 도달했는지 확인하고 이 정보를 사용하여 올바른 결과를 얻을 수 있습니다.
- **추론에 관한 안내 제공**: 특히 긴 출력을 원하는 경우 프롬프트에서 모델이 사용하는 사고량을 제한하는 안내를 제공하는 것이 좋습니다. 이렇게 하면 대답에 더 많은 토큰 출력을 예약할 수 있습니다.

### 작업 복잡성에 따른 Thinking 사용

#### 간단한 작업 (사고를 사용하지 않음)

사실 검색이나 분류와 같이 복잡한 추론이 필요하지 않은 간단한 요청에는 사고가 필요하지 않습니다. `thinking_budget=0`으로 설정하여 비용을 절감할 수 있습니다.

예시:
- "DeepMind는 어디에서 설립되었어?"
- "이 이메일은 회의를 요청하는 건가요, 아니면 정보를 제공하는 건가요?"

```python
# 간단한 작업에는 thinking 비활성화
wrapper = VLMWrapper(
    model="gemini-2.5-flash",
    thinking_budget=0
)
```

#### 중간 작업 (기본값/약간의 사고)

많은 일반적인 요청은 단계별 처리 또는 더 깊은 이해가 필요합니다. Gemini는 다음과 같은 작업에 사고 능력을 유연하게 사용할 수 있습니다.

예시:
- "광합성과 성장을 비유해 줘."
- "전기 자동차와 하이브리드 자동차를 비교 및 대조하세요."

```python
# 중간 작업에는 기본 thinking 사용
wrapper = VLMWrapper(
    model="gemini-2.5-flash",
    thinking_budget=None  # 기본값
)
```

#### 어려운 작업 (최대 사고 능력)

복잡한 수학 문제 풀기나 코딩 작업과 같은 매우 복잡한 과제의 경우 높은 사고 예산을 설정하는 것이 좋습니다.

예시:
- "AIME 2025의 문제 1을 풀어 보세요. 17b가 97b의 약수가 되는 모든 정수 밑 b > 9의 합을 구하세요."
- "사용자 인증을 포함하여 실시간 주식 시장 데이터를 시각화하는 웹 애플리케이션용 Python 코드를 작성하세요."

```python
# 복잡한 작업에는 높은 thinking budget 설정
wrapper = VLMWrapper(
    model="gemini-2.5-flash",
    thinking_budget=2000  # 높은 thinking budget
)
```

## 지원되는 모델

Thinking 기능은 모든 **Gemini 3 및 2.5 시리즈 모델**에서 지원됩니다.

### 현재 프로젝트에서 지원하는 모델

- `gemini-2.5-flash`: Thinking 기능 지원 (thinking_budget 설정 가능)
- `gemini-1.5-flash`: Thinking 기능 미지원
- `gemini-1.5-pro`: Thinking 기능 미지원
- 기타 모델: Thinking 기능 미지원

**참고**: 모델 개요 페이지에서 모든 모델 기능을 확인할 수 있습니다.

## 프로젝트에서 사용하기

### VLMWrapper 사용 예제

```python
from utils import VLMWrapper

# 1. Thinking 비활성화 (간단한 작업)
wrapper_fast = VLMWrapper(
    model="gemini-2.5-flash",
    thinking_budget=0,
    max_tokens=1000
)

# 2. 기본 Thinking 사용 (일반 작업)
wrapper_default = VLMWrapper(
    model="gemini-2.5-flash",
    thinking_budget=None,  # 또는 생략
    max_tokens=2000
)

# 3. 높은 Thinking Budget (복잡한 작업)
wrapper_complex = VLMWrapper(
    model="gemini-2.5-flash",
    thinking_budget=2000,
    max_tokens=4000
)

# 사용
response = wrapper_complex.generate(
    system_prompt="You are a helpful assistant.",
    user_prompt="복잡한 수학 문제를 단계별로 풀어주세요",
    debug=True  # 토큰 사용량 확인
)
```

### GeminiHandler 직접 사용 예제

```python
from utils.vlm.handlers.gemini_handler import GeminiHandler

# Thinking 비활성화
handler = GeminiHandler(
    model="gemini-2.5-flash",
    thinking_budget=0,
    max_tokens=1000
)

handler.initialize()

response, metadata = handler.generate(
    user_prompt="간단한 질문에 답해주세요",
    return_metadata=True
)

print(f"Response: {response}")
print(f"Thinking tokens: {metadata.get('thinking_tokens')}")
```

## 참고 자료

- [Gemini API 공식 문서 - Thinking](https://ai.google.dev/gemini-api/docs/thinking?hl=ko)
- [프로젝트 VLM 핸들러 가이드](../vlm-handlers.md)
- [프로젝트 VLM Wrapper API](../wrapper-api.md)

## 요약

- **Thinking 기능**: Gemini 2.5/3 시리즈 모델의 내부 사고 과정으로 복잡한 작업에 효과적
- **thinking_budget 파라미터**: 
  - `None`: 기본 thinking 사용 (기본값)
  - `0`: Thinking 비활성화 (빠르고 저렴)
  - 양의 정수: Thinking에 사용할 최대 토큰 수
- **지원 모델**: `gemini-2.5-flash`만 지원 (다른 모델에서는 에러 발생)
- **비용**: Thinking 토큰도 비용에 포함되므로, 간단한 작업에는 `thinking_budget=0` 권장

