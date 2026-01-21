# LLM API 문서

이 디렉토리에는 다양한 LLM (Large Language Model) API 사용 가이드가 포함되어 있습니다.

## 문서 목록

### API 설정
- [API Key 생성 및 설정 가이드](./api-key-setup.md) - OpenAI, Gemini, Vertex AI API Key 설정 방법

### Gemini API
- [Gemini Thinking 기능 가이드](./gemini-thinking.md) - Gemini 2.5/3 시리즈의 Thinking 기능 사용법

## 개요

이 프로젝트는 다양한 LLM API를 지원하며, 각 API의 고유 기능을 활용할 수 있습니다.

### 지원하는 LLM

- **OpenAI**: GPT-4o, GPT-4o-mini, GPT-4 등
- **Google Gemini**: gemini-2.5-flash, gemini-1.5-pro, gemini-1.5-flash 등
- **Qwen**: Qwen2-VL 시리즈
- **Gemma**: Gemma 2 시리즈

자세한 내용은 [VLM 핸들러 시스템 가이드](../vlm-handlers.md)를 참고하세요.

## 빠른 시작

### VLMWrapper 사용

```python
from utils import VLMWrapper

# Gemini 2.5 Flash with Thinking
wrapper = VLMWrapper(
    model="gemini-2.5-flash",
    thinking_budget=0  # Thinking 비활성화
)

response = wrapper.generate(
    system_prompt="You are a helpful assistant.",
    user_prompt="간단한 질문에 답해주세요"
)
```

## 관련 문서

- [VLM 핸들러 시스템 가이드](../vlm-handlers.md) - 다양한 VLM 모델 사용하기
- [Wrapper API](../wrapper-api.md) - Wrapper 클래스 API

