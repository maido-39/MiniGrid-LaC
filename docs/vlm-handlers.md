# VLM 핸들러 시스템 가이드

이 문서는 다양한 Vision Language Model (VLM)을 효율적으로 사용하기 위한 핸들러 시스템에 대한 가이드입니다.

## 목차

1. [개요](#개요)
2. [지원하는 VLM 모델](#지원하는-vlm-모델)
3. [기본 사용법](#기본-사용법)
4. [모델 파라미터 크기 선택](#모델-파라미터-크기-선택)
5. [핸들러별 상세 가이드](#핸들러별-상세-가이드)
6. [테스트](#테스트)
7. [문제 해결](#문제-해결)

## 개요

VLM 핸들러 시스템은 다양한 Vision Language Model을 통일된 인터페이스로 사용할 수 있게 해주는 확장 가능한 구조입니다.

### 주요 특징

- **통일된 인터페이스**: 모든 VLM을 동일한 방식으로 사용
- **확장 가능**: 새로운 VLM 핸들러를 쉽게 추가 가능
- **호환성 유지**: 기존 `ChatGPT4oVLMWrapper`와 완전 호환
- **유연한 설정**: 모델, 파라미터 크기, API 타입 등 자유롭게 선택

### 아키텍처

```
vlm/
├── handlers/
│   ├── base.py          # 베이스 핸들러 클래스
│   ├── openai_handler.py    # OpenAI GPT-4o 핸들러
│   ├── qwen_handler.py      # Qwen VLM 핸들러
│   └── gemma_handler.py      # Gemma VLM 핸들러
├── vlm_manager.py       # 핸들러 관리자
└── __init__.py
```

## 지원하는 VLM 모델

### OpenAI

- **gpt-4o-mini**: 경량 모델 (빠르지만 정확도 낮음)
- **gpt-4o**: 중간 모델 (균형잡힌 성능, 기본값)
- **gpt-4-turbo**: 대형 모델 (느리지만 정확도 높음)
- **gpt-4**: 레거시 대형 모델
- **gpt-5**: 최신 대형 모델 (가능한 경우)

### Qwen (Alibaba)

- **Qwen2-VL-2B-Instruct**: 2B 파라미터 (경량)
- **Qwen2-VL-7B-Instruct**: 7B 파라미터 (중간)
- **Qwen2-VL-72B-Instruct**: 72B 파라미터 (대형)
- **Qwen2.5-VL-3B-Instruct**: 3B 파라미터
- **Qwen2.5-VL-7B-Instruct**: 7B 파라미터
- **Qwen2.5-VL-32B-Instruct**: 32B 파라미터 (기본값)

### Gemma (Google)

- **google/gemma-2-2b-it**: 2B 파라미터 (경량)
- **google/gemma-2-9b-it**: 9B 파라미터 (중간, 기본값)
- **google/gemma-2-27b-it**: 27B 파라미터 (대형)

**주의**: Gemma는 기본적으로 비전을 지원하지 않지만, 일부 파인튜닝된 버전이나 멀티모달 확장이 있을 수 있습니다.

## 기본 사용법

### 방법 1: 기존 ChatGPT4oVLMWrapper 사용 (호환성 유지)

```python
# Actual path: lib.vlm.vlm_wrapper
from lib import ChatGPT4oVLMWrapper

# 기존 방식 그대로 사용 가능
wrapper = ChatGPT4oVLMWrapper(
    model="gpt-4o",
    temperature=0.0,
    max_tokens=1000
)

response = wrapper.generate(
    image=image,
    system_prompt="You are a helpful assistant.",
    user_prompt="Describe what you see in this image."
)
```

### 방법 2: VLMManager 사용 (권장)

```python
from vlm import VLMManager

# Manager 생성
manager = VLMManager()

# 핸들러 생성 및 등록
manager.create_handler(
    handler_type="openai",
    name="my_vlm",
    model="gpt-4o",
    temperature=0.0,
    max_tokens=1000
)

# 사용
response = manager.generate(
    image=image,
    system_prompt="You are a helpful assistant.",
    user_prompt="Describe what you see in this image."
)
```

### 방법 3: 핸들러 직접 사용

```python
from vlm.handlers import OpenAIHandler, QwenHandler, GemmaHandler

# OpenAI 핸들러
handler = OpenAIHandler(model="gpt-4o")
handler.initialize()
response = handler.generate(image=image, system_prompt="...", user_prompt="...")

# Qwen 핸들러
handler = QwenHandler(model="Qwen2-VL-7B-Instruct", api_type="dashscope")
handler.initialize()
response = handler.generate(image=image, system_prompt="...", user_prompt="...")

# Gemma 핸들러
handler = GemmaHandler(model="google/gemma-2-9b-it")
handler.initialize()
response = handler.generate(image=image, system_prompt="...", user_prompt="...")
```

## 모델 파라미터 크기 선택

모델의 파라미터 크기에 따라 성능과 속도가 달라집니다. 용도에 맞는 모델을 선택하세요.

### 경량 모델 (2B-3B 파라미터)

**특징**:
- 빠른 응답 속도
- 낮은 메모리 사용량
- 상대적으로 낮은 정확도

**사용 예시**:
```python
# OpenAI
handler = OpenAIHandler(model="gpt-4o-mini", max_tokens=500)

# Qwen
handler = QwenHandler(model="Qwen2-VL-2B-Instruct", api_type="dashscope")

# Gemma
handler = GemmaHandler(model="google/gemma-2-2b-it")
```

**권장 용도**: 빠른 프로토타이핑, 간단한 작업, 리소스 제약 환경

### 중간 모델 (7B-9B 파라미터)

**특징**:
- 균형잡힌 성능과 속도
- 적당한 메모리 사용량
- 좋은 정확도

**사용 예시**:
```python
# OpenAI
handler = OpenAIHandler(model="gpt-4o", max_tokens=1000)

# Qwen
handler = QwenHandler(model="Qwen2-VL-7B-Instruct", api_type="dashscope")

# Gemma
handler = GemmaHandler(model="google/gemma-2-9b-it")
```

**권장 용도**: 일반적인 작업, 프로덕션 환경, 대부분의 사용 사례

### 대형 모델 (27B-72B 파라미터)

**특징**:
- 높은 정확도
- 느린 응답 속도
- 높은 메모리 사용량

**사용 예시**:
```python
# OpenAI
handler = OpenAIHandler(model="gpt-4-turbo", max_tokens=2000)

# Qwen
handler = QwenHandler(model="Qwen2-VL-72B-Instruct", api_type="dashscope")

# Gemma
handler = GemmaHandler(model="google/gemma-2-27b-it")
```

**권장 용도**: 복잡한 작업, 높은 정확도가 필요한 경우, 연구 목적

## 핸들러별 상세 가이드

### OpenAI 핸들러

#### 설치

```bash
pip install openai python-dotenv
```

#### 환경 변수 설정

`.env` 파일에 추가:
```
OPENAI_API_KEY=your-api-key-here
```

또는 환경 변수로 설정:
```bash
export OPENAI_API_KEY=your-api-key-here
```

#### 사용 예시

```python
from vlm.handlers import OpenAIHandler

handler = OpenAIHandler(
    model="gpt-4o",           # 모델 선택
    temperature=0.0,          # 생성 온도
    max_tokens=1000          # 최대 토큰 수
)
handler.initialize()

response = handler.generate(
    image="path/to/image.png",
    system_prompt="You are a helpful assistant.",
    user_prompt="Describe this image."
)
```

#### 모델별 권장 설정

| 모델 | max_tokens 권장값 | 속도 | 정확도 |
|------|------------------|------|--------|
| gpt-4o-mini | 500-1000 | 빠름 | 보통 |
| gpt-4o | 1000-2000 | 보통 | 좋음 |
| gpt-4-turbo | 2000-4000 | 느림 | 매우 좋음 |

### Qwen 핸들러

#### 설치

**DashScope API 사용 시**:
```bash
pip install dashscope python-dotenv
```

**Hugging Face 사용 시**:
```bash
pip install transformers torch python-dotenv
```

#### 환경 변수 설정 (DashScope)

`.env` 파일에 추가:
```
DASHSCOPE_API_KEY=your-api-key-here
```

#### 사용 예시

```python
from vlm.handlers import QwenHandler

# DashScope API 사용
handler = QwenHandler(
    model="Qwen2-VL-7B-Instruct",
    api_type="dashscope",
    temperature=0.0,
    max_tokens=1000
)
handler.initialize()

response = handler.generate(
    image="path/to/image.png",
    system_prompt="You are a helpful assistant.",
    user_prompt="Describe this image."
)
```

#### 모델별 특징

| 모델 | 파라미터 | 속도 | 정확도 | 권장 용도 |
|------|---------|------|--------|----------|
| Qwen2-VL-2B-Instruct | 2B | 매우 빠름 | 보통 | 경량 작업 |
| Qwen2-VL-7B-Instruct | 7B | 빠름 | 좋음 | 일반 작업 |
| Qwen2-VL-72B-Instruct | 72B | 느림 | 매우 좋음 | 복잡한 작업 |
| Qwen2.5-VL-32B-Instruct | 32B | 보통 | 매우 좋음 | 고품질 작업 |

### Gemma 핸들러

#### 설치

```bash
pip install transformers torch python-dotenv
```

#### 사용 예시

```python
from vlm.handlers import GemmaHandler

handler = GemmaHandler(
    model="google/gemma-2-9b-it",
    temperature=0.0,
    max_tokens=1000,
    device="cuda"  # 또는 "cpu"
)
handler.initialize()

response = handler.generate(
    image="path/to/image.png",
    system_prompt="You are a helpful assistant.",
    user_prompt="Describe this image."
)
```

#### 주의사항

- Gemma는 기본적으로 비전을 지원하지 않습니다
- 비전 지원 버전이 있다면 해당 모델을 사용해야 합니다
- GPU 사용을 권장합니다 (CUDA)

#### 모델별 특징

| 모델 | 파라미터 | 속도 | 정확도 | GPU 메모리 |
|------|---------|------|--------|-----------|
| google/gemma-2-2b-it | 2B | 빠름 | 보통 | ~4GB |
| google/gemma-2-9b-it | 9B | 보통 | 좋음 | ~18GB |
| google/gemma-2-27b-it | 27B | 느림 | 매우 좋음 | ~54GB |

## 테스트

### 테스트 스크립트 사용

`test_vlm.py` 스크립트를 사용하여 다양한 모델을 쉽게 테스트할 수 있습니다.

```bash
python test_vlm.py
```

### 테스트 설정 변경

`test_vlm.py` 파일의 설정 섹션을 수정하여 테스트를 커스터마이즈할 수 있습니다:

```python
# 이미지 설정
USE_NUMPY_IMAGE = True
IMAGE_FILE_PATH = None  # 또는 "path/to/image.png"

# 프롬프트 설정
SYSTEM_PROMPT = "You are a helpful assistant."
USER_PROMPT = "Describe what you see in this image."

# 모델 설정
TEST_MODELS = [
    {
        "handler_type": "openai",
        "name": "gpt-4o",
        "model": "gpt-4o",
        "temperature": 0.0,
        "max_tokens": 1000,
    },
    # 추가 모델...
]
```

### 여러 모델 비교 테스트

```python
from vlm import VLMManager

manager = VLMManager()

# 여러 모델 등록
models = [
    ("gpt-4o-mini", "openai", "gpt-4o-mini"),
    ("gpt-4o", "openai", "gpt-4o"),
    ("qwen-7b", "qwen", "Qwen2-VL-7B-Instruct"),
]

for name, handler_type, model in models:
    manager.create_handler(
        handler_type=handler_type,
        name=name,
        model=model
    )

# 각 모델 테스트
for name in [m[0] for m in models]:
    response = manager.generate(
        image=image,
        system_prompt="...",
        user_prompt="...",
        handler_name=name
    )
    print(f"{name}: {response[:100]}...")
```

## 문제 해결

### ImportError: No module named 'openai'

```bash
pip install openai
```

### ImportError: No module named 'dashscope'

```bash
pip install dashscope
```

### ImportError: No module named 'transformers'

```bash
pip install transformers torch
```

### API 키 오류

환경 변수나 `.env` 파일에 API 키가 올바르게 설정되었는지 확인하세요.

### GPU 메모리 부족 (Gemma)

더 작은 모델을 사용하거나 `device="cpu"`로 설정하세요:

```python
handler = GemmaHandler(
    model="google/gemma-2-2b-it",  # 더 작은 모델
    device="cpu"  # CPU 사용
)
```

### Qwen DashScope API 오류

API 키가 올바른지, DashScope 계정이 활성화되어 있는지 확인하세요.

## 추가 리소스

- [OpenAI API 문서](https://platform.openai.com/docs)
- [Qwen VLM GitHub](https://github.com/QwenLM/Qwen-VL)
- [Gemma 문서](https://ai.google.dev/gemma)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)

## 예제 코드

더 많은 예제는 다음 파일들을 참고하세요:

- `vlm_example_new.py`: 새로운 핸들러 시스템 사용 예제
- `test_vlm.py`: VLM 테스트 스크립트
- `vlm_wrapper.py`: 기존 호환성 래퍼 예제

