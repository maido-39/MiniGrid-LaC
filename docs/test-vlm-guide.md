# VLM 테스트 스크립트 가이드

`test_vlm.py`는 다양한 VLM(Vision Language Model) 모델을 테스트하고 비교할 수 있는 스크립트입니다. 이미지, 프롬프트, 모델을 쉽게 변경하여 테스트할 수 있습니다.

## 개요

이 스크립트는 다음 기능을 제공합니다:

- **다양한 VLM 모델 지원**: OpenAI, Qwen, Gemma 등 여러 VLM 모델을 테스트
- **유연한 이미지 입력**: URL, 로컬 파일, 또는 자동 생성 이미지 사용
- **명령줄 인터페이스**: 이미지와 프롬프트를 명령줄 인자로 쉽게 지정
- **다중 모델 테스트**: 여러 모델을 한 번에 테스트하고 결과 비교

## 설치 요구사항

### 기본 요구사항

```bash
# minigrid conda 환경 활성화
conda activate minigrid

# 기본 패키지 (이미 설치되어 있을 수 있음)
pip install pillow numpy requests
```

### 모델별 추가 요구사항

#### OpenAI 모델 사용 시
```bash
pip install openai
```

#### Qwen 모델 사용 시 (로컬 실행)
```bash
pip install transformers torch torchvision
```

#### Gemma 모델 사용 시
```bash
pip install transformers torch
```

## 기본 사용법

### 1. 기본 이미지와 기본 프롬프트 사용

가장 간단한 사용법입니다. 기본 URL에서 이미지를 다운로드하고 기본 프롬프트를 사용합니다.

```bash
cd src
python test_script/etc/test_vlm.py
```
```

### 2. 로컬 이미지 파일 사용

로컬에 있는 이미지 파일을 사용하려면 `--image` 또는 `-i` 옵션을 사용합니다.

```bash
cd src
python test_script/etc/test_vlm.py
``` --image path/to/image.jpg
python test_vlm.py -i logs/scenario2_20260107_103053/step_0001.png
```

### 3. URL에서 이미지 다운로드

이미지 URL을 직접 지정할 수 있습니다.

```bash
cd src
python test_script/etc/test_vlm.py
``` --image https://picsum.photos/400/300
python test_vlm.py -i https://example.com/image.jpg
```

### 4. 사용자 프롬프트 지정

사용자 프롬프트를 명령줄 인자로 지정할 수 있습니다.

```bash
cd src
python test_script/etc/test_vlm.py
``` --prompt "What objects are in this image?"
python test_vlm.py --command "Describe the colors in this image"
```

### 5. 시스템 프롬프트 지정

시스템 프롬프트를 지정하여 VLM의 역할을 변경할 수 있습니다.

```bash
cd src
python test_script/etc/test_vlm.py
``` --system "You are an expert image analyst."
```

### 6. 모든 옵션 조합

이미지, 시스템 프롬프트, 사용자 프롬프트를 모두 지정할 수 있습니다.

```bash
cd src
python test_script/etc/test_vlm.py
``` \
  --image path/to/image.jpg \
  --system "You are a color expert." \
  --prompt "List all the colors you see in this image, one per line."
```

## 명령줄 옵션

| 옵션 | 단축형 | 설명 |
|------|--------|------|
| `--image` | `-i` | 이미지 파일 경로 또는 URL |
| `--system-prompt` | `--system` | 시스템 프롬프트 |
| `--user-prompt` | `--prompt`, `--command` | 사용자 프롬프트/명령어 |
| `--help` | `-h` | 도움말 메시지 표시 |

## 모델 설정

스크립트 내부에서 테스트할 모델을 설정할 수 있습니다. `test_vlm.py` 파일을 열어 `TEST_MODELS` 리스트를 수정하세요.

### OpenAI 모델 예시

```python
TEST_MODELS = [
    {
        "handler_type": "openai",
        "name": "gpt-4o",
        "model": "gpt-4o",
        "temperature": 0.0,
        "max_tokens": 1000,
    },
]
```

### Qwen 모델 예시 (로컬 실행)

```python
TEST_MODELS = [
    {
        "handler_type": "qwen",
        "name": "qwen-2b",
        "model": "Qwen/Qwen2-VL-2B-Instruct",
        "api_type": "huggingface",
        "temperature": 0.0,
        "max_tokens": 1000,
    },
]
```

### Gemma 모델 예시

```python
TEST_MODELS = [
    {
        "handler_type": "gemma",
        "name": "gemma-9b",
        "model": "google/gemma-2-9b-it",
        "temperature": 0.0,
        "max_tokens": 1000,
    },
]
```

### 여러 모델 동시 테스트

여러 모델을 한 번에 테스트하여 결과를 비교할 수 있습니다.

```python
TEST_MODELS = [
    {
        "handler_type": "openai",
        "name": "gpt-4o",
        "model": "gpt-4o",
        "temperature": 0.0,
        "max_tokens": 1000,
    },
    {
        "handler_type": "qwen",
        "name": "qwen-2b",
        "model": "Qwen/Qwen2-VL-2B-Instruct",
        "api_type": "huggingface",
        "temperature": 0.0,
        "max_tokens": 1000,
    },
]
```

## 지원 모델 목록

### OpenAI 모델

- `gpt-4o-mini`: 경량 모델 (빠르지만 정확도 낮음)
- `gpt-4o`: 중간 모델 (균형잡힌 성능, 기본값)
- `gpt-4-turbo`: 대형 모델 (느리지만 정확도 높음)
- `gpt-4`: 레거시 대형 모델
- `gpt-5`: 최신 대형 모델 (가능한 경우)

### Qwen 모델 (Hugging Face 로컬 실행)

- `Qwen/Qwen2-VL-2B-Instruct`: 2B 파라미터 (경량)
- `Qwen/Qwen2-VL-7B-Instruct`: 7B 파라미터 (중간)
- `Qwen/Qwen2-VL-72B-Instruct`: 72B 파라미터 (대형)
- `Qwen/Qwen2.5-VL-3B-Instruct`: 3B 파라미터
- `Qwen/Qwen2.5-VL-7B-Instruct`: 7B 파라미터
- `Qwen/Qwen2.5-VL-32B-Instruct`: 32B 파라미터

### Gemma 모델 (Hugging Face)

- `google/gemma-2-2b-it`: 2B 파라미터 (경량)
- `google/gemma-2-9b-it`: 9B 파라미터 (중간, 기본값)
- `google/gemma-2-27b-it`: 27B 파라미터 (대형)

## 출력 예시

스크립트를 실행하면 다음과 같은 정보가 출력됩니다:

```
================================================================================
VLM 모델 테스트 스크립트
================================================================================

[1] 테스트 이미지 준비
URL에서 이미지 다운로드 중: https://picsum.photos/300/200
✓ 이미지 다운로드 완료: (300, 200)
이미지 타입: <class 'PIL.Image.Image'>
이미지 size: (300, 200)

[2] 프롬프트 설정
System Prompt: You are a helpful assistant that describes images in detail.
User Prompt: Describe what you see in this image. Be specific about colors, shapes, and any objects present.

[3] 모델 테스트 시작
테스트할 모델 수: 1

[1/1] 모델 테스트

================================================================================
모델 테스트: qwen-2b
핸들러 타입: qwen
모델명: Qwen/Qwen2-VL-2B-Instruct
================================================================================
핸들러 초기화 중...
✓ 초기화 완료

VLM 호출 중...
System Prompt: You are a helpful assistant that describes images in detail.
User Prompt: Describe what you see in this image. Be specific about colors, shapes, and any objects present.

✓ 응답 수신 완료

응답 길이: 471 문자

응답 내용:
--------------------------------------------------------------------------------
The image depicts a complex pattern composed of numerous small, irregularly shaped, and colored dots...
--------------------------------------------------------------------------------

================================================================================
테스트 결과 요약
================================================================================

성공: 1/1
  ✓ qwen-2b: 성공 (응답 길이: 471 문자)

================================================================================
테스트 완료
================================================================================
```

## 고급 설정

### 기본 이미지 URL 변경

스크립트 상단에서 기본 이미지 URL을 변경할 수 있습니다.

```python
# 기본 이미지 URL (명령줄 인자가 없을 때 사용)
DEFAULT_IMAGE_URL = "https://picsum.photos/300/200"
```

### 기본 프롬프트 변경

기본 시스템 프롬프트와 사용자 프롬프트를 변경할 수 있습니다.

```python
# 프롬프트 설정 (기본값, 명령줄 인자로 덮어쓸 수 있음)
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that describes images in detail."
DEFAULT_USER_PROMPT = "Describe what you see in this image. Be specific about colors, shapes, and any objects present."
```

## 문제 해결

### ImportError: No module named 'transformers'

Qwen 또는 Gemma 모델을 사용하려면 transformers 라이브러리가 필요합니다.

```bash
conda activate minigrid
pip install transformers torch torchvision
```

### ImportError: No module named 'openai'

OpenAI 모델을 사용하려면 openai 라이브러리가 필요합니다.

```bash
conda activate minigrid
pip install openai
```

### 이미지 다운로드 실패

URL에서 이미지를 다운로드할 수 없는 경우:

1. 인터넷 연결을 확인하세요
2. URL이 올바른지 확인하세요
3. 로컬 이미지 파일을 사용하세요: `python test_vlm.py --image path/to/image.jpg`

### 모델 로드 실패

로컬 모델(Qwen, Gemma)을 로드할 수 없는 경우:

1. 충분한 메모리(VRAM)가 있는지 확인하세요
2. 모델 이름이 올바른지 확인하세요
3. 더 작은 모델을 시도해보세요 (예: 2B 파라미터 모델)

## 관련 문서

- [VLM 핸들러 문서](vlm-handlers.md) - VLM 핸들러 상세 설명
- [README.md](../README.md) - 프로젝트 전체 개요

