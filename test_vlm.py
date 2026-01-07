"""
VLM 테스트 스크립트

다양한 VLM 모델을 테스트하고 비교할 수 있는 스크립트입니다.
이미지, 프롬프트, 모델을 쉽게 변경하여 테스트할 수 있습니다.

사용법:
    # 기본 이미지 URL과 기본 프롬프트 사용
    python test_vlm.py
    
    # 로컬 이미지 파일 사용
    python test_vlm.py --image path/to/image.jpg
    
    # URL에서 이미지 다운로드
    python test_vlm.py --image https://example.com/image.jpg
    
    # 사용자 프롬프트 지정
    python test_vlm.py --prompt "What objects are in this image?"
    
    # 시스템 프롬프트와 사용자 프롬프트 모두 지정
    python test_vlm.py --system "You are an expert image analyst." --prompt "Analyze this image in detail."
    
    # 이미지와 프롬프트 모두 지정
    python test_vlm.py -i path/to/image.jpg --command "Describe the colors in this image"
"""

import numpy as np
from PIL import Image
from pathlib import Path
import argparse
import requests
from io import BytesIO

# ============================================================================
# 설정 섹션: 여기서 이미지, 프롬프트, 모델을 쉽게 변경할 수 있습니다
# ============================================================================

# 기본 이미지 URL (명령줄 인자가 없을 때 사용)
DEFAULT_IMAGE_URL = "https://picsum.photos/300/200"

# 이미지 설정 (명령줄 인자가 없을 때 사용)
# 옵션 1: URL에서 이미지 다운로드
USE_URL_IMAGE = True

# 옵션 2: NumPy 배열로 생성된 테스트 이미지 사용
USE_NUMPY_IMAGE = False
NUMPY_IMAGE_SHAPE = (100, 100, 3)  # (height, width, channels)

# 옵션 3: 이미지 파일 경로 사용
IMAGE_FILE_PATH = None  # 예: "test_image.png" 또는 "path/to/image.jpg"

# 옵션 4: PIL Image 직접 생성
USE_PIL_IMAGE = False
PIL_IMAGE_SIZE = (200, 200)  # (width, height)
PIL_IMAGE_COLOR = "lightblue"

# 프롬프트 설정 (기본값, 명령줄 인자로 덮어쓸 수 있음)
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that describes images in detail."
DEFAULT_USER_PROMPT = "Describe what you see in this image. Be specific about colors, shapes, and any objects present."

# 모델 설정
# 사용할 VLM 모델 선택 (아래 중 하나 선택)
# OpenAI 모델 예시:
#   - "gpt-4o-mini": 경량 모델 (빠르지만 정확도 낮음)
#   - "gpt-4o": 중간 모델 (균형잡힌 성능, 기본값)
#   - "gpt-4-turbo": 대형 모델 (느리지만 정확도 높음)
#   - "gpt-4": 레거시 대형 모델
#   - "gpt-5": 최신 대형 모델 (가능한 경우)

# Qwen 모델 예시 (Hugging Face 로컬 실행):
#   - "Qwen/Qwen2-VL-2B-Instruct": 2B 파라미터 (경량)
#   - "Qwen/Qwen2-VL-7B-Instruct": 7B 파라미터 (중간)
#   - "Qwen/Qwen2-VL-72B-Instruct": 72B 파라미터 (대형)
#   - "Qwen/Qwen2.5-VL-3B-Instruct": 3B 파라미터
#   - "Qwen/Qwen2.5-VL-7B-Instruct": 7B 파라미터
#   - "Qwen/Qwen2.5-VL-32B-Instruct": 32B 파라미터 (기본값)

# Gemma 모델 예시 (Hugging Face 사용):
#   - "google/gemma-2-2b-it": 2B 파라미터 (경량)
#   - "google/gemma-2-9b-it": 9B 파라미터 (중간, 기본값)
#   - "google/gemma-2-27b-it": 27B 파라미터 (대형)

# 테스트할 모델 리스트 (여러 모델을 한 번에 테스트 가능)
TEST_MODELS = [
    # {
    #     "handler_type": "openai",
    #     "name": "gpt-4o",
    #     "model": "gpt-4o",
    #     "temperature": 0.0,
    #     "max_tokens": 1000,
    # },
    # 다른 모델 예시 (주석 해제하여 사용):
    # {
    #     "handler_type": "openai",
    #     "name": "gpt-4o-mini",
    #     "model": "gpt-4o-mini",
    #     "temperature": 0.0,
    #     "max_tokens": 500,
    # },
    {
        "handler_type": "qwen",
        "name": "qwen-2b",
        "model": "Qwen/Qwen2-VL-2B-Instruct",
        "api_type": "huggingface",
        "temperature": 0.0,
        "max_tokens": 1000,
    },
    # {
    #     "handler_type": "gemma",
    #     "name": "gemma-9b",
    #     "model": "google/gemma-2-9b-it",
    #     "temperature": 0.0,
    #     "max_tokens": 1000,
    # },
]

# ============================================================================
# 테스트 코드 (설정 아래는 수정하지 않아도 됩니다)
# ============================================================================


def load_image_from_url(url: str) -> Image.Image:
    """URL에서 이미지 다운로드"""
    try:
        print(f"URL에서 이미지 다운로드 중: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        print(f"✓ 이미지 다운로드 완료: {image.size}")
        return image
    except Exception as e:
        print(f"❌ URL에서 이미지 다운로드 실패: {e}")
        raise


def create_test_image(image_path_or_url: str = None):
    """
    테스트 이미지 생성
    
    Args:
        image_path_or_url: 이미지 파일 경로 또는 URL (None이면 기본 설정 사용)
    
    Returns:
        PIL Image 또는 numpy array
    """
    # 명령줄 인자로 이미지가 제공된 경우
    if image_path_or_url:
        # URL인지 확인 (http:// 또는 https://로 시작)
        if image_path_or_url.startswith(('http://', 'https://')):
            return load_image_from_url(image_path_or_url)
        # 파일 경로인 경우
        elif Path(image_path_or_url).exists():
            print(f"이미지 파일 로드: {image_path_or_url}")
            return Image.open(image_path_or_url).convert("RGB")
        else:
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path_or_url}")
    
    # 기본 설정 사용
    if IMAGE_FILE_PATH and Path(IMAGE_FILE_PATH).exists():
        # 파일에서 로드
        print(f"이미지 파일 로드: {IMAGE_FILE_PATH}")
        return Image.open(IMAGE_FILE_PATH).convert("RGB")
    
    elif USE_URL_IMAGE:
        # URL에서 다운로드
        return load_image_from_url(DEFAULT_IMAGE_URL)
    
    elif USE_NUMPY_IMAGE:
        # NumPy 배열 생성
        print(f"NumPy 배열 이미지 생성: {NUMPY_IMAGE_SHAPE}")
        return np.random.randint(0, 255, NUMPY_IMAGE_SHAPE, dtype=np.uint8)
    
    elif USE_PIL_IMAGE:
        # PIL Image 생성
        print(f"PIL Image 생성: {PIL_IMAGE_SIZE}, 색상: {PIL_IMAGE_COLOR}")
        return Image.new('RGB', PIL_IMAGE_SIZE, color=PIL_IMAGE_COLOR)
    
    else:
        # 기본값: URL에서 다운로드
        return load_image_from_url(DEFAULT_IMAGE_URL)


def test_vlm_model(model_config, image, system_prompt, user_prompt):
    """단일 VLM 모델 테스트"""
    handler_type = model_config.get("handler_type", "openai")
    name = model_config.get("name", "unknown")
    model = model_config.get("model", "gpt-4o")
    
    print(f"\n{'='*80}")
    print(f"모델 테스트: {name}")
    print(f"핸들러 타입: {handler_type}")
    print(f"모델명: {model}")
    print(f"{'='*80}")
    
    try:
        # 핸들러 직접 생성
        if handler_type == "openai":
            from vlm.handlers import OpenAIHandler
            handler = OpenAIHandler(
                model=model,
                temperature=model_config.get("temperature", 0.0),
                max_tokens=model_config.get("max_tokens", 1000),
            )
        
        elif handler_type == "qwen":
            from vlm.handlers import QwenHandler
            handler = QwenHandler(
                model=model,
                api_type=model_config.get("api_type", "huggingface"),
                temperature=model_config.get("temperature", 0.0),
                max_tokens=model_config.get("max_tokens", 1000),
            )
        
        elif handler_type == "gemma":
            from vlm.handlers import GemmaHandler
            handler = GemmaHandler(
                model=model,
                temperature=model_config.get("temperature", 0.0),
                max_tokens=model_config.get("max_tokens", 1000),
                device=model_config.get("device", None),
            )
        
        else:
            raise ValueError(f"지원하지 않는 핸들러 타입: {handler_type}")
        
        # 초기화
        print("핸들러 초기화 중...")
        handler.initialize()
        print("✓ 초기화 완료")
        
        # VLM 호출
        print("\nVLM 호출 중...")
        print(f"System Prompt: {system_prompt}")
        print(f"User Prompt: {user_prompt}")
        
        response = handler.generate(
            image=image,
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        
        print(f"\n✓ 응답 수신 완료")
        print(f"\n응답 길이: {len(response)} 문자")
        print(f"\n응답 내용:")
        print("-" * 80)
        print(response)
        print("-" * 80)
        
        return {
            "success": True,
            "model": name,
            "response": response,
            "response_length": len(response)
        }
    
    except ImportError as e:
        print(f"❌ 라이브러리 import 실패: {e}")
        print("필요한 라이브러리를 설치하세요:")
        if handler_type == "openai":
            print("  pip install openai")
        elif handler_type == "qwen":
            print("  pip install transformers torch torchvision")
        elif handler_type == "gemma":
            print("  pip install transformers torch")
        return {"success": False, "model": name, "error": str(e)}
    
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "model": name, "error": str(e)}


def main():
    """메인 함수"""
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(
        description="VLM 모델 테스트 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 이미지 URL과 기본 프롬프트 사용
  python test_vlm.py
  
  # 로컬 이미지 파일 사용
  python test_vlm.py --image path/to/image.jpg
  
  # URL에서 이미지 다운로드
  python test_vlm.py --image https://example.com/image.jpg
  
  # 사용자 프롬프트 지정
  python test_vlm.py --prompt "What objects are in this image?"
  
  # 시스템 프롬프트와 사용자 프롬프트 모두 지정
  python test_vlm.py --system "You are an expert image analyst." --prompt "Analyze this image in detail."
  
  # 이미지와 프롬프트 모두 지정
  python test_vlm.py -i path/to/image.jpg --command "Describe the colors in this image"
        """
    )
    parser.add_argument(
        '--image', '-i',
        type=str,
        default=None,
        help='이미지 파일 경로 또는 URL (기본값: 기본 URL에서 다운로드)'
    )
    parser.add_argument(
        '--system-prompt', '--system',
        type=str,
        default=None,
        help='시스템 프롬프트 (기본값: 기본 시스템 프롬프트 사용)'
    )
    parser.add_argument(
        '--user-prompt', '--prompt', '--command',
        type=str,
        default=None,
        help='사용자 프롬프트/명령어 (기본값: 기본 사용자 프롬프트 사용)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("VLM 모델 테스트 스크립트")
    print("=" * 80)
    
    # 테스트 이미지 생성
    print("\n[1] 테스트 이미지 준비")
    image = create_test_image(args.image)
    print(f"이미지 타입: {type(image)}")
    if isinstance(image, np.ndarray):
        print(f"이미지 shape: {image.shape}")
    elif isinstance(image, Image.Image):
        print(f"이미지 size: {image.size}")
    
    # 프롬프트 설정 (명령줄 인자가 있으면 사용, 없으면 기본값)
    system_prompt = args.system_prompt if args.system_prompt else DEFAULT_SYSTEM_PROMPT
    user_prompt = args.user_prompt if args.user_prompt else DEFAULT_USER_PROMPT
    
    # 프롬프트 출력
    print("\n[2] 프롬프트 설정")
    print(f"System Prompt: {system_prompt}")
    print(f"User Prompt: {user_prompt}")
    
    # 모델 테스트
    print("\n[3] 모델 테스트 시작")
    print(f"테스트할 모델 수: {len(TEST_MODELS)}")
    
    results = []
    for i, model_config in enumerate(TEST_MODELS, 1):
        print(f"\n[{i}/{len(TEST_MODELS)}] 모델 테스트")
        result = test_vlm_model(model_config, image, system_prompt, user_prompt)
        results.append(result)
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("테스트 결과 요약")
    print("=" * 80)
    
    success_count = sum(1 for r in results if r.get("success", False))
    print(f"\n성공: {success_count}/{len(results)}")
    
    for result in results:
        model_name = result.get("model", "unknown")
        if result.get("success", False):
            response_length = result.get("response_length", 0)
            print(f"  ✓ {model_name}: 성공 (응답 길이: {response_length} 문자)")
        else:
            error = result.get("error", "Unknown error")
            print(f"  ❌ {model_name}: 실패 ({error})")
    
    print("\n" + "=" * 80)
    print("테스트 완료")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()

