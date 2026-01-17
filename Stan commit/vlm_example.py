"""
VLM Wrapper 사용 예제 스크립트

vlm_wrapper의 입력과 출력만 테스트하는 간단한 예제입니다.
"""

import numpy as np
from PIL import Image
from vlm_wrapper import ChatGPT4oVLMWrapper


def example_with_numpy_image():
    """NumPy 배열로 만든 간단한 이미지 사용 예제"""
    print("=" * 50)
    print("예제 1: NumPy 배열 이미지 사용")
    print("=" * 50)
    
    # 간단한 테스트 이미지 생성 (RGB, 100x100)
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Wrapper 초기화
    wrapper = ChatGPT4oVLMWrapper()
    
    # System Prompt와 User Prompt 설정
    system_prompt = "You are a helpful assistant."
    user_prompt = "Describe what you see in this image."
    
    print(f"\n[입력]")
    print(f"  이미지: NumPy 배열 (shape: {test_image.shape})")
    print(f"  System Prompt: {system_prompt}")
    print(f"  User Prompt: {user_prompt}")
    print(f"\nVLM 호출 중...")
    
    # VLM 호출
    response = wrapper.generate(
        image=test_image,
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )
    
    # 결과 출력 (원본 응답)
    print(f"\n[출력]")
    print(f"  원본 응답:")
    print(f"  {response}")
    print()


def example_with_pil_image():
    """PIL Image 사용 예제"""
    print("=" * 50)
    print("예제 2: PIL Image 사용")
    print("=" * 50)
    
    # 간단한 테스트 이미지 생성
    test_image = Image.new('RGB', (200, 200), color='lightblue')
    
    # Wrapper 초기화
    wrapper = ChatGPT4oVLMWrapper()
    
    # 간단한 프롬프트
    user_prompt = "What do you see in this image?"
    
    print(f"\n[입력]")
    print(f"  이미지: PIL Image ({test_image.size})")
    print(f"  System Prompt: (빈 문자열)")
    print(f"  User Prompt: {user_prompt}")
    print(f"\nVLM 호출 중...")
    
    # VLM 호출
    response = wrapper(
        image=test_image,
        system_prompt="",
        user_prompt=user_prompt
    )
    
    # 결과 출력
    print(f"\n[출력]")
    print(f"  원본 응답:")
    print(f"  {response}")
    print()


def example_without_image():
    """이미지 없이 텍스트만 사용하는 예제"""
    print("=" * 50)
    print("예제 3: 이미지 없이 텍스트만 사용")
    print("=" * 50)
    
    # Wrapper 초기화
    wrapper = ChatGPT4oVLMWrapper()
    
    # 프롬프트 설정
    system_prompt = "You are a helpful assistant."
    user_prompt = "What is the capital of France?"
    
    print(f"\n[입력]")
    print(f"  이미지: None (이미지 없음)")
    print(f"  System Prompt: {system_prompt}")
    print(f"  User Prompt: {user_prompt}")
    print(f"\nVLM 호출 중...")
    
    # VLM 호출 (이미지 없이)
    response = wrapper.generate(
        image=None,  # 또는 image 파라미터를 생략해도 됨
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )
    
    # 결과 출력
    print(f"\n[출력]")
    print(f"  원본 응답:")
    print(f"  {response}")
    print()


def example_with_image_file():
    """이미지 파일 경로 사용 예제"""
    print("=" * 50)
    print("예제 4: 이미지 파일 경로 사용")
    print("=" * 50)
    
    # 이미지 파일 경로 (실제 파일이 있는 경우)
    image_path = "test_image.png"  # 실제 이미지 파일 경로로 변경하세요
    
    print(f"\n[입력]")
    print(f"  이미지 경로: {image_path}")
    print(f"  System Prompt: (빈 문자열)")
    print(f"  User Prompt: 'Analyze this image'")
    print(f"\nVLM 호출 중...")
    
    try:
        wrapper = ChatGPT4oVLMWrapper()
        response = wrapper.generate(
            image=image_path,
            system_prompt="",
            user_prompt="Analyze this image."
        )
        
        # 결과 출력
        print(f"\n[출력]")
        print(f"  원본 응답:")
        print(f"  {response}")
    except FileNotFoundError:
        print(f"\n⚠️  이미지 파일을 찾을 수 없습니다: {image_path}")
        print("   다른 예제를 실행하거나 이미지 경로를 수정하세요.")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("VLM Wrapper 입력/출력 테스트")
    print("=" * 50 + "\n")
    
    # 예제 1: NumPy 배열 사용
    try:
        example_with_numpy_image()
    except Exception as e:
        print(f"❌ 예제 1 실행 중 오류: {e}\n")
        import traceback
        traceback.print_exc()
    
    # 예제 2: PIL Image 사용
    try:
        example_with_pil_image()
    except Exception as e:
        print(f"❌ 예제 2 실행 중 오류: {e}\n")
        import traceback
        traceback.print_exc()
    
    # 예제 3: 이미지 없이 텍스트만 사용
    try:
        example_without_image()
    except Exception as e:
        print(f"❌ 예제 3 실행 중 오류: {e}\n")
        import traceback
        traceback.print_exc()
    
    # 예제 4: 이미지 파일 사용 (선택적)
    # example_with_image_file()
    
    print("=" * 50)
    print("예제 실행 완료")
    print("=" * 50)

