"""
VLM 핸들러 시스템 사용 예제

새로운 핸들러 기반 VLM 시스템의 사용법을 보여줍니다.
"""

import numpy as np
from PIL import Image

# 방법 1: 기존 ChatGPT4oVLMWrapper 사용 (호환성 유지)
from vlm_wrapper import ChatGPT4oVLMWrapper

# 방법 2: 새로운 VLMManager 사용
from vlm import VLMManager

# 방법 3: 핸들러 직접 사용
from vlm.handlers import OpenAIHandler


def example_1_legacy_wrapper():
    """예제 1: 기존 ChatGPT4oVLMWrapper 사용 (호환성 유지)"""
    print("=" * 60)
    print("예제 1: 기존 ChatGPT4oVLMWrapper 사용")
    print("=" * 60)
    
    # 기존 방식 그대로 사용 가능
    wrapper = ChatGPT4oVLMWrapper(
        model="gpt-4o",
        temperature=0.0,
        max_tokens=1000
    )
    
    # 사용법은 동일
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    print("\n[입력]")
    print(f"  이미지: NumPy 배열 (shape: {test_image.shape})")
    print(f"  System Prompt: 'You are a helpful assistant.'")
    print(f"  User Prompt: 'Describe what you see in this image.'")
    print(f"\nVLM 호출 중...")
    
    try:
        response = wrapper.generate(
            image=test_image,
            system_prompt="You are a helpful assistant.",
            user_prompt="Describe what you see in this image."
        )
        
        print(f"\n[출력]")
        print(f"  원본 응답:")
        print(f"  {response[:200]}...")
    except Exception as e:
        print(f"  오류: {e}")
    
    print()


def example_2_vlm_manager():
    """예제 2: 새로운 VLMManager 사용"""
    print("=" * 60)
    print("예제 2: 새로운 VLMManager 사용")
    print("=" * 60)
    
    # VLMManager 생성
    manager = VLMManager()
    
    # 핸들러 생성 및 등록
    manager.create_handler(
        handler_type="openai",
        name="my_openai",
        set_as_default=True,
        model="gpt-4o",
        temperature=0.0,
        max_tokens=1000
    )
    
    print("\n등록된 핸들러:", manager.list_handlers())
    
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    print("\n[입력]")
    print(f"  이미지: NumPy 배열 (shape: {test_image.shape})")
    print(f"  System Prompt: 'You are a helpful assistant.'")
    print(f"  User Prompt: 'What do you see?'")
    print(f"\nVLM 호출 중...")
    
    try:
        # Manager를 통해 호출
        response = manager.generate(
            image=test_image,
            system_prompt="You are a helpful assistant.",
            user_prompt="What do you see?"
        )
        
        print(f"\n[출력]")
        print(f"  원본 응답:")
        print(f"  {response[:200]}...")
    except Exception as e:
        print(f"  오류: {e}")
    
    print()


def example_3_multiple_handlers():
    """예제 3: 여러 핸들러 등록 및 사용"""
    print("=" * 60)
    print("예제 3: 여러 핸들러 등록 및 사용")
    print("=" * 60)
    
    manager = VLMManager()
    
    # 여러 핸들러 등록
    manager.create_handler(
        handler_type="openai",
        name="gpt4o",
        model="gpt-4o",
        temperature=0.0,
        max_tokens=1000
    )
    
    manager.create_handler(
        handler_type="openai",
        name="gpt4o_mini",
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=500
    )
    
    print("\n등록된 핸들러:", manager.list_handlers())
    
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # 특정 핸들러 사용
    print("\n[gpt4o 핸들러 사용]")
    try:
        response1 = manager.generate(
            image=test_image,
            system_prompt="You are a helpful assistant.",
            user_prompt="Describe the image.",
            handler_name="gpt4o"
        )
        print(f"  응답 길이: {len(response1)} 문자")
    except Exception as e:
        print(f"  오류: {e}")
    
    print("\n[gpt4o_mini 핸들러 사용]")
    try:
        response2 = manager.generate(
            image=test_image,
            system_prompt="You are a helpful assistant.",
            user_prompt="Describe the image.",
            handler_name="gpt4o_mini"
        )
        print(f"  응답 길이: {len(response2)} 문자")
    except Exception as e:
        print(f"  오류: {e}")
    
    print()


def example_4_direct_handler():
    """예제 4: 핸들러 직접 사용"""
    print("=" * 60)
    print("예제 4: 핸들러 직접 사용")
    print("=" * 60)
    
    # 핸들러 직접 생성
    handler = OpenAIHandler(
        model="gpt-4o",
        temperature=0.0,
        max_tokens=1000
    )
    
    # 초기화
    handler.initialize()
    
    print(f"\n모델명: {handler.get_model_name()}")
    print(f"지원 이미지 형식: {handler.get_supported_image_formats()}")
    
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    print("\n[입력]")
    print(f"  이미지: NumPy 배열 (shape: {test_image.shape})")
    print(f"  System Prompt: 'You are a helpful assistant.'")
    print(f"  User Prompt: 'Analyze this image.'")
    print(f"\nVLM 호출 중...")
    
    try:
        # 핸들러 직접 호출
        response = handler.generate(
            image=test_image,
            system_prompt="You are a helpful assistant.",
            user_prompt="Analyze this image."
        )
        
        print(f"\n[출력]")
        print(f"  원본 응답:")
        print(f"  {response[:200]}...")
    except Exception as e:
        print(f"  오류: {e}")
    
    print()


def example_5_callable():
    """예제 5: 호출 가능한 객체로 사용"""
    print("=" * 60)
    print("예제 5: 호출 가능한 객체로 사용")
    print("=" * 60)
    
    # Manager를 호출 가능한 객체로 사용
    manager = VLMManager()
    manager.create_handler(
        handler_type="openai",
        name="default",
        set_as_default=True,
        model="gpt-4o",
        temperature=0.0,
        max_tokens=1000
    )
    
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    print("\n[입력]")
    print(f"  이미지: NumPy 배열 (shape: {test_image.shape})")
    print(f"  System Prompt: 'You are a helpful assistant.'")
    print(f"  User Prompt: 'What is this?'")
    print(f"\nVLM 호출 중...")
    
    try:
        # 함수처럼 호출
        response = manager(
            image=test_image,
            system_prompt="You are a helpful assistant.",
            user_prompt="What is this?"
        )
        
        print(f"\n[출력]")
        print(f"  원본 응답:")
        print(f"  {response[:200]}...")
    except Exception as e:
        print(f"  오류: {e}")
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("VLM 핸들러 시스템 사용 예제")
    print("=" * 60 + "\n")
    
    # 예제 실행 (각 예제는 독립적으로 실행 가능)
    try:
        example_1_legacy_wrapper()
    except Exception as e:
        print(f"❌ 예제 1 실행 중 오류: {e}\n")
        import traceback
        traceback.print_exc()
    
    try:
        example_2_vlm_manager()
    except Exception as e:
        print(f"❌ 예제 2 실행 중 오류: {e}\n")
        import traceback
        traceback.print_exc()
    
    try:
        example_3_multiple_handlers()
    except Exception as e:
        print(f"❌ 예제 3 실행 중 오류: {e}\n")
        import traceback
        traceback.print_exc()
    
    try:
        example_4_direct_handler()
    except Exception as e:
        print(f"❌ 예제 4 실행 중 오류: {e}\n")
        import traceback
        traceback.print_exc()
    
    try:
        example_5_callable()
    except Exception as e:
        print(f"❌ 예제 5 실행 중 오류: {e}\n")
        import traceback
        traceback.print_exc()
    
    print("=" * 60)
    print("예제 실행 완료")
    print("=" * 60)

