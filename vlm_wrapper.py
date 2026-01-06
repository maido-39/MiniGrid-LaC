"""
ChatGPT 4o (VLM) Python Wrapper

이 모듈은 OpenAI의 GPT-4o Vision 모델을 쉽게 사용할 수 있는 Wrapper 클래스를 제공합니다.
VLM 호출과 기본적인 입력/출력 처리만 담당합니다.

주요 기능:
- 이미지 처리 및 인코딩
- System Prompt, User Prompt 입력
- API 호출 및 매개변수 지정
- 원본 응답 메시지 반환

이 모듈은 내부적으로 새로운 핸들러 시스템(vlm.handlers)을 사용합니다.
호환성을 위해 기존 API는 그대로 유지됩니다.
"""

from typing import Optional, Union
from pathlib import Path
import numpy as np
from PIL import Image

# 새로운 핸들러 시스템 import
from vlm.handlers import OpenAIHandler
from vlm.vlm_manager import VLMManager

# 하위 호환성을 위해 VLMManager도 export
__all__ = ["ChatGPT4oVLMWrapper", "VLMManager"]


class ChatGPT4oVLMWrapper:
    """
    ChatGPT 4o Vision Language Model Wrapper (호환성 래퍼)
    
    이미지와 텍스트 프롬프트를 입력받아 VLM API를 호출하고 원본 응답을 반환합니다.
    후처리(파싱, 검증 등)는 별도 모듈에서 처리해야 합니다.
    
    이 클래스는 기존 코드와의 호환성을 위해 제공되며,
    내부적으로 새로운 핸들러 시스템(OpenAIHandler)을 사용합니다.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 1000
    ):
        """
        Wrapper 초기화
        
        Args:
            api_key: OpenAI API 키. None이면 환경변수 OPENAI_API_KEY 또는 .env 파일에서 자동 로드
            model: 사용할 모델명 (기본값: "gpt-4o")
            temperature: 생성 온도 (기본값: 0.0)
            max_tokens: 최대 토큰 수 (기본값: 1000)
        """
        # 내부적으로 OpenAIHandler 사용
        self._handler = OpenAIHandler(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        self._handler.initialize()
        
        # 호환성을 위해 속성 유지
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.is_gpt5_model = model.startswith("gpt-5") if model else False
    
    def _encode_image(self, image: Union[str, Path, np.ndarray, Image.Image]) -> str:
        """
        이미지를 base64로 인코딩 (호환성을 위해 유지)
        
        Args:
            image: 이미지 경로(str/Path), numpy 배열, 또는 PIL Image
            
        Returns:
            base64 인코딩된 이미지 문자열
        """
        return self._handler.encode_image(image)
    
    def generate(
        self,
        image: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
        system_prompt: str = "",
        user_prompt: str = ""
    ) -> str:
        """
        이미지와 프롬프트를 입력받아 VLM API를 호출하고 원본 응답을 반환
        
        Args:
            image: 입력 이미지 (경로, numpy 배열, 또는 PIL Image). None이면 이미지 없이 텍스트만 전송
            system_prompt: 시스템 프롬프트
            user_prompt: 사용자 프롬프트
            
        Returns:
            원본 응답 텍스트 (str)
        """
        return self._handler.generate(image, system_prompt, user_prompt)
    
    def __call__(
        self,
        image: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
        system_prompt: str = "",
        user_prompt: str = ""
    ) -> str:
        """
        호출 가능한 객체로 사용 (간편한 사용을 위해)
        
        Args:
            image: 입력 이미지 (None이면 이미지 없이 텍스트만 전송)
            system_prompt: 시스템 프롬프트
            user_prompt: 사용자 프롬프트
            
        Returns:
            원본 응답 텍스트 (str)
        """
        return self.generate(image, system_prompt, user_prompt)


# 사용 예제
if __name__ == "__main__":
    # Wrapper 초기화
    wrapper = ChatGPT4oVLMWrapper()
    
    # 예제 사용법
    response = wrapper.generate(
        # image="path/to/image.png",
        system_prompt="You are a helpful assistant.",
        user_prompt="describe the miku's characteristics"
    )
    print(response)

