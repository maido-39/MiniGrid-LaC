"""
ChatGPT 4o (VLM) Python Wrapper

이 모듈은 OpenAI의 GPT-4o Vision 모델을 쉽게 사용할 수 있는 Wrapper 클래스를 제공합니다.
VLM 호출과 기본적인 입력/출력 처리만 담당합니다.

주요 기능:
- 이미지 처리 및 인코딩
- System Prompt, User Prompt 입력
- API 호출 및 매개변수 지정
- 원본 응답 메시지 반환
"""

import base64
import os
from typing import Dict, Optional, Union, List
from pathlib import Path
import numpy as np
from PIL import Image
import io

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "openai 라이브러리가 필요합니다. 다음 명령어로 설치하세요: pip install openai"
    )

try:
    from dotenv import load_dotenv
except ImportError:
    raise ImportError(
        "python-dotenv 라이브러리가 필요합니다. 다음 명령어로 설치하세요: pip install python-dotenv"
    )

# .env 파일 자동 로드
load_dotenv()


class ChatGPT4oVLMWrapper:
    """
    ChatGPT 4o Vision Language Model Wrapper
    
    이미지와 텍스트 프롬프트를 입력받아 VLM API를 호출하고 원본 응답을 반환합니다.
    후처리(파싱, 검증 등)는 별도 모듈에서 처리해야 합니다.
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
        # API 키 우선순위: 인자 > 환경변수 > .env 파일
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError(
                    "API 키가 제공되지 않았습니다. "
                    "다음 중 하나를 사용하세요:\n"
                    "1. __init__(api_key='your-key')로 직접 전달\n"
                    "2. 환경변수 OPENAI_API_KEY 설정\n"
                    "3. .env 파일에 OPENAI_API_KEY=your-key 추가"
                )
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # GPT-5 계열 모델인지 확인 (모델명이 "gpt-5"로 시작하는 경우)
        self.is_gpt5_model = model.startswith("gpt-5") if model else False
    
    def _encode_image(self, image: Union[str, Path, np.ndarray, Image.Image]) -> str:
        """
        이미지를 base64로 인코딩
        
        Args:
            image: 이미지 경로(str/Path), numpy 배열, 또는 PIL Image
            
        Returns:
            base64 인코딩된 이미지 문자열
        """
        # 경로 문자열인 경우
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        
        # PIL Image인 경우
        elif isinstance(image, Image.Image):
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # numpy 배열인 경우
        elif isinstance(image, np.ndarray):
            # numpy 배열을 PIL Image로 변환
            if image.dtype != np.uint8:
                # 0-1 범위의 float 배열인 경우 0-255로 변환
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            pil_image = Image.fromarray(image)
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        else:
            raise TypeError(
                f"지원하지 않는 이미지 타입입니다: {type(image)}. "
                "str, Path, np.ndarray, 또는 PIL.Image를 사용하세요."
            )
    
    def _build_messages(
        self,
        image: Optional[Union[str, Path, np.ndarray, Image.Image]],
        system_prompt: str,
        user_prompt: str
    ) -> List[Dict]:
        """
        API 호출을 위한 메시지 리스트 생성
        
        Args:
            image: 입력 이미지 (None이면 이미지 없이 텍스트만 전송)
            system_prompt: 시스템 프롬프트
            user_prompt: 사용자 프롬프트
            
        Returns:
            OpenAI API 형식의 메시지 리스트
        """
        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        
        # 이미지가 있는 경우
        if image is not None:
            # 이미지 인코딩
            base64_image = self._encode_image(image)
            
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            })
        else:
            # 이미지가 없는 경우 텍스트만 전송
            messages.append({
                "role": "user",
                "content": user_prompt
            })
        
        return messages
    
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
        # 메시지 생성
        messages = self._build_messages(image, system_prompt, user_prompt)
        
        # API 호출
        try:
            # GPT-5 계열 모델은 max_completion_tokens 사용, 그 외는 max_tokens 사용
            api_params = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature
            }
            
            if self.is_gpt5_model:
                # GPT-5 계열: max_completion_tokens 사용
                api_params["max_completion_tokens"] = self.max_tokens
            else:
                # GPT-4 계열: max_tokens 사용
                api_params["max_tokens"] = self.max_tokens
            
            response = self.client.chat.completions.create(**api_params)
            
            # 응답 텍스트 추출 및 반환
            raw_response = response.choices[0].message.content
            return raw_response
                
        except Exception as e:
            raise RuntimeError(f"API 호출 중 오류 발생: {e}")
    
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

