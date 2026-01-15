"""
Qwen VLM 핸들러

Alibaba의 Qwen Vision Language Model을 위한 핸들러 구현
DashScope API 또는 Hugging Face를 통해 사용 가능합니다.
"""

import base64
import os
from typing import Optional, Union
from pathlib import Path
import numpy as np
from PIL import Image
import io

try:
    from dotenv import load_dotenv
except ImportError:
    raise ImportError(
        "python-dotenv 라이브러리가 필요합니다. 다음 명령어로 설치하세요: pip install python-dotenv"
    )

from .base import VLMHandler

# .env 파일 자동 로드
load_dotenv()


class QwenHandler(VLMHandler):
    """
    Qwen Vision Language Model 핸들러
    
    지원 모델 (파라미터 크기별):
    - Qwen2-VL-2B-Instruct: 2B 파라미터 (경량)
    - Qwen2-VL-7B-Instruct: 7B 파라미터 (중간)
    - Qwen2-VL-72B-Instruct: 72B 파라미터 (대형)
    - Qwen2.5-VL-3B-Instruct: 3B 파라미터
    - Qwen2.5-VL-7B-Instruct: 7B 파라미터
    - Qwen2.5-VL-32B-Instruct: 32B 파라미터 (기본값)
    
    사용 예시:
        # 작은 모델 사용 (빠르지만 정확도 낮음)
        handler = QwenHandler(model="Qwen2-VL-2B-Instruct", api_type="dashscope")
        
        # 중간 모델 사용 (균형잡힌 성능)
        handler = QwenHandler(model="Qwen2-VL-7B-Instruct", api_type="dashscope")
        
        # 큰 모델 사용 (느리지만 정확도 높음)
        handler = QwenHandler(model="Qwen2-VL-72B-Instruct", api_type="dashscope")
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "Qwen2.5-VL-32B-Instruct",
        api_type: str = "dashscope",  # "dashscope" 또는 "huggingface"
        temperature: float = 0.0,
        max_tokens: int = 1000,
        **kwargs
    ):
        """
        Qwen 핸들러 초기화
        
        Args:
            api_key: API 키 (DashScope 사용 시 필요)
            model: 사용할 모델명
                - DashScope: "Qwen2-VL-2B-Instruct", "Qwen2-VL-7B-Instruct", "Qwen2-VL-72B-Instruct"
                - HuggingFace: "Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-7B-Instruct" 등
            api_type: API 타입 ("dashscope" 또는 "huggingface")
            temperature: 생성 온도 (기본값: 0.0)
            max_tokens: 최대 토큰 수 (기본값: 1000)
            **kwargs: 추가 설정
        """
        super().__init__(
            api_key=api_key,
            model=model,
            api_type=api_type,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        self.api_key = api_key
        self.model = model
        self.api_type = api_type
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.client = None
        self._model = None  # HuggingFace 모델용
    
    def initialize(self) -> bool:
        """
        Qwen 클라이언트 초기화
        
        Returns:
            초기화 성공 여부
        """
        if self.api_type == "dashscope":
            try:
                import dashscope
                from dashscope import MultiModalConversation
                
                if self.api_key is None:
                    self.api_key = os.getenv("DASHSCOPE_API_KEY")
                    if self.api_key is None:
                        raise ValueError(
                            "DashScope API 키가 필요합니다. "
                            "다음 중 하나를 사용하세요:\n"
                            "1. __init__(api_key='your-key')로 직접 전달\n"
                            "2. 환경변수 DASHSCOPE_API_KEY 설정\n"
                            "3. .env 파일에 DASHSCOPE_API_KEY=your-key 추가"
                        )
                
                dashscope.api_key = self.api_key
                self.client = MultiModalConversation
                return True
            except ImportError:
                raise ImportError(
                    "dashscope 라이브러리가 필요합니다. "
                    "다음 명령어로 설치하세요: pip install dashscope"
                )
        
        elif self.api_type == "huggingface":
            try:
                import torch
            except ImportError:
                raise ImportError(
                    "torch 라이브러리가 필요합니다. "
                    "다음 명령어로 설치하세요: pip install torch"
                )
            
            try:
                from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            except ImportError as e:
                error_msg = str(e)
                if "torchvision" in error_msg.lower():
                    raise ImportError(
                        f"Qwen2-VL 모델을 사용하려면 torchvision이 필요합니다. "
                        f"다음 명령어로 설치하세요: pip install torchvision\n"
                        f"원본 오류: {e}"
                    )
                else:
                    raise ImportError(
                        f"Qwen2VLForConditionalGeneration을 import할 수 없습니다. "
                        f"transformers 라이브러리의 최신 버전이 필요할 수 있습니다. "
                        f"다음 명령어로 업그레이드하세요: pip install --upgrade transformers\n"
                        f"원본 오류: {e}"
                    )
            
            try:
                self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                self.processor = AutoProcessor.from_pretrained(self.model)
                return True
            except Exception as e:
                raise RuntimeError(
                    f"Qwen 모델 로드 실패: {e}\n"
                    f"모델명: {self.model}\n"
                    f"transformers 버전을 확인하거나 모델명이 올바른지 확인하세요."
                )
        else:
            raise ValueError(f"지원하지 않는 API 타입: {self.api_type}. 'dashscope' 또는 'huggingface'를 사용하세요.")
    
    def encode_image(self, image: Union[str, Path, np.ndarray, Image.Image]) -> Union[str, bytes]:
        """
        이미지를 인코딩
        
        Args:
            image: 입력 이미지
            
        Returns:
            인코딩된 이미지
        """
        # DashScope는 base64, HuggingFace는 PIL Image 사용
        if self.api_type == "dashscope":
            # base64 인코딩
            if isinstance(image, (str, Path)):
                image_path = Path(image)
                if not image_path.exists():
                    raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
            
            elif isinstance(image, Image.Image):
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                return base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            elif isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                
                pil_image = Image.fromarray(image)
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                return base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            else:
                raise TypeError(f"지원하지 않는 이미지 타입: {type(image)}")
        
        else:  # huggingface
            # PIL Image로 변환
            if isinstance(image, (str, Path)):
                return Image.open(image).convert("RGB")
            elif isinstance(image, Image.Image):
                return image.convert("RGB")
            elif isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                return Image.fromarray(image).convert("RGB")
            else:
                raise TypeError(f"지원하지 않는 이미지 타입: {type(image)}")
    
    def generate(
        self,
        image: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
        system_prompt: str = "",
        user_prompt: str = ""
    ) -> str:
        """
        이미지와 프롬프트를 입력받아 VLM API를 호출하고 원본 응답을 반환
        
        Args:
            image: 입력 이미지
            system_prompt: 시스템 프롬프트
            user_prompt: 사용자 프롬프트
            
        Returns:
            원본 응답 텍스트 (str)
        """
        if self.client is None and self._model is None:
            self.initialize()
        
        # 프롬프트 결합
        full_prompt = f"{system_prompt}\n{user_prompt}".strip() if system_prompt else user_prompt
        
        try:
            if self.api_type == "dashscope":
                # DashScope API 사용
                if image is None:
                    messages = [{"role": "user", "content": [{"text": full_prompt}]}]
                else:
                    image_data = self.encode_image(image)
                    messages = [{
                        "role": "user",
                        "content": [
                            {"image": f"data:image/png;base64,{image_data}"},
                            {"text": full_prompt}
                        ]
                    }]
                
                response = self.client.call(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                if response.status_code == 200:
                    return response.output.choices[0].message.content[0].text
                else:
                    raise RuntimeError(f"DashScope API 호출 실패: {response.message}")
            
            else:  # huggingface
                # HuggingFace Transformers 사용
                import torch
                
                if image is None:
                    raise ValueError("HuggingFace 모드는 이미지가 필요합니다.")
                
                pil_image = self.encode_image(image)
                
                # 프롬프트 포맷팅
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_image},
                            {"type": "text", "text": full_prompt}
                        ]
                    }
                ]
                
                # apply_chat_template으로 텍스트 생성
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                # processor를 직접 호출하여 이미지와 텍스트를 함께 처리
                inputs = self.processor(
                    text=[text],
                    images=[pil_image],
                    padding=True,
                    return_tensors="pt"
                )
                
                inputs = inputs.to(self._model.device)
                
                with torch.no_grad():
                    generated_ids = self._model.generate(**inputs, max_new_tokens=self.max_tokens)
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                
                return output_text[0]
                
        except Exception as e:
            raise RuntimeError(f"API 호출 중 오류 발생: {e}")

