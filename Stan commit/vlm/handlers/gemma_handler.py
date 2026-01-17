"""
Gemma VLM 핸들러

Google의 Gemma Vision Language Model을 위한 핸들러 구현
Hugging Face Transformers를 통해 사용 가능합니다.
"""

import os
from typing import Optional, Union
from pathlib import Path
import numpy as np
from PIL import Image

try:
    from dotenv import load_dotenv
except ImportError:
    raise ImportError(
        "python-dotenv 라이브러리가 필요합니다. 다음 명령어로 설치하세요: pip install python-dotenv"
    )

from .base import VLMHandler

# .env 파일 자동 로드
load_dotenv()


class GemmaHandler(VLMHandler):
    """
    Gemma Vision Language Model 핸들러
    
    지원 모델 (파라미터 크기별):
    - google/gemma-2b-it: 2B 파라미터 (경량, 비전 미지원)
    - google/gemma-7b-it: 7B 파라미터 (중간, 비전 미지원)
    - google/gemma-2-2b-it: 2B 파라미터 (Gemma 2)
    - google/gemma-2-9b-it: 9B 파라미터 (Gemma 2)
    - google/gemma-2-27b-it: 27B 파라미터 (Gemma 2, 대형)
    
    주의: Gemma는 기본적으로 비전을 지원하지 않지만, 
    일부 파인튜닝된 버전이나 멀티모달 확장이 있을 수 있습니다.
    
    사용 예시:
        # 작은 모델 사용 (빠르지만 정확도 낮음)
        handler = GemmaHandler(model="google/gemma-2-2b-it")
        
        # 중간 모델 사용 (균형잡힌 성능)
        handler = GemmaHandler(model="google/gemma-2-9b-it")
        
        # 큰 모델 사용 (느리지만 정확도 높음)
        handler = GemmaHandler(model="google/gemma-2-27b-it")
        
    참고: 실제 비전 지원 여부는 모델에 따라 다를 수 있습니다.
    """
    
    def __init__(
        self,
        model: str = "google/gemma-2-9b-it",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Gemma 핸들러 초기화
        
        Args:
            model: 사용할 모델명
                - "google/gemma-2-2b-it": 2B 파라미터
                - "google/gemma-2-9b-it": 9B 파라미터 (기본값)
                - "google/gemma-2-27b-it": 27B 파라미터
            temperature: 생성 온도 (기본값: 0.0)
            max_tokens: 최대 토큰 수 (기본값: 1000)
            device: 사용할 디바이스 ("cuda", "cpu", None=자동)
            **kwargs: 추가 설정
        """
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            device=device,
            **kwargs
        )
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.device = device
        
        self._model = None
        self._tokenizer = None
        self._processor = None
    
    def initialize(self) -> bool:
        """
        Gemma 모델 초기화
        
        Returns:
            초기화 성공 여부
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
            import torch
            
            # 디바이스 설정
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # 모델 로드
            # 주의: Gemma는 기본적으로 비전을 지원하지 않으므로,
            # 비전 지원 버전이 있다면 해당 모델을 사용해야 합니다.
            try:
                # 비전 프로세서 시도
                self._processor = AutoProcessor.from_pretrained(self.model)
                self._tokenizer = self._processor.tokenizer
            except:
                # 일반 토크나이저 사용
                self._tokenizer = AutoTokenizer.from_pretrained(self.model)
                self._processor = None
            
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                self._model = self._model.to(self.device)
            
            return True
            
        except ImportError:
            raise ImportError(
                "transformers 라이브러리가 필요합니다. "
                "다음 명령어로 설치하세요: pip install transformers torch"
            )
        except Exception as e:
            raise RuntimeError(f"Gemma 모델 초기화 실패: {e}")
    
    def encode_image(self, image: Union[str, Path, np.ndarray, Image.Image]) -> Image.Image:
        """
        이미지를 PIL Image로 변환
        
        Args:
            image: 입력 이미지
            
        Returns:
            PIL Image
        """
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
            image: 입력 이미지 (Gemma는 비전 미지원이므로 None 가능)
            system_prompt: 시스템 프롬프트
            user_prompt: 사용자 프롬프트
            
        Returns:
            원본 응답 텍스트 (str)
        """
        if self._model is None:
            self.initialize()
        
        import torch
        
        # 프롬프트 결합
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
        else:
            full_prompt = user_prompt
        
        try:
            # 이미지가 있는 경우 처리
            if image is not None:
                # 주의: Gemma는 기본적으로 비전을 지원하지 않습니다.
                # 비전 지원 버전이 있다면 해당 프로세서를 사용해야 합니다.
                if self._processor is not None:
                    # 비전 프로세서 사용
                    pil_image = self.encode_image(image)
                    inputs = self._processor(
                        text=full_prompt,
                        images=[pil_image],
                        return_tensors="pt"
                    )
                else:
                    # 이미지는 무시하고 텍스트만 처리
                    print("경고: 이 Gemma 모델은 비전을 지원하지 않습니다. 텍스트만 처리합니다.")
                    inputs = self._tokenizer(full_prompt, return_tensors="pt")
            else:
                # 텍스트만 처리
                inputs = self._tokenizer(full_prompt, return_tensors="pt")
            
            # 디바이스로 이동
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 생성
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature if self.temperature > 0 else None,
                    do_sample=self.temperature > 0,
                    pad_token_id=self._tokenizer.eos_token_id
                )
            
            # 디코딩
            generated_text = self._tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            raise RuntimeError(f"API 호출 중 오류 발생: {e}")

