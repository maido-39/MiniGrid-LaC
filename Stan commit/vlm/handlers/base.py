"""
VLM 핸들러 베이스 클래스

모든 VLM 핸들러가 상속받아야 하는 추상 베이스 클래스를 정의합니다.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Union, List
from pathlib import Path
import numpy as np
from PIL import Image


class VLMHandler(ABC):
    """
    VLM 핸들러 추상 베이스 클래스
    
    모든 VLM 핸들러는 이 클래스를 상속받아 구현해야 합니다.
    """
    
    def __init__(self, **kwargs):
        """
        핸들러 초기화
        
        Args:
            **kwargs: 각 핸들러별 초기화 파라미터
        """
        self.config = kwargs
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        핸들러 초기화 (API 클라이언트 생성 등)
        
        Returns:
            초기화 성공 여부
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def encode_image(
        self,
        image: Union[str, Path, np.ndarray, Image.Image]
    ) -> Union[str, bytes]:
        """
        이미지를 VLM API에 전송 가능한 형식으로 인코딩
        
        Args:
            image: 입력 이미지
            
        Returns:
            인코딩된 이미지 (형식은 핸들러마다 다를 수 있음)
        """
        pass
    
    def get_model_name(self) -> str:
        """
        사용 중인 모델명 반환
        
        Returns:
            모델명 문자열
        """
        return self.config.get("model", "unknown")
    
    def get_supported_image_formats(self) -> List[str]:
        """
        지원하는 이미지 형식 반환
        
        Returns:
            지원 형식 리스트 (예: ["png", "jpg", "jpeg"])
        """
        return ["png", "jpg", "jpeg"]
    
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

