"""
VLM 매니저

다양한 VLM 핸들러를 등록하고 관리하는 중앙 관리 클래스
"""

from typing import Dict, Optional, Union, List
from pathlib import Path
import numpy as np
from PIL import Image

from .handlers import VLMHandler, AVAILABLE_HANDLERS


class VLMManager:
    """
    VLM 핸들러 관리자
    
    다양한 VLM 핸들러를 등록하고, 모델명이나 핸들러 이름으로 적절한 핸들러를 선택하여 사용합니다.
    """
    
    def __init__(self):
        """VLM 매니저 초기화"""
        self._handlers: Dict[str, VLMHandler] = {}
        self._default_handler: Optional[str] = None
    
    def register_handler(
        self,
        name: str,
        handler: VLMHandler,
        set_as_default: bool = False
    ):
        """
        핸들러 등록
        
        Args:
            name: 핸들러 이름 (예: "openai", "gpt-4o")
            handler: VLMHandler 인스턴스
            set_as_default: True이면 기본 핸들러로 설정
        """
        if not isinstance(handler, VLMHandler):
            raise TypeError(f"handler는 VLMHandler 인스턴스여야 합니다. 받은 타입: {type(handler)}")
        
        self._handlers[name] = handler
        
        if set_as_default or self._default_handler is None:
            self._default_handler = name
    
    def get_handler(self, name: Optional[str] = None) -> VLMHandler:
        """
        핸들러 가져오기
        
        Args:
            name: 핸들러 이름. None이면 기본 핸들러 반환
            
        Returns:
            VLMHandler 인스턴스
            
        Raises:
            ValueError: 핸들러를 찾을 수 없는 경우
        """
        if name is None:
            if self._default_handler is None:
                raise ValueError("등록된 핸들러가 없습니다. register_handler()로 핸들러를 등록하세요.")
            name = self._default_handler
        
        if name not in self._handlers:
            raise ValueError(
                f"핸들러 '{name}'를 찾을 수 없습니다. "
                f"등록된 핸들러: {list(self._handlers.keys())}"
            )
        
        return self._handlers[name]
    
    def create_handler(
        self,
        handler_type: str,
        name: Optional[str] = None,
        set_as_default: bool = False,
        **kwargs
    ) -> VLMHandler:
        """
        핸들러 생성 및 등록
        
        Args:
            handler_type: 핸들러 타입 (예: "openai", "gpt-4o")
            name: 등록할 이름. None이면 handler_type 사용
            set_as_default: True이면 기본 핸들러로 설정
            **kwargs: 핸들러 초기화 파라미터
            
        Returns:
            생성된 VLMHandler 인스턴스
            
        Raises:
            ValueError: 지원하지 않는 핸들러 타입인 경우
        """
        if handler_type not in AVAILABLE_HANDLERS:
            raise ValueError(
                f"지원하지 않는 핸들러 타입: {handler_type}. "
                f"사용 가능한 핸들러: {list(AVAILABLE_HANDLERS.keys())}"
            )
        
        handler_class = AVAILABLE_HANDLERS[handler_type]
        handler = handler_class(**kwargs)
        handler.initialize()
        
        if name is None:
            name = handler_type
        
        self.register_handler(name, handler, set_as_default)
        
        return handler
    
    def generate(
        self,
        image: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
        system_prompt: str = "",
        user_prompt: str = "",
        handler_name: Optional[str] = None
    ) -> str:
        """
        VLM에 요청 전송 (기본 핸들러 사용)
        
        Args:
            image: 입력 이미지 (None이면 이미지 없이 텍스트만 전송)
            system_prompt: 시스템 프롬프트
            user_prompt: 사용자 프롬프트
            handler_name: 사용할 핸들러 이름. None이면 기본 핸들러 사용
            
        Returns:
            원본 응답 텍스트 (str)
        """
        handler = self.get_handler(handler_name)
        return handler.generate(image, system_prompt, user_prompt)
    
    def list_handlers(self) -> List[str]:
        """
        등록된 핸들러 목록 반환
        
        Returns:
            핸들러 이름 리스트
        """
        return list(self._handlers.keys())
    
    def remove_handler(self, name: str):
        """
        핸들러 제거
        
        Args:
            name: 제거할 핸들러 이름
        """
        if name in self._handlers:
            del self._handlers[name]
            if self._default_handler == name:
                self._default_handler = None
                if self._handlers:
                    # 다른 핸들러를 기본으로 설정
                    self._default_handler = list(self._handlers.keys())[0]
    
    def __call__(
        self,
        image: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
        system_prompt: str = "",
        user_prompt: str = "",
        handler_name: Optional[str] = None
    ) -> str:
        """
        호출 가능한 객체로 사용 (간편한 사용을 위해)
        
        Args:
            image: 입력 이미지 (None이면 이미지 없이 텍스트만 전송)
            system_prompt: 시스템 프롬프트
            user_prompt: 사용자 프롬프트
            handler_name: 사용할 핸들러 이름. None이면 기본 핸들러 사용
            
        Returns:
            원본 응답 텍스트 (str)
        """
        return self.generate(image, system_prompt, user_prompt, handler_name)

