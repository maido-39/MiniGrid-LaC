"""
VLM Handler Base Class

Defines the abstract base class that all VLM handlers must inherit from.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Union, List
from pathlib import Path
import numpy as np
from PIL import Image


class VLMHandler(ABC):
    """
    VLM Handler Abstract Base Class
    
    All VLM handlers must inherit from this class and implement it.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize handler
        
        Args:
            **kwargs: Initialization parameters for each handler
        """
        self.config = kwargs
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize handler (create API client, etc.)
        
        Returns:
            Whether initialization succeeded
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
        Call VLM API with image and prompts, return raw response
        
        Args:
            image: Input image (path, numpy array, or PIL Image). If None, send text only without image
            system_prompt: System prompt
            user_prompt: User prompt
            
        Returns:
            Raw response text (str)
        """
        pass
    
    @abstractmethod
    def encode_image(
        self,
        image: Union[str, Path, np.ndarray, Image.Image]
    ) -> Union[str, bytes]:
        """
        Encode image to format suitable for VLM API
        
        Args:
            image: Input image
            
        Returns:
            Encoded image (format may vary by handler)
        """
        pass
    
    def get_model_name(self) -> str:
        """
        Return model name in use
        
        Returns:
            Model name string
        """
        return self.config.get("model", "unknown")
    
    def get_supported_image_formats(self) -> List[str]:
        """
        Return supported image formats
        
        Returns:
            List of supported formats (e.g., ["png", "jpg", "jpeg"])
        """
        return ["png", "jpg", "jpeg"]
    
    def __call__(
        self,
        image: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
        system_prompt: str = "",
        user_prompt: str = ""
    ) -> str:
        """
        Use as callable object (for convenience)
        
        Args:
            image: Input image (if None, send text only without image)
            system_prompt: System prompt
            user_prompt: User prompt
            
        Returns:
            Raw response text (str)
        """
        return self.generate(image, system_prompt, user_prompt)

