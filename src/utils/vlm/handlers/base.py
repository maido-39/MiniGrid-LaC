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
    
    def _to_pil_image(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        convert_to_rgb: bool = True
    ) -> Image.Image:
        """
        Convert various image formats to PIL Image
        
        Common utility method for all handlers to convert images to PIL format.
        This reduces code duplication across handlers.
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            convert_to_rgb: Whether to convert to RGB mode (default: True)
            
        Returns:
            PIL Image object
            
        Raises:
            FileNotFoundError: If image path doesn't exist
            TypeError: If image type is not supported
        """
        # If path string
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            pil_image = Image.open(image_path)
            if convert_to_rgb:
                pil_image = pil_image.convert("RGB")
            return pil_image
        
        # If PIL Image
        elif isinstance(image, Image.Image):
            if convert_to_rgb:
                return image.convert("RGB")
            return image
        
        # If numpy array
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            if image.dtype != np.uint8:
                # Convert 0-1 range float array to 0-255
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            pil_image = Image.fromarray(image)
            if convert_to_rgb:
                pil_image = pil_image.convert("RGB")
            return pil_image
        
        else:
            raise TypeError(
                f"Unsupported image type: {type(image)}. "
                "Use str, Path, np.ndarray, or PIL.Image."
            )
    
    def _resize_image(
        self,
        image: Image.Image,
        max_size: int = 1024,
        maintain_aspect_ratio: bool = True
    ) -> Image.Image:
        """
        Resize image if it exceeds max_size
        
        Common utility method for all handlers to resize large images.
        
        Args:
            image: PIL Image
            max_size: Maximum size for width or height
            maintain_aspect_ratio: Whether to maintain aspect ratio (default: True)
            
        Returns:
            Resized PIL Image
        """
        width, height = image.size
        if width <= max_size and height <= max_size:
            return image
        
        if maintain_aspect_ratio:
            # Calculate new size maintaining aspect ratio
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            # Square resize
            return image.resize((max_size, max_size), Image.Resampling.LANCZOS)
    
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

