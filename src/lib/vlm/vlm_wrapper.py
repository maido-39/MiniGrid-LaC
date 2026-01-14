"""
ChatGPT 4o (VLM) Python Wrapper

This module provides a Wrapper class for easy use of OpenAI's GPT-4o Vision model.
Handles only VLM calls and basic input/output processing.

Key features:
- Image processing and encoding
- System Prompt, User Prompt input
- API calls and parameter specification
- Returns raw response messages

This module internally uses the new handler system (vlm.handlers).
The existing API is maintained for compatibility.
"""

from typing import Optional, Union
from pathlib import Path
import numpy as np
from PIL import Image

# Import new handler system
from .handlers import OpenAIHandler
from .vlm_manager import VLMManager

# Export VLMManager for backward compatibility
__all__ = ["ChatGPT4oVLMWrapper", "VLMManager"]


class ChatGPT4oVLMWrapper:
    """
    ChatGPT 4o Vision Language Model Wrapper (compatibility wrapper)
    
    Receives image and text prompts, calls VLM API, and returns raw response.
    Post-processing (parsing, validation, etc.) should be handled in a separate module.
    
    This class is provided for compatibility with existing code,
    and internally uses the new handler system (OpenAIHandler).
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 1000
    ):
        """
        Initialize wrapper
        
        Args:
            api_key: OpenAI API key. If None, automatically load from environment variable OPENAI_API_KEY or .env file
            model: Model name to use (default: "gpt-4o")
            temperature: Generation temperature (default: 0.0)
            max_tokens: Maximum number of tokens (default: 1000)
        """
        # Use OpenAIHandler internally
        self._handler = OpenAIHandler(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        self._handler.initialize()
        
        # Maintain attributes for compatibility
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.is_gpt5_model = model.startswith("gpt-5") if model else False
    
    def _encode_image(self, image: Union[str, Path, np.ndarray, Image.Image]) -> str:
        """
        Encode image to base64 (maintained for compatibility)
        
        Args:
            image: Image path (str/Path), numpy array, or PIL Image
            
        Returns:
            base64 encoded image string
        """
        return self._handler.encode_image(image)
    
    def generate(
        self,
        image: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
        system_prompt: str = "",
        user_prompt: str = ""
    ) -> str:
        """Generate a response from the VLM using image and text prompts.
        
        Sends the provided image and prompts to the OpenAI GPT-4o Vision API
        and returns the raw text response. This method handles image encoding
        and API communication automatically.
        
        Args:
            image: Input image for vision analysis. Can be:
                - str: Path to image file (e.g., "path/to/image.png")
                - Path: Path object to image file
                - numpy.ndarray: RGB image array of shape (H, W, 3) with dtype uint8
                - PIL.Image: PIL Image object
                - None: No image (text-only request)
                Defaults to None.
            system_prompt: System-level prompt that defines the assistant's
                behavior and context. This sets the overall role and guidelines
                for the VLM. Defaults to "".
            user_prompt: User-level prompt containing the specific request
                or question. This is the main input that the VLM will respond to.
                Defaults to "".
        
        Returns:
            str: Raw text response from the VLM. This is the unprocessed response
                and may need parsing depending on the use case.
        
        Raises:
            RuntimeError: If the API call fails (network error, API error, etc.).
            TypeError: If the image type is not supported.
        
        Examples:
            >>> wrapper = ChatGPT4oVLMWrapper()
            >>> 
            >>> # Text-only request
            >>> response = wrapper.generate(
            ...     system_prompt="You are a helpful assistant.",
            ...     user_prompt="What is the capital of France?"
            ... )
            >>> 
            >>> # Image + text request
            >>> import numpy as np
            >>> image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            >>> response = wrapper.generate(
            ...     image=image,
            ...     system_prompt="You are a robot controller.",
            ...     user_prompt="What action should I take?"
            ... )
            >>> 
            >>> # Using file path
            >>> response = wrapper.generate(
            ...     image="path/to/image.png",
            ...     system_prompt="Analyze this image.",
            ...     user_prompt="Describe what you see."
            ... )
        
        Note:
            The response is returned as raw text. For structured responses
            (e.g., JSON), use VLMResponsePostProcessor to parse the response.
        """
        return self._handler.generate(image, system_prompt, user_prompt)
    
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


# Usage example
if __name__ == "__main__":
    # Initialize wrapper
    wrapper = ChatGPT4oVLMWrapper()
    
    # Example usage
    response = wrapper.generate(
        # image="path/to/image.png",
        system_prompt="You are a helpful assistant.",
        user_prompt="describe the miku's characteristics"
    )
    print(response)

