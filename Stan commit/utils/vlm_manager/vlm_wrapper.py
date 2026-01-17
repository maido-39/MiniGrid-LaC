######################################################
#                                                    #
#                         VLM                        #
#                       WRAPPER                      #
#                                                    #
######################################################


"""
ChatGPT 4o (VLM) Python Wrapper

This module provides a wrapper class for easily using OpenAI's GPT-4o Vision model.
It handles only VLM calls and basic input/output processing.

Key features:
- Image processing and encoding
- System Prompt and User Prompt input
- API calls and parameter specification
- Return of raw response messages

This module internally utilizes a new handler system (vlm.handlers).
The existing API remains unchanged for compatibility.
"""




######################################################
#                                                    #
#                      LIBRARIES                     #
#                                                    #
######################################################


import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Union

# New Handler System import
from vlm.handlers import OpenAIHandler
from vlm.vlm_manager import VLMManager

# For backward compatibility, VLMManager is also exported.
__all__ = ["ChatGPT4oVLMWrapper", "VLMManager"]




######################################################
#                                                    #
#                       CLASS                        #
#                                                    #
######################################################


class ChatGPT4oVLMWrapper:
    """
    ChatGPT 4o Vision Language Model Wrapper (Compatibility Wrapper)
    
    Accepts image and text prompts, calls the VLM API, and returns the raw response.
    Post-processing (parsing, validation, etc.) must be handled in a separate module.
    
    This class is provided for compatibility with existing code,
    but internally uses the new handler system (OpenAIHandler).
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 1000
    ):
        """
        Wrapper Initialization
        
        Args:
            api_key: OpenAI API key. If None, automatically loaded from environment variable OPENAI_API_KEY or .env file
            model: Model name to use (default: “gpt-4o”)
            temperature: Generation temperature (default: 0.0)
            max_tokens: Maximum token count (default: 1000)
        """
        
        # Internally using OpenAIHandler
        self._handler = OpenAIHandler(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        self._handler.initialize()
        
        # Preserve attributes for compatibility
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.is_gpt5_model = model.startswith("gpt-5") if model else False
    
    def _encode_image(self, image: Union[str, Path, np.ndarray, Image.Image]) -> str:
        """
        Encode image as base64 (maintained for compatibility)
        
        Args:
            image: Image path (str/Path), numpy array, or PIL Image
            
        Returns:
            base64-encoded image string
        """
        
        return self._handler.encode_image(image)
    
    def generate(self,
                 image: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
                 system_prompt: str = "",
                 user_prompt: str = ""
                ) -> str:
        """
        Takes an image and prompt as input, calls the VLM API, and returns the original response.
        
        Args:
            image: Input image (path, numpy array, or PIL Image). If None, sends only text without an image.
            system_prompt: System prompt
            user_prompt: User prompt
            
        Returns:
            Original response text (str)
        """
        
        return self._handler.generate(image, system_prompt, user_prompt)
    
    def __call__(self,
                 image: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
                 system_prompt: str = "",
                 user_prompt: str = ""
                ) -> str:
        """
        Usable as a callable object (for easy usage)
        
        Args:
            image: Input image (None sends text only without an image)
            system_prompt: System prompt
            user_prompt: User prompt
            
        Returns:
            Original response text (str)
        """
        
        return self.generate(image, system_prompt, user_prompt)




######################################################
#                                                    #
#                        MAIN                        #
#                                                    #
######################################################


# Usage Example
if __name__ == "__main__":
    # Wrapper Initialization
    wrapper = ChatGPT4oVLMWrapper()
    
    # Example Usage
    response = wrapper.generate(
        # image="path/to/image.png",
        system_prompt="You are a helpful assistant.",
        user_prompt="describe the miku's characteristics"
    )
    print(response)



