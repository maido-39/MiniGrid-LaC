"""
VLM (Vision Language Model) Python Wrapper

This module provides a Wrapper class for easy use of various VLM models (OpenAI GPT-4o, Gemini, etc.).
Handles only VLM calls and basic input/output processing.

Key features:
- Image processing and encoding
- System Prompt, User Prompt input
- API calls and parameter specification
- Returns raw response messages
- Automatic handler selection based on model name

This module internally uses the new handler system (vlm.handlers).
The existing API is maintained for compatibility.
"""

import time
from typing import Optional, Union
from pathlib import Path
import numpy as np
from PIL import Image

# Import new handler system
from .handlers import OpenAIHandler
from .vlm_manager import VLMManager

# Import GeminiHandler (always check, will raise ImportError if not available)
from .handlers.gemini_handler import GeminiHandler

# Export VLMManager for backward compatibility
__all__ = ["VLMWrapper", "VLMManager", "ChatGPT4oVLMWrapper"]  # ChatGPT4oVLMWrapper is alias for backward compatibility


class VLMWrapper:
    """
    Vision Language Model Wrapper
    
    Receives image and text prompts, calls VLM API, and returns raw response.
    Post-processing (parsing, validation, etc.) should be handled in a separate module.
    
    This class automatically selects the appropriate handler based on model name:
    - Models starting with "gemini": Uses GeminiHandler
    - Other models (gpt-4o, gpt-4, etc.): Uses OpenAIHandler
    
    Supports multiple VLM providers:
    - OpenAI: gpt-4o, gpt-4o-mini, gpt-4, gpt-4-turbo, etc.
    - Google Gemini: gemini-2.5-flash, gemini-1.5-pro, gemini-1.5-flash, etc.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        thinking_budget: Optional[int] = None
    ):
        """
        Initialize wrapper
        
        Args:
            api_key: API key. If None, automatically load from environment variable:
                - For OpenAI models: OPENAI_API_KEY
                - For Gemini models: GEMINI_API_KEY or GOOGLE_API_KEY
            model: Model name to use (default: "gpt-4o")
                - OpenAI models: "gpt-4o", "gpt-4o-mini", "gpt-4", etc.
                - Gemini models: "gemini-2.5-flash", "gemini-1.5-pro", "gemini-1.5-flash", etc.
            temperature: Generation temperature (default: 0.0)
            max_tokens: Maximum number of tokens (default: 1000)
            thinking_budget: Thinking budget for Gemini 2.5 Flash model (default: None)
                - None: Use default thinking (enabled by default for gemini-2.5-flash)
                - 0: Disable thinking (faster, lower cost)
                - Positive integer: Set thinking budget in tokens
                - Note: Only supported for gemini-2.5-flash model
        """
        # Select handler based on model name
        model_lower = model.lower() if model else ""
        
        if model_lower.startswith("gemini"):
            # Use GeminiHandler for Gemini models
            self._handler = GeminiHandler(
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                thinking_budget=thinking_budget
            )
        else:
            # Use OpenAIHandler for OpenAI models (default)
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
        user_prompt: str = "",
        debug: bool = False
    ) -> str:
        """Generate a response from the VLM using image and text prompts.
        
        Sends the provided image and prompts to the VLM API
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
            debug: If True, print detailed debug information about the response.
                Defaults to False.
        
        Returns:
            str: Raw text response from the VLM. This is the unprocessed response
                and may need parsing depending on the use case.
        
        Raises:
            RuntimeError: If the API call fails (network error, API error, etc.).
            TypeError: If the image type is not supported.
        
        Examples:
            >>> wrapper = VLMWrapper()
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
            
            >>> # With debug output
            >>> response = wrapper.generate(
            ...     image=image,
            ...     system_prompt="You are a robot controller.",
            ...     user_prompt="What action should I take?",
            ...     debug=True
            ... )
        
        Note:
            The response is returned as raw text. For structured responses
            (e.g., JSON), use VLMResponsePostProcessor to parse the response.
        """
        # Measure inference time
        start_time = time.time()
        
        # Call handler with metadata if debug is enabled
        if debug:
            result = self._handler.generate(image, system_prompt, user_prompt, return_metadata=True)
            if isinstance(result, tuple):
                response, metadata = result
            else:
                response = result
                metadata = {}
        else:
            response = self._handler.generate(image, system_prompt, user_prompt, return_metadata=False)
            metadata = {}
        
        inference_time = time.time() - start_time
        
        # Debug output
        if debug:
            print("\n" + "="*80)
            print("[DEBUG] RAW VLM RESPONSE:")
            print("="*80)
            print(response)
            print("\n" + "="*80)
            print("[DEBUG] RESPONSE ANALYSIS:")
            print("="*80)
            print(f"  Length: {len(response)} characters")
            print(f"  First 100 chars: {response[:100]}")
            print(f"  Last 100 chars: {response[-100:]}")
            print(f"  Contains '```json': {'```json' in response}")
            print(f"  Number of '```': {response.count('```')}")
            print(f"  Number of '{{': {response.count('{')}")
            print(f"  Number of '}}': {response.count('}')}")
            if "```json" in response:
                start = response.find("```json")
                end = response.find("```", start + 7)
                if end != -1:
                    json_part = response[start+7:end].strip()
                    print(f"  Extracted JSON length: {len(json_part)}")
                    print(f"  Extracted JSON (full):")
                    print(json_part)
                else:
                    print(f"  ⚠️  Found ```json but NO closing ```")
                    json_part = response[start+7:].strip()
                    print(f"  Extracted JSON (no closing, full):")
                    print(json_part)
            print("\n" + "="*80)
            print("[DEBUG] API METADATA:")
            print("="*80)
            print(f"  Inference Time: {inference_time:.3f} seconds")
            if metadata:
                if metadata.get('input_tokens') is not None:
                    print(f"  Input Tokens: {metadata['input_tokens']}")
                if metadata.get('output_tokens') is not None:
                    print(f"  Output Tokens: {metadata['output_tokens']}")
                if metadata.get('total_tokens') is not None:
                    print(f"  Total Tokens: {metadata['total_tokens']}")
                if metadata.get('thinking_tokens') is not None:
                    print(f"  Thinking Tokens: {metadata['thinking_tokens']}")
                if metadata.get('thinking_content'):
                    print(f"  Thinking Content (full):")
                    print(metadata['thinking_content'])
                elif metadata.get('thinking_content') is None and metadata.get('thinking_tokens') is None:
                    print(f"  Thinking: Not available (may require thinking_config)")
            else:
                print(f"  Token information: Not available")
            print("="*80 + "\n")
        
        return response
    
    def __call__(
        self,
        image: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
        system_prompt: str = "",
        user_prompt: str = "",
        debug: bool = False
    ) -> str:
        """
        Use as callable object (for convenience)
        
        Args:
            image: Input image (if None, send text only without image)
            system_prompt: System prompt
            user_prompt: User prompt
            debug: If True, print detailed debug information about the response.
            
        Returns:
            Raw response text (str)
        """
        return self.generate(image, system_prompt, user_prompt, debug=debug)


# Backward compatibility alias
ChatGPT4oVLMWrapper = VLMWrapper

# Usage example
if __name__ == "__main__":
    # Initialize wrapper
    wrapper = VLMWrapper()
    
    # Example usage
    response = wrapper.generate(
        # image="path/to/image.png",
        system_prompt="You are a helpful assistant.",
        user_prompt="describe the miku's characteristics"
    )
    print(response)

