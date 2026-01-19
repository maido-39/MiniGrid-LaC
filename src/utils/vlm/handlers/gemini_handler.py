"""
Gemini VLM Handler

Handler implementation for Google's Gemini Vision Language Model
Based on previous implementation with retry logic and image optimization
"""

import os
import re
import time
import random
from typing import Optional, Union
from pathlib import Path
import numpy as np
from PIL import Image
import io

try:
    import google.generativeai as genai
except ImportError as exc:
    raise ImportError(
        "google-generativeai library is required. Install with: pip install google-generativeai"
    ) from exc

try:
    from dotenv import load_dotenv
except ImportError as exc:
    raise ImportError(
        "python-dotenv library is required. Install with: pip install python-dotenv"
    ) from exc

from .base import VLMHandler

# Automatically load .env file
load_dotenv()


class GeminiHandler(VLMHandler):
    """
    Google Gemini Vision Language Model Handler
    
    Supported models:
    - gemini-1.5-flash: Fast model (balanced performance, default)
    - gemini-1.5-pro: Large model (high accuracy)
    - gemini-1.5-flash-latest: Latest flash model
    - gemini-1.5-pro-latest: Latest pro model
    - gemini-pro: Legacy model
    - gemini-pro-vision: Legacy vision model
    
    Usage examples:
        # Use fast model (balanced performance)
        handler = GeminiHandler(model="gemini-1.5-flash", max_tokens=1000)
        
        # Use large model (high accuracy)
        handler = GeminiHandler(model="gemini-1.5-pro", max_tokens=2000)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-flash",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        **kwargs
    ):
        """
        Initialize Gemini handler
        
        Args:
            api_key: Google API key. If None, automatically loaded from environment variable GEMINI_API_KEY or GOOGLE_API_KEY or .env file
            model: Model name to use (default: "gemini-1.5-flash")
                - "gemini-1.5-flash": Fast model (default)
                - "gemini-1.5-pro": Large model
                - "gemini-1.5-flash-latest": Latest flash model
                - "gemini-1.5-pro-latest": Latest pro model
                - "gemini-pro": Legacy model
                - "gemini-pro-vision": Legacy vision model
            temperature: Generation temperature (default: 0.0)
            max_tokens: Maximum token count (default: 1000)
                - Flash model: 1000-2000 tokens recommended
                - Pro model: 2000-4000 tokens recommended
            **kwargs: Additional settings
        """
        super().__init__(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.client = None
        self.generative_model = None
    
    def initialize(self) -> bool:
        """
        Initialize Gemini client
        
        Returns:
            Whether initialization succeeded
        """
        # API key priority: argument > environment variable > .env file
        if self.api_key is None:
            # Try GEMINI_API_KEY first, then GOOGLE_API_KEY
            self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if self.api_key is None:
                raise ValueError(
                    "API key not provided. "
                    "Use one of the following:\n"
                    "1. Pass directly via __init__(api_key='your-key')\n"
                    "2. Set environment variable GEMINI_API_KEY or GOOGLE_API_KEY\n"
                    "3. Add GEMINI_API_KEY=your-key to .env file"
                )
        
        try:
            genai.configure(api_key=self.api_key)
            self.generative_model = genai.GenerativeModel(self.model)
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini client: {e}") from e
    
    def encode_image(self, image: Union[str, Path, np.ndarray, Image.Image], max_size: int = 1024) -> bytes:
        """
        Encode image to bytes (Gemini API accepts bytes directly)
        Automatically resizes large images to optimize API usage
        
        Uses base handler's _to_pil_image and _resize_image methods for common functionality.
        
        Args:
            image: Image path (str/Path), numpy array, or PIL Image
            max_size: Maximum size for width or height (default: 1024)
            
        Returns:
            Image bytes (JPEG format)
        """
        # Use base handler's utility method to convert to PIL Image
        pil_image = self._to_pil_image(image, convert_to_rgb=True)
        
        # Use base handler's utility method to resize if needed
        pil_image = self._resize_image(pil_image, max_size=max_size)
        
        # Save as JPEG bytes
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG", quality=85)
        return buffered.getvalue()
    
    def _parse_retry_after(self, error_str: str, default_delay: Optional[float] = None) -> float:
        """
        Parse retry delay from error message
        
        Args:
            error_str: Error message string
            default_delay: Default delay in seconds
        
        Returns:
            Retry delay in seconds
        """
        # "Please retry in X.XXXXXXs" pattern
        retry_pattern = r'retry in ([\d.]+)s'
        match = re.search(retry_pattern, error_str, re.IGNORECASE)
        
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        
        # Use default if provided
        if default_delay is not None:
            return default_delay
        
        # Default to 5 seconds
        return 5.0
    
    def generate(
        self,
        image: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
        system_prompt: str = "",
        user_prompt: str = "",
        max_retries: int = 3,
        retry_delay: Optional[float] = None
    ) -> str:
        """
        Call Gemini API with image and prompts, return raw response
        Includes automatic retry logic for rate limiting
        
        Args:
            image: Input image (path, numpy array, or PIL Image). If None, send text only without image
            system_prompt: System prompt (Gemini uses system_instruction parameter)
            user_prompt: User prompt
            max_retries: Maximum number of retries for rate limit errors (default: 3)
            retry_delay: Base retry delay in seconds (default: None, auto-parsed from error)
            
        Returns:
            Raw response text (str)
        """
        # Initialize client if not initialized
        if self.generative_model is None:
            self.initialize()
        
        # Combine system prompt and user prompt
        # Gemini doesn't have separate system/user roles, so we combine them
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
        else:
            full_prompt = user_prompt
        
        # Prepare generation config
        generation_config = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }
        
        # Prepare image part if image exists
        image_part = None
        if image is not None:
            # Encode image (with automatic resizing)
            image_bytes = self.encode_image(image, max_size=1024)
            
            # Create image part
            image_part = {
                "mime_type": "image/jpeg",
                "data": image_bytes
            }
        
        # Retry logic
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                if image_part is not None:
                    # Generate with image
                    response = self.generative_model.generate_content(
                        [full_prompt, image_part],
                        generation_config=generation_config
                    )
                else:
                    # Generate text only
                    response = self.generative_model.generate_content(
                        full_prompt,
                        generation_config=generation_config
                    )
                
                # Extract and return response text
                if response.text:
                    return response.text.strip()
                else:
                    raise RuntimeError("Empty response from Gemini API")
                    
            except (RuntimeError, ValueError):
                # Re-raise non-API errors immediately
                raise
            except Exception as e:  # pylint: disable=broad-except
                # Catch all other exceptions (API errors, network errors, etc.)
                last_error = e
                error_str = str(e)
                
                # Handle 429 error (quota/rate limit)
                if '429' in error_str or 'quota' in error_str.lower() or 'rate limit' in error_str.lower():
                    # Parse retry delay from error message
                    retry_after = self._parse_retry_after(error_str, retry_delay)
                    
                    if attempt < max_retries:
                        # Add jitter to prevent simultaneous retries
                        jitter = random.uniform(0.1, 0.5)
                        wait_time = retry_after + jitter
                        
                        print(f"[WARNING] Rate limit error. Retrying in {wait_time:.2f}s ({attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
                    else:
                        # Max retries exceeded
                        raise RuntimeError(
                            f"Rate limit error after {max_retries} retries. "
                            f"Error: {error_str}"
                        ) from e
                else:
                    # Non-rate-limit errors are raised immediately
                    raise RuntimeError(f"Error occurred during API call: {e}") from e
        
        # Should not reach here, but safety check
        if last_error:
            raise last_error
        raise RuntimeError("Unexpected error occurred")

