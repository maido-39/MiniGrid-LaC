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
    from google import genai
    from google.genai import types
except ImportError as exc:
    raise ImportError(
        "google-genai library is required. Install with: pip install google-genai"
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
    - gemini-2.5-flash: Latest 2.5 flash model
    - gemini-pro: Legacy model
    - gemini-pro-vision: Legacy vision model
    
    Usage examples:
        # Use fast model (balanced performance)
        handler = GeminiHandler(model="gemini-1.5-flash", max_tokens=1000)
        
        # Use large model (high accuracy)
        handler = GeminiHandler(model="gemini-1.5-pro", max_tokens=2000)
        
        # Use latest 2.5 flash model
        handler = GeminiHandler(model="gemini-2.5-flash", max_tokens=1000)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-flash",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        thinking_budget: Optional[int] = None,
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
                - "gemini-2.5-flash": Latest 2.5 flash model
                - "gemini-pro": Legacy model
                - "gemini-pro-vision": Legacy vision model
            temperature: Generation temperature (default: 0.0)
            max_tokens: Maximum token count (default: 1000)
                - Flash model: 1000-2000 tokens recommended
                - Pro model: 2000-4000 tokens recommended
            thinking_budget: Thinking budget for Gemini 2.5 Flash model (default: None)
                - None: Use default thinking (enabled by default for gemini-2.5-flash)
                - 0: Disable thinking (faster, lower cost)
                - Positive integer: Set thinking budget in tokens
                - Note: Only supported for gemini-2.5-flash model
            **kwargs: Additional settings
        """
        super().__init__(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Validate thinking_budget
        if thinking_budget is not None:
            model_lower = model.lower()
            if model_lower != "gemini-2.5-flash":
                raise ValueError(
                    f"thinking_budget is only supported for gemini-2.5-flash model. "
                    f"Current model: {model}"
                )
            if thinking_budget < 0:
                raise ValueError(
                    f"thinking_budget must be non-negative. Got: {thinking_budget}"
                )
        
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking_budget = thinking_budget
        
        self.client = None
    
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
            # Initialize client with API key
            # If api_key is provided, use it directly; otherwise client will read from environment
            if self.api_key:
                self.client = genai.Client(api_key=self.api_key)
            else:
                # Client will automatically read from GEMINI_API_KEY environment variable
                self.client = genai.Client()
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
        retry_delay: Optional[float] = None,
        return_metadata: bool = False
    ) -> Union[str, tuple]:
        """
        Call Gemini API with image and prompts, return raw response
        Includes automatic retry logic for rate limiting
        
        Args:
            image: Input image (path, numpy array, or PIL Image). If None, send text only without image
            system_prompt: System prompt (Gemini uses system_instruction parameter)
            user_prompt: User prompt
            max_retries: Maximum number of retries for rate limit errors (default: 3)
            retry_delay: Base retry delay in seconds (default: None, auto-parsed from error)
            return_metadata: If True, return tuple (response, metadata_dict). If False, return only response string.
            
        Returns:
            Raw response text (str) or tuple (str, dict) if return_metadata=True
        """
        # Initialize client if not initialized
        if self.client is None:
            self.initialize()
        
        # Prepare generation config with system_instruction and thinking_budget support
        config_kwargs = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }
        
        # Add system_instruction if provided
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt
        
        # Add thinking_config for gemini-2.5-flash
        model_lower = self.model.lower()
        if model_lower == "gemini-2.5-flash":
            if self.thinking_budget is not None:
                # Explicit thinking_budget specified
                # include_thoughts=True to get thinking content in response
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=self.thinking_budget,
                    include_thoughts=True
                )
            else:
                # None: Use default dynamic thinking (-1)
                # include_thoughts=True to get thinking content in response
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=-1,
                    include_thoughts=True
                )
        elif self.thinking_budget is not None:
            print("[WARNING] thinking_budget is only supported for gemini-2.5-flash. Ignoring setting.")
        
        # Disable all safety filters
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        config_kwargs["safety_settings"] = safety_settings
        
        config = types.GenerateContentConfig(**config_kwargs)
        
        # Prepare contents with image if image exists
        if image is not None:
            # Encode image (with automatic resizing)
            image_bytes = self.encode_image(image, max_size=1024)
            
            # For new genai API, use Part with inline_data
            from google.genai.types import Part
            image_part = Part(inline_data={"mime_type": "image/jpeg", "data": image_bytes})
            contents = [image_part, user_prompt]
        else:
            contents = user_prompt
        
        # Retry logic
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                # Generate with new genai API
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config
                )
                
                # Extract response text
                if not response.text:
                    raise RuntimeError("Empty response from Gemini API")
                
                raw_response = response.text.strip()
                
                # Extract metadata if requested
                if return_metadata:
                    metadata = {}
                    
                    # Extract usage information
                    if hasattr(response, 'usage_metadata'):
                        usage = response.usage_metadata
                        metadata['input_tokens'] = getattr(usage, 'prompt_token_count', None)
                        metadata['output_tokens'] = getattr(usage, 'candidates_token_count', None)
                        metadata['total_tokens'] = getattr(usage, 'total_token_count', None)
                        # Thinking tokens (if available)
                        metadata['thinking_tokens'] = getattr(usage, 'thoughts_token_count', None)
                    
                    # Extract thinking content (if available)
                    # According to Gemini API docs, thinking content is in parts where part.thought == True
                    # part.thought is a boolean flag, and the actual content is in part.text
                    thinking_content = None
                    if hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                            # Look for thought parts
                            thoughts = []
                            for part in candidate.content.parts:
                                # Check if part.thought is True (boolean flag)
                                if hasattr(part, 'thought') and part.thought is True:
                                    # The thinking content is in part.text when part.thought is True
                                    if hasattr(part, 'text') and part.text:
                                        thoughts.append(str(part.text))
                            
                            if thoughts:
                                thinking_content = '\n'.join(thoughts)
                    
                    metadata['thinking_content'] = thinking_content
                    
                    return raw_response, metadata
                
                return raw_response
                    
            except (RuntimeError, ValueError):
                # Re-raise non-API errors immediately
                raise
            except Exception as e:  # pylint: disable=broad-except
                # Catch all other exceptions (API errors, network errors, etc.)
                last_error = e
                error_str = str(e)
                
                # Handle 429 error (quota/rate limit) and 503 error (service unavailable/overloaded)
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
                elif '503' in error_str or 'unavailable' in error_str.lower() or 'overloaded' in error_str.lower():
                    # Handle 503 service unavailable / model overloaded
                    if attempt < max_retries:
                        # Exponential backoff for 503 errors
                        base_delay = 2.0  # Start with 2 seconds
                        wait_time = base_delay * (2 ** attempt) + random.uniform(0.1, 1.0)
                        
                        print(f"[WARNING] Service unavailable (model overloaded). Retrying in {wait_time:.2f}s ({attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
                    else:
                        # Max retries exceeded
                        raise RuntimeError(
                            f"Service unavailable after {max_retries} retries. "
                            f"The model may be overloaded. Please try again later. "
                            f"Error: {error_str}"
                        ) from e
                else:
                    # Non-retryable errors are raised immediately
                    raise RuntimeError(f"Error occurred during API call: {e}") from e
        
        # Should not reach here, but safety check
        if last_error:
            raise last_error
        raise RuntimeError("Unexpected error occurred")

