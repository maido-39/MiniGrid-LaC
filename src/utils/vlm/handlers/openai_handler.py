"""
OpenAI VLM Handler

Handler implementation for OpenAI's GPT-4o Vision model
"""

import base64
import os
from typing import Dict, Optional, Union, List
from pathlib import Path
import numpy as np
from PIL import Image
import io

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "openai library is required. Install with: pip install openai"
    )

try:
    from dotenv import load_dotenv
except ImportError:
    raise ImportError(
        "python-dotenv library is required. Install with: pip install python-dotenv"
    )

from .base import VLMHandler

# Automatically load .env file
load_dotenv()


class OpenAIHandler(VLMHandler):
    """
    OpenAI GPT-4o Vision Language Model Handler
    
    Supported models (by parameter size):
    - gpt-4o-mini: Lightweight model (fast but lower accuracy)
    - gpt-4o: Medium model (balanced performance, default)
    - gpt-4-turbo: Large model (slow but high accuracy)
    - gpt-4: Legacy large model
    - gpt-5: Latest large model (if available)
    
    Usage examples:
        # Use lightweight model (fast but lower accuracy)
        handler = OpenAIHandler(model="gpt-4o-mini", max_tokens=500)
        
        # Use medium model (balanced performance)
        handler = OpenAIHandler(model="gpt-4o", max_tokens=1000)
        
        # Use large model (slow but high accuracy)
        handler = OpenAIHandler(model="gpt-4-turbo", max_tokens=2000)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        **kwargs
    ):
        """
        Initialize OpenAI handler
        
        Args:
            api_key: OpenAI API key. If None, automatically loaded from environment variable OPENAI_API_KEY or .env file
            model: Model name to use (default: "gpt-4o")
                - "gpt-4o-mini": Lightweight model
                - "gpt-4o": Medium model (default)
                - "gpt-4-turbo": Large model
                - "gpt-4": Legacy large model
                - "gpt-5": Latest large model
            temperature: Generation temperature (default: 0.0)
            max_tokens: Maximum token count (default: 1000)
                - Lightweight model: 500-1000 tokens recommended
                - Medium model: 1000-2000 tokens recommended
                - Large model: 2000-4000 tokens recommended
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
        
        # Check if GPT-5 series model (model name starts with "gpt-5")
        self.is_gpt5_model = model.startswith("gpt-5") if model else False
        
        self.client = None
    
    def initialize(self) -> bool:
        """
        Initialize OpenAI client
        
        Returns:
            Whether initialization succeeded
        """
        # API key priority: argument > environment variable > .env file
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
            if self.api_key is None:
                raise ValueError(
                    "API key not provided. "
                    "Use one of the following:\n"
                    "1. Pass directly via __init__(api_key='your-key')\n"
                    "2. Set environment variable OPENAI_API_KEY\n"
                    "3. Add OPENAI_API_KEY=your-key to .env file"
                )
        
        try:
            self.client = OpenAI(api_key=self.api_key)
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
    
    def encode_image(self, image: Union[str, Path, np.ndarray, Image.Image]) -> str:
        """
        Encode image to base64
        
        Args:
            image: Image path (str/Path), numpy array, or PIL Image
            
        Returns:
            base64 encoded image string
        """
        # If path string
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        
        # If PIL Image
        elif isinstance(image, Image.Image):
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        
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
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        else:
            raise TypeError(
                f"Unsupported image type: {type(image)}. "
                "Use str, Path, np.ndarray, or PIL.Image."
            )
    
    def _build_messages(
        self,
        image: Optional[Union[str, Path, np.ndarray, Image.Image]],
        system_prompt: str,
        user_prompt: str
    ) -> List[Dict]:
        """
        Build message list for API call
        
        Args:
            image: Input image (if None, send text only without image)
            system_prompt: System prompt
            user_prompt: User prompt
            
        Returns:
            Message list in OpenAI API format
        """
        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        
        # If image exists
        if image is not None:
            # Encode image
            base64_image = self.encode_image(image)
            
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            })
        else:
            # Send text only if no image
            messages.append({
                "role": "user",
                "content": user_prompt
            })
        
        return messages
    
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
        # Initialize client if not initialized
        if self.client is None:
            self.initialize()
        
        # Build messages
        messages = self._build_messages(image, system_prompt, user_prompt)
        
        # Call API
        try:
            # GPT-5 series models use max_completion_tokens, others use max_tokens
            api_params = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature
            }
            
            if self.is_gpt5_model:
                # GPT-5 series: use max_completion_tokens
                api_params["max_completion_tokens"] = self.max_tokens
            else:
                # GPT-4 series: use max_tokens
                api_params["max_tokens"] = self.max_tokens
            
            response = self.client.chat.completions.create(**api_params)
            
            # Extract and return response text
            raw_response = response.choices[0].message.content
            return raw_response
                
        except Exception as e:
            raise RuntimeError(f"Error occurred during API call: {e}")

