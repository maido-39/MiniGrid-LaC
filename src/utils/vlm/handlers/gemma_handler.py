"""
Gemma VLM Handler

Handler implementation for Google's Gemma Vision Language Model
Available through Hugging Face Transformers.
"""

import os
from typing import Optional, Union
from pathlib import Path
import numpy as np
from PIL import Image

try:
    from dotenv import load_dotenv
except ImportError:
    raise ImportError(
        "python-dotenv library is required. Install with: pip install python-dotenv"
    )

from .base import VLMHandler

# Automatically load .env file
load_dotenv()


class GemmaHandler(VLMHandler):
    """
    Gemma Vision Language Model Handler
    
    Supported models (by parameter size):
    - google/gemma-2b-it: 2B parameters (lightweight, no vision support)
    - google/gemma-7b-it: 7B parameters (medium, no vision support)
    - google/gemma-2-2b-it: 2B parameters (Gemma 2)
    - google/gemma-2-9b-it: 9B parameters (Gemma 2)
    - google/gemma-2-27b-it: 27B parameters (Gemma 2, large)
    
    Note: Gemma does not support vision by default, but some fine-tuned versions
    or multimodal extensions may be available.
    
    Usage examples:
        # Use small model (fast but lower accuracy)
        handler = GemmaHandler(model="google/gemma-2-2b-it")
        
        # Use medium model (balanced performance)
        handler = GemmaHandler(model="google/gemma-2-9b-it")
        
        # Use large model (slow but high accuracy)
        handler = GemmaHandler(model="google/gemma-2-27b-it")
        
    Note: Actual vision support may vary by model.
    """
    
    def __init__(
        self,
        model: str = "google/gemma-2-9b-it",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Gemma handler
        
        Args:
            model: Model name to use
                - "google/gemma-2-2b-it": 2B parameters
                - "google/gemma-2-9b-it": 9B parameters (default)
                - "google/gemma-2-27b-it": 27B parameters
            temperature: Generation temperature (default: 0.0)
            max_tokens: Maximum number of tokens (default: 1000)
            device: Device to use ("cuda", "cpu", None=auto)
            **kwargs: Additional settings
        """
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            device=device,
            **kwargs
        )
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.device = device
        
        self._model = None
        self._tokenizer = None
        self._processor = None
    
    def initialize(self) -> bool:
        """
        Initialize Gemma model
        
        Returns:
            Whether initialization succeeded
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
            import torch
            
            # Set device
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load model
            # Note: Gemma does not support vision by default,
            # so if there is a vision-supporting version, that model should be used.
            try:
                # Try vision processor
                self._processor = AutoProcessor.from_pretrained(self.model)
                self._tokenizer = self._processor.tokenizer
            except:
                # Use regular tokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self.model)
                self._processor = None
            
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                self._model = self._model.to(self.device)
            
            return True
            
        except ImportError:
            raise ImportError(
                "transformers library is required. "
                "Install with: pip install transformers torch"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemma model: {e}")
    
    def encode_image(self, image: Union[str, Path, np.ndarray, Image.Image]) -> Image.Image:
        """
        Convert image to PIL Image
        
        Args:
            image: Input image
            
        Returns:
            PIL Image
        """
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            return image.convert("RGB")
        elif isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            return Image.fromarray(image).convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
    
    def generate(
        self,
        image: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
        system_prompt: str = "",
        user_prompt: str = ""
    ) -> str:
        """
        Call VLM API with image and prompts, return raw response
        
        Args:
            image: Input image (Gemma does not support vision, so None is allowed)
            system_prompt: System prompt
            user_prompt: User prompt
            
        Returns:
            Raw response text (str)
        """
        if self._model is None:
            self.initialize()
        
        import torch
        
        # Combine prompts
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
        else:
            full_prompt = user_prompt
        
        try:
            # Process if image exists
            if image is not None:
                # Note: Gemma does not support vision by default.
                # If there is a vision-supporting version, that processor should be used.
                if self._processor is not None:
                    # Use vision processor
                    pil_image = self.encode_image(image)
                    inputs = self._processor(
                        text=full_prompt,
                        images=[pil_image],
                        return_tensors="pt"
                    )
                else:
                    # Ignore image and process text only
                    print("Warning: This Gemma model does not support vision. Processing text only.")
                    inputs = self._tokenizer(full_prompt, return_tensors="pt")
            else:
                # Process text only
                inputs = self._tokenizer(full_prompt, return_tensors="pt")
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature if self.temperature > 0 else None,
                    do_sample=self.temperature > 0,
                    pad_token_id=self._tokenizer.eos_token_id
                )
            
            # Decode
            generated_text = self._tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            raise RuntimeError(f"Error occurred during API call: {e}")

