"""
Qwen VLM Handler

Handler implementation for Alibaba's Qwen Vision Language Model
Available through DashScope API or Hugging Face.
"""

import base64
import os
from typing import Optional, Union
from pathlib import Path
import numpy as np
from PIL import Image
import io

try:
    from dotenv import load_dotenv
except ImportError:
    raise ImportError(
        "python-dotenv library is required. Install with: pip install python-dotenv"
    )

from .base import VLMHandler

# Automatically load .env file
load_dotenv()


class QwenHandler(VLMHandler):
    """
    Qwen Vision Language Model Handler
    
    Supported models (by parameter size):
    - Qwen2-VL-2B-Instruct: 2B parameters (lightweight)
    - Qwen2-VL-7B-Instruct: 7B parameters (medium)
    - Qwen2-VL-72B-Instruct: 72B parameters (large)
    - Qwen2.5-VL-3B-Instruct: 3B parameters
    - Qwen2.5-VL-7B-Instruct: 7B parameters
    - Qwen2.5-VL-32B-Instruct: 32B parameters (default)
    
    Usage examples:
        # Use small model (fast but lower accuracy)
        handler = QwenHandler(model="Qwen2-VL-2B-Instruct", api_type="dashscope")
        
        # Use medium model (balanced performance)
        handler = QwenHandler(model="Qwen2-VL-7B-Instruct", api_type="dashscope")
        
        # Use large model (slow but high accuracy)
        handler = QwenHandler(model="Qwen2-VL-72B-Instruct", api_type="dashscope")
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "Qwen2.5-VL-32B-Instruct",
        api_type: str = "dashscope",  # "dashscope" or "huggingface"
        temperature: float = 0.0,
        max_tokens: int = 1000,
        **kwargs
    ):
        """
        Initialize Qwen handler
        
        Args:
            api_key: API key (required for DashScope)
            model: Model name to use
                - DashScope: "Qwen2-VL-2B-Instruct", "Qwen2-VL-7B-Instruct", "Qwen2-VL-72B-Instruct"
                - HuggingFace: "Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-7B-Instruct", etc.
            api_type: API type ("dashscope" or "huggingface")
            temperature: Generation temperature (default: 0.0)
            max_tokens: Maximum token count (default: 1000)
            **kwargs: Additional settings
        """
        super().__init__(
            api_key=api_key,
            model=model,
            api_type=api_type,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        self.api_key = api_key
        self.model = model
        self.api_type = api_type
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.client = None
        self._model = None  # For HuggingFace model
    
    def initialize(self) -> bool:
        """
        Initialize Qwen client
        
        Returns:
            Whether initialization succeeded
        """
        if self.api_type == "dashscope":
            try:
                import dashscope
                from dashscope import MultiModalConversation
                
                if self.api_key is None:
                    self.api_key = os.getenv("DASHSCOPE_API_KEY")
                    if self.api_key is None:
                        raise ValueError(
                            "DashScope API key is required. "
                            "Use one of the following:\n"
                            "1. Pass directly via __init__(api_key='your-key')\n"
                            "2. Set environment variable DASHSCOPE_API_KEY\n"
                            "3. Add DASHSCOPE_API_KEY=your-key to .env file"
                        )
                
                dashscope.api_key = self.api_key
                self.client = MultiModalConversation
                return True
            except ImportError:
                raise ImportError(
                    "dashscope library is required. "
                    "Install with: pip install dashscope"
                )
        
        elif self.api_type == "huggingface":
            try:
                import torch
            except ImportError:
                raise ImportError(
                    "torch library is required. "
                    "Install with: pip install torch"
                )
            
            try:
                from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            except ImportError as e:
                error_msg = str(e)
                if "torchvision" in error_msg.lower():
                    raise ImportError(
                        f"torchvision is required to use Qwen2-VL model. "
                        f"Install with: pip install torchvision\n"
                        f"Original error: {e}"
                    )
                else:
                    raise ImportError(
                        f"Cannot import Qwen2VLForConditionalGeneration. "
                        f"Latest version of transformers library may be required. "
                        f"Upgrade with: pip install --upgrade transformers\n"
                        f"Original error: {e}"
                    )
            
            try:
                self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                self.processor = AutoProcessor.from_pretrained(self.model)
                return True
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load Qwen model: {e}\n"
                    f"Model name: {self.model}\n"
                    f"Check transformers version or verify model name is correct."
                )
        else:
            raise ValueError(f"Unsupported API type: {self.api_type}. Use 'dashscope' or 'huggingface'.")
    
    def encode_image(self, image: Union[str, Path, np.ndarray, Image.Image]) -> Union[str, bytes]:
        """
        Encode image
        
        Args:
            image: Input image
            
        Returns:
            Encoded image
        """
        # DashScope uses base64, HuggingFace uses PIL Image
        if self.api_type == "dashscope":
            # base64 encoding
            if isinstance(image, (str, Path)):
                image_path = Path(image)
                if not image_path.exists():
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
            
            elif isinstance(image, Image.Image):
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                return base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            elif isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                
                pil_image = Image.fromarray(image)
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                return base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            else:
                raise TypeError(f"Unsupported image type: {type(image)}")
        
        else:  # huggingface
            # Convert to PIL Image
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
            image: Input image
            system_prompt: System prompt
            user_prompt: User prompt
            
        Returns:
            Raw response text (str)
        """
        if self.client is None and self._model is None:
            self.initialize()
        
        # Combine prompts
        full_prompt = f"{system_prompt}\n{user_prompt}".strip() if system_prompt else user_prompt
        
        try:
            if self.api_type == "dashscope":
                # Use DashScope API
                if image is None:
                    messages = [{"role": "user", "content": [{"text": full_prompt}]}]
                else:
                    image_data = self.encode_image(image)
                    messages = [{
                        "role": "user",
                        "content": [
                            {"image": f"data:image/png;base64,{image_data}"},
                            {"text": full_prompt}
                        ]
                    }]
                
                response = self.client.call(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                if response.status_code == 200:
                    return response.output.choices[0].message.content[0].text
                else:
                    raise RuntimeError(f"DashScope API call failed: {response.message}")
            
            else:  # huggingface
                # Use HuggingFace Transformers
                import torch
                
                if image is None:
                    raise ValueError("HuggingFace mode requires an image.")
                
                pil_image = self.encode_image(image)
                
                # Format prompts
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_image},
                            {"type": "text", "text": full_prompt}
                        ]
                    }
                ]
                
                # Generate text with apply_chat_template
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                # Call processor directly to process image and text together
                inputs = self.processor(
                    text=[text],
                    images=[pil_image],
                    padding=True,
                    return_tensors="pt"
                )
                
                inputs = inputs.to(self._model.device)
                
                with torch.no_grad():
                    generated_ids = self._model.generate(**inputs, max_new_tokens=self.max_tokens)
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                
                return output_text[0]
                
        except Exception as e:
            raise RuntimeError(f"Error occurred during API call: {e}")

