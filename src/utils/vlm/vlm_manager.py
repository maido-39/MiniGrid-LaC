"""
VLM Manager

Central management class for registering and managing various VLM handlers
"""

from typing import Dict, Optional, Union, List
from pathlib import Path
import numpy as np
from PIL import Image

from .handlers import VLMHandler, AVAILABLE_HANDLERS


class VLMManager:
    """
    VLM Handler Manager
    
    Registers various VLM handlers and selects appropriate handlers by model name or handler name.
    """
    
    def __init__(self):
        """Initialize VLM manager"""
        self._handlers: Dict[str, VLMHandler] = {}
        self._default_handler: Optional[str] = None
    
    def register_handler(
        self,
        name: str,
        handler: VLMHandler,
        set_as_default: bool = False
    ):
        """
        Register handler
        
        Args:
            name: Handler name (e.g., "openai", "gpt-4o")
            handler: VLMHandler instance
            set_as_default: If True, set as default handler
        """
        if not isinstance(handler, VLMHandler):
            raise TypeError(f"handler must be a VLMHandler instance. Received type: {type(handler)}")
        
        self._handlers[name] = handler
        
        if set_as_default or self._default_handler is None:
            self._default_handler = name
    
    def get_handler(self, name: Optional[str] = None) -> VLMHandler:
        """
        Get handler
        
        Args:
            name: Handler name. If None, return default handler
            
        Returns:
            VLMHandler instance
            
        Raises:
            ValueError: If handler not found
        """
        if name is None:
            if self._default_handler is None:
                raise ValueError("No registered handlers. Register a handler with register_handler().")
            name = self._default_handler
        
        if name not in self._handlers:
            raise ValueError(
                f"Handler '{name}' not found. "
                f"Registered handlers: {list(self._handlers.keys())}"
            )
        
        return self._handlers[name]
    
    def create_handler(
        self,
        handler_type: str,
        name: Optional[str] = None,
        set_as_default: bool = False,
        **kwargs
    ) -> VLMHandler:
        """Create and register a VLM handler.
        
        Creates a new VLM handler instance, initializes it, and registers it
        with the manager. This is a convenience method that combines handler
        creation and registration in one step.
        
        Args:
            handler_type: Type of handler to create. Supported types:
                - "openai" or "gpt-4o": OpenAI GPT-4o Vision handler
                - "qwen": Qwen VLM handler (DashScope or HuggingFace)
                - "gemma": Gemma handler (HuggingFace)
                See AVAILABLE_HANDLERS for full list.
            name: Name to register the handler under. If None, uses handler_type.
                This allows multiple handlers of the same type with different
                configurations. Defaults to None.
            set_as_default: If True, sets this handler as the default handler
                for generate() calls. Defaults to False.
            **kwargs: Handler-specific initialization parameters. These are
                passed directly to the handler's __init__ method. Common
                parameters include:
                - api_key: API key for the service
                - model: Model name/identifier
                - temperature: Generation temperature
                - max_tokens: Maximum response tokens
                - device: Device for local models ("cuda", "cpu")
        
        Returns:
            VLMHandler: The created and initialized handler instance.
        
        Raises:
            ValueError: If handler_type is not in AVAILABLE_HANDLERS.
            RuntimeError: If handler initialization fails.
        
        Examples:
            >>> manager = VLMManager()
            >>> 
            >>> # Create OpenAI handler
            >>> handler = manager.create_handler(
            ...     "openai",
            ...     model="gpt-4o",
            ...     temperature=0.0
            ... )
            >>> 
            >>> # Create Qwen handler with DashScope API
            >>> handler = manager.create_handler(
            ...     "qwen",
            ...     name="qwen-dashscope",
            ...     api_type="dashscope",
            ...     model="qwen-vl-max"
            ... )
            >>> 
            >>> # Create and set as default
            >>> handler = manager.create_handler(
            ...     "openai",
            ...     set_as_default=True
            ... )
            >>> # Now manager.generate() uses this handler by default
        """
        if handler_type not in AVAILABLE_HANDLERS:
            raise ValueError(
                f"Unsupported handler type: {handler_type}. "
                f"Available handlers: {list(AVAILABLE_HANDLERS.keys())}"
            )
        
        handler_class = AVAILABLE_HANDLERS[handler_type]
        handler = handler_class(**kwargs)
        handler.initialize()
        
        if name is None:
            name = handler_type
        
        self.register_handler(name, handler, set_as_default)
        
        return handler
    
    def generate(
        self,
        image: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
        system_prompt: str = "",
        user_prompt: str = "",
        handler_name: Optional[str] = None
    ) -> str:
        """Generate a response from a VLM handler.
        
        Sends a request to the specified VLM handler (or default handler)
        with the provided image and prompts, and returns the raw text response.
        This is the main method for interacting with VLMs through the manager.
        
        Args:
            image: Input image for vision analysis. Can be:
                - str: Path to image file
                - Path: Path object to image file
                - numpy.ndarray: RGB image array (H, W, 3), dtype uint8
                - PIL.Image: PIL Image object
                - None: Text-only request (no image)
                Defaults to None.
            system_prompt: System-level prompt defining the VLM's role and
                behavior. Defaults to "".
            user_prompt: User-level prompt with the specific request or question.
                Defaults to "".
            handler_name: Name of the handler to use. If None, uses the
                default handler (set via register_handler or create_handler).
                Defaults to None.
        
        Returns:
            str: Raw text response from the VLM. This is unprocessed and may
                need parsing depending on the use case.
        
        Raises:
            ValueError: If no handlers are registered or if handler_name is
                not found.
            RuntimeError: If the VLM API call fails.
            TypeError: If the image type is not supported by the handler.
        
        Examples:
            >>> manager = VLMManager()
            >>> manager.create_handler("openai", set_as_default=True)
            >>> 
            >>> # Text-only request
            >>> response = manager.generate(
            ...     system_prompt="You are helpful.",
            ...     user_prompt="What is AI?"
            ... )
            >>> 
            >>> # Image + text request
            >>> import numpy as np
            >>> image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            >>> response = manager.generate(
            ...     image=image,
            ...     system_prompt="Analyze this image.",
            ...     user_prompt="What do you see?"
            ... )
            >>> 
            >>> # Use specific handler
            >>> response = manager.generate(
            ...     image=image,
            ...     user_prompt="Describe this.",
            ...     handler_name="qwen-dashscope"
            ... )
        
        Note:
            The manager must have at least one handler registered before calling
            this method. Use create_handler() or register_handler() first.
        """
        handler = self.get_handler(handler_name)
        return handler.generate(image, system_prompt, user_prompt)
    
    def list_handlers(self) -> List[str]:
        """
        Return list of registered handlers
        
        Returns:
            List of handler names
        """
        return list(self._handlers.keys())
    
    def remove_handler(self, name: str):
        """
        Remove handler
        
        Args:
            name: Handler name to remove
        """
        if name in self._handlers:
            del self._handlers[name]
            if self._default_handler == name:
                self._default_handler = None
                if self._handlers:
                    # Set another handler as default
                    self._default_handler = list(self._handlers.keys())[0]
    
    def __call__(
        self,
        image: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
        system_prompt: str = "",
        user_prompt: str = "",
        handler_name: Optional[str] = None
    ) -> str:
        """
        Use as callable object (for convenience)
        
        Args:
            image: Input image (if None, send text only without image)
            system_prompt: System prompt
            user_prompt: User prompt
            handler_name: Handler name to use. If None, use default handler
            
        Returns:
            Raw response text (str)
        """
        return self.generate(image, system_prompt, user_prompt, handler_name)

