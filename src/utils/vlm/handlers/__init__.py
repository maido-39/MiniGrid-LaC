"""
VLM Handler Package

Registers and manages handlers for each VLM provider.
"""

from .base import VLMHandler
from .openai_handler import OpenAIHandler
from .gemini_handler import GeminiHandler

# Qwen and Gemma handlers are optional imports (required libraries may not be available)
try:
    from .qwen_handler import QwenHandler
except ImportError:
    QwenHandler = None

try:
    from .gemma_handler import GemmaHandler
except ImportError:
    GemmaHandler = None

# List of available handlers
AVAILABLE_HANDLERS = {
    "openai": OpenAIHandler,
    "gpt-4o": OpenAIHandler,
    "gpt-4o-mini": OpenAIHandler,
    "gpt-4": OpenAIHandler,
    "gpt-4-turbo": OpenAIHandler,
    "gpt-5": OpenAIHandler,
    "gemini": GeminiHandler,
    "gemini-1.5-flash": GeminiHandler,
    "gemini-1.5-pro": GeminiHandler,
    "gemini-1.5-flash-latest": GeminiHandler,
    "gemini-1.5-pro-latest": GeminiHandler,
    "gemini-pro": GeminiHandler,
    "gemini-pro-vision": GeminiHandler,
}

# Register Qwen handler
if QwenHandler is not None:
    AVAILABLE_HANDLERS.update({
        "qwen": QwenHandler,
        "qwen2-vl-2b": QwenHandler,
        "qwen2-vl-7b": QwenHandler,
        "qwen2-vl-72b": QwenHandler,
        "qwen2.5-vl-3b": QwenHandler,
        "qwen2.5-vl-7b": QwenHandler,
        "qwen2.5-vl-32b": QwenHandler,
    })

# Register Gemma handler
if GemmaHandler is not None:
    AVAILABLE_HANDLERS.update({
        "gemma": GemmaHandler,
        "gemma-2-2b": GemmaHandler,
        "gemma-2-9b": GemmaHandler,
        "gemma-2-27b": GemmaHandler,
    })

__all__ = [
    "VLMHandler",
    "OpenAIHandler",
    "GeminiHandler",
    "AVAILABLE_HANDLERS",
]

# Add optional handlers
if QwenHandler is not None:
    __all__.append("QwenHandler")
if GemmaHandler is not None:
    __all__.append("GemmaHandler")
