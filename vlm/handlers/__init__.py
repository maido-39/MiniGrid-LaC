"""
VLM 핸들러 패키지

각 VLM 제공업체별 핸들러를 등록하고 관리합니다.
"""

from .base import VLMHandler
from .openai_handler import OpenAIHandler

# Qwen과 Gemma 핸들러는 선택적 import (필요한 라이브러리가 없을 수 있음)
try:
    from .qwen_handler import QwenHandler
except ImportError:
    QwenHandler = None

try:
    from .gemma_handler import GemmaHandler
except ImportError:
    GemmaHandler = None

# 등록 가능한 핸들러 목록
AVAILABLE_HANDLERS = {
    "openai": OpenAIHandler,
    "gpt-4o": OpenAIHandler,
    "gpt-4o-mini": OpenAIHandler,
    "gpt-4": OpenAIHandler,
    "gpt-4-turbo": OpenAIHandler,
    "gpt-5": OpenAIHandler,
}

# Qwen 핸들러 등록
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

# Gemma 핸들러 등록
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
    "AVAILABLE_HANDLERS",
]

# 선택적 핸들러 추가
if QwenHandler is not None:
    __all__.append("QwenHandler")
if GemmaHandler is not None:
    __all__.append("GemmaHandler")

