"""
Library Package

Core library modules for MiniGrid-LaC project.
Map manager (minigrid) is lazy-loaded so scripts that only use VLM/prompt do not load gymnasium/minigrid.
"""

# VLM exports (eager: no minigrid dependency)
from .vlm.vlm_wrapper import VLMWrapper, ChatGPT4oVLMWrapper
from .vlm.vlm_postprocessor import VLMResponsePostProcessor
from .vlm.vlm_controller import VLMController
from .vlm.vlm_manager import VLMManager

# Lazy (via __getattr__): MiniGridEmojiWrapper, load_emoji_map_from_json
__all__ = [
    "VLMWrapper",
    "ChatGPT4oVLMWrapper",
    "VLMResponsePostProcessor",
    "VLMController",
    "VLMManager",
]


def __getattr__(name):
    """Lazy-load map_manager (minigrid) so dev-memory etc. do not load gymnasium/minigrid."""
    if name == "MiniGridEmojiWrapper":
        from .map_manager.minigrid_customenv_emoji import MiniGridEmojiWrapper
        return MiniGridEmojiWrapper
    if name == "load_emoji_map_from_json":
        from .map_manager.emoji_map_loader import load_emoji_map_from_json
        return load_emoji_map_from_json
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

