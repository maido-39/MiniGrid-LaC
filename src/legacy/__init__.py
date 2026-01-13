"""
Legacy Package

Legacy code modules maintained for backward compatibility.
"""

# Relative Movement exports
# Actual path: legacy.relative_movement.custom_environment
from .relative_movement.custom_environment import CustomRoomWrapper
# Actual path: legacy.relative_movement.custom_environment_relative_movement
from .relative_movement.custom_environment_relative_movement import CustomRoomWrapper as CustomRoomWrapperRelative

# VLM Relations exports
# Actual path: legacy.vlm_rels.minigrid_vlm_controller
from .vlm_rels.minigrid_vlm_controller import MiniGridVLMController
# Actual path: legacy.vlm_rels.minigrid_vlm_helpers
from .vlm_rels.minigrid_vlm_helpers import visualize_minigrid_grid_cli

__all__ = [
    "CustomRoomWrapper",
    "CustomRoomWrapperRelative",
    "MiniGridVLMController",
    "visualize_minigrid_grid_cli",
]

