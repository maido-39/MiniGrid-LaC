"""
Legacy Relative Movement Package

Legacy modules for relative movement-based robot control.
These modules are maintained for backward compatibility.
New projects should use lib.map_manager.minigrid_customenv_emoji instead.
"""

# Actual path: legacy.relative_movement.custom_environment
from .custom_environment import CustomRoomWrapper
# Actual path: legacy.relative_movement.custom_environment_relative_movement
from .custom_environment_relative_movement import CustomRoomWrapper as CustomRoomWrapperRelative

__all__ = [
    "CustomRoomWrapper",
    "CustomRoomWrapperRelative",
]

