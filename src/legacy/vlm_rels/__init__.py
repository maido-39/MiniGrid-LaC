"""
Legacy VLM Relations Package

Legacy VLM-related modules maintained for backward compatibility.
New projects should use lib.vlm modules instead.
"""

# Actual path: legacy.vlm_rels.minigrid_vlm_controller
from .minigrid_vlm_controller import MiniGridVLMController
# Actual path: legacy.vlm_rels.minigrid_vlm_helpers
from .minigrid_vlm_helpers import visualize_minigrid_grid_cli

__all__ = [
    "MiniGridVLMController",
    "visualize_minigrid_grid_cli",
]

