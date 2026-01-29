"""
TWED (Time Warp Edit Distance) metric.

Edit distance with explicit penalty for time warping (stiffness parameter).
Uses legacy custom implementation (distancia requires timestamps which we don't have).
"""

import numpy as np
from typing import Union, Optional

# Import legacy implementation
import sys
from pathlib import Path
legacy_path = Path(__file__).parent / 'legacy' / 'twed.py'
if legacy_path.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("legacy_twed", legacy_path)
    legacy_twed_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy_twed_module)
    legacy_twed_distance = legacy_twed_module.twed_distance
else:
    # Fallback: define a simple implementation
    def legacy_twed_distance(traj1, traj2, nu, lambda_param):
        raise ImportError("Legacy TWED implementation not found")


def twed_distance(
    trajectory1: np.ndarray,
    trajectory2: np.ndarray,
    nu: float = 0.5,
    lambda_param: float = 1.0
) -> float:
    """
    Calculate TWED distance between two trajectories using legacy implementation.
    
    Note: distancia library requires timestamps which we don't have in our trajectory data.
    Therefore, we use the proven legacy custom implementation.
    
    TWED adds explicit penalty for time warping through the stiffness parameter (nu).
    Higher nu means less tolerance for time distortion.
    
    Args:
        trajectory1: First trajectory, shape (N, 2) with [x, y] coordinates
        trajectory2: Second trajectory, shape (M, 2) with [x, y] coordinates
        nu: Stiffness parameter (penalty for time warping). Higher = less warping allowed.
            Default: 0.5
        lambda_param: Penalty for deleting/inserting elements. Default: 1.0
        
    Returns:
        TWED distance (non-negative float)
    """
    if len(trajectory1) == 0 or len(trajectory2) == 0:
        return np.inf
    
    # Use legacy custom implementation
    return legacy_twed_distance(trajectory1, trajectory2, nu, lambda_param)
