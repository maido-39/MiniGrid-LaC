"""
Fréchet Distance metric.

Discrete Fréchet Distance (also known as "dog leash distance").
Uses similaritymeasures library for computation.
"""

import numpy as np
from typing import Union

try:
    import similaritymeasures
    HAS_SIMILARITYMEASURES = True
except ImportError:
    HAS_SIMILARITYMEASURES = False
    print("Warning: similaritymeasures not available. Install with: pip install similaritymeasures")


def frechet_distance(trajectory1: np.ndarray, trajectory2: np.ndarray) -> float:
    """
    Calculate discrete Fréchet distance between two trajectories using similaritymeasures.
    
    The Fréchet distance is the minimum leash length needed for a person
    to walk along one trajectory while their dog walks along the other.
    
    Args:
        trajectory1: First trajectory, shape (N, 2) with [x, y] coordinates
        trajectory2: Second trajectory, shape (M, 2) with [x, y] coordinates
        
    Returns:
        Fréchet distance (non-negative float)
        
    Raises:
        ImportError: If similaritymeasures is not installed
    """
    if not HAS_SIMILARITYMEASURES:
        raise ImportError(
            "similaritymeasures is required. Install with: pip install similaritymeasures"
        )
    
    if len(trajectory1) == 0 or len(trajectory2) == 0:
        return np.inf
    
    try:
        # similaritymeasures.frechet_dist expects 2D arrays
        # Returns the Fréchet distance
        distance = similaritymeasures.frechet_dist(trajectory1, trajectory2)
        return float(distance)
    except Exception as e:
        print(f"Error computing Fréchet distance: {e}")
        return np.inf
