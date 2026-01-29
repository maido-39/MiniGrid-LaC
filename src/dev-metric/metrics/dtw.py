"""
DTW (Dynamic Time Warping) metric.

Standard DTW algorithm using dtaidistance library (C-based fast computation).
"""

import numpy as np
from typing import Union

try:
    from dtaidistance import dtw_ndim
    HAS_DTAIDISTANCE = True
except ImportError:
    HAS_DTAIDISTANCE = False
    print("Warning: dtaidistance not available. Install with: pip install dtaidistance")


def dtw_distance(trajectory1: np.ndarray, trajectory2: np.ndarray) -> float:
    """
    Calculate DTW distance between two trajectories using dtaidistance library.
    
    Uses C-based fast computation for improved performance.
    
    Args:
        trajectory1: First trajectory, shape (N, 2) with [x, y] coordinates
        trajectory2: Second trajectory, shape (M, 2) with [x, y] coordinates
        
    Returns:
        DTW distance (non-negative float)
        
    Raises:
        ImportError: If dtaidistance is not installed
    """
    if not HAS_DTAIDISTANCE:
        raise ImportError(
            "dtaidistance is required. Install with: pip install dtaidistance"
        )
    
    if len(trajectory1) == 0 or len(trajectory2) == 0:
        return np.inf
    
    # dtaidistance expects 2D array where each row is a time point
    # and columns are features (x, y coordinates)
    # This matches our trajectory format
    
    try:
        # dtaidistance requires double precision for 2D trajectories
        traj1_double = np.asarray(trajectory1, dtype=np.double)
        traj2_double = np.asarray(trajectory2, dtype=np.double)
        
        # Use dtw_ndim for multi-dimensional trajectories
        distance = dtw_ndim.distance_fast(traj1_double, traj2_double)
        return float(distance)
    except Exception as e:
        # Fallback: if there's an error, return inf
        print(f"Error computing DTW: {e}")
        return np.inf
