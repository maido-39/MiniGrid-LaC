"""
DDTW (Derivative Dynamic Time Warping) metric.

Uses derivatives (velocity/direction vectors) instead of raw positions for DTW.
This naturally penalizes stops (derivative=0) and backtracking (opposite direction).

Uses dtaidistance library for DTW computation (C-based fast computation).
"""

import numpy as np
from typing import Union

try:
    from dtaidistance import dtw_ndim
    HAS_DTAIDISTANCE = True
except ImportError:
    HAS_DTAIDISTANCE = False
    print("Warning: dtaidistance not available. Install with: pip install dtaidistance")


def compute_derivatives(trajectory: np.ndarray) -> np.ndarray:
    """
    Compute derivatives using Keogh & Pazzani's DDTW formula.
    
    Based on: Keogh, E. J., & Pazzani, M. J. (2001). 
    Derivative Dynamic Time Warping (SIAM ICDM 2002).
    
    Formula: D(x_i) = ((x_{i+1} - x_{i-1}) + (x_i - x_{i-1})/2) / 2
    
    Args:
        trajectory: Trajectory, shape (N, 2) with [x, y] coordinates
        
    Returns:
        Derivatives, shape (N-1, 2) with [dx, dy] velocity vectors
        (Note: Standard DDTW returns len-1 derivatives)
    """
    if len(trajectory) == 0:
        return np.array([]).reshape(0, 2)
    
    if len(trajectory) == 1:
        # Single point: no derivative
        return np.array([]).reshape(0, 2)
    
    n = len(trajectory)
    derivatives = np.zeros((n - 1, trajectory.shape[1]))
    
    for i in range(n - 1):
        if i == 0:
            # First point: forward difference (fallback when i-1 doesn't exist)
            derivatives[i] = trajectory[i + 1] - trajectory[i]
        else:
            # Keogh & Pazzani formula
            # D(x_i) = ((x_{i+1} - x_{i-1}) + (x_i - x_{i-1})/2) / 2
            forward_part = trajectory[i + 1] - trajectory[i - 1]
            backward_part = (trajectory[i] - trajectory[i - 1]) / 2.0
            derivatives[i] = (forward_part + backward_part) / 2.0
    
    return derivatives


def ddtw_distance(trajectory1: np.ndarray, trajectory2: np.ndarray) -> float:
    """
    Calculate DDTW distance between two trajectories.
    
    DDTW computes derivatives (velocity vectors) from both trajectories
    and then applies DTW on the derivative sequences using dtaidistance.
    
    This approach:
    - Penalizes stops: if robot stops (derivative=0) while reference moves, large cost
    - Penalizes backtracking: opposite direction vectors result in large distance
    - Preserves kinematic constraints in the comparison
    
    Args:
        trajectory1: First trajectory, shape (N, 2) with [x, y] coordinates
        trajectory2: Second trajectory, shape (M, 2) with [x, y] coordinates
        
    Returns:
        DDTW distance (non-negative float)
        
    Raises:
        ImportError: If dtaidistance is not installed
    """
    if not HAS_DTAIDISTANCE:
        raise ImportError(
            "dtaidistance is required. Install with: pip install dtaidistance"
        )
    
    if len(trajectory1) == 0 or len(trajectory2) == 0:
        return np.inf
    
    # Compute derivatives
    deriv1 = compute_derivatives(trajectory1)
    deriv2 = compute_derivatives(trajectory2)
    
    if len(deriv1) == 0 or len(deriv2) == 0:
        return np.inf
    
    # Apply DTW on derivatives using dtaidistance
    try:
        # dtaidistance requires double precision for 2D trajectories
        deriv1_double = np.asarray(deriv1, dtype=np.double)
        deriv2_double = np.asarray(deriv2, dtype=np.double)
        
        # Use dtw_ndim for multi-dimensional trajectories
        distance = dtw_ndim.distance_fast(deriv1_double, deriv2_double)
        return float(distance)
    except Exception as e:
        print(f"Error computing DDTW: {e}")
        return np.inf
