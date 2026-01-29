"""
DDTW (Derivative Dynamic Time Warping) metric.

Uses derivatives (velocity/direction vectors) instead of raw positions for DTW.
This naturally penalizes stops (derivative=0) and backtracking (opposite direction).
"""

import numpy as np
from typing import Union
from .dtw import dtw_distance


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
    and then applies DTW on the derivative sequences.
    
    This approach:
    - Penalizes stops: if robot stops (derivative=0) while reference moves, large cost
    - Penalizes backtracking: opposite direction vectors result in large distance
    - Preserves kinematic constraints in the comparison
    
    Args:
        trajectory1: First trajectory, shape (N, 2) with [x, y] coordinates
        trajectory2: Second trajectory, shape (M, 2) with [x, y] coordinates
        
    Returns:
        DDTW distance (non-negative float)
    """
    if len(trajectory1) == 0 or len(trajectory2) == 0:
        return np.inf
    
    # Compute derivatives
    deriv1 = compute_derivatives(trajectory1)
    deriv2 = compute_derivatives(trajectory2)
    
    # Apply DTW on derivatives
    return dtw_distance(deriv1, deriv2)
