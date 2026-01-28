"""
TWED (Time Warp Edit Distance) metric.

Edit distance with explicit penalty for time warping (stiffness parameter).
"""

import numpy as np
from typing import Union, Optional


def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate Euclidean distance between two points."""
    return float(np.linalg.norm(p1 - p2))


def twed_distance(
    trajectory1: np.ndarray,
    trajectory2: np.ndarray,
    nu: float = 0.5,
    lambda_param: float = 1.0
) -> float:
    """
    Calculate TWED distance between two trajectories.
    
    TWED adds explicit penalty for time warping through the stiffness parameter (nu).
    Higher nu means less tolerance for time distortion.
    
    Args:
        trajectory1: First trajectory, shape (N, 2) with [x, y] coordinates
        trajectory2: Second trajectory, shape (M, 2) with [x, y] coordinates
        nu: Stiffness parameter (penalty for time warping). Higher = less warping allowed.
        lambda_param: Penalty for deleting/inserting elements
        
    Returns:
        TWED distance (non-negative float)
    """
    if len(trajectory1) == 0 or len(trajectory2) == 0:
        return np.inf
    
    n = len(trajectory1)
    m = len(trajectory2)
    
    # Initialize cost matrix
    cost_matrix = np.full((n + 1, m + 1), np.inf)
    cost_matrix[0, 0] = 0.0
    
    # Fill cost matrix
    for i in range(0, n + 1):
        for j in range(0, m + 1):
            if i == 0 and j == 0:
                continue
            
            # Option 1: Match trajectory1[i-1] with trajectory2[j-1]
            if i > 0 and j > 0:
                match_cost = (
                    cost_matrix[i-1, j-1] +
                    euclidean_distance(trajectory1[i-1], trajectory2[j-1]) +
                    nu * abs(i - j)  # Time warping penalty
                )
            else:
                match_cost = np.inf
            
            # Option 2: Delete from trajectory1 (match trajectory1[i-1] with nothing)
            if i > 0:
                delete_cost = (
                    cost_matrix[i-1, j] +
                    euclidean_distance(trajectory1[i-1], trajectory1[i-2] if i > 1 else trajectory1[i-1]) +
                    lambda_param +
                    nu * abs(i - 1 - j)  # Time warping penalty
                )
            else:
                delete_cost = np.inf
            
            # Option 3: Insert into trajectory1 (match nothing with trajectory2[j-1])
            if j > 0:
                insert_cost = (
                    cost_matrix[i, j-1] +
                    euclidean_distance(trajectory2[j-1], trajectory2[j-2] if j > 1 else trajectory2[j-1]) +
                    lambda_param +
                    nu * abs(i - (j - 1))  # Time warping penalty
                )
            else:
                insert_cost = np.inf
            
            cost_matrix[i, j] = min(match_cost, delete_cost, insert_cost)
    
    return float(cost_matrix[n, m])
