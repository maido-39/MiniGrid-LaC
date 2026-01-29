"""
DTW (Dynamic Time Warping) metric.

Standard DTW algorithm with Euclidean distance.
"""

import numpy as np
from typing import Union


def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate Euclidean distance between two points."""
    return float(np.linalg.norm(p1 - p2))


def dtw_distance(trajectory1: np.ndarray, trajectory2: np.ndarray) -> float:
    """
    Calculate DTW distance between two trajectories.
    
    Args:
        trajectory1: First trajectory, shape (N, 2) with [x, y] coordinates
        trajectory2: Second trajectory, shape (M, 2) with [x, y] coordinates
        
    Returns:
        DTW distance (non-negative float)
    """
    if len(trajectory1) == 0 or len(trajectory2) == 0:
        return np.inf
    
    n = len(trajectory1)
    m = len(trajectory2)
    
    # Initialize cost matrix
    cost_matrix = np.full((n + 1, m + 1), np.inf)
    cost_matrix[0, 0] = 0.0
    
    # Fill cost matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Cost of matching trajectory1[i-1] with trajectory2[j-1]
            cost = euclidean_distance(trajectory1[i-1], trajectory2[j-1])
            
            # Take minimum of three possible paths
            cost_matrix[i, j] = cost + min(
                cost_matrix[i-1, j],      # Insertion
                cost_matrix[i, j-1],      # Deletion
                cost_matrix[i-1, j-1]    # Match
            )
    
    return float(cost_matrix[n, m])
