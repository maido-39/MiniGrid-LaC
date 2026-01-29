"""
RMSE (Root Mean Square Error) metric.

Element-wise Euclidean distance assuming time synchronization.
"""

import numpy as np
from typing import Union


def rmse_distance(trajectory1: np.ndarray, trajectory2: np.ndarray) -> float:
    """
    Calculate RMSE between two trajectories.
    
    Assumes trajectories are time-synchronized (same length).
    If lengths differ, the shorter trajectory is padded or truncated.
    
    Args:
        trajectory1: First trajectory, shape (N, 2) with [x, y] coordinates
        trajectory2: Second trajectory, shape (M, 2) with [x, y] coordinates
        
    Returns:
        RMSE value (non-negative float)
    """
    if len(trajectory1) == 0 or len(trajectory2) == 0:
        return np.inf
    
    # Ensure same length by taking minimum
    min_len = min(len(trajectory1), len(trajectory2))
    traj1 = trajectory1[:min_len]
    traj2 = trajectory2[:min_len]
    
    # Calculate Euclidean distances for each point pair
    distances = np.linalg.norm(traj1 - traj2, axis=1)
    
    # RMSE
    rmse = np.sqrt(np.mean(distances ** 2))
    
    return float(rmse)
