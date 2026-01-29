"""
Fréchet Distance metric.

Discrete Fréchet Distance (also known as "dog leash distance").
"""

import numpy as np
from typing import Union


def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate Euclidean distance between two points."""
    return float(np.linalg.norm(p1 - p2))


def frechet_distance_recursive(
    traj1: np.ndarray,
    traj2: np.ndarray,
    i: int,
    j: int,
    memo: np.ndarray
) -> float:
    """
    Recursive helper function for Fréchet distance calculation.
    
    Uses memoization to avoid redundant calculations.
    """
    if memo[i, j] >= 0:
        return memo[i, j]
    
    if i == 0 and j == 0:
        dist = euclidean_distance(traj1[0], traj2[0])
    elif i == 0:
        dist = max(
            frechet_distance_recursive(traj1, traj2, 0, j-1, memo),
            euclidean_distance(traj1[0], traj2[j])
        )
    elif j == 0:
        dist = max(
            frechet_distance_recursive(traj1, traj2, i-1, 0, memo),
            euclidean_distance(traj1[i], traj2[0])
        )
    else:
        dist = max(
            min(
                frechet_distance_recursive(traj1, traj2, i-1, j, memo),
                frechet_distance_recursive(traj1, traj2, i-1, j-1, memo),
                frechet_distance_recursive(traj1, traj2, i, j-1, memo)
            ),
            euclidean_distance(traj1[i], traj2[j])
        )
    
    memo[i, j] = dist
    return dist


def frechet_distance(trajectory1: np.ndarray, trajectory2: np.ndarray) -> float:
    """
    Calculate discrete Fréchet distance between two trajectories.
    
    The Fréchet distance is the minimum leash length needed for a person
    to walk along one trajectory while their dog walks along the other.
    
    Args:
        trajectory1: First trajectory, shape (N, 2) with [x, y] coordinates
        trajectory2: Second trajectory, shape (M, 2) with [x, y] coordinates
        
    Returns:
        Fréchet distance (non-negative float)
    """
    if len(trajectory1) == 0 or len(trajectory2) == 0:
        return np.inf
    
    n = len(trajectory1)
    m = len(trajectory2)
    
    # Memoization table (initialized to -1)
    memo = np.full((n, m), -1.0)
    
    return frechet_distance_recursive(trajectory1, trajectory2, n-1, m-1, memo)
