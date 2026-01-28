"""
ERP (Edit Distance on Real sequence) metric.

Edit distance specialized for handling gaps in real-valued sequences.
"""

import numpy as np
from typing import Union, Optional


def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate Euclidean distance between two points."""
    return float(np.linalg.norm(p1 - p2))


def erp_distance(
    trajectory1: np.ndarray,
    trajectory2: np.ndarray,
    gap_penalty: Optional[float] = None
) -> float:
    """
    Calculate ERP distance between two trajectories.
    
    ERP handles gaps by using a gap element (g) and calculating the cost
    of matching a point with the gap element.
    
    Args:
        trajectory1: First trajectory, shape (N, 2) with [x, y] coordinates
        trajectory2: Second trajectory, shape (M, 2) with [x, y] coordinates
        gap_penalty: Penalty for matching with gap. If None, uses mean of trajectory1.
        
    Returns:
        ERP distance (non-negative float)
    """
    if len(trajectory1) == 0 or len(trajectory2) == 0:
        return np.inf
    
    n = len(trajectory1)
    m = len(trajectory2)
    
    # Default gap element: zero vector (standard in ERP literature)
    # Using mean makes the metric asymmetric, zero vector is standard
    if gap_penalty is None:
        gap_element = np.zeros(trajectory1.shape[1])  # [0, 0] for 2D
    else:
        # If gap_penalty is a scalar, create a point at that value
        # If it's an array, use it as gap element
        if isinstance(gap_penalty, (int, float)):
            gap_element = np.full(trajectory1.shape[1], gap_penalty)
        else:
            gap_element = np.array(gap_penalty)
    
    # Initialize cost matrix
    cost_matrix = np.zeros((n + 1, m + 1))
    
    # Initialize first row and column (matching with gaps)
    for i in range(1, n + 1):
        cost_matrix[i, 0] = cost_matrix[i-1, 0] + euclidean_distance(trajectory1[i-1], gap_element)
    
    for j in range(1, m + 1):
        cost_matrix[0, j] = cost_matrix[0, j-1] + euclidean_distance(trajectory2[j-1], gap_element)
    
    # Fill cost matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Three operations:
            # 1. Match trajectory1[i-1] with trajectory2[j-1]
            match_cost = cost_matrix[i-1, j-1] + euclidean_distance(trajectory1[i-1], trajectory2[j-1])
            
            # 2. Match trajectory1[i-1] with gap
            gap1_cost = cost_matrix[i-1, j] + euclidean_distance(trajectory1[i-1], gap_element)
            
            # 3. Match trajectory2[j-1] with gap
            gap2_cost = cost_matrix[i, j-1] + euclidean_distance(trajectory2[j-1], gap_element)
            
            cost_matrix[i, j] = min(match_cost, gap1_cost, gap2_cost)
    
    return float(cost_matrix[n, m])
