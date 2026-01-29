"""
Sobolev Metric (H¹ Norm).

Combines position error and velocity error in a weighted sum.
Mathematically rigorous definition of "position and velocity similarity".
"""

import numpy as np
from typing import Union, Optional


def compute_velocity(trajectory: np.ndarray) -> np.ndarray:
    """
    Compute velocity vectors from trajectory using np.gradient.
    
    np.gradient provides more accurate derivatives than simple differences,
    especially at boundaries.
    
    Args:
        trajectory: Trajectory, shape (N, 2) with [x, y] coordinates
        
    Returns:
        Velocity vectors, shape (N, 2) with [vx, vy] components
    """
    if len(trajectory) == 0:
        return np.array([]).reshape(0, 2)
    
    if len(trajectory) == 1:
        return np.array([[0.0, 0.0]])
    
    velocities = np.zeros_like(trajectory)
    
    # Use np.gradient for more accurate derivatives
    # np.gradient handles boundaries automatically
    for dim in range(trajectory.shape[1]):
        velocities[:, dim] = np.gradient(trajectory[:, dim])
    
    return velocities


def interpolate_to_same_length(
    traj1: np.ndarray,
    traj2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate trajectories to same length using linear interpolation.
    
    Args:
        traj1: First trajectory
        traj2: Second trajectory
        
    Returns:
        Tuple of (interpolated_traj1, interpolated_traj2) with same length
    """
    if len(traj1) == len(traj2):
        return traj1, traj2
    
    # Use longer length
    target_len = max(len(traj1), len(traj2))
    
    # Create interpolation indices
    indices1 = np.linspace(0, len(traj1) - 1, target_len)
    indices2 = np.linspace(0, len(traj2) - 1, target_len)
    
    # Interpolate
    interp_traj1 = np.array([
        np.interp(indices1, np.arange(len(traj1)), traj1[:, dim])
        for dim in range(traj1.shape[1])
    ]).T
    
    interp_traj2 = np.array([
        np.interp(indices2, np.arange(len(traj2)), traj2[:, dim])
        for dim in range(traj2.shape[1])
    ]).T
    
    return interp_traj1, interp_traj2


def sobolev_distance(
    trajectory1: np.ndarray,
    trajectory2: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0
) -> float:
    """
    Calculate Sobolev distance (H¹ norm) between two trajectories.
    
    Sobolev distance combines:
    - Position error: ||traj1 - traj2||²
    - Velocity error: ||vel1 - vel2||²
    
    Distance = sqrt(alpha * position_error² + beta * velocity_error²)
    
    Args:
        trajectory1: First trajectory, shape (N, 2) with [x, y] coordinates
        trajectory2: Second trajectory, shape (M, 2) with [x, y] coordinates
        alpha: Weight for position error (default: 1.0)
        beta: Weight for velocity error (default: 1.0)
        
    Returns:
        Sobolev distance (non-negative float)
    """
    if len(trajectory1) == 0 or len(trajectory2) == 0:
        return np.inf
    
    # Interpolate to same length
    traj1_interp, traj2_interp = interpolate_to_same_length(trajectory1, trajectory2)
    
    # Compute velocities
    vel1 = compute_velocity(traj1_interp)
    vel2 = compute_velocity(traj2_interp)
    
    # Position error (L² norm)
    position_diff = traj1_interp - traj2_interp
    position_error_sq = np.sum(position_diff ** 2)
    
    # Velocity error (L² norm)
    velocity_diff = vel1 - vel2
    velocity_error_sq = np.sum(velocity_diff ** 2)
    
    # Sobolev norm (H¹)
    sobolev_dist_sq = alpha * position_error_sq + beta * velocity_error_sq
    sobolev_dist = np.sqrt(sobolev_dist_sq)
    
    return float(sobolev_dist)
