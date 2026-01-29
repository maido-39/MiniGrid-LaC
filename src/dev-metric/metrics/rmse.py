"""
RMSE (Root Mean Square Error) metric.

Element-wise Euclidean distance assuming time synchronization.
Uses sklearn.metrics.mean_squared_error for computation.
"""

import numpy as np
from typing import Union

try:
    from sklearn.metrics import mean_squared_error
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")


def rmse_distance(trajectory1: np.ndarray, trajectory2: np.ndarray) -> float:
    """
    Calculate RMSE between two trajectories using sklearn.
    
    Assumes trajectories are time-synchronized (same length).
    If lengths differ, the shorter trajectory is padded or truncated.
    
    Args:
        trajectory1: First trajectory, shape (N, 2) with [x, y] coordinates
        trajectory2: Second trajectory, shape (M, 2) with [x, y] coordinates
        
    Returns:
        RMSE value (non-negative float)
        
    Raises:
        ImportError: If scikit-learn is not installed
    """
    if not HAS_SKLEARN:
        raise ImportError(
            "scikit-learn is required. Install with: pip install scikit-learn"
        )
    
    if len(trajectory1) == 0 or len(trajectory2) == 0:
        return np.inf
    
    # Ensure same length by taking minimum
    min_len = min(len(trajectory1), len(trajectory2))
    traj1 = trajectory1[:min_len]
    traj2 = trajectory2[:min_len]
    
    try:
        # Flatten for sklearn (it expects 1D or 2D arrays)
        # We'll compute MSE for each coordinate and then take mean
        # Or compute MSE on flattened arrays
        mse = mean_squared_error(traj1, traj2, multioutput='uniform_average')
        rmse = np.sqrt(mse)
        return float(rmse)
    except Exception as e:
        print(f"Error computing RMSE: {e}")
        return np.inf
