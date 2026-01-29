"""
ERP (Edit Distance on Real sequence) metric.

Edit distance specialized for handling gaps in real-valued sequences.
Uses similaritymeasures library for computation.
"""

import numpy as np
from typing import Union, Optional

try:
    from aeon.distances import erp_distance as aeon_erp_distance
    HAS_AEON = True
except ImportError:
    HAS_AEON = False

# Always try to load legacy implementation as fallback
try:
    import sys
    from pathlib import Path
    legacy_path = Path(__file__).parent / 'legacy' / 'erp.py'
    if legacy_path.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("legacy_erp", legacy_path)
        legacy_erp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy_erp_module)
        legacy_erp_distance = legacy_erp_module.erp_distance
        HAS_LEGACY_ERP = True
    else:
        HAS_LEGACY_ERP = False
except Exception:
    HAS_LEGACY_ERP = False

if not HAS_AEON and not HAS_LEGACY_ERP:
    print("Warning: Neither aeon nor legacy ERP implementation available")


def erp_distance(
    trajectory1: np.ndarray,
    trajectory2: np.ndarray,
    gap_penalty: Optional[float] = None
) -> float:
    """
    Calculate ERP distance between two trajectories using similaritymeasures library.
    
    ERP handles gaps by using a gap element (g) and calculating the cost
    of matching a point with the gap element.
    
    Args:
        trajectory1: First trajectory, shape (N, 2) with [x, y] coordinates
        trajectory2: Second trajectory, shape (M, 2) with [x, y] coordinates
        gap_penalty: Penalty for matching with gap. If None, uses zero vector (standard).
        
    Returns:
        ERP distance (non-negative float)
        
    Raises:
        ImportError: If similaritymeasures is not installed
    """
    if len(trajectory1) == 0 or len(trajectory2) == 0:
        return np.inf
    
    # Default gap element: zero (standard in ERP literature)
    if gap_penalty is None:
        g_value = 0.0
    else:
        # If gap_penalty is a scalar, use it
        # If it's an array, use the first value or mean
        if isinstance(gap_penalty, (int, float)):
            g_value = float(gap_penalty)
        else:
            gap_element = np.array(gap_penalty)
            g_value = float(np.mean(gap_element))
    
    # Try aeon first, fallback to legacy implementation
    if HAS_AEON:
        try:
            # aeon's erp_distance signature:
            # erp_distance(x, y, window=None, g=0.0, g_arr=None, ...)
            # For 2D trajectories, we can use g_arr for per-dimension gap
            if gap_penalty is None:
                distance = aeon_erp_distance(trajectory1, trajectory2, g=0.0)
            else:
                # If gap_penalty is an array, use g_arr
                if isinstance(gap_penalty, (int, float)):
                    distance = aeon_erp_distance(trajectory1, trajectory2, g=g_value)
                else:
                    gap_arr = np.array(gap_penalty)
                    distance = aeon_erp_distance(trajectory1, trajectory2, g_arr=gap_arr)
            return float(distance)
        except Exception as e:
            print(f"aeon ERP error, using legacy: {e}")
            if HAS_LEGACY_ERP:
                return legacy_erp_distance(trajectory1, trajectory2, gap_penalty)
            else:
                raise
    elif HAS_LEGACY_ERP:
        # Use legacy custom implementation
        return legacy_erp_distance(trajectory1, trajectory2, gap_penalty)
    else:
        raise ImportError(
            "ERP requires either aeon or legacy implementation. "
            "Install with: pip install aeon"
        )
