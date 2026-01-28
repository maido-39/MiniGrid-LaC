"""
Step-by-step trajectory analyzer.

Analyzes metrics at each step of the trajectory to understand
local behavior and metric sensitivity.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from collections import defaultdict

# Import metrics
try:
    from metrics import (
        ddtw_distance, twed_distance, sobolev_distance, dtw_distance,
        frechet_distance, erp_distance, rmse_distance
    )
except ImportError:
    from .metrics import (
        ddtw_distance, twed_distance, sobolev_distance, dtw_distance,
        frechet_distance, erp_distance, rmse_distance
    )


class StepAnalyzer:
    """Analyzer for step-by-step trajectory analysis."""
    
    def __init__(self):
        """Initialize step analyzer."""
        self.metric_functions = {
            'RMSE': self._compute_rmse_stepwise,
            'DTW': self._compute_dtw_stepwise,
            'Fréchet': self._compute_frechet_stepwise,
            'ERP': self._compute_erp_stepwise,
            'DDTW': self._compute_ddtw_stepwise,
            'TWED': self._compute_twed_stepwise,
            'Sobolev': self._compute_sobolev_stepwise,
        }
    
    def _compute_rmse_stepwise(
        self, trajectory: np.ndarray, gt_trajectory: np.ndarray
    ) -> np.ndarray:
        """Compute RMSE at each step."""
        min_len = min(len(trajectory), len(gt_trajectory))
        errors = []
        for i in range(min_len):
            error = np.linalg.norm(trajectory[i] - gt_trajectory[i])
            errors.append(error)
        return np.array(errors)
    
    def _compute_dtw_stepwise(
        self, trajectory: np.ndarray, gt_trajectory: np.ndarray
    ) -> np.ndarray:
        """Compute DTW cost at each step (cumulative)."""
        # DTW is global, but we can compute cumulative cost
        # This is an approximation: compute DTW for trajectory up to step i
        costs = []
        for i in range(1, len(trajectory) + 1):
            sub_traj = trajectory[:i]
            # Find best match length in GT
            best_cost = np.inf
            for j in range(1, len(gt_trajectory) + 1):
                sub_gt = gt_trajectory[:j]
                try:
                    cost = dtw_distance(sub_traj, sub_gt)
                    best_cost = min(best_cost, cost)
                except:
                    pass
            costs.append(best_cost if best_cost != np.inf else 0)
        return np.array(costs)
    
    def _compute_frechet_stepwise(
        self, trajectory: np.ndarray, gt_trajectory: np.ndarray
    ) -> np.ndarray:
        """Compute Fréchet distance at each step (cumulative)."""
        # Similar to DTW, compute for trajectory up to step i
        costs = []
        for i in range(1, len(trajectory) + 1):
            sub_traj = trajectory[:i]
            best_cost = np.inf
            for j in range(1, len(gt_trajectory) + 1):
                sub_gt = gt_trajectory[:j]
                try:
                    cost = frechet_distance(sub_traj, sub_gt)
                    best_cost = min(best_cost, cost)
                except:
                    pass
            costs.append(best_cost if best_cost != np.inf else 0)
        return np.array(costs)
    
    def _compute_erp_stepwise(
        self, trajectory: np.ndarray, gt_trajectory: np.ndarray
    ) -> np.ndarray:
        """Compute ERP at each step (cumulative)."""
        costs = []
        for i in range(1, len(trajectory) + 1):
            sub_traj = trajectory[:i]
            best_cost = np.inf
            for j in range(1, len(gt_trajectory) + 1):
                sub_gt = gt_trajectory[:j]
                try:
                    cost = erp_distance(sub_traj, sub_gt)
                    best_cost = min(best_cost, cost)
                except:
                    pass
            costs.append(best_cost if best_cost != np.inf else 0)
        return np.array(costs)
    
    def _compute_ddtw_stepwise(
        self, trajectory: np.ndarray, gt_trajectory: np.ndarray
    ) -> np.ndarray:
        """Compute DDTW at each step (cumulative)."""
        costs = []
        for i in range(2, len(trajectory) + 1):  # Need at least 2 points for derivative
            sub_traj = trajectory[:i]
            best_cost = np.inf
            for j in range(2, len(gt_trajectory) + 1):
                sub_gt = gt_trajectory[:j]
                try:
                    cost = ddtw_distance(sub_traj, sub_gt)
                    best_cost = min(best_cost, cost)
                except:
                    pass
            costs.append(best_cost if best_cost != np.inf else 0)
        # Pad first step
        if len(costs) > 0:
            costs = [costs[0]] + costs
        return np.array(costs)
    
    def _compute_twed_stepwise(
        self, trajectory: np.ndarray, gt_trajectory: np.ndarray
    ) -> np.ndarray:
        """Compute TWED at each step (cumulative)."""
        costs = []
        for i in range(1, len(trajectory) + 1):
            sub_traj = trajectory[:i]
            best_cost = np.inf
            for j in range(1, len(gt_trajectory) + 1):
                sub_gt = gt_trajectory[:j]
                try:
                    cost = twed_distance(sub_traj, sub_gt)
                    best_cost = min(best_cost, cost)
                except:
                    pass
            costs.append(best_cost if best_cost != np.inf else 0)
        return np.array(costs)
    
    def _compute_sobolev_stepwise(
        self, trajectory: np.ndarray, gt_trajectory: np.ndarray
    ) -> np.ndarray:
        """Compute Sobolev at each step (cumulative)."""
        costs = []
        for i in range(2, len(trajectory) + 1):  # Need at least 2 points
            sub_traj = trajectory[:i]
            best_cost = np.inf
            for j in range(2, len(gt_trajectory) + 1):
                sub_gt = gt_trajectory[:j]
                try:
                    cost = sobolev_distance(sub_traj, sub_gt)
                    best_cost = min(best_cost, cost)
                except:
                    pass
            costs.append(best_cost if best_cost != np.inf else 0)
        # Pad first step
        if len(costs) > 0:
            costs = [costs[0]] + costs
        return np.array(costs)
    
    def compute_stepwise_metrics(
        self, trajectory: np.ndarray, gt_trajectory: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute all metrics step-by-step.
        
        Args:
            trajectory: Robot trajectory
            gt_trajectory: Ground truth trajectory
            
        Returns:
            Dictionary mapping metric names to stepwise values
        """
        results = {}
        for metric_name, func in self.metric_functions.items():
            try:
                values = func(trajectory, gt_trajectory)
                results[metric_name] = values.tolist()
            except Exception as e:
                print(f"Error computing {metric_name}: {e}")
                results[metric_name] = []
        
        return results
    
    def compute_trajectory_features_stepwise(
        self, trajectory: np.ndarray, gt_trajectory: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute trajectory features at each step.
        
        Args:
            trajectory: Robot trajectory
            gt_trajectory: Ground truth trajectory
            
        Returns:
            Dictionary of stepwise features
        """
        features = {}
        
        # Position error at each step
        min_len = min(len(trajectory), len(gt_trajectory))
        position_errors = []
        for i in range(min_len):
            error = np.linalg.norm(trajectory[i] - gt_trajectory[i])
            position_errors.append(error)
        features['position_error'] = position_errors
        
        # Velocity (displacement) at each step
        if len(trajectory) > 1:
            velocities = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
            gt_velocities = np.linalg.norm(np.diff(gt_trajectory, axis=0), axis=1)
            min_vel_len = min(len(velocities), len(gt_velocities))
            velocity_errors = []
            for i in range(min_vel_len):
                error = abs(velocities[i] - gt_velocities[i])
                velocity_errors.append(error)
            features['velocity_error'] = velocity_errors
        
        # Direction (angle) at each step
        if len(trajectory) > 1:
            traj_dirs = np.arctan2(
                np.diff(trajectory[:, 1]), np.diff(trajectory[:, 0])
            )
            gt_dirs = np.arctan2(
                np.diff(gt_trajectory[:, 1]), np.diff(gt_trajectory[:, 0])
            )
            min_dir_len = min(len(traj_dirs), len(gt_dirs))
            direction_errors = []
            for i in range(min_dir_len):
                error = abs(traj_dirs[i] - gt_dirs[i])
                # Wrap to [0, pi]
                error = min(error, 2 * np.pi - error)
                direction_errors.append(error)
            features['direction_error'] = direction_errors
        
        # Cumulative path length
        if len(trajectory) > 1:
            cum_path_length = np.cumsum(
                np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
            )
            features['cumulative_path_length'] = cum_path_length.tolist()
        
        # Distance from start
        if len(trajectory) > 0:
            distances_from_start = [
                np.linalg.norm(trajectory[i] - trajectory[0])
                for i in range(len(trajectory))
            ]
            features['distance_from_start'] = distances_from_start
        
        return features
    
    def analyze_metric_sensitivity(
        self, 
        stepwise_metrics: Dict[str, List[float]],
        stepwise_features: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze which features each metric is sensitive to.
        
        Args:
            stepwise_metrics: Stepwise metric values
            stepwise_features: Stepwise feature values
            
        Returns:
            Dictionary mapping metric names to feature correlations
        """
        sensitivity = {}
        
        for metric_name, metric_values in stepwise_metrics.items():
            if len(metric_values) == 0:
                continue
            
            metric_array = np.array(metric_values)
            correlations = {}
            
            for feature_name, feature_values in stepwise_features.items():
                if len(feature_values) == 0:
                    continue
                
                feature_array = np.array(feature_values)
                # Align lengths
                min_len = min(len(metric_array), len(feature_array))
                if min_len < 2:
                    continue
                
                metric_aligned = metric_array[:min_len]
                feature_aligned = feature_array[:min_len]
                
                # Compute correlation
                if np.std(metric_aligned) > 0 and np.std(feature_aligned) > 0:
                    correlation = np.corrcoef(metric_aligned, feature_aligned)[0, 1]
                    if not np.isnan(correlation):
                        correlations[feature_name] = float(correlation)
            
            sensitivity[metric_name] = correlations
        
        return sensitivity
