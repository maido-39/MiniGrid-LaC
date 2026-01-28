"""
Statistical analyzer for trajectory metrics.

Performs comprehensive statistical analysis including:
- Trajectory characteristics (speed, direction, path length, curvature)
- Correlation analysis between metrics and trajectory features
- Statistical significance tests
- Regression analysis
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Try to import scipy, but use numpy fallbacks if not available
try:
    from scipy import stats
    from scipy.stats import pearsonr, spearmanr, kendalltau, mannwhitneyu
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    # Define fallback functions using numpy
    def pearsonr(x, y):
        """Pearson correlation coefficient."""
        x = np.array(x)
        y = np.array(y)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
        if denominator == 0:
            return 0.0, 1.0
        r = numerator / denominator
        # Simple p-value approximation (not exact but reasonable)
        n = len(x)
        if n < 3:
            p = 1.0
        else:
            t = r * np.sqrt((n - 2) / (1 - r**2)) if abs(r) < 0.999 else np.inf
            # Approximate p-value using t-distribution (two-tailed)
            # This is a simplified approximation
            p = 2 * (1 - min(abs(t) / 3.0, 1.0))  # Rough approximation
        return float(r), float(p)
    
    def spearmanr(x, y):
        """Spearman rank correlation coefficient."""
        x = np.array(x)
        y = np.array(y)
        # Rank the data
        x_ranks = np.argsort(np.argsort(x)) + 1
        y_ranks = np.argsort(np.argsort(y)) + 1
        # Use Pearson on ranks
        return pearsonr(x_ranks, y_ranks)
    
    def mannwhitneyu(x, y, alternative='two-sided'):
        """Mann-Whitney U test (simplified version)."""
        x = np.array(x)
        y = np.array(y)
        n1, n2 = len(x), len(y)
        # Combine and rank
        combined = np.concatenate([x, y])
        ranks = np.argsort(np.argsort(combined)) + 1
        # Sum of ranks for x
        R1 = np.sum(ranks[:n1])
        # U statistic
        U1 = n1 * n2 + n1 * (n1 + 1) / 2 - R1
        U2 = n1 * n2 - U1
        U = min(U1, U2)
        # Approximate p-value (normal approximation)
        mean_U = n1 * n2 / 2
        var_U = n1 * n2 * (n1 + n2 + 1) / 12
        if var_U == 0:
            return U, 1.0
        z = (U - mean_U) / np.sqrt(var_U)
        # Two-tailed p-value approximation
        p = 2 * (1 - min(abs(z) / 2.0, 1.0))  # Rough approximation
        return float(U), float(p)
    
    class LinregressResult:
        def __init__(self, slope, intercept, rvalue, pvalue, stderr):
            self.slope = slope
            self.intercept = intercept
            self.rvalue = rvalue
            self.pvalue = pvalue
            self.stderr = stderr
    
    def linregress(x, y):
        """Linear regression."""
        x = np.array(x)
        y = np.array(y)
        n = len(x)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean)**2)
        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        # R-squared
        y_pred = intercept + slope * x
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - y_mean)**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        r_value = np.sqrt(r_squared) if slope >= 0 else -np.sqrt(r_squared)
        # P-value (simplified)
        if n < 3:
            pvalue = 1.0
        else:
            t = r_value * np.sqrt((n - 2) / (1 - r_value**2)) if abs(r_value) < 0.999 else np.inf
            pvalue = 2 * (1 - min(abs(t) / 3.0, 1.0))
        # Standard error
        stderr = np.sqrt(ss_res / (n - 2)) / np.sqrt(np.sum((x - x_mean)**2)) if denominator > 0 and n > 2 else 0
        return LinregressResult(slope, intercept, r_value, pvalue, stderr)


class StatisticalAnalyzer:
    """Comprehensive statistical analyzer for trajectory metrics."""
    
    def __init__(self, logs_dir: Path):
        """
        Initialize statistical analyzer.
        
        Args:
            logs_dir: Path to logs_good directory
        """
        self.logs_dir = Path(logs_dir)
        self.trajectory_data = {}
        self.metric_data = {}
        
    def load_trajectory_data(self, group_results_dir: Path):
        """
        Load trajectory data from JSON results.
        
        Args:
            group_results_dir: Directory containing detailed_results.json
        """
        from data_loader import load_all_episodes, get_reference_path
        
        # Load all episodes
        all_episodes = load_all_episodes(self.logs_dir)
        
        # Load metric results
        results_json_path = group_results_dir / 'detailed_results.json'
        if not results_json_path.exists():
            return {}
        
        with open(results_json_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        group_name = results.get('group', 'unknown')
        gt_episode = results.get('gt_episode', '')
        gt_trajectory = None
        
        # Get GT trajectory
        for ep_name, ep_data in all_episodes.items():
            if ep_name == gt_episode:
                gt_trajectory = ep_data['trajectory']
                break
        
        if gt_trajectory is None:
            return {}
        
        trajectory_info = {}
        
        for ep_name, ep_result in results.get('episodes', {}).items():
            # Find trajectory
            trajectory = None
            for ep_name_full, ep_data in all_episodes.items():
                if ep_name in ep_name_full or ep_name_full == ep_name:
                    trajectory = ep_data['trajectory']
                    break
            
            if trajectory is None:
                continue
            
            # Compute trajectory characteristics
            characteristics = self.compute_trajectory_characteristics(
                trajectory, gt_trajectory
            )
            
            trajectory_info[ep_name] = {
                'episode_number': ep_result.get('episode_number', 0),
                'trajectory_length': ep_result.get('trajectory_length', 0),
                'metrics': ep_result.get('metrics', {}),
                'characteristics': characteristics,
                'trajectory': trajectory,
                'gt_trajectory': gt_trajectory
            }
        
        return trajectory_info
    
    def compute_trajectory_characteristics(
        self, trajectory: np.ndarray, gt_trajectory: np.ndarray
    ) -> Dict:
        """
        Compute detailed trajectory characteristics.
        
        Args:
            trajectory: Robot trajectory (N, 2)
            gt_trajectory: Ground truth trajectory (M, 2)
            
        Returns:
            Dictionary of characteristics
        """
        if len(trajectory) < 2:
            return {}
        
        # Compute velocities (displacements)
        velocities = np.diff(trajectory, axis=0)
        speeds = np.linalg.norm(velocities, axis=1)
        
        # GT velocities
        gt_velocities = np.diff(gt_trajectory, axis=0)
        gt_speeds = np.linalg.norm(gt_velocities, axis=1)
        
        # Compute directions (angles)
        directions = np.arctan2(velocities[:, 1], velocities[:, 0])
        gt_directions = np.arctan2(gt_velocities[:, 1], gt_velocities[:, 0])
        
        # Compute curvature (change in direction)
        if len(directions) > 1:
            curvature = np.abs(np.diff(directions))
            curvature = np.minimum(curvature, 2 * np.pi - curvature)  # Wrap to [0, pi]
        else:
            curvature = np.array([])
        
        # Compute path length
        path_length = np.sum(speeds)
        gt_path_length = np.sum(gt_speeds)
        
        # Compute total displacement
        total_displacement = np.linalg.norm(trajectory[-1] - trajectory[0])
        gt_total_displacement = np.linalg.norm(gt_trajectory[-1] - gt_trajectory[0])
        
        # Compute efficiency (displacement / path_length)
        efficiency = total_displacement / path_length if path_length > 0 else 0
        gt_efficiency = gt_total_displacement / gt_path_length if gt_path_length > 0 else 0
        
        # Compute stops (zero velocity)
        num_stops = np.sum(speeds < 0.01)
        num_gt_stops = np.sum(gt_speeds < 0.01)
        
        # Compute backtracking (negative dot product with forward direction)
        if len(velocities) > 1:
            forward_dirs = velocities[:-1]
            next_dirs = velocities[1:]
            dot_products = np.sum(forward_dirs * next_dirs, axis=1)
            num_backtracks = np.sum(dot_products < -0.1)  # Threshold for backtracking
        else:
            num_backtracks = 0
        
        # Compute average speed
        avg_speed = np.mean(speeds) if len(speeds) > 0 else 0
        gt_avg_speed = np.mean(gt_speeds) if len(gt_speeds) > 0 else 0
        
        # Compute speed variance
        speed_variance = np.var(speeds) if len(speeds) > 0 else 0
        gt_speed_variance = np.var(gt_speeds) if len(gt_speeds) > 0 else 0
        
        # Compute average curvature
        avg_curvature = np.mean(curvature) if len(curvature) > 0 else 0
        
        # Compute direction similarity (cosine similarity of velocity vectors)
        # Interpolate to same length for comparison
        min_len = min(len(velocities), len(gt_velocities))
        if min_len > 0:
            traj_vel = velocities[:min_len]
            gt_vel = gt_velocities[:min_len]
            # Normalize
            traj_norm = np.linalg.norm(traj_vel, axis=1, keepdims=True)
            gt_norm = np.linalg.norm(gt_vel, axis=1, keepdims=True)
            traj_norm[traj_norm == 0] = 1
            gt_norm[gt_norm == 0] = 1
            traj_vel_norm = traj_vel / traj_norm
            gt_vel_norm = gt_vel / gt_norm
            direction_similarity = np.mean(np.sum(traj_vel_norm * gt_vel_norm, axis=1))
        else:
            direction_similarity = 0
        
        # Compute path length ratio
        length_ratio = len(trajectory) / len(gt_trajectory) if len(gt_trajectory) > 0 else 0
        
        return {
            'path_length': float(path_length),
            'gt_path_length': float(gt_path_length),
            'path_length_ratio': float(length_ratio),
            'total_displacement': float(total_displacement),
            'gt_total_displacement': float(gt_total_displacement),
            'efficiency': float(efficiency),
            'gt_efficiency': float(gt_efficiency),
            'avg_speed': float(avg_speed),
            'gt_avg_speed': float(gt_avg_speed),
            'speed_variance': float(speed_variance),
            'gt_speed_variance': float(gt_speed_variance),
            'num_stops': int(num_stops),
            'num_gt_stops': int(num_gt_stops),
            'num_backtracks': int(num_backtracks),
            'avg_curvature': float(avg_curvature),
            'direction_similarity': float(direction_similarity),
            'trajectory_length': len(trajectory),
            'gt_trajectory_length': len(gt_trajectory),
        }
    
    def analyze_group(self, group_results_dir: Path) -> Dict:
        """
        Perform comprehensive statistical analysis for a group.
        
        Args:
            group_results_dir: Directory containing detailed_results.json
            
        Returns:
            Dictionary of analysis results
        """
        trajectory_info = self.load_trajectory_data(group_results_dir)
        
        if len(trajectory_info) == 0:
            return {}
        
        # Prepare data for analysis
        episodes = []
        episode_numbers = []
        trajectory_lengths = []
        metrics_data = defaultdict(list)
        characteristics_data = defaultdict(list)
        
        for ep_name, info in trajectory_info.items():
            episodes.append(ep_name)
            episode_numbers.append(info['episode_number'])
            trajectory_lengths.append(info['trajectory_length'])
            
            # Metrics
            for metric_name, metric_value in info['metrics'].items():
                if metric_value is not None:
                    metrics_data[metric_name].append(metric_value)
                else:
                    metrics_data[metric_name].append(np.nan)
            
            # Characteristics
            for char_name, char_value in info['characteristics'].items():
                characteristics_data[char_name].append(char_value)
        
        # Convert to arrays
        episode_numbers = np.array(episode_numbers)
        trajectory_lengths = np.array(trajectory_lengths)
        
        # Statistical analysis
        analysis_results = {
            'episodes': episodes,
            'episode_numbers': episode_numbers.tolist(),
            'trajectory_lengths': trajectory_lengths.tolist(),
            'metrics': {},
            'characteristics': {},
            'correlations': {},
            'regression': {},
            'statistical_tests': {}
        }
        
        # Analyze each metric
        for metric_name, metric_values in metrics_data.items():
            metric_array = np.array(metric_values)
            valid_mask = ~np.isnan(metric_array) & ~np.isinf(metric_array)
            
            if np.sum(valid_mask) < 2:
                continue
            
            valid_values = metric_array[valid_mask]
            
            # Basic statistics
            analysis_results['metrics'][metric_name] = {
                'mean': float(np.mean(valid_values)),
                'std': float(np.std(valid_values)),
                'median': float(np.median(valid_values)),
                'min': float(np.min(valid_values)),
                'max': float(np.max(valid_values)),
                'cv': float(np.std(valid_values) / np.mean(valid_values)) if np.mean(valid_values) > 0 else np.inf,
                'values': valid_values.tolist()
            }
            
            # Trend analysis (linear regression with episode number)
            if len(episode_numbers[valid_mask]) > 1:
                if HAS_SCIPY:
                    result = stats.linregress(
                        episode_numbers[valid_mask], valid_values
                    )
                    slope, intercept, r_value, p_value, std_err = (
                        result.slope, result.intercept, result.rvalue, 
                        result.pvalue, result.stderr
                    )
                else:
                    result = linregress(
                        episode_numbers[valid_mask], valid_values
                    )
                    slope, intercept, r_value, p_value, std_err = (
                        result.slope, result.intercept, result.rvalue,
                        result.pvalue, result.stderr
                    )
                analysis_results['metrics'][metric_name]['trend'] = {
                    'slope': float(slope),
                    'intercept': float(intercept),
                    'r_squared': float(r_value ** 2),
                    'p_value': float(p_value),
                    'std_err': float(std_err)
                }
        
        # Analyze characteristics
        for char_name, char_values in characteristics_data.items():
            char_array = np.array(char_values)
            valid_mask = ~np.isnan(char_array) & ~np.isinf(char_array)
            
            if np.sum(valid_mask) < 2:
                continue
            
            valid_values = char_array[valid_mask]
            
            analysis_results['characteristics'][char_name] = {
                'mean': float(np.mean(valid_values)),
                'std': float(np.std(valid_values)),
                'median': float(np.median(valid_values)),
                'min': float(np.min(valid_values)),
                'max': float(np.max(valid_values)),
                'values': valid_values.tolist()
            }
        
        # Correlation analysis
        # Correlations between metrics and characteristics
        for metric_name, metric_values in metrics_data.items():
            metric_array = np.array(metric_values)
            valid_metric = ~np.isnan(metric_array) & ~np.isinf(metric_array)
            
            if np.sum(valid_metric) < 2:
                continue
            
            correlations = {}
            
            # Correlation with episode number
            if len(episode_numbers[valid_metric]) > 1:
                r_pearson, p_pearson = pearsonr(
                    episode_numbers[valid_metric], metric_array[valid_metric]
                )
                r_spearman, p_spearman = spearmanr(
                    episode_numbers[valid_metric], metric_array[valid_metric]
                )
                correlations['episode_number'] = {
                    'pearson_r': float(r_pearson),
                    'pearson_p': float(p_pearson),
                    'spearman_r': float(r_spearman),
                    'spearman_p': float(p_spearman)
                }
            
            # Correlation with trajectory length
            if len(trajectory_lengths[valid_metric]) > 1:
                r_pearson, p_pearson = pearsonr(
                    trajectory_lengths[valid_metric], metric_array[valid_metric]
                )
                r_spearman, p_spearman = spearmanr(
                    trajectory_lengths[valid_metric], metric_array[valid_metric]
                )
                correlations['trajectory_length'] = {
                    'pearson_r': float(r_pearson),
                    'pearson_p': float(p_pearson),
                    'spearman_r': float(r_spearman),
                    'spearman_p': float(p_spearman)
                }
            
            # Correlation with characteristics
            for char_name, char_values in characteristics_data.items():
                char_array = np.array(char_values)
                valid_char = ~np.isnan(char_array) & ~np.isinf(char_array)
                valid_both = valid_metric & valid_char
                
                if np.sum(valid_both) > 1:
                    r_pearson, p_pearson = pearsonr(
                        char_array[valid_both], metric_array[valid_both]
                    )
                    r_spearman, p_spearman = spearmanr(
                        char_array[valid_both], metric_array[valid_both]
                    )
                    correlations[char_name] = {
                        'pearson_r': float(r_pearson),
                        'pearson_p': float(p_pearson),
                        'spearman_r': float(r_spearman),
                        'spearman_p': float(p_spearman)
                    }
            
            analysis_results['correlations'][metric_name] = correlations
        
        # Inter-metric correlations
        metric_names = list(metrics_data.keys())
        inter_metric_corr = {}
        for i, metric1 in enumerate(metric_names):
            for metric2 in metric_names[i+1:]:
                values1 = np.array(metrics_data[metric1])
                values2 = np.array(metrics_data[metric2])
                valid = ~np.isnan(values1) & ~np.isnan(values2) & ~np.isinf(values1) & ~np.isinf(values2)
                
                if np.sum(valid) > 1:
                    r_pearson, p_pearson = pearsonr(values1[valid], values2[valid])
                    r_spearman, p_spearman = spearmanr(values1[valid], values2[valid])
                    inter_metric_corr[f"{metric1}_vs_{metric2}"] = {
                        'pearson_r': float(r_pearson),
                        'pearson_p': float(p_pearson),
                        'spearman_r': float(r_spearman),
                        'spearman_p': float(p_spearman)
                    }
        
        analysis_results['inter_metric_correlations'] = inter_metric_corr
        
        # Statistical significance tests
        # Test if metrics change significantly with episode number
        statistical_tests = {}
        for metric_name, metric_values in metrics_data.items():
            metric_array = np.array(metric_values)
            valid_metric = ~np.isnan(metric_array) & ~np.isinf(metric_array)
            
            if np.sum(valid_metric) < 2:
                continue
            
            # Mann-Whitney U test (non-parametric) for early vs late episodes
            if len(episode_numbers[valid_metric]) >= 4:
                # Split into early and late
                sorted_indices = np.argsort(episode_numbers[valid_metric])
                mid_point = len(sorted_indices) // 2
                early_indices = sorted_indices[:mid_point]
                late_indices = sorted_indices[mid_point:]
                
                early_values = metric_array[valid_metric][early_indices]
                late_values = metric_array[valid_metric][late_indices]
                
                if len(early_values) > 0 and len(late_values) > 0:
                    if HAS_SCIPY:
                        u_statistic, u_p_value = stats.mannwhitneyu(
                            early_values, late_values, alternative='two-sided'
                        )
                    else:
                        u_statistic, u_p_value = mannwhitneyu(
                            early_values, late_values, alternative='two-sided'
                        )
                    statistical_tests[metric_name] = {
                        'mann_whitney_u': float(u_statistic),
                        'mann_whitney_p': float(u_p_value),
                        'early_mean': float(np.mean(early_values)),
                        'late_mean': float(np.mean(late_values)),
                        'early_std': float(np.std(early_values)),
                        'late_std': float(np.std(late_values))
                    }
        
        analysis_results['statistical_tests'] = statistical_tests
        
        return analysis_results
    
    def save_analysis(self, analysis_results: Dict, output_path: Path):
        """
        Save analysis results to JSON file.
        
        Args:
            analysis_results: Analysis results dictionary
            output_path: Path to save JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        
        print(f"Saved statistical analysis to {output_path}")
