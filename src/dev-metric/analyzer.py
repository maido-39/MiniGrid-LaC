"""
Trajectory metric analyzer.

Computes all metrics for comparing trajectories and performs episode clustering.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

from data_loader import load_all_episodes, get_reference_path
from metrics import (
    ddtw_distance,
    twed_distance,
    sobolev_distance,
    dtw_distance,
    frechet_distance,
    erp_distance,
    rmse_distance
)


class TrajectoryAnalyzer:
    """Analyzer for computing trajectory comparison metrics."""
    
    def __init__(self, logs_dir: Path, reference_name: str = 'Episode_1_1'):
        """
        Initialize analyzer.
        
        Args:
            logs_dir: Path to logs_good directory
            reference_name: Name of episode to use as reference
        """
        self.logs_dir = Path(logs_dir)
        self.reference_name = reference_name
        self.episodes = {}
        self.reference_trajectory = None
        self.results = {}
        
    def load_data(self):
        """Load all episode data."""
        print(f"Loading episodes from {self.logs_dir}...")
        self.episodes = load_all_episodes(self.logs_dir)
        print(f"Loaded {len(self.episodes)} episodes")
        
        # Get reference trajectory
        self.reference_trajectory = get_reference_path(self.episodes, self.reference_name)
        if self.reference_trajectory is None:
            print(f"Warning: Reference episode '{self.reference_name}' not found.")
            print(f"Available episodes: {list(self.episodes.keys())[:10]}...")
            if len(self.episodes) > 0:
                # Use first episode as fallback
                first_ep = list(self.episodes.keys())[0]
                self.reference_trajectory = self.episodes[first_ep]['trajectory']
                self.reference_name = first_ep
                print(f"Using '{first_ep}' as reference instead.")
        else:
            print(f"Using '{self.reference_name}' as reference trajectory")
            print(f"Reference trajectory length: {len(self.reference_trajectory)}")
    
    def compute_all_metrics(
        self,
        trajectory: np.ndarray,
        reference: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute all metrics for a trajectory compared to reference.
        
        Args:
            trajectory: Trajectory to evaluate
            reference: Reference trajectory (default: self.reference_trajectory)
            
        Returns:
            Dictionary of metric names to values
        """
        if reference is None:
            reference = self.reference_trajectory
        
        if reference is None:
            raise ValueError("No reference trajectory available")
        
        metrics = {}
        
        try:
            metrics['RMSE'] = rmse_distance(reference, trajectory)
        except Exception as e:
            metrics['RMSE'] = np.nan
            print(f"Error computing RMSE: {e}")
        
        try:
            metrics['DTW'] = dtw_distance(reference, trajectory)
        except Exception as e:
            metrics['DTW'] = np.nan
            print(f"Error computing DTW: {e}")
        
        try:
            metrics['Fréchet'] = frechet_distance(reference, trajectory)
        except Exception as e:
            metrics['Fréchet'] = np.nan
            print(f"Error computing Fréchet: {e}")
        
        try:
            metrics['ERP'] = erp_distance(reference, trajectory)
        except Exception as e:
            metrics['ERP'] = np.nan
            print(f"Error computing ERP: {e}")
        
        try:
            metrics['DDTW'] = ddtw_distance(reference, trajectory)
        except Exception as e:
            metrics['DDTW'] = np.nan
            print(f"Error computing DDTW: {e}")
        
        try:
            metrics['TWED'] = twed_distance(reference, trajectory, nu=0.5)
        except Exception as e:
            metrics['TWED'] = np.nan
            print(f"Error computing TWED: {e}")
        
        try:
            metrics['Sobolev'] = sobolev_distance(reference, trajectory)
        except Exception as e:
            metrics['Sobolev'] = np.nan
            print(f"Error computing Sobolev: {e}")
        
        return metrics
    
    def analyze_all_episodes(self):
        """Compute metrics for all episodes."""
        if self.reference_trajectory is None:
            raise ValueError("Reference trajectory not loaded. Call load_data() first.")
        
        print("\nComputing metrics for all episodes...")
        self.results = {}
        
        for ep_name, ep_data in self.episodes.items():
            if ep_name == self.reference_name:
                # Skip reference episode
                continue
            
            print(f"  Processing {ep_name}...")
            trajectory = ep_data['trajectory']
            metrics = self.compute_all_metrics(trajectory)
            
            self.results[ep_name] = {
                'metrics': metrics,
                'group': ep_data['group'],
                'trajectory_length': len(trajectory),
                'reference_length': len(self.reference_trajectory)
            }
        
        print(f"\nComputed metrics for {len(self.results)} episodes")
    
    def get_results_by_group(self) -> Dict[str, List[Dict]]:
        """
        Group results by episode group.
        
        Returns:
            Dictionary mapping group names to lists of results
        """
        grouped = {}
        for ep_name, result in self.results.items():
            group = result['group']
            if group not in grouped:
                grouped[group] = []
            grouped[group].append({
                'episode': ep_name,
                **result
            })
        return grouped
    
    def get_metric_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute statistics for each metric across all episodes.
        
        Returns:
            Dictionary mapping metric names to statistics (mean, std, min, max)
        """
        if not self.results:
            return {}
        
        metric_names = ['RMSE', 'DTW', 'Fréchet', 'ERP', 'DDTW', 'TWED', 'Sobolev']
        stats = {}
        
        for metric_name in metric_names:
            values = []
            for result in self.results.values():
                value = result['metrics'].get(metric_name)
                if value is not None and not np.isnan(value) and not np.isinf(value):
                    values.append(value)
            
            if values:
                stats[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'count': len(values)
                }
            else:
                stats[metric_name] = {
                    'mean': np.nan,
                    'std': np.nan,
                    'min': np.nan,
                    'max': np.nan,
                    'median': np.nan,
                    'count': 0
                }
        
        return stats
    
    def save_results(self, output_dir: Path):
        """
        Save analysis results to files.
        
        Args:
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results (JSON)
        results_json = {
            'reference_episode': self.reference_name,
            'reference_length': len(self.reference_trajectory) if self.reference_trajectory is not None else 0,
            'analysis_date': datetime.now().isoformat(),
            'episodes': {}
        }
        
        for ep_name, result in self.results.items():
            results_json['episodes'][ep_name] = {
                'group': result['group'],
                'trajectory_length': result['trajectory_length'],
                'metrics': {k: float(v) if not (np.isnan(v) or np.isinf(v)) else None
                           for k, v in result['metrics'].items()}
            }
        
        json_path = output_dir / 'detailed_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_json, f, indent=2, ensure_ascii=False)
        print(f"Saved detailed results to {json_path}")
        
        # Save CSV
        import csv
        csv_path = output_dir / 'metrics_summary.csv'
        metric_names = ['RMSE', 'DTW', 'Fréchet', 'ERP', 'DDTW', 'TWED', 'Sobolev']
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Group', 'Trajectory_Length'] + metric_names)
            
            for ep_name, result in self.results.items():
                row = [ep_name, result['group'], result['trajectory_length']]
                for metric_name in metric_names:
                    value = result['metrics'].get(metric_name)
                    if value is not None and not (np.isnan(value) or np.isinf(value)):
                        row.append(f"{value:.6f}")
                    else:
                        row.append("")
                writer.writerow(row)
        
        print(f"Saved CSV summary to {csv_path}")
        
        # Save statistics
        stats = self.get_metric_statistics()
        stats_path = output_dir / 'statistics.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"Saved statistics to {stats_path}")
