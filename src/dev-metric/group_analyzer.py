"""
Group-wise trajectory analyzer.

Analyzes trajectories by group (hogun, hogun_0125, other) with Episode 1 as GT.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

from data_loader import load_all_episodes, extract_episode_number
from analyzer import TrajectoryAnalyzer


class GroupWiseAnalyzer:
    """Analyzer for group-wise trajectory comparison."""
    
    def __init__(self, logs_dir: Path):
        """
        Initialize group-wise analyzer.
        
        Args:
            logs_dir: Path to logs_good directory
        """
        self.logs_dir = Path(logs_dir)
        self.all_episodes = {}
        self.group_results = {}
        
    def load_data(self):
        """Load all episode data."""
        print(f"Loading episodes from {self.logs_dir}...")
        self.all_episodes = load_all_episodes(self.logs_dir)
        print(f"Loaded {len(self.all_episodes)} episodes")
        
    def find_episode_1(self, group_episodes: Dict[str, Dict]) -> Optional[str]:
        """
        Find Episode 1 in a group.
        
        Args:
            group_episodes: Dictionary of episodes in the group
            
        Returns:
            Name of Episode 1, or None if not found
        """
        # Try to find episode1 or Episode_1_* pattern
        for ep_name, ep_data in group_episodes.items():
            ep_num = extract_episode_number(ep_name)
            if ep_num == 1:
                return ep_name
        
        # Fallback: look for names containing 'episode1' or 'Episode_1'
        for ep_name in group_episodes.keys():
            if 'episode1' in ep_name.lower() or 'Episode_1' in ep_name:
                return ep_name
        
        return None
    
    def analyze_group(self, group_name: str) -> Dict:
        """
        Analyze a specific group.
        
        Args:
            group_name: Name of the group ('hogun', 'hogun_0125', or 'other')
            
        Returns:
            Dictionary containing analysis results for the group
        """
        print(f"\n{'='*60}")
        print(f"Analyzing group: {group_name}")
        print(f"{'='*60}")
        
        # Filter episodes by group
        group_episodes = {
            name: data for name, data in self.all_episodes.items()
            if data['group'] == group_name
        }
        
        if len(group_episodes) == 0:
            print(f"No episodes found for group: {group_name}")
            return {}
        
        print(f"Found {len(group_episodes)} episodes in group {group_name}")
        
        # Find Episode 1 (GT)
        gt_episode_name = self.find_episode_1(group_episodes)
        if gt_episode_name is None:
            print(f"Warning: Episode 1 not found in group {group_name}")
            print(f"Available episodes: {list(group_episodes.keys())}")
            return {}
        
        print(f"Using '{gt_episode_name}' as Ground Truth (Episode 1)")
        gt_trajectory = group_episodes[gt_episode_name]['trajectory']
        print(f"GT trajectory length: {len(gt_trajectory)}")
        
        # Create analyzer for this group
        analyzer = TrajectoryAnalyzer(self.logs_dir, reference_name=gt_episode_name)
        analyzer.episodes = group_episodes
        analyzer.reference_trajectory = gt_trajectory
        analyzer.reference_name = gt_episode_name
        
        # Compute metrics for all other episodes
        results = {}
        episode_numbers = []
        
        for ep_name, ep_data in group_episodes.items():
            if ep_name == gt_episode_name:
                continue  # Skip GT itself
            
            ep_num = extract_episode_number(ep_name)
            if ep_num is None:
                print(f"Warning: Could not extract episode number from '{ep_name}', skipping")
                continue
            
            print(f"  Processing {ep_name} (Episode {ep_num})...")
            trajectory = ep_data['trajectory']
            
            try:
                metrics = analyzer.compute_all_metrics(trajectory, gt_trajectory)
                results[ep_name] = {
                    'episode_number': ep_num,
                    'metrics': metrics,
                    'trajectory_length': len(trajectory),
                    'gt_length': len(gt_trajectory)
                }
                episode_numbers.append(ep_num)
            except Exception as e:
                print(f"    Error computing metrics for {ep_name}: {e}")
                continue
        
        # Sort by episode number
        sorted_results = dict(sorted(
            results.items(),
            key=lambda x: x[1]['episode_number']
        ))
        
        print(f"\nComputed metrics for {len(sorted_results)} episodes in group {group_name}")
        
        return {
            'group_name': group_name,
            'gt_episode': gt_episode_name,
            'gt_trajectory': gt_trajectory,
            'results': sorted_results,
            'episode_numbers': sorted(episode_numbers)
        }
    
    def analyze_all_groups(self):
        """Analyze all groups."""
        groups = ['hogun', 'hogun_0125', 'other']
        
        for group_name in groups:
            group_result = self.analyze_group(group_name)
            if group_result:
                self.group_results[group_name] = group_result
    
    def save_group_results(self, output_dir: Path, group_name: str):
        """
        Save results for a specific group.
        
        Args:
            output_dir: Base output directory
            group_name: Name of the group
        """
        if group_name not in self.group_results:
            return
        
        group_result = self.group_results[group_name]
        group_output_dir = output_dir / group_name
        group_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results (JSON)
        results_json = {
            'group': group_name,
            'gt_episode': group_result['gt_episode'],
            'gt_length': len(group_result['gt_trajectory']),
            'analysis_date': datetime.now().isoformat(),
            'episodes': {}
        }
        
        for ep_name, ep_result in group_result['results'].items():
            results_json['episodes'][ep_name] = {
                'episode_number': ep_result['episode_number'],
                'trajectory_length': ep_result['trajectory_length'],
                'metrics': {
                    k: float(v) if not (np.isnan(v) or np.isinf(v)) else None
                    for k, v in ep_result['metrics'].items()
                }
            }
        
        json_path = group_output_dir / 'detailed_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_json, f, indent=2, ensure_ascii=False)
        print(f"Saved detailed results to {json_path}")
        
        # Save CSV
        import csv
        csv_path = group_output_dir / 'metrics_by_episode.csv'
        metric_names = ['RMSE', 'DTW', 'Fréchet', 'ERP', 'DDTW', 'TWED', 'Sobolev']
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Episode_Number', 'Trajectory_Length'] + metric_names)
            
            for ep_name, ep_result in group_result['results'].items():
                row = [
                    ep_name,
                    ep_result['episode_number'],
                    ep_result['trajectory_length']
                ]
                for metric_name in metric_names:
                    value = ep_result['metrics'].get(metric_name)
                    if value is not None and not (np.isnan(value) or np.isinf(value)):
                        row.append(f"{value:.6f}")
                    else:
                        row.append("")
                writer.writerow(row)
        
        print(f"Saved CSV to {csv_path}")
    
    def save_all_results(self, output_dir: Path):
        """
        Save results for all groups.
        
        Args:
            output_dir: Base output directory
        """
        for group_name in self.group_results.keys():
            self.save_group_results(output_dir, group_name)
