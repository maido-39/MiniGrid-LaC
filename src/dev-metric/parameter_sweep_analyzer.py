"""
Parameter sweep analyzer for tunable metrics.

Analyzes how parameter changes affect metric values for TWED, Sobolev, and ERP.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from data_loader import load_all_episodes, extract_episode_number
from metrics import twed_distance, sobolev_distance, erp_distance


class ParameterSweepAnalyzer:
    """Analyzer for parameter sensitivity analysis."""
    
    def __init__(self, logs_dir: Path, results_dir: Path):
        """
        Initialize parameter sweep analyzer.
        
        Args:
            logs_dir: Path to logs_good directory
            results_dir: Path to analysis_reports directory
        """
        self.logs_dir = Path(logs_dir)
        self.results_dir = Path(results_dir)
        self.all_episodes = {}
        self.sweep_results = {}
        
    def load_data(self):
        """Load all episode data."""
        print(f"Loading episodes from {self.logs_dir}...")
        self.all_episodes = load_all_episodes(self.logs_dir)
        print(f"Loaded {len(self.all_episodes)} episodes")
        
    def find_episode_1(self, group_episodes: Dict[str, Dict]) -> Optional[str]:
        """Find Episode 1 in a group."""
        for ep_name, ep_data in group_episodes.items():
            ep_num = extract_episode_number(ep_name)
            if ep_num == 1:
                return ep_name
        
        for ep_name in group_episodes.keys():
            if 'episode1' in ep_name.lower() or 'Episode_1' in ep_name:
                return ep_name
        
        return None
    
    def sweep_twed_parameters(
        self,
        trajectory: np.ndarray,
        reference: np.ndarray,
        nu_values: List[float] = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
        lambda_values: List[float] = [0.5, 1.0, 2.0, 5.0]
    ) -> Dict:
        """Sweep TWED parameters."""
        results = {}
        
        for nu in nu_values:
            for lambda_param in lambda_values:
                try:
                    distance = twed_distance(trajectory, reference, nu=nu, lambda_param=lambda_param)
                    key = f"nu={nu:.2f},lambda={lambda_param:.2f}"
                    results[key] = {
                        'nu': nu,
                        'lambda': lambda_param,
                        'distance': float(distance)
                    }
                except Exception as e:
                    print(f"Error computing TWED (nu={nu}, lambda={lambda_param}): {e}")
                    results[key] = {
                        'nu': nu,
                        'lambda': lambda_param,
                        'distance': np.nan
                    }
        
        return results
    
    def sweep_sobolev_parameters(
        self,
        trajectory: np.ndarray,
        reference: np.ndarray,
        alpha_values: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0],
        beta_values: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0]
    ) -> Dict:
        """Sweep Sobolev parameters."""
        results = {}
        
        for alpha in alpha_values:
            for beta in beta_values:
                try:
                    distance = sobolev_distance(trajectory, reference, alpha=alpha, beta=beta)
                    key = f"alpha={alpha:.2f},beta={beta:.2f}"
                    results[key] = {
                        'alpha': alpha,
                        'beta': beta,
                        'distance': float(distance)
                    }
                except Exception as e:
                    print(f"Error computing Sobolev (alpha={alpha}, beta={beta}): {e}")
                    results[key] = {
                        'alpha': alpha,
                        'beta': beta,
                        'distance': np.nan
                    }
        
        return results
    
    def sweep_erp_parameters(
        self,
        trajectory: np.ndarray,
        reference: np.ndarray,
        gap_values: Optional[List[float]] = None
    ) -> Dict:
        """Sweep ERP parameters."""
        if gap_values is None:
            # Default: 0.0, mean of reference, and some multiples
            ref_mean = np.mean(reference, axis=0)
            ref_mean_norm = np.linalg.norm(ref_mean)
            gap_values = [0.0, 0.5, 1.0, 2.0, ref_mean_norm, ref_mean_norm * 2]
        
        results = {}
        
        for gap in gap_values:
            try:
                # For scalar gap, create a 2D point
                if isinstance(gap, (int, float)):
                    gap_point = np.array([gap, gap]) if gap != 0.0 else None
                else:
                    gap_point = gap
                
                distance = erp_distance(trajectory, reference, gap_penalty=gap_point)
                key = f"gap={gap:.3f}" if isinstance(gap, (int, float)) else f"gap=custom"
                results[key] = {
                    'gap': float(gap) if isinstance(gap, (int, float)) else float(np.linalg.norm(gap_point)),
                    'distance': float(distance)
                }
            except Exception as e:
                print(f"Error computing ERP (gap={gap}): {e}")
                results[key] = {
                    'gap': float(gap) if isinstance(gap, (int, float)) else np.nan,
                    'distance': np.nan
                }
        
        return results
    
    def analyze_group(self, group_name: str) -> Dict:
        """Analyze parameter sensitivity for a group."""
        print(f"\n{'='*60}")
        print(f"Parameter Sweep Analysis for group: {group_name}")
        print(f"{'='*60}")
        
        # Filter episodes by group
        group_episodes = {
            name: data for name, data in self.all_episodes.items()
            if data['group'] == group_name
        }
        
        if len(group_episodes) == 0:
            print(f"No episodes found for group: {group_name}")
            return {}
        
        # Find GT
        gt_episode_name = self.find_episode_1(group_episodes)
        if gt_episode_name is None:
            print(f"Warning: Episode 1 not found in group {group_name}")
            return {}
        
        gt_trajectory = group_episodes[gt_episode_name]['trajectory']
        print(f"GT: {gt_episode_name} (length: {len(gt_trajectory)})")
        
        # Analyze each non-GT episode
        group_results = {
            'group': group_name,
            'gt_episode': gt_episode_name,
            'gt_length': len(gt_trajectory),
            'episodes': {}
        }
        
        for ep_name, ep_data in group_episodes.items():
            if ep_name == gt_episode_name:
                continue
            
            print(f"  Analyzing: {ep_name}...")
            trajectory = ep_data['trajectory']
            
            # Parameter sweeps
            twed_results = self.sweep_twed_parameters(trajectory, gt_trajectory)
            sobolev_results = self.sweep_sobolev_parameters(trajectory, gt_trajectory)
            erp_results = self.sweep_erp_parameters(trajectory, gt_trajectory)
            
            group_results['episodes'][ep_name] = {
                'episode_number': extract_episode_number(ep_name),
                'trajectory_length': len(trajectory),
                'TWED': twed_results,
                'Sobolev': sobolev_results,
                'ERP': erp_results
            }
        
        return group_results
    
    def visualize_twed_sweep(
        self,
        sweep_results: Dict,
        output_path: Path,
        group_name: str
    ):
        """Visualize TWED parameter sweep results."""
        # Extract data
        nu_values = sorted(set(r['nu'] for r in sweep_results.values()))
        lambda_values = sorted(set(r['lambda'] for r in sweep_results.values()))
        
        # Create heatmap data
        heatmap_data = np.zeros((len(nu_values), len(lambda_values)))
        for key, result in sweep_results.items():
            if not np.isnan(result['distance']):
                nu_idx = nu_values.index(result['nu'])
                lambda_idx = lambda_values.index(result['lambda'])
                heatmap_data[nu_idx, lambda_idx] = result['distance']
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Heatmap
        im = axes[0].imshow(heatmap_data, aspect='auto', cmap='viridis', origin='lower')
        axes[0].set_xticks(range(len(lambda_values)))
        axes[0].set_xticklabels([f'{l:.2f}' for l in lambda_values])
        axes[0].set_yticks(range(len(nu_values)))
        axes[0].set_yticklabels([f'{n:.2f}' for n in nu_values])
        axes[0].set_xlabel('Lambda (λ)')
        axes[0].set_ylabel('Nu (ν)')
        axes[0].set_title(f'TWED Parameter Sweep - {group_name}\nHeatmap')
        plt.colorbar(im, ax=axes[0], label='Distance')
        
        # Line plots for different nu values
        for nu in nu_values:
            nu_data = [
                r['distance'] for r in sweep_results.values()
                if r['nu'] == nu and not np.isnan(r['distance'])
            ]
            lambda_data = [
                r['lambda'] for r in sweep_results.values()
                if r['nu'] == nu and not np.isnan(r['distance'])
            ]
            if len(nu_data) > 0:
                # Sort by lambda
                sorted_pairs = sorted(zip(lambda_data, nu_data))
                lambda_sorted, nu_sorted = zip(*sorted_pairs)
                axes[1].plot(lambda_sorted, nu_sorted, marker='o', label=f'ν={nu:.2f}')
        
        axes[1].set_xlabel('Lambda (λ)')
        axes[1].set_ylabel('TWED Distance')
        axes[1].set_title('TWED Distance vs Lambda\n(for different Nu values)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved TWED sweep visualization to {output_path}")
    
    def visualize_sobolev_sweep(
        self,
        sweep_results: Dict,
        output_path: Path,
        group_name: str
    ):
        """Visualize Sobolev parameter sweep results."""
        # Extract data
        alpha_values = sorted(set(r['alpha'] for r in sweep_results.values()))
        beta_values = sorted(set(r['beta'] for r in sweep_results.values()))
        
        # Create heatmap data
        heatmap_data = np.zeros((len(alpha_values), len(beta_values)))
        for key, result in sweep_results.items():
            if not np.isnan(result['distance']):
                alpha_idx = alpha_values.index(result['alpha'])
                beta_idx = beta_values.index(result['beta'])
                heatmap_data[alpha_idx, beta_idx] = result['distance']
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Heatmap
        im = axes[0].imshow(heatmap_data, aspect='auto', cmap='plasma', origin='lower')
        axes[0].set_xticks(range(len(beta_values)))
        axes[0].set_xticklabels([f'{b:.2f}' for b in beta_values])
        axes[0].set_yticks(range(len(alpha_values)))
        axes[0].set_yticklabels([f'{a:.2f}' for a in alpha_values])
        axes[0].set_xlabel('Beta (β) - Velocity Weight')
        axes[0].set_ylabel('Alpha (α) - Position Weight')
        axes[0].set_title(f'Sobolev Parameter Sweep - {group_name}\nHeatmap')
        plt.colorbar(im, ax=axes[0], label='Distance')
        
        # Line plots for different alpha values
        for alpha in alpha_values:
            alpha_data = [
                r['distance'] for r in sweep_results.values()
                if r['alpha'] == alpha and not np.isnan(r['distance'])
            ]
            beta_data = [
                r['beta'] for r in sweep_results.values()
                if r['alpha'] == alpha and not np.isnan(r['distance'])
            ]
            if len(alpha_data) > 0:
                # Sort by beta
                sorted_pairs = sorted(zip(beta_data, alpha_data))
                beta_sorted, alpha_sorted = zip(*sorted_pairs)
                axes[1].plot(beta_sorted, alpha_sorted, marker='o', label=f'α={alpha:.2f}')
        
        axes[1].set_xlabel('Beta (β)')
        axes[1].set_ylabel('Sobolev Distance')
        axes[1].set_title('Sobolev Distance vs Beta\n(for different Alpha values)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved Sobolev sweep visualization to {output_path}")
    
    def visualize_erp_sweep(
        self,
        sweep_results: Dict,
        output_path: Path,
        group_name: str
    ):
        """Visualize ERP parameter sweep results."""
        # Extract data
        gap_values = sorted([r['gap'] for r in sweep_results.values() if not np.isnan(r['gap'])])
        distances = [
            r['distance'] for r in sweep_results.values()
            if not np.isnan(r['distance'])
        ]
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Plot
        ax.plot(gap_values, distances, marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('Gap Penalty (g)')
        ax.set_ylabel('ERP Distance')
        ax.set_title(f'ERP Parameter Sweep - {group_name}\nDistance vs Gap Penalty')
        ax.grid(True, alpha=0.3)
        
        # Highlight default (g=0.0)
        if 0.0 in gap_values:
            idx = gap_values.index(0.0)
            ax.axvline(x=0.0, color='r', linestyle='--', alpha=0.5, label='Default (g=0.0)')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved ERP sweep visualization to {output_path}")
    
    def generate_aggregate_visualizations(self, all_results: Dict, output_dir: Path):
        """Generate aggregate visualizations across all groups."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Aggregate data across all groups
        twed_aggregate = defaultdict(list)
        sobolev_aggregate = defaultdict(list)
        erp_aggregate = defaultdict(list)
        
        for group_name, group_data in all_results.items():
            for ep_name, ep_data in group_data['episodes'].items():
                # TWED
                for key, result in ep_data['TWED'].items():
                    if not np.isnan(result['distance']):
                        twed_key = (result['nu'], result['lambda'])
                        twed_aggregate[twed_key].append(result['distance'])
                
                # Sobolev
                for key, result in ep_data['Sobolev'].items():
                    if not np.isnan(result['distance']):
                        sobolev_key = (result['alpha'], result['beta'])
                        sobolev_aggregate[sobolev_key].append(result['distance'])
                
                # ERP
                for key, result in ep_data['ERP'].items():
                    if not np.isnan(result['distance']):
                        erp_aggregate[result['gap']].append(result['distance'])
        
        # Create aggregate visualizations
        # TWED: Mean distance for each parameter combination
        nu_values = sorted(set(k[0] for k in twed_aggregate.keys()))
        lambda_values = sorted(set(k[1] for k in twed_aggregate.keys()))
        twed_mean = np.zeros((len(nu_values), len(lambda_values)))
        twed_std = np.zeros((len(nu_values), len(lambda_values)))
        
        for i, nu in enumerate(nu_values):
            for j, lambda_val in enumerate(lambda_values):
                key = (nu, lambda_val)
                if key in twed_aggregate:
                    twed_mean[i, j] = np.mean(twed_aggregate[key])
                    twed_std[i, j] = np.std(twed_aggregate[key])
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        im1 = axes[0].imshow(twed_mean, aspect='auto', cmap='viridis', origin='lower')
        axes[0].set_xticks(range(len(lambda_values)))
        axes[0].set_xticklabels([f'{l:.2f}' for l in lambda_values])
        axes[0].set_yticks(range(len(nu_values)))
        axes[0].set_yticklabels([f'{n:.2f}' for n in nu_values])
        axes[0].set_xlabel('Lambda (λ)')
        axes[0].set_ylabel('Nu (ν)')
        axes[0].set_title('TWED: Mean Distance Across All Episodes')
        plt.colorbar(im1, ax=axes[0], label='Mean Distance')
        
        im2 = axes[1].imshow(twed_std, aspect='auto', cmap='plasma', origin='lower')
        axes[1].set_xticks(range(len(lambda_values)))
        axes[1].set_xticklabels([f'{l:.2f}' for l in lambda_values])
        axes[1].set_yticks(range(len(nu_values)))
        axes[1].set_yticklabels([f'{n:.2f}' for n in nu_values])
        axes[1].set_xlabel('Lambda (λ)')
        axes[1].set_ylabel('Nu (ν)')
        axes[1].set_title('TWED: Std Dev Across All Episodes')
        plt.colorbar(im2, ax=axes[1], label='Std Dev')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'twed_aggregate_sweep.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Similar for Sobolev and ERP...
        print(f"Saved aggregate visualizations to {output_dir}")
    
    def save_results(self, all_results: Dict, output_dir: Path):
        """Save parameter sweep results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        json_path = output_dir / 'parameter_sweep_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"Saved parameter sweep results to {json_path}")
        
        # Generate visualizations for each group
        for group_name, group_data in all_results.items():
            group_output_dir = output_dir / group_name / 'parameter_sweep'
            group_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Aggregate results across episodes in group
            twed_agg = defaultdict(list)
            sobolev_agg = defaultdict(list)
            erp_agg = defaultdict(list)
            
            for ep_name, ep_data in group_data['episodes'].items():
                for key, result in ep_data['TWED'].items():
                    if not np.isnan(result['distance']):
                        twed_agg[(result['nu'], result['lambda'])].append(result['distance'])
                
                for key, result in ep_data['Sobolev'].items():
                    if not np.isnan(result['distance']):
                        sobolev_agg[(result['alpha'], result['beta'])].append(result['distance'])
                
                for key, result in ep_data['ERP'].items():
                    if not np.isnan(result['distance']):
                        erp_agg[result['gap']].append(result['distance'])
            
            # Create aggregate results for visualization
            twed_agg_result = {}
            for (nu, lambda_val), distances in twed_agg.items():
                key = f"nu={nu:.2f},lambda={lambda_val:.2f}"
                twed_agg_result[key] = {
                    'nu': nu,
                    'lambda': lambda_val,
                    'distance': np.mean(distances),
                    'std': np.std(distances),
                    'count': len(distances)
                }
            
            sobolev_agg_result = {}
            for (alpha, beta), distances in sobolev_agg.items():
                key = f"alpha={alpha:.2f},beta={beta:.2f}"
                sobolev_agg_result[key] = {
                    'alpha': alpha,
                    'beta': beta,
                    'distance': np.mean(distances),
                    'std': np.std(distances),
                    'count': len(distances)
                }
            
            erp_agg_result = {}
            for gap, distances in erp_agg.items():
                key = f"gap={gap:.3f}"
                erp_agg_result[key] = {
                    'gap': gap,
                    'distance': np.mean(distances),
                    'std': np.std(distances),
                    'count': len(distances)
                }
            
            # Visualize
            self.visualize_twed_sweep(
                twed_agg_result,
                group_output_dir / 'twed_parameter_sweep.png',
                group_name
            )
            self.visualize_sobolev_sweep(
                sobolev_agg_result,
                group_output_dir / 'sobolev_parameter_sweep.png',
                group_name
            )
            self.visualize_erp_sweep(
                erp_agg_result,
                group_output_dir / 'erp_parameter_sweep.png',
                group_name
            )
    
    def analyze_all_groups(self) -> Dict:
        """Analyze all groups."""
        all_results = {}
        
        for group_name in ['hogun', 'hogun_0125', 'other']:
            result = self.analyze_group(group_name)
            if result:
                all_results[group_name] = result
        
        return all_results
