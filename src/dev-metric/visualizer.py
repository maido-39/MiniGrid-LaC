"""
Visualization module for trajectory metric analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from typing import Dict, List, Optional

# Publication-quality plot settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'axes.unicode_minus': False,
})


class TrajectoryVisualizer:
    """Visualizer for trajectory analysis results."""
    
    def __init__(self, analyzer):
        """
        Initialize visualizer.
        
        Args:
            analyzer: TrajectoryAnalyzer instance with computed results
        """
        self.analyzer = analyzer
    
    def plot_trajectory_comparison(
        self,
        episode_name: str,
        output_path: Optional[Path] = None,
        max_episodes: int = 10
    ):
        """
        Plot reference trajectory vs. episode trajectory.
        
        Args:
            episode_name: Name of episode to compare
            output_path: Path to save figure
            max_episodes: Maximum number of episodes to plot in one figure
        """
        if episode_name not in self.analyzer.results:
            print(f"Episode {episode_name} not found in results")
            return
        
        ref_traj = self.analyzer.reference_trajectory
        ep_traj = self.analyzer.episodes[episode_name]['trajectory']
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot reference trajectory
        ax.plot(ref_traj[:, 0], ref_traj[:, 1], 'b-', linewidth=2, 
               label=f'Reference ({self.analyzer.reference_name})', alpha=0.7)
        ax.scatter(ref_traj[0, 0], ref_traj[0, 1], c='green', s=150, 
                  marker='s', label='Start', zorder=10, edgecolors='black', linewidths=2)
        ax.scatter(ref_traj[-1, 0], ref_traj[-1, 1], c='red', s=150, 
                  marker='*', label='End', zorder=10, edgecolors='black', linewidths=1)
        
        # Plot episode trajectory
        ax.plot(ep_traj[:, 0], ep_traj[:, 1], 'r--', linewidth=2, 
               label=f'Episode ({episode_name})', alpha=0.7)
        ax.scatter(ep_traj[0, 0], ep_traj[0, 1], c='green', s=150, 
                  marker='s', zorder=10, edgecolors='black', linewidths=2)
        ax.scatter(ep_traj[-1, 0], ep_traj[-1, 1], c='red', s=150, 
                  marker='*', zorder=10, edgecolors='black', linewidths=1)
        
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.set_title(f'Trajectory Comparison: {episode_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Match MiniGrid coordinate system
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved trajectory comparison to {output_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_metrics_comparison(
        self,
        output_path: Optional[Path] = None,
        metric_names: Optional[List[str]] = None
    ):
        """
        Plot metric values comparison across episodes.
        
        Args:
            output_path: Path to save figure
            metric_names: List of metrics to plot (default: all)
        """
        if metric_names is None:
            metric_names = ['RMSE', 'DTW', 'Fréchet', 'ERP', 'DDTW', 'TWED', 'Sobolev']
        
        # Prepare data
        episodes = list(self.analyzer.results.keys())
        n_metrics = len(metric_names)
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for idx, metric_name in enumerate(metric_names):
            ax = axes[idx]
            values = []
            labels = []
            
            for ep_name in episodes:
                value = self.analyzer.results[ep_name]['metrics'].get(metric_name)
                if value is not None and not (np.isnan(value) or np.isinf(value)):
                    values.append(value)
                    labels.append(ep_name)
            
            if values:
                ax.bar(range(len(values)), values, alpha=0.7)
                ax.set_xticks(range(len(values)))
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
                ax.set_ylabel(metric_name, fontsize=11)
                ax.set_title(f'{metric_name} by Episode', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
        
        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Metric Comparison Across Episodes', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved metrics comparison to {output_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_metrics_boxplot(
        self,
        output_path: Optional[Path] = None,
        group_by: str = 'group'
    ):
        """
        Plot box plots of metrics grouped by episode group.
        
        Args:
            output_path: Path to save figure
            group_by: 'group' or 'episode' for grouping
        """
        metric_names = ['RMSE', 'DTW', 'Fréchet', 'ERP', 'DDTW', 'TWED', 'Sobolev']
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        # Get groups
        if group_by == 'group':
            groups = {}
            for ep_name, result in self.analyzer.results.items():
                group = result['group']
                if group not in groups:
                    groups[group] = []
                groups[group].append(ep_name)
        else:
            # Group by episode number (extract from name)
            groups = {}
            for ep_name in self.analyzer.results.keys():
                # Try to extract episode number
                import re
                match = re.search(r'[Ee]pisode[_\s]*(\d+)', ep_name)
                if match:
                    ep_num = match.group(1)
                    if ep_num not in groups:
                        groups[ep_num] = []
                    groups[ep_num].append(ep_name)
                else:
                    if 'other' not in groups:
                        groups['other'] = []
                    groups['other'].append(ep_name)
        
        for idx, metric_name in enumerate(metric_names):
            ax = axes[idx]
            box_data = []
            group_labels = []
            
            for group_name, ep_list in groups.items():
                values = []
                for ep_name in ep_list:
                    if ep_name in self.analyzer.results:
                        value = self.analyzer.results[ep_name]['metrics'].get(metric_name)
                        if value is not None and not (np.isnan(value) or np.isinf(value)):
                            values.append(value)
                
                if values:
                    box_data.append(values)
                    group_labels.append(f"{group_name}\n(n={len(values)})")
            
            if box_data:
                bp = ax.boxplot(box_data, tick_labels=group_labels, patch_artist=True)
                colors = plt.cm.Set3(np.linspace(0, 1, len(box_data)))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                
                ax.set_ylabel(metric_name, fontsize=11)
                ax.set_title(f'{metric_name} Distribution', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
        
        # Hide unused subplots
        for idx in range(len(metric_names), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Metric Distribution by {group_by.capitalize()}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved boxplot to {output_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_metrics_heatmap(
        self,
        output_path: Optional[Path] = None
    ):
        """
        Plot heatmap of metric values across episodes.
        
        Args:
            output_path: Path to save figure
        """
        metric_names = ['RMSE', 'DTW', 'Fréchet', 'ERP', 'DDTW', 'TWED', 'Sobolev']
        episodes = list(self.analyzer.results.keys())
        
        # Prepare data matrix
        data_matrix = []
        for ep_name in episodes:
            row = []
            for metric_name in metric_names:
                value = self.analyzer.results[ep_name]['metrics'].get(metric_name)
                if value is not None and not (np.isnan(value) or np.isinf(value)):
                    row.append(value)
                else:
                    row.append(np.nan)
            data_matrix.append(row)
        
        data_matrix = np.array(data_matrix)
        
        # Normalize each metric (0-1 scale) for better visualization
        normalized_matrix = np.zeros_like(data_matrix)
        for j, metric_name in enumerate(metric_names):
            col = data_matrix[:, j]
            valid_col = col[~np.isnan(col)]
            if len(valid_col) > 0:
                min_val = np.min(valid_col)
                max_val = np.max(valid_col)
                if max_val > min_val:
                    normalized_matrix[:, j] = (col - min_val) / (max_val - min_val)
                else:
                    normalized_matrix[:, j] = 0.5
        
        fig, ax = plt.subplots(figsize=(12, max(8, len(episodes) * 0.5)))
        
        im = ax.imshow(normalized_matrix, cmap='YlOrRd', aspect='auto')
        
        ax.set_xticks(np.arange(len(metric_names)))
        ax.set_xticklabels(metric_names, rotation=45, ha='right')
        ax.set_yticks(np.arange(len(episodes)))
        ax.set_yticklabels(episodes, fontsize=9)
        
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Episodes', fontsize=12)
        ax.set_title('Metric Values Heatmap (Normalized)', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='Normalized Value')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved heatmap to {output_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_statistics_summary(
        self,
        output_path: Optional[Path] = None
    ):
        """
        Plot summary statistics for all metrics.
        
        Args:
            output_path: Path to save figure
        """
        stats = self.analyzer.get_metric_statistics()
        metric_names = list(stats.keys())
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for idx, metric_name in enumerate(metric_names):
            ax = axes[idx]
            stat = stats[metric_name]
            
            if stat['count'] > 0:
                # Bar plot with error bars
                ax.bar([0], [stat['mean']], yerr=[stat['std']], 
                      capsize=10, alpha=0.7, color='steelblue')
                ax.axhline(y=stat['median'], color='red', linestyle='--', 
                         linewidth=2, label='Median')
                ax.axhline(y=stat['min'], color='green', linestyle=':', 
                         alpha=0.5, label='Min')
                ax.axhline(y=stat['max'], color='orange', linestyle=':', 
                         alpha=0.5, label='Max')
                
                ax.set_ylabel(metric_name, fontsize=11)
                ax.set_title(f'{metric_name} Statistics\n(n={stat["count"]})', 
                           fontsize=12, fontweight='bold')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_xticks([])
        
        # Hide unused subplots
        for idx in range(len(metric_names), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Metric Statistics Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved statistics summary to {output_path}")
        else:
            plt.show()
        plt.close()
    
    def generate_all_visualizations(self, output_dir: Path):
        """
        Generate all visualizations and save to output directory.
        
        Args:
            output_dir: Directory to save all visualizations
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\nGenerating visualizations...")
        
        # Metrics comparison
        self.plot_metrics_comparison(output_dir / '01_metrics_comparison.png')
        
        # Box plots by group
        self.plot_metrics_boxplot(output_dir / '02_metrics_boxplot_by_group.png', group_by='group')
        
        # Heatmap
        self.plot_metrics_heatmap(output_dir / '03_metrics_heatmap.png')
        
        # Statistics summary
        self.plot_statistics_summary(output_dir / '04_statistics_summary.png')
        
        # Trajectory comparisons (sample)
        sample_episodes = list(self.analyzer.results.keys())[:5]
        for ep_name in sample_episodes:
            ep_safe = ep_name.replace('/', '_').replace('\\', '_')
            self.plot_trajectory_comparison(
                ep_name,
                output_dir / f'trajectory_comparison_{ep_safe}.png'
            )
        
        print(f"\nAll visualizations saved to {output_dir}")
