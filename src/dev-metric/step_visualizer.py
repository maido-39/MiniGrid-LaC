"""
Rich visualizations for step-by-step analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class StepVisualizer:
    """Rich visualizer for step-by-step analysis."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize step visualizer.
        
        Args:
            output_dir: Output directory for figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid' if hasattr(plt.style, 'available') and 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    
    def plot_trajectory_comparison_with_metrics(
        self,
        trajectory: np.ndarray,
        gt_trajectory: np.ndarray,
        stepwise_metrics: Dict[str, np.ndarray],
        episode_name: str,
        save_path: Optional[Path] = None
    ):
        """
        Plot trajectory comparison with stepwise metrics overlaid.
        
        Args:
            trajectory: Robot trajectory
            gt_trajectory: Ground truth trajectory
            stepwise_metrics: Stepwise metric values
            episode_name: Name of episode
            save_path: Path to save figure
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Trajectory comparison (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], 'b-', linewidth=2, 
                label='Ground Truth', alpha=0.7)
        ax1.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, 
                label='Robot Path', alpha=0.7)
        ax1.scatter(gt_trajectory[0, 0], gt_trajectory[0, 1], c='green', 
                   s=100, marker='o', label='Start', zorder=5)
        ax1.scatter(gt_trajectory[-1, 0], gt_trajectory[-1, 1], c='red', 
                   s=100, marker='s', label='End', zorder=5)
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_title(f'Trajectory Comparison: {episode_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', adjustable='box')
        
        # 2. Stepwise metrics (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        for metric_name, values in stepwise_metrics.items():
            if len(values) > 0:
                ax2.plot(values, label=metric_name, alpha=0.7, linewidth=1.5)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Metric Value')
        ax2.set_title('Stepwise Metric Values')
        ax2.legend(fontsize=8, ncol=2)
        ax2.grid(True, alpha=0.3)
        
        # 3. Position error over steps (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        min_len = min(len(trajectory), len(gt_trajectory))
        position_errors = [
            np.linalg.norm(trajectory[i] - gt_trajectory[i])
            for i in range(min_len)
        ]
        ax3.plot(position_errors, 'purple', linewidth=2, label='Position Error')
        ax3.fill_between(range(len(position_errors)), position_errors, alpha=0.3)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Position Error')
        ax3.set_title('Position Error Over Steps')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Velocity comparison (middle right)
        ax4 = fig.add_subplot(gs[1, 1])
        if len(trajectory) > 1 and len(gt_trajectory) > 1:
            traj_velocities = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
            gt_velocities = np.linalg.norm(np.diff(gt_trajectory, axis=0), axis=1)
            min_vel_len = min(len(traj_velocities), len(gt_velocities))
            ax4.plot(traj_velocities[:min_vel_len], 'r-', linewidth=2, 
                    label='Robot Velocity', alpha=0.7)
            ax4.plot(gt_velocities[:min_vel_len], 'b-', linewidth=2, 
                    label='GT Velocity', alpha=0.7)
            ax4.set_xlabel('Step')
            ax4.set_ylabel('Velocity')
            ax4.set_title('Velocity Comparison')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. Metric heatmap (bottom left)
        ax5 = fig.add_subplot(gs[2, 0])
        metric_names = []
        metric_data = []
        for metric_name, values in stepwise_metrics.items():
            if len(values) > 0:
                metric_names.append(metric_name)
                metric_data.append(values)
        
        if metric_data:
            # Find max length and pad all arrays to same length
            max_len = max(len(data) for data in metric_data)
            normalized_data = []
            for data in metric_data:
                data_array = np.array(data)
                # Pad with last value
                if len(data_array) < max_len:
                    padded = np.pad(data_array, (0, max_len - len(data_array)), 
                                  mode='edge')
                else:
                    padded = data_array[:max_len]
                
                # Normalize to [0, 1]
                if np.max(padded) > np.min(padded):
                    normalized = (padded - np.min(padded)) / (
                        np.max(padded) - np.min(padded)
                    )
                else:
                    normalized = padded
                normalized_data.append(normalized)
            
            if normalized_data:
                normalized_matrix = np.array(normalized_data)
                im = ax5.imshow(normalized_matrix, aspect='auto', cmap='viridis', 
                               interpolation='nearest')
                ax5.set_yticks(range(len(metric_names)))
                ax5.set_yticklabels(metric_names)
                ax5.set_xlabel('Step')
                ax5.set_title('Normalized Metric Values (Heatmap)')
                plt.colorbar(im, ax=ax5)
        
        # 6. Cumulative error (bottom right)
        ax6 = fig.add_subplot(gs[2, 1])
        if len(position_errors) > 0:
            cumulative_error = np.cumsum(position_errors)
            ax6.plot(cumulative_error, 'orange', linewidth=2, 
                    label='Cumulative Position Error')
            ax6.set_xlabel('Step')
            ax6.set_ylabel('Cumulative Error')
            ax6.set_title('Cumulative Position Error')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        plt.suptitle(f'Comprehensive Step-by-Step Analysis: {episode_name}', 
                    fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {save_path}")
        else:
            save_path = self.output_dir / f'{episode_name}_stepwise_analysis.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {save_path}")
        
        plt.close()
    
    def plot_metric_sensitivity_heatmap(
        self,
        sensitivity_matrix: Dict[str, Dict[str, float]],
        save_path: Optional[Path] = None
    ):
        """
        Plot heatmap showing metric sensitivity to different features.
        
        Args:
            sensitivity_matrix: Dictionary mapping metrics to feature correlations
            save_path: Path to save figure
        """
        if not sensitivity_matrix:
            return
        
        # Prepare data
        metric_names = list(sensitivity_matrix.keys())
        all_features = set()
        for correlations in sensitivity_matrix.values():
            all_features.update(correlations.keys())
        feature_names = sorted(list(all_features))
        
        # Create matrix
        matrix = np.zeros((len(metric_names), len(feature_names)))
        for i, metric_name in enumerate(metric_names):
            for j, feature_name in enumerate(feature_names):
                if feature_name in sensitivity_matrix[metric_name]:
                    matrix[i, j] = sensitivity_matrix[metric_name][feature_name]
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r', 
                      vmin=-1, vmax=1, interpolation='nearest')
        
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.set_yticks(range(len(metric_names)))
        ax.set_yticklabels(metric_names)
        ax.set_xlabel('Trajectory Features')
        ax.set_ylabel('Metrics')
        ax.set_title('Metric Sensitivity to Trajectory Features\n(Correlation Coefficient)')
        
        # Add text annotations
        for i in range(len(metric_names)):
            for j in range(len(feature_names)):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax, label='Correlation Coefficient')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = self.output_dir / 'metric_sensitivity_heatmap.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        print(f"Saved sensitivity heatmap to {save_path}")
        plt.close()
    
    def plot_metric_response_to_features(
        self,
        stepwise_metrics: Dict[str, np.ndarray],
        stepwise_features: Dict[str, np.ndarray],
        save_path: Optional[Path] = None
    ):
        """
        Plot how metrics respond to different features.
        
        Args:
            stepwise_metrics: Stepwise metric values
            stepwise_features: Stepwise feature values
            save_path: Path to save figure
        """
        n_metrics = len(stepwise_metrics)
        n_features = len(stepwise_features)
        
        if n_metrics == 0 or n_features == 0:
            return
        
        fig, axes = plt.subplots(n_metrics, n_features, 
                                figsize=(4*n_features, 3*n_metrics))
        
        if n_metrics == 1:
            axes = axes.reshape(1, -1)
        if n_features == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (metric_name, metric_values) in enumerate(stepwise_metrics.items()):
            if len(metric_values) == 0:
                continue
            
            for j, (feature_name, feature_values) in enumerate(stepwise_features.items()):
                if len(feature_values) == 0:
                    continue
                
                ax = axes[i, j]
                
                # Align lengths
                min_len = min(len(metric_values), len(feature_values))
                if min_len < 2:
                    continue
                
                metric_aligned = np.array(metric_values[:min_len])
                feature_aligned = np.array(feature_values[:min_len])
                
                # Scatter plot
                ax.scatter(feature_aligned, metric_aligned, alpha=0.5, s=20)
                
                # Add trend line
                if np.std(feature_aligned) > 0:
                    z = np.polyfit(feature_aligned, metric_aligned, 1)
                    p = np.poly1d(z)
                    ax.plot(feature_aligned, p(feature_aligned), 
                           "r--", alpha=0.8, linewidth=2)
                
                ax.set_xlabel(feature_name)
                if j == 0:
                    ax.set_ylabel(metric_name)
                if i == 0:
                    ax.set_title(feature_name)
        
        plt.suptitle('Metric Response to Trajectory Features', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = self.output_dir / 'metric_response_to_features.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        print(f"Saved metric response plot to {save_path}")
        plt.close()
