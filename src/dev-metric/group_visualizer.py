"""
Group-wise visualization module.

Creates visualizations for each group with Episode 1 as GT.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.font_manager as fm

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


class GroupVisualizer:
    """Visualizer for group-wise analysis results."""
    
    def __init__(self, group_analyzer):
        """
        Initialize group visualizer.
        
        Args:
            group_analyzer: GroupWiseAnalyzer instance with computed results
        """
        self.analyzer = group_analyzer
    
    def plot_episode_trends(self, group_name: str, output_dir: Path):
        """
        Plot episode trends for each metric.
        
        Args:
            group_name: Name of the group
            output_dir: Output directory for the group
        """
        if group_name not in self.analyzer.group_results:
            return
        
        group_result = self.analyzer.group_results[group_name]
        results = group_result['results']
        
        if len(results) == 0:
            return
        
        metric_names = ['RMSE', 'DTW', 'Fréchet', 'ERP', 'DDTW', 'TWED', 'Sobolev']
        n_metrics = len(metric_names)
        
        # Create subplots: 2 rows, 4 columns
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for idx, metric_name in enumerate(metric_names):
            ax = axes[idx]
            
            # Extract episode numbers and metric values
            episode_numbers = []
            metric_values = []
            episode_names = []
            
            for ep_name, ep_result in results.items():
                ep_num = ep_result['episode_number']
                value = ep_result['metrics'].get(metric_name)
                
                if value is not None and not (np.isnan(value) or np.isinf(value)):
                    episode_numbers.append(ep_num)
                    metric_values.append(value)
                    episode_names.append(ep_name)
            
            if len(episode_numbers) > 0:
                # Sort by episode number
                sorted_data = sorted(zip(episode_numbers, metric_values, episode_names))
                ep_nums_sorted, values_sorted, _ = zip(*sorted_data)
                
                # Plot line with markers
                ax.plot(ep_nums_sorted, values_sorted, 'o-', linewidth=2, markersize=8, alpha=0.7)
                
                # Add horizontal reference line (ideal: should be constant)
                if len(values_sorted) > 0:
                    mean_val = np.mean(values_sorted)
                    ax.axhline(y=mean_val, color='red', linestyle='--', 
                              alpha=0.5, linewidth=1, label='Mean')
                
                ax.set_xlabel('Episode Number', fontsize=11)
                ax.set_ylabel(metric_name, fontsize=11)
                ax.set_title(f'{metric_name} by Episode', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=9)
                
                # Set x-axis to integer ticks
                if len(ep_nums_sorted) > 0:
                    ax.set_xticks(ep_nums_sorted)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{metric_name} by Episode', fontsize=12, fontweight='bold')
        
        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'{group_name.upper()}: Metric Trends by Episode', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = output_dir / 'visualizations' / 'episode_trends.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved episode trends to {output_path}")
    
    def plot_all_metrics_comparison(self, group_name: str, output_dir: Path):
        """
        Plot all metrics in one figure for comparison.
        
        Args:
            group_name: Name of the group
            output_dir: Output directory for the group
        """
        if group_name not in self.analyzer.group_results:
            return
        
        group_result = self.analyzer.group_results[group_name]
        results = group_result['results']
        
        if len(results) == 0:
            return
        
        metric_names = ['RMSE', 'DTW', 'Fréchet', 'ERP', 'DDTW', 'TWED', 'Sobolev']
        
        # Extract data
        episode_numbers = []
        metric_data = {name: [] for name in metric_names}
        
        for ep_name, ep_result in results.items():
            ep_num = ep_result['episode_number']
            episode_numbers.append(ep_num)
            
            for metric_name in metric_names:
                value = ep_result['metrics'].get(metric_name)
                if value is not None and not (np.isnan(value) or np.isinf(value)):
                    metric_data[metric_name].append(value)
                else:
                    metric_data[metric_name].append(np.nan)
        
        # Sort by episode number
        sorted_indices = sorted(range(len(episode_numbers)), key=lambda i: episode_numbers[i])
        ep_nums_sorted = [episode_numbers[i] for i in sorted_indices]
        
        # Normalize each metric to [0, 1] for comparison
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(metric_names)))
        
        for idx, metric_name in enumerate(metric_names):
            values = [metric_data[metric_name][i] for i in sorted_indices]
            valid_values = [v for v in values if not np.isnan(v)]
            
            if len(valid_values) > 0:
                # Normalize
                min_val = min(valid_values)
                max_val = max(valid_values)
                if max_val > min_val:
                    normalized = [(v - min_val) / (max_val - min_val) if not np.isnan(v) else np.nan 
                                 for v in values]
                else:
                    normalized = [0.5 if not np.isnan(v) else np.nan for v in values]
                
                ax.plot(ep_nums_sorted, normalized, 'o-', label=metric_name, 
                       linewidth=2, markersize=6, color=colors[idx], alpha=0.7)
        
        ax.set_xlabel('Episode Number', fontsize=12)
        ax.set_ylabel('Normalized Metric Value', fontsize=12)
        ax.set_title(f'{group_name.upper()}: All Metrics Comparison (Normalized)', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(ep_nums_sorted)
        
        plt.tight_layout()
        
        output_path = output_dir / 'visualizations' / 'all_metrics_comparison.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved all metrics comparison to {output_path}")
    
    def plot_individual_metric_trends(self, group_name: str, output_dir: Path):
        """
        Plot individual trend plot for each metric.
        
        Args:
            group_name: Name of the group
            output_dir: Output directory for the group
        """
        if group_name not in self.analyzer.group_results:
            return
        
        group_result = self.analyzer.group_results[group_name]
        results = group_result['results']
        
        if len(results) == 0:
            return
        
        metric_names = ['RMSE', 'DTW', 'Fréchet', 'ERP', 'DDTW', 'TWED', 'Sobolev']
        
        for metric_name in metric_names:
            # Extract data
            episode_numbers = []
            metric_values = []
            
            for ep_name, ep_result in results.items():
                ep_num = ep_result['episode_number']
                value = ep_result['metrics'].get(metric_name)
                
                if value is not None and not (np.isnan(value) or np.isinf(value)):
                    episode_numbers.append(ep_num)
                    metric_values.append(value)
            
            if len(episode_numbers) == 0:
                continue
            
            # Sort by episode number
            sorted_data = sorted(zip(episode_numbers, metric_values))
            ep_nums_sorted, values_sorted = zip(*sorted_data)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot
            ax.plot(ep_nums_sorted, values_sorted, 'o-', linewidth=2.5, markersize=10, 
                   color='steelblue', alpha=0.8)
            
            # Add mean line
            mean_val = np.mean(values_sorted)
            ax.axhline(y=mean_val, color='red', linestyle='--', 
                      linewidth=2, alpha=0.7, label=f'Mean: {mean_val:.4f}')
            
            # Add std band
            std_val = np.std(values_sorted)
            ax.fill_between(ep_nums_sorted, 
                          [mean_val - std_val] * len(ep_nums_sorted),
                          [mean_val + std_val] * len(ep_nums_sorted),
                          alpha=0.2, color='red', label=f'±1 Std: {std_val:.4f}')
            
            ax.set_xlabel('Episode Number', fontsize=12)
            ax.set_ylabel(metric_name, fontsize=12)
            ax.set_title(f'{group_name.upper()}: {metric_name} Trend by Episode', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            ax.set_xticks(ep_nums_sorted)
            
            plt.tight_layout()
            
            # Save
            metric_safe = metric_name.lower().replace('é', 'e')
            output_path = output_dir / 'visualizations' / f'{metric_safe}_trend.png'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"Saved {metric_name} trend to {output_path}")
    
    def generate_all_visualizations(self, output_dir: Path):
        """
        Generate all visualizations for all groups.
        
        Args:
            output_dir: Base output directory
        """
        for group_name in self.analyzer.group_results.keys():
            group_output_dir = output_dir / group_name
            print(f"\nGenerating visualizations for group: {group_name}")
            
            self.plot_episode_trends(group_name, group_output_dir)
            self.plot_all_metrics_comparison(group_name, group_output_dir)
            self.plot_individual_metric_trends(group_name, group_output_dir)
