#!/usr/bin/env python3
"""
Hogun Episode 1, 2, 3 Comprehensive Analysis Script

Analysis:
1. Step-wise Entropy/Trust value analysis
2. Episode-wise statistical analysis
3. Cross-episode comparison visualization
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import csv

# Publication-quality plot settings with Arial font
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


# Path configuration
BASE_DIR = Path(__file__).parent.parent
LOGS_DIR = BASE_DIR / "logs_good" / "Hogun"
OUTPUT_DIR = Path(__file__).parent  # src/analyze

EPISODES = {
    "Episode 1": LOGS_DIR / "episode1" / "experiment_log.json",
    "Episode 2": LOGS_DIR / "episode2" / "experiment_log.json",
    "Episode 3": LOGS_DIR / "episode3" / "experiment_log.json",
}


def load_episode_data(json_path: Path) -> List[Dict]:
    """Load episode JSON data"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_values(data: List[Dict]) -> Dict[str, np.ndarray]:
    """Extract Entropy/Trust values from episode data"""
    steps = []
    H_X = []
    H_X_given_S = []
    H_X_given_LS = []
    trust_T = []
    
    for step_data in data:
        step_num = step_data.get('step')
        if step_num is None:
            continue
        
        steps.append(step_num)
        H_X.append(step_data.get('entropy_H_X'))
        H_X_given_S.append(step_data.get('entropy_H_X_given_S'))
        H_X_given_LS.append(step_data.get('entropy_H_X_given_LS'))
        trust_T.append(step_data.get('trust_T'))
    
    # Convert None to NaN
    def to_nan_array(values):
        return np.array([np.nan if v is None else v for v in values])
    
    return {
        'steps': np.array(steps),
        'H_X': to_nan_array(H_X),
        'H_X_given_S': to_nan_array(H_X_given_S),
        'H_X_given_LS': to_nan_array(H_X_given_LS),
        'trust_T': to_nan_array(trust_T),
    }


def extract_trajectory(data: List[Dict]) -> Dict[str, np.ndarray]:
    """Extract robot trajectory from episode data"""
    steps = []
    positions_x = []
    positions_y = []
    directions = []
    actions = []
    
    for step_data in data:
        step_num = step_data.get('step')
        if step_num is None:
            continue
        
        state = step_data.get('state', {})
        agent_pos = state.get('agent_pos', [None, None])
        agent_dir = state.get('agent_dir', None)
        action = step_data.get('action', {}).get('name', '')
        
        steps.append(step_num)
        positions_x.append(agent_pos[0] if agent_pos else None)
        positions_y.append(agent_pos[1] if agent_pos else None)
        directions.append(agent_dir)
        actions.append(action)
    
    return {
        'steps': np.array(steps),
        'x': np.array(positions_x, dtype=float),
        'y': np.array(positions_y, dtype=float),
        'dir': np.array(directions),
        'actions': actions,
    }


def analyze_trajectory(trajectory: Dict) -> Dict:
    """Trajectory analysis statistics"""
    x = trajectory['x']
    y = trajectory['y']
    
    # Extract valid positions only
    valid_mask = ~np.isnan(x) & ~np.isnan(y)
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    
    if len(x_valid) < 2:
        return {'total_steps': len(x), 'valid_positions': len(x_valid)}
    
    # Total travel distance (Manhattan distance)
    total_distance = 0
    for i in range(1, len(x_valid)):
        total_distance += abs(x_valid[i] - x_valid[i-1]) + abs(y_valid[i] - y_valid[i-1])
    
    # Start and end positions
    start_pos = (x_valid[0], y_valid[0])
    end_pos = (x_valid[-1], y_valid[-1])
    
    # Direct distance (start to end)
    direct_distance = abs(end_pos[0] - start_pos[0]) + abs(end_pos[1] - start_pos[1])
    
    # Efficiency (direct distance / total distance)
    efficiency = direct_distance / total_distance if total_distance > 0 else 0
    
    # Number of unique positions visited
    unique_positions = set(zip(x_valid, y_valid))
    
    # Revisit count
    revisit_count = len(x_valid) - len(unique_positions)
    
    # Movement range
    x_range = (x_valid.min(), x_valid.max())
    y_range = (y_valid.min(), y_valid.max())
    
    return {
        'total_steps': len(x),
        'valid_positions': len(x_valid),
        'start_pos': start_pos,
        'end_pos': end_pos,
        'total_distance': total_distance,
        'direct_distance': direct_distance,
        'efficiency': efficiency,
        'unique_positions': len(unique_positions),
        'revisit_count': revisit_count,
        'x_range': x_range,
        'y_range': y_range,
    }


def calculate_statistics(values: np.ndarray, name: str) -> Dict:
    """Calculate statistics"""
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return {
            'name': name,
            'count': 0,
            'valid_count': 0,
            'null_count': len(values),
            'mean': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'median': np.nan,
            'q25': np.nan,
            'q75': np.nan,
        }
    
    return {
        'name': name,
        'count': len(values),
        'valid_count': len(valid),
        'null_count': len(values) - len(valid),
        'null_ratio': (len(values) - len(valid)) / len(values) * 100,
        'mean': np.mean(valid),
        'std': np.std(valid),
        'min': np.min(valid),
        'max': np.max(valid),
        'median': np.median(valid),
        'q25': np.percentile(valid, 25),
        'q75': np.percentile(valid, 75),
    }


def filter_outliers(values: np.ndarray, n_std: float = 2.0) -> np.ndarray:
    """Filter outliers (preserving NaN)"""
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return values.copy()
    
    mean = np.mean(valid)
    std = np.std(valid)
    lower = mean - n_std * std
    upper = mean + n_std * std
    
    filtered = values.copy()
    for i, v in enumerate(filtered):
        if not np.isnan(v) and (v < lower or v > upper):
            filtered[i] = np.nan
    
    return filtered


def plot_episode_comparison(all_data: Dict[str, Dict], output_path: Path):
    """Cross-episode comparison visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {'Episode 1': '#1f77b4', 'Episode 2': '#ff7f0e', 'Episode 3': '#2ca02c'}
    markers = {'Episode 1': 'o', 'Episode 2': 's', 'Episode 3': '^'}
    
    metrics = [
        ('H_X', 'H(X)', axes[0, 0]),
        ('H_X_given_S', 'H(X|S)', axes[0, 1]),
        ('H_X_given_LS', 'H(X|L,S)', axes[1, 0]),
        ('trust_T', 'Trust T', axes[1, 1]),
    ]
    
    for key, label, ax in metrics:
        for ep_name, data in all_data.items():
            steps = data['steps']
            values = filter_outliers(data[key], n_std=2.5)
            valid_mask = ~np.isnan(values)
            
            if np.any(valid_mask):
                ax.plot(steps[valid_mask], values[valid_mask], 
                       f'{markers[ep_name]}-', color=colors[ep_name], 
                       label=ep_name, markersize=4, linewidth=1, alpha=0.7)
                
                # Mean line
                mean_val = np.nanmean(values)
                ax.axhline(y=mean_val, color=colors[ep_name], linestyle='--', 
                          alpha=0.3, linewidth=1.5)
        
        ax.set_xlabel('Step', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f'{label} by Step', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Hogun Episodes: Entropy & Trust Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_statistics_comparison(all_stats: List[Dict], output_path: Path):
    """Statistical comparison visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['H_X', 'H_X_given_S', 'H_X_given_LS', 'trust_T']
    labels = ['H(X)', 'H(X|S)', 'H(X|L,S)', 'Trust T']
    episodes = ['Episode 1', 'Episode 2', 'Episode 3']
    
    for idx, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[idx // 2, idx % 2]
        
        means = []
        stds = []
        for ep in episodes:
            stat = next((s for s in all_stats if s['episode'] == ep and s['metric'] == metric), None)
            means.append(stat['mean'] if stat else np.nan)
            stds.append(stat['std'] if stat else np.nan)
        
        x = np.arange(len(episodes))
        bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7,
                     color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        
        ax.set_xticks(x)
        ax.set_xticklabels(episodes, fontsize=10)
        ax.set_ylabel(f'{label} (Mean ± Std)', fontsize=11)
        ax.set_title(f'{label} Statistics by Episode', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Value labels
        for i, (m, s) in enumerate(zip(means, stds)):
            if not np.isnan(m):
                ax.text(i, m + s + 0.001, f'{m:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Hogun Episodes: Statistical Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_null_ratio(all_stats: List[Dict], output_path: Path):
    """Null ratio visualization"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['H_X', 'H_X_given_S', 'H_X_given_LS', 'trust_T']
    labels = ['H(X)', 'H(X|S)', 'H(X|L,S)', 'Trust T']
    episodes = ['Episode 1', 'Episode 2', 'Episode 3']
    
    x = np.arange(len(labels))
    width = 0.25
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, ep in enumerate(episodes):
        null_ratios = []
        for m in metrics:
            stat = next((s for s in all_stats if s['episode'] == ep and s['metric'] == m), None)
            null_ratios.append(stat['null_ratio'] if stat else 0)
        ax.bar(x + i * width, null_ratios, width, label=ep, color=colors[i], alpha=0.7)
    
    ax.set_xlabel('Metric', fontsize=11)
    ax.set_ylabel('Null Ratio (%)', fontsize=11)
    ax.set_title('Hogun Episodes: Null Value Ratio Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_box_comparison(all_data: Dict[str, Dict], output_path: Path):
    """Box plot comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['H_X', 'H_X_given_S', 'H_X_given_LS', 'trust_T']
    labels = ['H(X)', 'H(X|S)', 'H(X|L,S)', 'Trust T']
    episodes = list(all_data.keys())
    
    for idx, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[idx // 2, idx % 2]
        
        box_data = []
        for ep in episodes:
            values = filter_outliers(all_data[ep][metric], n_std=2.5)
            valid = values[~np.isnan(values)]
            box_data.append(valid)
        
        bp = ax.boxplot(box_data, tick_labels=episodes, patch_artist=True)
        
        colors = ['#a6cee3', '#fdbf6f', '#b2df8a']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f'{label} Distribution Comparison', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Hogun Episodes: Distribution Comparison (Box Plot)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_trajectories(all_trajectories: Dict[str, Dict], output_path: Path):
    """Episode-wise trajectory visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    colors = {'Episode 1': '#1f77b4', 'Episode 2': '#ff7f0e', 'Episode 3': '#2ca02c'}
    
    for idx, (ep_name, traj) in enumerate(all_trajectories.items()):
        ax = axes[idx]
        x = traj['x']
        y = traj['y']
        
        # Valid positions only
        valid_mask = ~np.isnan(x) & ~np.isnan(y)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        if len(x_valid) < 2:
            ax.set_title(f'{ep_name}: Insufficient data')
            continue
        
        # Draw trajectory path
        ax.plot(x_valid, y_valid, '-', color=colors[ep_name], alpha=0.5, linewidth=1)
        
        # Plot positions with color gradient by time
        scatter = ax.scatter(x_valid, y_valid, c=np.arange(len(x_valid)), 
                            cmap='viridis', s=30, zorder=5, edgecolors='white', linewidths=0.5)
        
        # Highlight start and end points
        ax.scatter(x_valid[0], y_valid[0], c='green', s=150, marker='s', 
                  label='Start', zorder=10, edgecolors='black', linewidths=2)
        ax.scatter(x_valid[-1], y_valid[-1], c='red', s=150, marker='*', 
                  label='End', zorder=10, edgecolors='black', linewidths=1)
        
        # Direction arrows (sparse)
        arrow_interval = max(1, len(x_valid) // 10)
        for i in range(0, len(x_valid) - 1, arrow_interval):
            dx = x_valid[i+1] - x_valid[i]
            dy = y_valid[i+1] - y_valid[i]
            if dx != 0 or dy != 0:
                ax.annotate('', xy=(x_valid[i+1], y_valid[i+1]), 
                           xytext=(x_valid[i], y_valid[i]),
                           arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7, lw=1))
        
        ax.set_xlabel('X Coordinate', fontsize=11)
        ax.set_ylabel('Y Coordinate', fontsize=11)
        ax.set_title(f'{ep_name} Trajectory ({len(x_valid)} steps)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Match MiniGrid coordinate system
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, label='Step #')
    
    plt.suptitle('Hogun Episodes: Robot Trajectory', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_trajectory_overlay(all_trajectories: Dict[str, Dict], output_path: Path):
    """Overlay all episode trajectories on single plot"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    colors = {'Episode 1': '#1f77b4', 'Episode 2': '#ff7f0e', 'Episode 3': '#2ca02c'}
    
    for ep_name, traj in all_trajectories.items():
        x = traj['x']
        y = traj['y']
        
        valid_mask = ~np.isnan(x) & ~np.isnan(y)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        if len(x_valid) < 2:
            continue
        
        # Draw trajectory
        ax.plot(x_valid, y_valid, '-o', color=colors[ep_name], alpha=0.6, 
               linewidth=1.5, markersize=3, label=f'{ep_name} ({len(x_valid)} steps)')
        
        # Start point
        ax.scatter(x_valid[0], y_valid[0], c=colors[ep_name], s=200, marker='s', 
                  edgecolors='black', linewidths=2, zorder=10)
        # End point
        ax.scatter(x_valid[-1], y_valid[-1], c=colors[ep_name], s=200, marker='*', 
                  edgecolors='black', linewidths=1, zorder=10)
    
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title('Hogun Episodes: Trajectory Comparison (Overlay)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_trajectory_heatmap(all_trajectories: Dict[str, Dict], output_path: Path):
    """Position visit frequency heatmap"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (ep_name, traj) in enumerate(all_trajectories.items()):
        ax = axes[idx]
        x = traj['x']
        y = traj['y']
        
        valid_mask = ~np.isnan(x) & ~np.isnan(y)
        x_valid = x[valid_mask].astype(int)
        y_valid = y[valid_mask].astype(int)
        
        if len(x_valid) < 2:
            ax.set_title(f'{ep_name}: Insufficient data')
            continue
        
        # Calculate visit frequency
        x_range = (x_valid.min(), x_valid.max())
        y_range = (y_valid.min(), y_valid.max())
        
        heatmap = np.zeros((y_range[1] - y_range[0] + 1, x_range[1] - x_range[0] + 1))
        for xi, yi in zip(x_valid, y_valid):
            heatmap[yi - y_range[0], xi - x_range[0]] += 1
        
        im = ax.imshow(heatmap, cmap='YlOrRd', origin='upper', aspect='auto')
        
        # Axis labels
        ax.set_xticks(np.arange(heatmap.shape[1]))
        ax.set_xticklabels(np.arange(x_range[0], x_range[1] + 1))
        ax.set_yticks(np.arange(heatmap.shape[0]))
        ax.set_yticklabels(np.arange(y_range[0], y_range[1] + 1))
        
        ax.set_xlabel('X Coordinate', fontsize=11)
        ax.set_ylabel('Y Coordinate', fontsize=11)
        ax.set_title(f'{ep_name} Visit Frequency', fontsize=12, fontweight='bold')
        
        plt.colorbar(im, ax=ax, label='Visit Count')
    
    plt.suptitle('Hogun Episodes: Position Visit Frequency Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("=" * 60)
    print("Hogun Episodes Comprehensive Analysis")
    print("=" * 60)
    
    # Timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_dir = OUTPUT_DIR / f"hogun_analysis_{timestamp}"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    all_data = {}
    all_stats = []
    all_trajectories = {}
    all_traj_stats = {}
    
    for ep_name, json_path in EPISODES.items():
        print(f"\n[{ep_name}] Loading data...")
        data = load_episode_data(json_path)
        values = extract_values(data)
        all_data[ep_name] = values
        
        # Extract trajectory
        trajectory = extract_trajectory(data)
        all_trajectories[ep_name] = trajectory
        traj_stats = analyze_trajectory(trajectory)
        all_traj_stats[ep_name] = traj_stats
        
        print(f"  - Total steps: {len(values['steps'])}")
        print(f"  - Trajectory: Start{traj_stats.get('start_pos', 'N/A')} -> End{traj_stats.get('end_pos', 'N/A')}")
        print(f"  - Total distance: {traj_stats.get('total_distance', 0):.0f}, Unique positions: {traj_stats.get('unique_positions', 0)}")
        
        # Calculate statistics
        for metric in ['H_X', 'H_X_given_S', 'H_X_given_LS', 'trust_T']:
            stats = calculate_statistics(values[metric], metric)
            stats['episode'] = ep_name
            stats['metric'] = metric
            all_stats.append(stats)
    
    # Helper function to get stat
    def get_stat(ep: str, metric: str) -> Dict:
        return next((s for s in all_stats if s['episode'] == ep and s['metric'] == metric), {})
    
    # Save statistics (CSV)
    stats_path = analysis_dir / "statistics_summary.csv"
    with open(stats_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['episode', 'metric', 'count', 'valid_count', 'null_count', 'null_ratio', 
                     'mean', 'std', 'min', 'max', 'median', 'q25', 'q75']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for stat in all_stats:
            writer.writerow({k: stat.get(k, '') for k in fieldnames})
    print(f"\nStatistics saved: {stats_path}")
    
    # Write analysis report
    report_path = analysis_dir / "analysis_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Hogun Episodes Analysis Report\n\n")
        f.write(f"**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        f.write("## 1. Episode Overview\n\n")
        f.write("| Episode | Total Steps |\n")
        f.write("|---------|-------------|\n")
        for ep_name, data in all_data.items():
            f.write(f"| {ep_name} | {len(data['steps'])} |\n")
        f.write("\n")
        
        f.write("## 2. Statistical Summary\n\n")
        
        for metric, label in [('H_X', 'H(X)'), ('H_X_given_S', 'H(X|S)'), 
                              ('H_X_given_LS', 'H(X|L,S)'), ('trust_T', 'Trust T')]:
            f.write(f"### {label}\n\n")
            f.write("| Episode | Valid Count | Null Ratio | Mean | Std | Min | Max | Median |\n")
            f.write("|---------|-------------|------------|------|-----|-----|-----|--------|\n")
            
            for ep in ['Episode 1', 'Episode 2', 'Episode 3']:
                row = get_stat(ep, metric)
                f.write(f"| {ep} | {row.get('valid_count', 0):.0f} | {row.get('null_ratio', 0):.1f}% | ")
                f.write(f"{row.get('mean', np.nan):.6f} | {row.get('std', np.nan):.6f} | ")
                f.write(f"{row.get('min', np.nan):.6f} | {row.get('max', np.nan):.6f} | {row.get('median', np.nan):.6f} |\n")
            f.write("\n")
        
        f.write("## 3. Analysis Results\n\n")
        
        # H(X) analysis
        f.write("### H(X) - Overall Entropy\n\n")
        hx_means = [get_stat(f'Episode {i}', 'H_X').get('mean', np.nan) for i in [1, 2, 3]]
        best_hx = np.nanargmin(hx_means) + 1
        f.write(f"- **Lowest mean H(X)**: Episode {best_hx} ({hx_means[best_hx-1]:.6f})\n")
        f.write(f"- Lower H(X) indicates higher confidence in VLM's first action token.\n\n")
        
        # H(X|L,S) analysis
        f.write("### H(X|L,S) - Conditional Entropy based on Logprobs\n\n")
        hxls_means = [get_stat(f'Episode {i}', 'H_X_given_LS').get('mean', np.nan) for i in [1, 2, 3]]
        best_hxls = np.nanargmin(hxls_means) + 1
        f.write(f"- **Lowest mean H(X|L,S)**: Episode {best_hxls} ({hxls_means[best_hxls-1]:.6f})\n")
        f.write(f"- H(X|L,S) reflects action decision certainty from actual logprobs distribution.\n\n")
        
        # Trust analysis
        f.write("### Trust T\n\n")
        trust_means = [get_stat(f'Episode {i}', 'trust_T').get('mean', np.nan) for i in [1, 2, 3]]
        trust_stds = [get_stat(f'Episode {i}', 'trust_T').get('std', np.nan) for i in [1, 2, 3]]
        f.write(f"- Episode 1: Mean Trust = {trust_means[0]:.4f} (±{trust_stds[0]:.4f})\n")
        f.write(f"- Episode 2: Mean Trust = {trust_means[1]:.4f} (±{trust_stds[1]:.4f})\n")
        f.write(f"- Episode 3: Mean Trust = {trust_means[2]:.4f} (±{trust_stds[2]:.4f})\n\n")
        
        # Null ratio analysis
        f.write("### Null Value Ratio Analysis\n\n")
        for metric, label in [('H_X', 'H(X)'), ('H_X_given_S', 'H(X|S)'), 
                              ('H_X_given_LS', 'H(X|L,S)'), ('trust_T', 'Trust T')]:
            null_ratios = [get_stat(f'Episode {i}', metric).get('null_ratio', 0) for i in [1, 2, 3]]
            avg_null = np.mean(null_ratios)
            f.write(f"- **{label}**: Average Null Ratio = {avg_null:.1f}%\n")
        
        f.write("\n## 4. Trajectory Analysis\n\n")
        f.write("| Episode | Start Position | End Position | Total Distance | Direct Distance | Efficiency | Unique Positions | Revisit Count |\n")
        f.write("|---------|----------------|--------------|----------------|-----------------|------------|------------------|---------------|\n")
        for ep_name, traj_stats in all_traj_stats.items():
            start = traj_stats.get('start_pos', ('N/A', 'N/A'))
            end = traj_stats.get('end_pos', ('N/A', 'N/A'))
            f.write(f"| {ep_name} | ({start[0]:.0f}, {start[1]:.0f}) | ({end[0]:.0f}, {end[1]:.0f}) | ")
            f.write(f"{traj_stats.get('total_distance', 0):.0f} | {traj_stats.get('direct_distance', 0):.0f} | ")
            f.write(f"{traj_stats.get('efficiency', 0)*100:.1f}% | {traj_stats.get('unique_positions', 0)} | ")
            f.write(f"{traj_stats.get('revisit_count', 0)} |\n")
        f.write("\n")
        
        f.write("### Path Efficiency Analysis\n\n")
        f.write("- **Efficiency** = Direct Distance / Total Distance × 100%\n")
        f.write("- Higher efficiency indicates a more direct path to the goal.\n")
        f.write("- Higher revisit count suggests inefficient path planning.\n\n")
        
        f.write("## 5. Conclusions\n\n")
        f.write("Based on the analysis:\n\n")
        f.write("1. **Entropy Comparison**: Compare mean entropy values across episodes to evaluate VLM's decision confidence.\n")
        f.write("2. **Trust Patterns**: Higher Trust variance indicates more situation-dependent reliability changes.\n")
        f.write("3. **Data Completeness**: Higher null ratio indicates more missing data for metric calculation.\n")
        f.write("4. **Path Efficiency**: Episodes with higher efficiency and lower revisit count achieved better path planning.\n")
    
    print(f"Report saved: {report_path}")
    
    # Visualization
    print("\nGenerating visualizations...")
    
    plot_episode_comparison(all_data, analysis_dir / "01_step_comparison.png")
    plot_statistics_comparison(all_stats, analysis_dir / "02_statistics_comparison.png")
    plot_null_ratio(all_stats, analysis_dir / "03_null_ratio_comparison.png")
    plot_box_comparison(all_data, analysis_dir / "04_box_plot_comparison.png")
    
    # Trajectory visualization
    plot_trajectories(all_trajectories, analysis_dir / "05_trajectory_individual.png")
    plot_trajectory_overlay(all_trajectories, analysis_dir / "06_trajectory_overlay.png")
    plot_trajectory_heatmap(all_trajectories, analysis_dir / "07_trajectory_heatmap.png")
    
    print("\n" + "=" * 60)
    print(f"Analysis complete! Results: {analysis_dir}")
    print("=" * 60)
    
    # Statistics summary
    print("\n[Statistics Summary]")
    for metric, label in [('H_X', 'H(X)'), ('H_X_given_S', 'H(X|S)'), 
                          ('H_X_given_LS', 'H(X|L,S)'), ('trust_T', 'Trust T')]:
        print(f"\n{label}:")
        for ep in ['Episode 1', 'Episode 2', 'Episode 3']:
            row = get_stat(ep, metric)
            print(f"  {ep}: Mean={row.get('mean', np.nan):.6f}, Std={row.get('std', np.nan):.6f}, Null={row.get('null_ratio', 0):.1f}%")


if __name__ == "__main__":
    main()
