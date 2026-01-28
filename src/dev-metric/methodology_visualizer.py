"""
Methodology visualization module.

Visualizes how each metric compares trajectories intuitively.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

# Import metrics for internal calculations
# Use relative imports only if needed, otherwise import directly
try:
    from metrics.dtw import dtw_distance
    from metrics.ddtw import compute_derivatives
    from metrics.sobolev import compute_velocity
except ImportError:
    from .metrics.dtw import dtw_distance
    from .metrics.ddtw import compute_derivatives
    from .metrics.sobolev import compute_velocity

# Publication-quality plot settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'axes.unicode_minus': False,
})


class MethodologyVisualizer:
    """Visualizes how each metric compares trajectories."""
    
    def visualize_rmse_comparison(
        self,
        ref_traj: np.ndarray,
        robot_traj: np.ndarray,
        output_path: Optional[Path] = None
    ):
        """
        RMSE: Time-synchronized point pair matching.
        
        Shows how RMSE matches points at the same time step.
        """
        # Interpolate to same length (take minimum)
        min_len = min(len(ref_traj), len(robot_traj))
        ref = ref_traj[:min_len]
        robot = robot_traj[:min_len]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot trajectories
        ax.plot(ref[:, 0], ref[:, 1], 'b-', linewidth=3, label='Reference (GT)', alpha=0.7)
        ax.plot(robot[:, 0], robot[:, 1], 'r--', linewidth=3, label='Robot', alpha=0.7)
        
        # Draw connections between matched points
        distances = np.linalg.norm(ref - robot, axis=1)
        max_dist = np.max(distances) if len(distances) > 0 else 1.0
        
        for i in range(min_len):
            dist = distances[i]
            # Color by distance (red = far, green = close)
            color_intensity = dist / max_dist if max_dist > 0 else 0
            color = plt.cm.RdYlGn(1 - color_intensity)  # Red-Yellow-Green colormap
            
            ax.plot([ref[i, 0], robot[i, 0]], [ref[i, 1], robot[i, 1]], 
                   '--', color=color, linewidth=1.5, alpha=0.6)
            ax.scatter([ref[i, 0], robot[i, 0]], [ref[i, 1], robot[i, 1]], 
                      c=[color, color], s=50, zorder=5, edgecolors='black', linewidths=1)
        
        ax.set_xlabel('X Coordinate', fontsize=11)
        ax.set_ylabel('Y Coordinate', fontsize=11)
        ax.set_title('RMSE: Time-Synchronized Point Matching\n(Line color = distance)', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        else:
            plt.show()
    
    def visualize_dtw_warping(
        self,
        ref_traj: np.ndarray,
        robot_traj: np.ndarray,
        output_path: Optional[Path] = None
    ):
        """
        DTW: Warping path matrix and optimal path.
        
        Shows how DTW warps time axis non-linearly.
        """
        try:
            from metrics.dtw import euclidean_distance
        except ImportError:
            from .metrics.dtw import euclidean_distance
        
        n = len(ref_traj)
        m = len(robot_traj)
        
        # Compute cost matrix
        cost_matrix = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                cost_matrix[i, j] = euclidean_distance(ref_traj[i], robot_traj[j])
        
        # Compute DTW path (simplified - just show cost matrix)
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Left: Cost matrix heatmap
        ax1 = axes[0]
        im = ax1.imshow(cost_matrix, cmap='YlOrRd', origin='lower', aspect='auto')
        ax1.set_xlabel('Robot Trajectory Index', fontsize=11)
        ax1.set_ylabel('Reference Trajectory Index', fontsize=11)
        ax1.set_title('DTW Cost Matrix', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax1, label='Distance')
        
        # Right: Trajectories with warping visualization
        ax2 = axes[1]
        ax2.plot(ref_traj[:, 0], ref_traj[:, 1], 'b-', linewidth=3, 
                label='Reference', alpha=0.7, marker='o', markersize=4)
        ax2.plot(robot_traj[:, 0], robot_traj[:, 1], 'r--', linewidth=3, 
                label='Robot', alpha=0.7, marker='s', markersize=4)
        
        # Sample warping connections (every 5th point for clarity)
        step = max(1, min(n, m) // 10)
        for i in range(0, min(n, m), step):
            j = i  # Simplified: show diagonal warping
            if i < n and j < m:
                ax2.plot([ref_traj[i, 0], robot_traj[j, 0]], 
                        [ref_traj[i, 1], robot_traj[j, 1]], 
                        'g--', alpha=0.3, linewidth=1)
        
        ax2.set_xlabel('X Coordinate', fontsize=11)
        ax2.set_ylabel('Y Coordinate', fontsize=11)
        ax2.set_title('DTW: Time Warping Visualization', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        ax2.invert_yaxis()
        
        plt.suptitle('DTW: Dynamic Time Warping', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        else:
            plt.show()
    
    def visualize_frechet_matching(
        self,
        ref_traj: np.ndarray,
        robot_traj: np.ndarray,
        output_path: Optional[Path] = None
    ):
        """
        Fréchet: "Dog leash" analogy visualization.
        
        Shows how Fréchet distance finds the minimum leash length.
        """
        try:
            from metrics.frechet import euclidean_distance
        except ImportError:
            from .metrics.frechet import euclidean_distance
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot trajectories
        ax.plot(ref_traj[:, 0], ref_traj[:, 1], 'b-', linewidth=3, 
               label='Reference (Person)', alpha=0.7, marker='o', markersize=5)
        ax.plot(robot_traj[:, 0], robot_traj[:, 1], 'r--', linewidth=3, 
               label='Robot (Dog)', alpha=0.7, marker='s', markersize=5)
        
        # Find matching points (simplified: match by progress ratio)
        n_ref = len(ref_traj)
        n_robot = len(robot_traj)
        max_len = max(n_ref, n_robot)
        
        distances = []
        max_leash = 0
        max_leash_idx = 0
        
        # Sample matching points
        for k in range(0, max_len, max(1, max_len // 20)):
            i = min(int(k * n_ref / max_len), n_ref - 1)
            j = min(int(k * n_robot / max_len), n_robot - 1)
            
            dist = euclidean_distance(ref_traj[i], robot_traj[j])
            distances.append((i, j, dist))
            
            if dist > max_leash:
                max_leash = dist
                max_leash_idx = len(distances) - 1
        
        # Draw connections
        for idx, (i, j, dist) in enumerate(distances):
            if idx == max_leash_idx:
                # Highlight maximum leash
                ax.plot([ref_traj[i, 0], robot_traj[j, 0]], 
                       [ref_traj[i, 1], robot_traj[j, 1]], 
                       'r-', linewidth=3, alpha=0.8, label='Max Leash (Fréchet Distance)')
            else:
                ax.plot([ref_traj[i, 0], robot_traj[j, 0]], 
                       [ref_traj[i, 1], robot_traj[j, 1]], 
                       'g--', linewidth=1, alpha=0.4)
        
        ax.set_xlabel('X Coordinate', fontsize=11)
        ax.set_ylabel('Y Coordinate', fontsize=11)
        ax.set_title(f'Fréchet Distance: "Dog Leash" Analogy\n(Max Leash = {max_leash:.2f})', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        else:
            plt.show()
    
    def visualize_ddtw_derivatives(
        self,
        ref_traj: np.ndarray,
        robot_traj: np.ndarray,
        output_path: Optional[Path] = None
    ):
        """
        DDTW: Original trajectory + derivative comparison.
        
        Shows how DDTW compares derivatives (velocity) instead of positions.
        """
        # Compute derivatives
        ref_deriv = compute_derivatives(ref_traj)
        robot_deriv = compute_derivatives(robot_traj)
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 12))
        
        # Top: Original trajectories
        ax1 = axes[0]
        ax1.plot(ref_traj[:, 0], ref_traj[:, 1], 'b-', linewidth=2, 
                label='Reference', alpha=0.7, marker='o', markersize=4)
        ax1.plot(robot_traj[:, 0], robot_traj[:, 1], 'r--', linewidth=2, 
                label='Robot', alpha=0.7, marker='s', markersize=4)
        ax1.set_xlabel('X Coordinate', fontsize=11)
        ax1.set_ylabel('Y Coordinate', fontsize=11)
        ax1.set_title('Original Trajectories', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        ax1.invert_yaxis()
        
        # Bottom: Derivative (velocity) vectors
        ax2 = axes[1]
        
        # Plot derivative vectors as arrows
        if len(ref_deriv) > 0:
            for i in range(0, len(ref_deriv), max(1, len(ref_deriv) // 20)):
                if i < len(ref_traj) - 1:
                    ax2.arrow(ref_traj[i, 0], ref_traj[i, 1],
                             ref_deriv[i, 0], ref_deriv[i, 1],
                             head_width=0.3, head_length=0.2, fc='blue', ec='blue', alpha=0.6)
        
        if len(robot_deriv) > 0:
            for i in range(0, len(robot_deriv), max(1, len(robot_deriv) // 20)):
                if i < len(robot_traj) - 1:
                    ax2.arrow(robot_traj[i, 0], robot_traj[i, 1],
                             robot_deriv[i, 0], robot_deriv[i, 1],
                             head_width=0.3, head_length=0.2, fc='red', ec='red', 
                             alpha=0.6, linestyle='--')
        
        # Highlight zero velocity (stops)
        for i in range(len(ref_deriv)):
            if np.linalg.norm(ref_deriv[i]) < 0.1 and i < len(ref_traj):
                ax2.scatter(ref_traj[i, 0], ref_traj[i, 1], c='green', s=100, 
                          marker='X', zorder=10, label='Stop (Zero Velocity)' if i == 0 else '')
        
        ax2.plot(ref_traj[:, 0], ref_traj[:, 1], 'b-', linewidth=1, alpha=0.3)
        ax2.plot(robot_traj[:, 0], robot_traj[:, 1], 'r--', linewidth=1, alpha=0.3)
        ax2.set_xlabel('X Coordinate', fontsize=11)
        ax2.set_ylabel('Y Coordinate', fontsize=11)
        ax2.set_title('Derivative (Velocity) Vectors\n(DDTW compares these instead of positions)', 
                     fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        ax2.invert_yaxis()
        
        plt.suptitle('DDTW: Derivative Dynamic Time Warping', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        else:
            plt.show()
    
    def visualize_sobolev_components(
        self,
        ref_traj: np.ndarray,
        robot_traj: np.ndarray,
        output_path: Optional[Path] = None
    ):
        """
        Sobolev: Position error + velocity error decomposition.
        
        Shows how Sobolev combines position and velocity errors.
        """
        try:
            from metrics.sobolev import interpolate_to_same_length
        except ImportError:
            from .metrics.sobolev import interpolate_to_same_length
        
        # Interpolate to same length
        ref_interp, robot_interp = interpolate_to_same_length(ref_traj, robot_traj)
        
        # Compute velocities
        ref_vel = compute_velocity(ref_interp)
        robot_vel = compute_velocity(robot_interp)
        
        # Compute errors
        pos_error = np.linalg.norm(ref_interp - robot_interp, axis=1)
        vel_error = np.linalg.norm(ref_vel - robot_vel, axis=1)
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 14))
        
        # Top: Position error
        ax1 = axes[0]
        ax1.plot(ref_interp[:, 0], ref_interp[:, 1], 'b-', linewidth=2, 
                label='Reference', alpha=0.7, marker='o', markersize=4)
        ax1.plot(robot_interp[:, 0], robot_interp[:, 1], 'r--', linewidth=2, 
                label='Robot', alpha=0.7, marker='s', markersize=4)
        
        # Color points by position error
        scatter1 = ax1.scatter(robot_interp[:, 0], robot_interp[:, 1], 
                              c=pos_error, cmap='Reds', s=50, 
                              edgecolors='black', linewidths=1, zorder=5)
        plt.colorbar(scatter1, ax=ax1, label='Position Error')
        
        ax1.set_xlabel('X Coordinate', fontsize=11)
        ax1.set_ylabel('Y Coordinate', fontsize=11)
        ax1.set_title('Position Error Component', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        ax1.invert_yaxis()
        
        # Middle: Velocity error
        ax2 = axes[1]
        ax2.plot(ref_interp[:, 0], ref_interp[:, 1], 'b-', linewidth=1, alpha=0.3)
        ax2.plot(robot_interp[:, 0], robot_interp[:, 1], 'r--', linewidth=1, alpha=0.3)
        
        # Plot velocity vectors with error coloring
        step = max(1, len(ref_interp) // 15)
        for i in range(0, len(ref_interp), step):
            if i < len(ref_vel) and i < len(robot_vel):
                # Reference velocity
                ax2.arrow(ref_interp[i, 0], ref_interp[i, 1],
                         ref_vel[i, 0], ref_vel[i, 1],
                         head_width=0.2, head_length=0.15, fc='blue', ec='blue', alpha=0.6)
                # Robot velocity
                ax2.arrow(robot_interp[i, 0], robot_interp[i, 1],
                         robot_vel[i, 0], robot_vel[i, 1],
                         head_width=0.2, head_length=0.15, fc='red', ec='red', alpha=0.6)
        
        scatter2 = ax2.scatter(robot_interp[:, 0], robot_interp[:, 1], 
                              c=vel_error, cmap='Oranges', s=50, 
                              edgecolors='black', linewidths=1, zorder=5)
        plt.colorbar(scatter2, ax=ax2, label='Velocity Error')
        
        ax2.set_xlabel('X Coordinate', fontsize=11)
        ax2.set_ylabel('Y Coordinate', fontsize=11)
        ax2.set_title('Velocity Error Component', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        ax2.invert_yaxis()
        
        # Bottom: Combined Sobolev distance
        ax3 = axes[2]
        sobolev_dist = np.sqrt(pos_error**2 + vel_error**2)
        scatter3 = ax3.scatter(robot_interp[:, 0], robot_interp[:, 1], 
                              c=sobolev_dist, cmap='Purples', s=50, 
                              edgecolors='black', linewidths=1, zorder=5)
        ax3.plot(ref_interp[:, 0], ref_interp[:, 1], 'b-', linewidth=1, alpha=0.3)
        plt.colorbar(scatter3, ax=ax3, label='Sobolev Distance')
        
        ax3.set_xlabel('X Coordinate', fontsize=11)
        ax3.set_ylabel('Y Coordinate', fontsize=11)
        ax3.set_title(f'Combined Sobolev Distance\n(Mean: {np.mean(sobolev_dist):.4f})', 
                     fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal')
        ax3.invert_yaxis()
        
        plt.suptitle('Sobolev Metric: Position + Velocity Error', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        else:
            plt.show()
    
    def generate_all_methodology_visualizations(
        self,
        group_name: str,
        ref_traj: np.ndarray,
        sample_robot_traj: np.ndarray,
        output_dir: Path
    ):
        """
        Generate all methodology visualizations for a group.
        
        Args:
            group_name: Name of the group
            ref_traj: Reference trajectory (GT, Episode 1)
            sample_robot_traj: Sample robot trajectory for visualization
            output_dir: Output directory for the group
        """
        methodology_dir = output_dir / 'visualizations' / 'methodology'
        methodology_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating methodology visualizations for {group_name}...")
        
        # RMSE
        self.visualize_rmse_comparison(
            ref_traj, sample_robot_traj,
            methodology_dir / 'rmse_comparison.png'
        )
        
        # DTW
        self.visualize_dtw_warping(
            ref_traj, sample_robot_traj,
            methodology_dir / 'dtw_warping.png'
        )
        
        # Fréchet
        self.visualize_frechet_matching(
            ref_traj, sample_robot_traj,
            methodology_dir / 'frechet_matching.png'
        )
        
        # DDTW
        self.visualize_ddtw_derivatives(
            ref_traj, sample_robot_traj,
            methodology_dir / 'ddtw_derivatives.png'
        )
        
        # Sobolev
        self.visualize_sobolev_components(
            ref_traj, sample_robot_traj,
            methodology_dir / 'sobolev_components.png'
        )
        
        print(f"Methodology visualizations saved to {methodology_dir}")
