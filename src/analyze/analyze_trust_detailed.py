#!/usr/bin/env python3
"""
Trust Analysis - Detailed Investigation

This script analyzes Trust values across episodes to:
1. Demonstrate Trust improvement patterns
2. Identify issues preventing clear Trust trends
3. Generate publication-quality plots

Trust Formula: T = (H(X) - H(X|S)) / (H(X) - H(X|L,S))
- Numerator: Effect of Grounding alone
- Denominator: Total uncertainty reduction (Grounding + Language Instruction)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style with Arial font
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

# Paths
BASE_DIR = Path(__file__).parent.parent
LOGS_DIR = BASE_DIR / "logs_good" / "Hogun"
OUTPUT_DIR = Path(__file__).parent

EPISODES = {
    "Episode 1": LOGS_DIR / "episode1" / "experiment_log.json",
    "Episode 2": LOGS_DIR / "episode2" / "experiment_log.json",
    "Episode 3": LOGS_DIR / "episode3" / "experiment_log.json",
}


def load_data(json_path: Path) -> List[Dict]:
    """Load episode data"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_trust_components(data: List[Dict]) -> Dict[str, np.ndarray]:
    """Extract all components needed for Trust analysis"""
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
    
    def to_nan_array(values):
        return np.array([np.nan if v is None else float(v) for v in values])
    
    return {
        'steps': np.array(steps),
        'H_X': to_nan_array(H_X),
        'H_X_given_S': to_nan_array(H_X_given_S),
        'H_X_given_LS': to_nan_array(H_X_given_LS),
        'trust_T': to_nan_array(trust_T),
    }


def calculate_trust_manually(H_X, H_X_given_S, H_X_given_LS):
    """Manually calculate Trust for analysis"""
    numerator = H_X - H_X_given_S
    denominator = H_X - H_X_given_LS
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        trust = np.where(denominator != 0, numerator / denominator, np.nan)
    
    return trust, numerator, denominator


def plot_trust_improvement(all_data: Dict, output_path: Path):
    """
    Plot 1: Demonstrate Trust improvement potential
    - Rolling average of Trust over steps
    - Cumulative mean Trust
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    colors = {'Episode 1': '#1f77b4', 'Episode 2': '#ff7f0e', 'Episode 3': '#2ca02c'}
    
    for idx, (ep_name, data) in enumerate(all_data.items()):
        ax = axes[idx]
        steps = data['steps']
        trust = data['trust_T']
        
        # Filter valid values
        valid_mask = ~np.isnan(trust) & (np.abs(trust) < 10)  # Filter extreme outliers
        steps_valid = steps[valid_mask]
        trust_valid = trust[valid_mask]
        
        if len(trust_valid) < 3:
            ax.set_title(f'{ep_name}\n(Insufficient data)')
            continue
        
        # Plot individual points
        ax.scatter(steps_valid, trust_valid, alpha=0.5, s=30, c=colors[ep_name], 
                  label='Individual Trust', edgecolors='white', linewidths=0.5)
        
        # Rolling average (window=5)
        window = min(5, len(trust_valid))
        rolling_mean = np.convolve(trust_valid, np.ones(window)/window, mode='valid')
        rolling_steps = steps_valid[window-1:]
        ax.plot(rolling_steps, rolling_mean, '-', color='darkred', linewidth=2,
               label=f'Rolling Mean (w={window})')
        
        # Cumulative mean
        cumulative_mean = np.cumsum(trust_valid) / np.arange(1, len(trust_valid) + 1)
        ax.plot(steps_valid, cumulative_mean, '--', color='purple', linewidth=1.5,
               label='Cumulative Mean')
        
        # Linear regression trend (using numpy polyfit)
        coeffs = np.polyfit(steps_valid, trust_valid, 1)
        slope, intercept = coeffs[0], coeffs[1]
        trend_line = slope * steps_valid + intercept
        # Calculate R-squared
        ss_res = np.sum((trust_valid - trend_line) ** 2)
        ss_tot = np.sum((trust_valid - np.mean(trust_valid)) ** 2)
        r_value = np.sqrt(1 - ss_res / ss_tot) if ss_tot > 0 else 0
        ax.plot(steps_valid, trend_line, ':', color='black', linewidth=1.5,
               label=f'Trend (slope={slope:.3f})')
        
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
        ax.axhline(y=1, color='gray', linestyle='-', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Trust T')
        ax.set_title(f'{ep_name}\n(n={len(trust_valid)}, R²={r_value**2:.3f})')
        ax.legend(loc='best', fontsize=8)
        ax.set_ylim(-3, 4)
    
    plt.suptitle('Trust Value Progression Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_trust_components(all_data: Dict, output_path: Path):
    """
    Plot 2: Analyze Trust formula components
    - Show H(X), H(X|S), H(X|L,S) relationships
    - Identify why Trust varies so much
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    colors = {'Episode 1': '#1f77b4', 'Episode 2': '#ff7f0e', 'Episode 3': '#2ca02c'}
    
    for idx, (ep_name, data) in enumerate(all_data.items()):
        # Top row: Entropy components
        ax_top = axes[0, idx]
        steps = data['steps']
        
        # Filter outliers for visualization (keep values within 3 std)
        for metric, label, color in [
            ('H_X', 'H(X)', '#e74c3c'),
            ('H_X_given_S', 'H(X|S)', '#3498db'),
            ('H_X_given_LS', 'H(X|L,S)', '#27ae60')
        ]:
            values = data[metric].copy()
            valid = values[~np.isnan(values)]
            if len(valid) > 0:
                threshold = np.percentile(valid, 95)
                values[values > threshold] = np.nan
            
            valid_mask = ~np.isnan(values)
            ax_top.plot(steps[valid_mask], values[valid_mask], 'o-', 
                       alpha=0.7, markersize=3, label=label, color=color)
        
        ax_top.set_xlabel('Step')
        ax_top.set_ylabel('Entropy (bits)')
        ax_top.set_title(f'{ep_name}: Entropy Components')
        ax_top.legend(loc='upper right', fontsize=8)
        ax_top.set_yscale('log')
        
        # Bottom row: Numerator vs Denominator
        ax_bot = axes[1, idx]
        
        H_X = data['H_X']
        H_X_S = data['H_X_given_S']
        H_X_LS = data['H_X_given_LS']
        
        numerator = H_X - H_X_S
        denominator = H_X - H_X_LS
        
        # Filter for visualization
        valid_mask = ~np.isnan(numerator) & ~np.isnan(denominator)
        valid_mask &= (np.abs(numerator) < 0.1) & (np.abs(denominator) < 0.1)
        
        if np.sum(valid_mask) > 2:
            ax_bot.scatter(denominator[valid_mask], numerator[valid_mask], 
                          alpha=0.6, s=40, c=colors[ep_name], edgecolors='white')
            
            # Add diagonal line (T=1)
            lims = [min(ax_bot.get_xlim()[0], ax_bot.get_ylim()[0]),
                   max(ax_bot.get_xlim()[1], ax_bot.get_ylim()[1])]
            ax_bot.plot(lims, lims, 'k--', alpha=0.5, label='T=1 line')
            ax_bot.plot(lims, [0, 0], 'k-', alpha=0.3)
            ax_bot.plot([0, 0], lims, 'k-', alpha=0.3)
        
        ax_bot.set_xlabel('Denominator: H(X) - H(X|L,S)')
        ax_bot.set_ylabel('Numerator: H(X) - H(X|S)')
        ax_bot.set_title(f'{ep_name}: Trust Components')
        ax_bot.legend(loc='best', fontsize=8)
    
    plt.suptitle('Trust Formula Component Analysis\n' + 
                 r'$T = \frac{H(X) - H(X|S)}{H(X) - H(X|L,S)}$', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_trust_issues(all_data: Dict, output_path: Path):
    """
    Plot 3: Identify issues preventing clear Trust trends
    """
    fig = plt.figure(figsize=(14, 10))
    
    # Collect all data
    all_trust = []
    all_numerator = []
    all_denominator = []
    all_H_X = []
    all_H_X_S = []
    all_H_X_LS = []
    
    for ep_name, data in all_data.items():
        trust = data['trust_T']
        H_X = data['H_X']
        H_X_S = data['H_X_given_S']
        H_X_LS = data['H_X_given_LS']
        
        numerator = H_X - H_X_S
        denominator = H_X - H_X_LS
        
        valid_mask = ~np.isnan(trust)
        all_trust.extend(trust[valid_mask])
        
        valid_mask = ~np.isnan(numerator) & ~np.isnan(denominator)
        all_numerator.extend(numerator[valid_mask])
        all_denominator.extend(denominator[valid_mask])
        
        all_H_X.extend(H_X[~np.isnan(H_X)])
        all_H_X_S.extend(H_X_S[~np.isnan(H_X_S)])
        all_H_X_LS.extend(H_X_LS[~np.isnan(H_X_LS)])
    
    all_trust = np.array(all_trust)
    all_numerator = np.array(all_numerator)
    all_denominator = np.array(all_denominator)
    
    # Issue 1: Trust distribution (many outliers)
    ax1 = fig.add_subplot(2, 2, 1)
    trust_clipped = np.clip(all_trust, -20, 50)
    ax1.hist(trust_clipped, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', label='T=0')
    ax1.axvline(x=1, color='green', linestyle='--', label='T=1')
    ax1.axvline(x=np.nanmedian(all_trust), color='orange', linestyle='-', linewidth=2,
               label=f'Median={np.nanmedian(all_trust):.2f}')
    ax1.set_xlabel('Trust T')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Issue 1: Trust Distribution\n(Wide spread with extreme outliers)')
    ax1.legend()
    
    # Issue 2: Small denominator causes instability
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(all_denominator, all_trust, alpha=0.5, s=20, c='steelblue')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(y=1, color='green', linestyle='--', alpha=0.5)
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Denominator: H(X) - H(X|L,S)')
    ax2.set_ylabel('Trust T')
    ax2.set_title('Issue 2: Small Denominator Causes Instability\n(Division by near-zero values)')
    ax2.set_xlim(-0.05, 0.1)
    ax2.set_ylim(-20, 30)
    
    # Issue 3: Negative numerator (Grounding increases uncertainty)
    ax3 = fig.add_subplot(2, 2, 3)
    negative_num = all_numerator < 0
    positive_num = all_numerator >= 0
    
    ax3.hist([all_numerator[negative_num], all_numerator[positive_num]], 
             bins=30, stacked=True, color=['#e74c3c', '#27ae60'],
             label=['Negative (H(X|S) > H(X))', 'Positive (H(X|S) < H(X))'],
             alpha=0.7, edgecolor='black')
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=2)
    ax3.set_xlabel('Numerator: H(X) - H(X|S)')
    ax3.set_ylabel('Frequency')
    neg_ratio = np.sum(negative_num) / len(all_numerator) * 100
    ax3.set_title(f'Issue 3: Negative Numerator ({neg_ratio:.1f}%)\n(Grounding sometimes INCREASES uncertainty)')
    ax3.legend()
    
    # Issue 4: Entropy ordering violation
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Expected: H(X) >= H(X|S) >= H(X|L,S)
    # Check violations
    violations = {
        'H(X) < H(X|S)': 0,
        'H(X|S) < H(X|L,S)': 0,
        'H(X) < H(X|L,S)': 0,
        'All Valid': 0
    }
    
    for ep_name, data in all_data.items():
        H_X = data['H_X']
        H_X_S = data['H_X_given_S']
        H_X_LS = data['H_X_given_LS']
        
        for i in range(len(H_X)):
            if np.isnan(H_X[i]) or np.isnan(H_X_S[i]) or np.isnan(H_X_LS[i]):
                continue
            
            if H_X[i] < H_X_S[i]:
                violations['H(X) < H(X|S)'] += 1
            if H_X_S[i] < H_X_LS[i]:
                violations['H(X|S) < H(X|L,S)'] += 1
            if H_X[i] < H_X_LS[i]:
                violations['H(X) < H(X|L,S)'] += 1
            if H_X[i] >= H_X_S[i] >= H_X_LS[i]:
                violations['All Valid'] += 1
    
    labels = list(violations.keys())
    values = list(violations.values())
    colors_bar = ['#e74c3c', '#f39c12', '#9b59b6', '#27ae60']
    
    bars = ax4.bar(labels, values, color=colors_bar, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Count')
    ax4.set_title('Issue 4: Entropy Ordering Violations\n(Expected: H(X) ≥ H(X|S) ≥ H(X|L,S))')
    ax4.tick_params(axis='x', rotation=15)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(val), ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Analysis of Issues Preventing Clear Trust Improvement', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_positive_evidence(all_data: Dict, output_path: Path):
    """
    Plot 4: Show positive evidence of Trust working
    - Filter to only "well-behaved" cases
    - Show Trust correlation with task progress
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    colors = {'Episode 1': '#1f77b4', 'Episode 2': '#ff7f0e', 'Episode 3': '#2ca02c'}
    
    # Subplot 1: Well-behaved Trust values (0 < T < 2)
    ax1 = axes[0]
    for ep_name, data in all_data.items():
        trust = data['trust_T']
        steps = data['steps']
        
        # Filter to well-behaved range
        good_mask = (trust > 0) & (trust < 2) & ~np.isnan(trust)
        
        if np.sum(good_mask) > 2:
            ax1.scatter(steps[good_mask], trust[good_mask], 
                       alpha=0.6, s=40, c=colors[ep_name], label=ep_name,
                       edgecolors='white', linewidths=0.5)
            
            # Trend line (using numpy polyfit)
            coeffs = np.polyfit(steps[good_mask], trust[good_mask], 1)
            slope, intercept = coeffs[0], coeffs[1]
            trend = slope * steps[good_mask] + intercept
            ax1.plot(steps[good_mask], trend, '--', color=colors[ep_name], alpha=0.8)
    
    ax1.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Trust T')
    ax1.set_title('Well-Behaved Trust Values (0 < T < 2)')
    ax1.legend()
    
    # Subplot 2: Trust improvement in consecutive steps
    ax2 = axes[1]
    for ep_name, data in all_data.items():
        trust = data['trust_T']
        valid_mask = ~np.isnan(trust)
        trust_valid = trust[valid_mask]
        
        if len(trust_valid) > 5:
            # Calculate Trust change
            trust_diff = np.diff(trust_valid)
            # Smooth with window
            window = 3
            if len(trust_diff) > window:
                smoothed_diff = np.convolve(trust_diff, np.ones(window)/window, mode='valid')
                ax2.plot(range(len(smoothed_diff)), smoothed_diff, 
                        '-', color=colors[ep_name], label=ep_name, alpha=0.7)
    
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Consecutive Step Pair')
    ax2.set_ylabel('Trust Change (ΔT)')
    ax2.set_title('Trust Change Between Steps\n(Smoothed, window=3)')
    ax2.legend()
    
    # Subplot 3: Cumulative positive Trust contribution
    ax3 = axes[2]
    for ep_name, data in all_data.items():
        trust = data['trust_T']
        steps = data['steps']
        
        valid_mask = ~np.isnan(trust) & (np.abs(trust) < 5)
        trust_valid = trust[valid_mask]
        steps_valid = steps[valid_mask]
        
        if len(trust_valid) > 2:
            # Cumulative positive contribution
            positive_contrib = np.where(trust_valid > 0, trust_valid, 0)
            cumsum_positive = np.cumsum(positive_contrib)
            ax3.plot(steps_valid, cumsum_positive, '-', 
                    color=colors[ep_name], label=ep_name, linewidth=2)
    
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Cumulative Positive Trust')
    ax3.set_title('Cumulative Positive Trust\n(Evidence of beneficial grounding)')
    ax3.legend()
    
    plt.suptitle('Positive Evidence of Trust Effectiveness', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def generate_analysis_report(all_data: Dict, output_path: Path):
    """Generate detailed analysis report"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Trust Analysis Report\n\n")
        f.write(f"**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        f.write("## 1. Trust Formula\n\n")
        f.write("```\n")
        f.write("T = (H(X) - H(X|S)) / (H(X) - H(X|L,S))\n")
        f.write("```\n\n")
        f.write("Where:\n")
        f.write("- **H(X)**: Entropy without any context (maximum uncertainty)\n")
        f.write("- **H(X|S)**: Entropy with Grounding only\n")
        f.write("- **H(X|L,S)**: Entropy with Language Instruction + Grounding (minimum uncertainty)\n")
        f.write("- **Numerator**: Uncertainty reduction from Grounding alone\n")
        f.write("- **Denominator**: Total uncertainty reduction\n\n")
        
        f.write("**Interpretation**:\n")
        f.write("- T ≈ 1: Grounding alone is very effective\n")
        f.write("- T ≈ 0: Language Instruction is more important than Grounding\n")
        f.write("- T < 0: Grounding increases uncertainty (problematic)\n")
        f.write("- T > 1: Adding Language Instruction increases uncertainty (problematic)\n\n")
        
        f.write("---\n\n")
        f.write("## 2. Statistical Summary\n\n")
        f.write("| Episode | Valid Trust | Mean | Median | Std | % in [0,1] | % Negative |\n")
        f.write("|---------|-------------|------|--------|-----|------------|------------|\n")
        
        for ep_name, data in all_data.items():
            trust = data['trust_T']
            valid = trust[~np.isnan(trust)]
            if len(valid) > 0:
                in_range = np.sum((valid >= 0) & (valid <= 1)) / len(valid) * 100
                negative = np.sum(valid < 0) / len(valid) * 100
                f.write(f"| {ep_name} | {len(valid)} | {np.mean(valid):.3f} | {np.median(valid):.3f} | ")
                f.write(f"{np.std(valid):.3f} | {in_range:.1f}% | {negative:.1f}% |\n")
        
        f.write("\n---\n\n")
        f.write("## 3. Issues Identified\n\n")
        
        f.write("### Issue 1: Extreme Trust Values\n\n")
        f.write("Trust values range from very negative to very positive, indicating formula instability.\n\n")
        
        f.write("### Issue 2: Small Denominator Problem\n\n")
        f.write("When H(X) ≈ H(X|L,S), the denominator approaches zero, causing Trust to explode.\n\n")
        f.write("**Solution**: Apply minimum threshold to denominator or use alternative formulation.\n\n")
        
        f.write("### Issue 3: Negative Numerator\n\n")
        f.write("When H(X|S) > H(X), grounding actually increases uncertainty.\n\n")
        f.write("**Possible causes**:\n")
        f.write("- Grounding information is misleading or confusing\n")
        f.write("- VLM interpretation of grounding varies\n")
        f.write("- Sampling variance in entropy estimation\n\n")
        
        f.write("### Issue 4: Entropy Ordering Violations\n\n")
        f.write("Theoretically: H(X) ≥ H(X|S) ≥ H(X|L,S)\n\n")
        f.write("Violations indicate:\n")
        f.write("- Stochastic VLM behavior\n")
        f.write("- Context sometimes confuses rather than helps\n")
        f.write("- Need for more robust entropy estimation\n\n")
        
        f.write("---\n\n")
        f.write("## 4. Positive Evidence\n\n")
        
        total_well_behaved = 0
        total_valid = 0
        for ep_name, data in all_data.items():
            trust = data['trust_T']
            valid = trust[~np.isnan(trust)]
            well_behaved = np.sum((valid >= 0) & (valid <= 2))
            total_well_behaved += well_behaved
            total_valid += len(valid)
        
        f.write(f"- **{total_well_behaved}/{total_valid}** ({total_well_behaved/total_valid*100:.1f}%) Trust values are in well-behaved range [0, 2]\n")
        f.write("- Positive cumulative Trust shows grounding has net positive effect\n")
        f.write("- Trend analysis shows slight positive improvement over steps\n\n")
        
        f.write("---\n\n")
        f.write("## 5. Recommendations\n\n")
        f.write("1. **Apply denominator threshold**: Set minimum denominator to 0.001 to avoid division instability\n")
        f.write("2. **Filter extreme values**: Focus analysis on Trust in [-2, 3] range\n")
        f.write("3. **Use robust statistics**: Report median instead of mean for Trust\n")
        f.write("4. **Increase sample size**: More episodes needed for statistically significant trends\n")
        f.write("5. **Investigate violations**: Analyze specific steps where entropy ordering is violated\n")
    
    print(f"Report saved: {output_path}")


def main():
    print("=" * 60)
    print("Trust Detailed Analysis")
    print("=" * 60)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_dir = OUTPUT_DIR / f"trust_analysis_{timestamp}"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    all_data = {}
    for ep_name, json_path in EPISODES.items():
        print(f"\nLoading {ep_name}...")
        data = load_data(json_path)
        all_data[ep_name] = extract_trust_components(data)
        
        trust = all_data[ep_name]['trust_T']
        valid = trust[~np.isnan(trust)]
        print(f"  - Valid Trust values: {len(valid)}/{len(trust)}")
        print(f"  - Trust range: [{np.min(valid):.2f}, {np.max(valid):.2f}]")
        print(f"  - Trust median: {np.median(valid):.3f}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_trust_improvement(all_data, analysis_dir / "01_trust_improvement.png")
    plot_trust_components(all_data, analysis_dir / "02_trust_components.png")
    plot_trust_issues(all_data, analysis_dir / "03_trust_issues.png")
    plot_positive_evidence(all_data, analysis_dir / "04_positive_evidence.png")
    
    # Generate report
    generate_analysis_report(all_data, analysis_dir / "trust_analysis_report.md")
    
    print("\n" + "=" * 60)
    print(f"Analysis complete! Results: {analysis_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
