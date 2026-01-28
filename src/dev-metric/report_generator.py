"""
Report generator for group-wise analysis.

Generates detailed analysis reports with insights about each metric.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List
from datetime import datetime


class ReportGenerator:
    """Generates detailed analysis reports."""
    
    def __init__(self, group_analyzer):
        """
        Initialize report generator.
        
        Args:
            group_analyzer: GroupWiseAnalyzer instance with computed results
        """
        self.analyzer = group_analyzer
    
    def analyze_metric_characteristics(self, group_name: str) -> Dict:
        """
        Analyze characteristics of each metric.
        
        Args:
            group_name: Name of the group
            
        Returns:
            Dictionary with metric characteristics analysis
        """
        if group_name not in self.analyzer.group_results:
            return {}
        
        group_result = self.analyzer.group_results[group_name]
        results = group_result['results']
        
        if len(results) == 0:
            return {}
        
        metric_names = ['RMSE', 'DTW', 'Fréchet', 'ERP', 'DDTW', 'TWED', 'Sobolev']
        characteristics = {}
        
        for metric_name in metric_names:
            values = []
            for ep_result in results.values():
                value = ep_result['metrics'].get(metric_name)
                if value is not None and not (np.isnan(value) or np.isinf(value)):
                    values.append(value)
            
            if len(values) > 0:
                # Calculate statistics
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / mean_val if mean_val > 0 else 0  # Coefficient of variation
                
                # Episode trend analysis
                episode_numbers = []
                metric_values = []
                for ep_name, ep_result in results.items():
                    ep_num = ep_result['episode_number']
                    value = ep_result['metrics'].get(metric_name)
                    if value is not None and not (np.isnan(value) or np.isinf(value)):
                        episode_numbers.append(ep_num)
                        metric_values.append(value)
                
                # Calculate trend (slope)
                trend = 0
                if len(episode_numbers) > 1:
                    sorted_data = sorted(zip(episode_numbers, metric_values))
                    ep_nums, vals = zip(*sorted_data)
                    # Linear regression slope
                    if len(ep_nums) > 1:
                        trend = np.polyfit(ep_nums, vals, 1)[0]
                
                characteristics[metric_name] = {
                    'mean': mean_val,
                    'std': std_val,
                    'cv': cv,  # Coefficient of variation (stability measure)
                    'min': np.min(values),
                    'max': np.max(values),
                    'trend': trend,  # Positive = increases with episode, Negative = decreases
                    'stability': 'High' if cv < 0.1 else 'Medium' if cv < 0.3 else 'Low'
                }
        
        return characteristics
    
    def generate_group_report(self, group_name: str, output_dir: Path):
        """
        Generate detailed analysis report for a group.
        
        Args:
            group_name: Name of the group
            output_dir: Output directory for the group
        """
        if group_name not in self.analyzer.group_results:
            return
        
        group_result = self.analyzer.group_results[group_name]
        results = group_result['results']
        gt_episode = group_result['gt_episode']
        
        if len(results) == 0:
            return
        
        # Analyze characteristics
        characteristics = self.analyze_metric_characteristics(group_name)
        
        # Generate report
        report_path = output_dir / 'analysis_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# {group_name.upper()} Group Analysis Report\n\n")
            f.write(f"**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Ground Truth**: {gt_episode} (Episode 1)\n\n")
            f.write(f"**Total Episodes Analyzed**: {len(results)}\n\n")
            f.write("---\n\n")
            
            # 1. Executive Summary
            f.write("## 1. Executive Summary\n\n")
            f.write("This report analyzes trajectory similarity metrics across different episodes, ")
            f.write("using Episode 1 as the Ground Truth (GT). ")
            f.write("Ideally, all metrics should remain constant regardless of episode number, ")
            f.write("indicating consistent performance. However, in practice, metrics may vary, ")
            f.write("revealing which aspects of trajectory following are most stable or sensitive.\n\n")
            
            # 2. Metric Characteristics
            f.write("## 2. Metric Characteristics Analysis\n\n")
            
            metric_descriptions = {
                'RMSE': {
                    'sensitive': 'Position errors, time synchronization',
                    'insensitive': 'Time warping, path shape variations',
                    'description': 'Measures point-to-point Euclidean distance assuming perfect time synchronization. Most sensitive to timing misalignment.'
                },
                'DTW': {
                    'sensitive': 'Path shape, overall trajectory pattern',
                    'insensitive': 'Temporary stops, speed variations',
                    'description': 'Allows non-linear time warping to match similar shapes. Can match trajectories even if robot stops temporarily.'
                },
                'Fréchet': {
                    'sensitive': 'Geometric shape, path topology',
                    'insensitive': 'Time, speed, temporary detours',
                    'description': 'Measures geometric similarity independent of time. Focuses on the "shape" of the path, like a dog leash analogy.'
                },
                'ERP': {
                    'sensitive': 'Large gaps, missing segments',
                    'insensitive': 'Small detours, local deviations',
                    'description': 'Handles gaps by matching points with a gap element. May treat small detours as gaps rather than errors.'
                },
                'DDTW': {
                    'sensitive': 'Stops, backtracking, velocity changes',
                    'insensitive': 'Position offsets, baseline shifts',
                    'description': 'Compares derivatives (velocity vectors) instead of positions. Naturally penalizes stops and reverse movements.'
                },
                'TWED': {
                    'sensitive': 'Time penalties, speed changes, stops',
                    'insensitive': 'Position-only errors',
                    'description': 'Adds explicit penalty for time warping. More sensitive to temporal aspects than pure DTW.'
                },
                'Sobolev': {
                    'sensitive': 'Both position and velocity errors',
                    'insensitive': 'Neither (comprehensive metric)',
                    'description': 'Combines position error and velocity error in a weighted sum. Most comprehensive but may be sensitive to both aspects.'
                }
            }
            
            for metric_name in ['RMSE', 'DTW', 'Fréchet', 'ERP', 'DDTW', 'TWED', 'Sobolev']:
                if metric_name not in characteristics:
                    continue
                
                char = characteristics[metric_name]
                desc = metric_descriptions.get(metric_name, {})
                
                f.write(f"### {metric_name}\n\n")
                f.write(f"**Description**: {desc.get('description', 'N/A')}\n\n")
                f.write(f"**Sensitive to**: {desc.get('sensitive', 'N/A')}\n\n")
                f.write(f"**Insensitive to**: {desc.get('insensitive', 'N/A')}\n\n")
                
                f.write("**Statistics**:\n")
                f.write(f"- Mean: {char['mean']:.6f}\n")
                f.write(f"- Std: {char['std']:.6f}\n")
                f.write(f"- Coefficient of Variation: {char['cv']:.4f} ({char['stability']} stability)\n")
                f.write(f"- Range: [{char['min']:.6f}, {char['max']:.6f}]\n")
                f.write(f"- Episode Trend: {char['trend']:+.6f} per episode ")
                if char['trend'] > 0.01:
                    f.write("(increasing)")
                elif char['trend'] < -0.01:
                    f.write("(decreasing)")
                else:
                    f.write("(stable)")
                f.write("\n\n")
            
            # 3. Episode Trend Analysis
            f.write("## 3. Episode Trend Analysis\n\n")
            f.write("The following analysis examines how each metric changes as episode number increases.\n\n")
            f.write("**Ideal Behavior**: All metrics should remain constant (horizontal line) regardless of episode number.\n\n")
            f.write("**Observed Behavior**:\n\n")
            
            # Sort metrics by stability
            sorted_metrics = sorted(
                characteristics.items(),
                key=lambda x: abs(x[1]['trend'])
            )
            
            f.write("### Most Stable Metrics (Lowest Episode Dependency)\n\n")
            for metric_name, char in sorted_metrics[:3]:
                f.write(f"- **{metric_name}**: Trend = {char['trend']:+.6f}, CV = {char['cv']:.4f}\n")
            f.write("\n")
            
            f.write("### Least Stable Metrics (Highest Episode Dependency)\n\n")
            for metric_name, char in sorted_metrics[-3:]:
                f.write(f"- **{metric_name}**: Trend = {char['trend']:+.6f}, CV = {char['cv']:.4f}\n")
            f.write("\n")
            
            # 4. Insights
            f.write("## 4. Key Insights\n\n")
            
            # Find most and least stable
            most_stable = min(characteristics.items(), key=lambda x: abs(x[1]['trend']))
            least_stable = max(characteristics.items(), key=lambda x: abs(x[1]['trend']))
            
            f.write(f"### Stability Analysis\n\n")
            f.write(f"- **Most Stable Metric**: {most_stable[0]} (trend: {most_stable[1]['trend']:+.6f})\n")
            f.write(f"  - This metric shows the least variation across episodes, suggesting it measures ")
            f.write(f"aspects that are consistent regardless of episode number.\n\n")
            
            f.write(f"- **Least Stable Metric**: {least_stable[0]} (trend: {least_stable[1]['trend']:+.6f})\n")
            f.write(f"  - This metric shows the most variation, indicating it is sensitive to aspects ")
            f.write(f"that change between episodes.\n\n")
            
            # Episode dependency analysis
            f.write("### Episode Dependency\n\n")
            increasing_metrics = [m for m, c in characteristics.items() if c['trend'] > 0.01]
            decreasing_metrics = [m for m, c in characteristics.items() if c['trend'] < -0.01]
            stable_metrics = [m for m, c in characteristics.items() if -0.01 <= c['trend'] <= 0.01]
            
            if increasing_metrics:
                f.write(f"- **Increasing with Episode**: {', '.join(increasing_metrics)}\n")
                f.write(f"  - These metrics suggest that later episodes deviate more from GT.\n\n")
            
            if decreasing_metrics:
                f.write(f"- **Decreasing with Episode**: {', '.join(decreasing_metrics)}\n")
                f.write(f"  - These metrics suggest that later episodes are closer to GT.\n\n")
            
            if stable_metrics:
                f.write(f"- **Stable across Episodes**: {', '.join(stable_metrics)}\n")
                f.write(f"  - These metrics remain relatively constant, indicating consistent performance.\n\n")
            
            # 5. Recommendations
            f.write("## 5. Recommendations\n\n")
            
            f.write("### For Trajectory Evaluation\n\n")
            f.write("Based on the analysis:\n\n")
            f.write("1. **For Overall Similarity**: Use metrics that show high stability (low CV) ")
            f.write("and low episode dependency.\n\n")
            f.write("2. **For Detecting Specific Issues**:\n")
            f.write("   - Use **DDTW** or **TWED** to detect stops and backtracking\n")
            f.write("   - Use **Fréchet** to evaluate geometric path similarity\n")
            f.write("   - Use **Sobolev** for comprehensive position+velocity evaluation\n\n")
            f.write("3. **For Time-Sensitive Analysis**: Use **RMSE** or **DTW** to evaluate ")
            f.write("temporal alignment.\n\n")
            
            f.write("### For Episode Comparison\n\n")
            if stable_metrics:
                f.write(f"Metrics {', '.join(stable_metrics)} show consistent values across episodes, ")
                f.write("suggesting that the evaluated aspect is stable regardless of episode number.\n\n")
            
            if increasing_metrics:
                f.write(f"Metrics {', '.join(increasing_metrics)} increase with episode number, ")
                f.write("which may indicate:\n")
                f.write("- Performance degradation in later episodes\n")
                f.write("- Accumulated errors\n")
                f.write("- Different experimental conditions\n\n")
        
        print(f"Generated report: {report_path}")
    
    def generate_all_reports(self, output_dir: Path):
        """
        Generate reports for all groups.
        
        Args:
            output_dir: Base output directory
        """
        for group_name in self.analyzer.group_results.keys():
            group_output_dir = output_dir / group_name
            self.generate_group_report(group_name, group_output_dir)
