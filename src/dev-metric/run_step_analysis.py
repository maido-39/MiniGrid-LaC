"""
Run step-by-step analysis for other group.
"""

import sys
from pathlib import Path
import json
import argparse

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from step_analyzer import StepAnalyzer
from step_visualizer import StepVisualizer
from data_loader import load_all_episodes, get_reference_path


def main():
    parser = argparse.ArgumentParser(description='Run step-by-step analysis')
    parser.add_argument(
        '--logs-dir',
        type=str,
        default='../logs_good',
        help='Path to logs_good directory'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='group_analysis_20260128_210325',
        help='Path to group analysis results directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='analysis_reports/other/step_analysis',
        help='Output directory for step analysis'
    )
    parser.add_argument(
        '--group',
        type=str,
        default='other',
        help='Group to analyze (default: other)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    base_dir = Path(__file__).parent.parent  # src/
    if Path(args.logs_dir).is_absolute():
        logs_dir = Path(args.logs_dir)
    else:
        if args.logs_dir.startswith('../'):
            rel_path = args.logs_dir[3:]
            logs_dir = base_dir / rel_path
        else:
            logs_dir = base_dir / args.logs_dir
    
    logs_dir = logs_dir.resolve()
    
    results_dir = Path(__file__).parent / args.results_dir
    output_dir = Path(__file__).parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Logs directory: {logs_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load data
    print(f"\nLoading episodes for group: {args.group}")
    all_episodes = load_all_episodes(logs_dir)
    
    # Filter by group
    group_episodes = {
        name: data for name, data in all_episodes.items()
        if data['group'] == args.group
    }
    
    if len(group_episodes) == 0:
        print(f"No episodes found for group: {args.group}")
        return
    
    # Load GT from results
    results_json_path = results_dir / args.group / 'detailed_results.json'
    if not results_json_path.exists():
        print(f"Warning: {results_json_path} does not exist")
        return
    
    with open(results_json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    gt_episode_name = results.get('gt_episode', '')
    gt_trajectory = None
    
    for ep_name, ep_data in group_episodes.items():
        if ep_name == gt_episode_name or gt_episode_name in ep_name:
            gt_trajectory = ep_data['trajectory']
            break
    
    if gt_trajectory is None:
        print(f"Warning: GT trajectory not found for {gt_episode_name}")
        return
    
    print(f"GT episode: {gt_episode_name}")
    print(f"GT trajectory length: {len(gt_trajectory)}")
    print(f"Analyzing {len(group_episodes)} episodes")
    
    # Initialize analyzers
    step_analyzer = StepAnalyzer()
    visualizer = StepVisualizer(output_dir)
    
    # Analyze each episode
    all_stepwise_metrics = {}
    all_stepwise_features = {}
    all_sensitivities = {}
    
    for ep_name, ep_data in group_episodes.items():
        if ep_name == gt_episode_name:
            continue  # Skip GT
        
        print(f"\nAnalyzing: {ep_name}")
        trajectory = ep_data['trajectory']
        
        # Compute stepwise metrics
        stepwise_metrics = step_analyzer.compute_stepwise_metrics(
            trajectory, gt_trajectory
        )
        all_stepwise_metrics[ep_name] = stepwise_metrics
        
        # Compute stepwise features
        stepwise_features = step_analyzer.compute_trajectory_features_stepwise(
            trajectory, gt_trajectory
        )
        all_stepwise_features[ep_name] = stepwise_features
        
        # Compute sensitivity
        sensitivity = step_analyzer.analyze_metric_sensitivity(
            stepwise_metrics, stepwise_features
        )
        all_sensitivities[ep_name] = sensitivity
        
        # Create visualizations
        vis_path = output_dir / f'{ep_name}_stepwise_analysis.png'
        visualizer.plot_trajectory_comparison_with_metrics(
            trajectory, gt_trajectory, stepwise_metrics, ep_name, vis_path
        )
    
    # Aggregate sensitivity analysis
    print("\nComputing aggregate sensitivity...")
    aggregated_sensitivity = {}
    for metric_name in ['RMSE', 'DTW', 'Fréchet', 'ERP', 'DDTW', 'TWED', 'Sobolev']:
        aggregated_sensitivity[metric_name] = {}
        for feature_name in ['position_error', 'velocity_error', 'direction_error']:
            correlations = []
            for ep_name, sensitivity in all_sensitivities.items():
                if (metric_name in sensitivity and 
                    feature_name in sensitivity[metric_name]):
                    correlations.append(sensitivity[metric_name][feature_name])
            if correlations:
                aggregated_sensitivity[metric_name][feature_name] = {
                    'mean': float(np.mean(correlations)),
                    'std': float(np.std(correlations)),
                    'values': [float(c) for c in correlations]
                }
    
    # Create aggregate visualizations
    visualizer.plot_metric_sensitivity_heatmap(
        {metric: {feat: data['mean'] 
                 for feat, data in features.items()}
         for metric, features in aggregated_sensitivity.items()},
        output_dir / 'metric_sensitivity_heatmap.png'
    )
    
    # Save results
    results_json = {
        'gt_episode': gt_episode_name,
        'episodes': {}
    }
    
    for ep_name in all_stepwise_metrics.keys():
        results_json['episodes'][ep_name] = {
            'stepwise_metrics': all_stepwise_metrics[ep_name],
            'stepwise_features': {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in all_stepwise_features[ep_name].items()
            },
            'sensitivity': all_sensitivities[ep_name]
        }
    
    results_json['aggregated_sensitivity'] = aggregated_sensitivity
    
    results_path = output_dir / 'stepwise_analysis_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved results to {results_path}")
    print(f"All visualizations saved to {output_dir}")


if __name__ == '__main__':
    import numpy as np
    main()
