#!/usr/bin/env python3
"""
Main script for group-wise trajectory metric analysis.

Usage:
    python main_group_analysis.py [--logs-dir LOGS_DIR] [--output OUTPUT_DIR]
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from group_analyzer import GroupWiseAnalyzer
from group_visualizer import GroupVisualizer
from report_generator import ReportGenerator
from methodology_visualizer import MethodologyVisualizer


def main():
    parser = argparse.ArgumentParser(
        description='Group-wise trajectory metric analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--logs-dir',
        type=str,
        default='../logs_good',
        help='Path to logs_good directory relative to src/ (default: ../logs_good)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for results (default: group_analysis_YYYYMMDD_HHMMSS)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    base_dir = Path(__file__).parent.parent  # src/
    logs_dir = base_dir / args.logs_dir
    
    # Handle '../logs_good' case
    if args.logs_dir.startswith('../'):
        rel_path = args.logs_dir[3:]  # Remove '../'
        logs_dir = base_dir / rel_path
    
    logs_dir = logs_dir.resolve()
    
    if not logs_dir.exists():
        print(f"Error: Logs directory not found: {logs_dir}")
        print(f"  Base directory: {base_dir}")
        print(f"  Resolved path: {logs_dir}")
        return 1
    
    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent / f'group_analysis_{timestamp}'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Group-wise Trajectory Metric Analysis")
    print("=" * 60)
    print(f"Logs directory: {logs_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = GroupWiseAnalyzer(logs_dir)
    
    # Load data
    try:
        analyzer.load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Analyze all groups
    try:
        analyzer.analyze_all_groups()
    except Exception as e:
        print(f"Error analyzing groups: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Save results
    try:
        analyzer.save_all_results(output_dir)
    except Exception as e:
        print(f"Error saving results: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Generate visualizations
    try:
        visualizer = GroupVisualizer(analyzer)
        visualizer.generate_all_visualizations(output_dir)
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Generate methodology visualizations
    try:
        methodology_viz = MethodologyVisualizer()
        
        for group_name, group_result in analyzer.group_results.items():
            if len(group_result['results']) == 0:
                continue
            
            # Use first episode as sample for methodology visualization
            first_ep_name = list(group_result['results'].keys())[0]
            first_ep_result = group_result['results'][first_ep_name]
            
            # Get trajectories
            gt_traj = group_result['gt_trajectory']
            robot_traj = analyzer.all_episodes[first_ep_name]['trajectory']
            
            group_output_dir = output_dir / group_name
            methodology_viz.generate_all_methodology_visualizations(
                group_name, gt_traj, robot_traj, group_output_dir
            )
    except Exception as e:
        print(f"Error generating methodology visualizations: {e}")
        import traceback
        traceback.print_exc()
        # Don't fail if methodology viz fails
    
    # Generate reports
    try:
        report_gen = ReportGenerator(analyzer)
        report_gen.generate_all_reports(output_dir)
    except Exception as e:
        print(f"Error generating reports: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    
    for group_name, group_result in analyzer.group_results.items():
        print(f"\n{group_name.upper()}:")
        print(f"  GT Episode: {group_result['gt_episode']}")
        print(f"  Episodes Analyzed: {len(group_result['results'])}")
        print(f"  Output: {output_dir / group_name}")
    
    print("\n" + "=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
