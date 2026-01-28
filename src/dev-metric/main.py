#!/usr/bin/env python3
"""
Main script for trajectory metric analysis.

Usage:
    python main.py [--logs-dir LOGS_DIR] [--reference REFERENCE] [--output OUTPUT_DIR]
"""

import argparse
from pathlib import Path
from datetime import datetime
import sys

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from analyzer import TrajectoryAnalyzer
from visualizer import TrajectoryVisualizer


def main():
    parser = argparse.ArgumentParser(
        description='Trajectory metric analysis for robot paths',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--logs-dir',
        type=str,
        default='../logs_good',
        help='Path to logs_good directory relative to src/ (default: ../logs_good)'
    )
    
    parser.add_argument(
        '--reference',
        type=str,
        default='Episode_1_1',
        help='Reference episode name (default: Episode_1_1)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for results (default: analysis_YYYYMMDD_HHMMSS)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    # If logs_dir is relative, resolve relative to src/ directory
    if Path(args.logs_dir).is_absolute():
        logs_dir = Path(args.logs_dir)
    else:
        # __file__ is at src/dev-metric/main.py, so parent.parent is src/
        base_dir = Path(__file__).parent.parent  # src/
        # Handle both '../logs_good' and 'logs_good' cases
        if args.logs_dir.startswith('../'):
            # Remove '../' and use base_dir directly
            rel_path = args.logs_dir[3:]  # Remove '../'
            logs_dir = base_dir / rel_path
        else:
            logs_dir = base_dir / args.logs_dir
    
    # Convert to absolute path
    logs_dir = logs_dir.resolve()
    
    if not logs_dir.exists():
        print(f"Error: Logs directory not found: {logs_dir}")
        print(f"  Base directory: {base_dir}")
        print(f"  Resolved path: {logs_dir}")
        print(f"  Please check the path and try again.")
        return 1
    
    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = base_dir / 'dev-metric' / f'analysis_{timestamp}'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Trajectory Metric Analysis")
    print("=" * 60)
    print(f"Logs directory: {logs_dir}")
    print(f"Reference episode: {args.reference}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = TrajectoryAnalyzer(logs_dir, reference_name=args.reference)
    
    # Load data
    try:
        analyzer.load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1
    
    # Analyze all episodes
    try:
        analyzer.analyze_all_episodes()
    except Exception as e:
        print(f"Error analyzing episodes: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Save results
    try:
        analyzer.save_results(output_dir)
    except Exception as e:
        print(f"Error saving results: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Generate visualizations
    try:
        visualizer = TrajectoryVisualizer(analyzer)
        visualizer.generate_all_visualizations(output_dir)
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    print(f"\nAnalyzed {len(analyzer.results)} episodes")
    
    # Print statistics
    stats = analyzer.get_metric_statistics()
    print("\nMetric Statistics:")
    print("-" * 60)
    for metric_name, stat in stats.items():
        if stat['count'] > 0:
            print(f"{metric_name:12s}: Mean={stat['mean']:10.4f}, Std={stat['std']:10.4f}, "
                  f"Min={stat['min']:10.4f}, Max={stat['max']:10.4f}, Count={stat['count']}")
    
    print("\n" + "=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
