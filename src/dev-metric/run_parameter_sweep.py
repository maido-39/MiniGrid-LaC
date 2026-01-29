#!/usr/bin/env python3
"""
Run parameter sweep analysis for tunable metrics.

Usage:
    python run_parameter_sweep.py [--logs-dir LOGS_DIR] [--results-dir RESULTS_DIR] [--output OUTPUT_DIR]
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from parameter_sweep_analyzer import ParameterSweepAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description='Parameter sweep analysis for tunable metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--logs-dir',
        type=str,
        default='../logs_good',
        help='Path to logs_good directory relative to src/ (default: ../logs_good)'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='analysis_reports',
        help='Path to analysis_reports directory (default: analysis_reports)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='analysis_reports/parameter_sweep',
        help='Output directory for parameter sweep results (default: analysis_reports/parameter_sweep)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    base_dir = Path(__file__).parent.parent  # src/
    logs_dir = base_dir / args.logs_dir
    
    if args.logs_dir.startswith('../'):
        rel_path = args.logs_dir[3:]
        logs_dir = base_dir / rel_path
    
    logs_dir = logs_dir.resolve()
    
    if not logs_dir.exists():
        print(f"Error: Logs directory not found: {logs_dir}")
        return 1
    
    results_dir = Path(args.results_dir).resolve()
    output_dir = Path(args.output).resolve()
    
    print("=" * 60)
    print("Parameter Sweep Analysis")
    print("=" * 60)
    print(f"Logs directory: {logs_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = ParameterSweepAnalyzer(logs_dir, results_dir)
    
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
        all_results = analyzer.analyze_all_groups()
    except Exception as e:
        print(f"Error analyzing groups: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Save results and generate visualizations
    try:
        analyzer.save_results(all_results, output_dir)
    except Exception as e:
        print(f"Error saving results: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("Parameter Sweep Analysis Complete!")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    
    for group_name, group_result in all_results.items():
        print(f"\n{group_name.upper()}:")
        print(f"  Episodes Analyzed: {len(group_result['episodes'])}")
        print(f"  Output: {output_dir / group_name / 'parameter_sweep'}")
    
    print("\n" + "=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
