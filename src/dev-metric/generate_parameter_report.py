#!/usr/bin/env python3
"""
Generate parameter sensitivity report from sweep results.

Usage:
    python generate_parameter_report.py [--sweep-results PATH] [--output PATH]
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from parameter_sensitivity_report import ParameterSensitivityReportGenerator


def main():
    parser = argparse.ArgumentParser(
        description='Generate parameter sensitivity analysis report',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--sweep-results',
        type=str,
        default='analysis_reports/parameter_sweep/parameter_sweep_results.json',
        help='Path to parameter_sweep_results.json'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='analysis_reports/parameter_sweep/파라미터_민감도_분석_보고서.md',
        help='Output path for the report'
    )
    
    args = parser.parse_args()
    
    sweep_results_path = Path(args.sweep_results)
    output_path = Path(args.output)
    
    if not sweep_results_path.exists():
        print(f"Error: Sweep results file not found: {sweep_results_path}")
        return 1
    
    print("=" * 60)
    print("Generating Parameter Sensitivity Report")
    print("=" * 60)
    print(f"Sweep results: {sweep_results_path}")
    print(f"Output: {output_path}")
    print("=" * 60)
    
    # Generate report
    try:
        generator = ParameterSensitivityReportGenerator(sweep_results_path)
        generator.generate_report(output_path)
    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "=" * 60)
    print("Report generation complete!")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
