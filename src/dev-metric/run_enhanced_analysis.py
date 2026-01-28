"""
Run enhanced statistical analysis and generate comprehensive reports.
"""

import sys
from pathlib import Path
import argparse

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from statistical_analyzer import StatisticalAnalyzer
from enhanced_report_generator import EnhancedReportGenerator


def main():
    parser = argparse.ArgumentParser(description='Run enhanced statistical analysis')
    parser.add_argument(
        '--logs-dir',
        type=str,
        default='../logs_good',
        help='Path to logs_good directory (relative to src/)'
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
        default='analysis_reports',
        help='Output directory for enhanced reports'
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
    
    print(f"Logs directory: {logs_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    
    # Initialize analyzer
    analyzer = StatisticalAnalyzer(logs_dir)
    report_generator = EnhancedReportGenerator()
    
    # Groups to analyze
    groups = ['hogun', 'hogun_0125', 'other']
    
    for group_name in groups:
        print(f"\n{'='*60}")
        print(f"Processing group: {group_name}")
        print(f"{'='*60}")
        
        group_results_dir = results_dir / group_name
        if not group_results_dir.exists():
            print(f"Warning: {group_results_dir} does not exist, skipping...")
            continue
        
        # Perform statistical analysis
        print("Performing statistical analysis...")
        analysis_results = analyzer.analyze_group(group_results_dir)
        
        if not analysis_results:
            print(f"Warning: No analysis results for {group_name}, skipping...")
            continue
        
        # Save statistical analysis
        output_group_dir = output_dir / group_name
        output_group_dir.mkdir(parents=True, exist_ok=True)
        
        stats_path = output_group_dir / 'statistical_analysis.json'
        analyzer.save_analysis(analysis_results, stats_path)
        
        # Generate enhanced report
        print("Generating enhanced report...")
        detailed_results_path = group_results_dir / 'detailed_results.json'
        
        enhanced_report = report_generator.generate_enhanced_report(
            group_name,
            detailed_results_path,
            stats_path
        )
        
        # Save enhanced report
        enhanced_report_path = output_group_dir / '종합_분석_보고서_Enhanced.md'
        with open(enhanced_report_path, 'w', encoding='utf-8') as f:
            f.write(enhanced_report)
        
        print(f"Saved enhanced report to {enhanced_report_path}")
    
    print("\n" + "="*60)
    print("Enhanced analysis complete!")
    print("="*60)


if __name__ == '__main__':
    main()
