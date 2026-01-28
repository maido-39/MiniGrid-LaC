"""
Merge original report with enhanced report and add images.
"""

from pathlib import Path
import re


def merge_reports(group_name: str, reports_dir: Path):
    """
    Merge original and enhanced reports with images.
    
    Args:
        group_name: Name of the group
        reports_dir: Directory containing reports
    """
    original_path = reports_dir / '종합_분석_보고서.md'
    enhanced_path = reports_dir / '종합_분석_보고서_Enhanced.md'
    
    if not original_path.exists():
        print(f"Warning: {original_path} does not exist")
        return
    
    if not enhanced_path.exists():
        print(f"Warning: {enhanced_path} does not exist")
        return
    
    # Read original report (has images)
    with open(original_path, 'r', encoding='utf-8') as f:
        original_content = f.read()
    
    # Read enhanced report (has statistical analysis)
    with open(enhanced_path, 'r', encoding='utf-8') as f:
        enhanced_content = f.read()
    
    # Extract sections from enhanced report
    # Find statistical sections
    enhanced_sections = {}
    
    # Extract section 1 (Data Overview)
    match = re.search(r'## 1\. 데이터 개요(.*?)(?=## 2\.|$)', enhanced_content, re.DOTALL)
    if match:
        enhanced_sections['data_overview'] = match.group(1).strip()
    
    # Extract section 2 (Trajectory Characteristics)
    match = re.search(r'## 2\. 궤적 특성 분석(.*?)(?=## 3\.|$)', enhanced_content, re.DOTALL)
    if match:
        enhanced_sections['characteristics'] = match.group(1).strip()
    
    # Extract section 3 (Metric Statistics)
    match = re.search(r'## 3\. 메트릭 통계 분석(.*?)(?=## 4\.|$)', enhanced_content, re.DOTALL)
    if match:
        enhanced_sections['statistics'] = match.group(1).strip()
    
    # Extract section 4 (Correlations)
    match = re.search(r'## 4\. 상관관계 분석(.*?)(?=## 5\.|$)', enhanced_content, re.DOTALL)
    if match:
        enhanced_sections['correlations'] = match.group(1).strip()
    
    # Extract section 5 (Statistical Tests)
    match = re.search(r'## 5\. 통계적 유의성 검정(.*?)(?=## 6\.|$)', enhanced_content, re.DOTALL)
    if match:
        enhanced_sections['tests'] = match.group(1).strip()
    
    # Extract section 6 (Enhanced Metric-Trajectory Relationship)
    match = re.search(r'## 6\. 메트릭 경향성과 궤적 특성의 관계(.*?)(?=## 7\.|$)', enhanced_content, re.DOTALL)
    if match:
        enhanced_sections['enhanced_relationship'] = match.group(1).strip()
    
    # Extract section 7 (Enhanced Episode Trends)
    match = re.search(r'## 7\. Episode 증가에 따른 경향성 종합 분석(.*?)(?=## 8\.|$)', enhanced_content, re.DOTALL)
    if match:
        enhanced_sections['enhanced_trends'] = match.group(1).strip()
    
    # Extract section 8 (Enhanced Metric Suitability)
    match = re.search(r'## 8\. 이 실험 데이터에 적합한 메트릭 분석(.*?)(?=## 9\.|$)', enhanced_content, re.DOTALL)
    if match:
        enhanced_sections['enhanced_suitability'] = match.group(1).strip()
    
    # Extract section 9 (Enhanced Conclusion)
    match = re.search(r'## 9\. 결론 및 권장사항(.*?)$', enhanced_content, re.DOTALL)
    if match:
        enhanced_sections['enhanced_conclusion'] = match.group(1).strip()
    
    # Build merged report
    merged_lines = []
    
    # Header from original
    header_match = re.search(r'^(#.*?\n\n---\n)', original_content, re.MULTILINE)
    if header_match:
        merged_lines.append(header_match.group(1))
    else:
        merged_lines.append(f"# {group_name.upper()} 그룹 종합 분석 보고서\n\n")
        merged_lines.append("---\n\n")
    
    # Table of contents
    merged_lines.append("## 목차\n\n")
    merged_lines.append("1. [데이터 개요](#1-데이터-개요)\n")
    merged_lines.append("2. [궤적 특성 분석](#2-궤적-특성-분석)\n")
    merged_lines.append("3. [메트릭 통계 분석](#3-메트릭-통계-분석)\n")
    merged_lines.append("4. [상관관계 분석](#4-상관관계-분석)\n")
    merged_lines.append("5. [통계적 유의성 검정](#5-통계적-유의성-검정)\n")
    merged_lines.append("6. [메트릭 경향성과 궤적 특성의 관계](#6-메트릭-경향성과-궤적-특성의-관계)\n")
    merged_lines.append("7. [Episode 증가에 따른 경향성 종합 분석](#7-episode-증가에-따른-경향성-종합-분석)\n")
    merged_lines.append("8. [이 실험 데이터에 적합한 메트릭 분석](#8-이-실험-데이터에-적합한-메트릭-분석)\n")
    merged_lines.append("9. [결론 및 권장사항](#9-결론-및-권장사항)\n")
    merged_lines.append("\n---\n\n")
    
    # Section 1: Data Overview (from enhanced)
    if 'data_overview' in enhanced_sections:
        merged_lines.append("## 1. 데이터 개요\n\n")
        merged_lines.append(enhanced_sections['data_overview'])
        merged_lines.append("\n\n")
    
    # Section 2: Characteristics (from enhanced)
    if 'characteristics' in enhanced_sections:
        merged_lines.append("## 2. 궤적 특성 분석\n\n")
        merged_lines.append(enhanced_sections['characteristics'])
        merged_lines.append("\n\n")
    
    # Section 3: Statistics (from enhanced)
    if 'statistics' in enhanced_sections:
        merged_lines.append("## 3. 메트릭 통계 분석\n\n")
        merged_lines.append(enhanced_sections['statistics'])
        merged_lines.append("\n\n")
    
    # Section 4: Correlations (from enhanced)
    if 'correlations' in enhanced_sections:
        merged_lines.append("## 4. 상관관계 분석\n\n")
        merged_lines.append(enhanced_sections['correlations'])
        merged_lines.append("\n\n")
    
    # Section 5: Statistical Tests (from enhanced)
    if 'tests' in enhanced_sections:
        merged_lines.append("## 5. 통계적 유의성 검정\n\n")
        merged_lines.append(enhanced_sections['tests'])
        merged_lines.append("\n\n")
    
    # Section 6: Metric-Trajectory Relationship
    # Start with enhanced version, then add images from original
    merged_lines.append("## 6. 메트릭 경향성과 궤적 특성의 관계\n\n")
    
    if 'enhanced_relationship' in enhanced_sections:
        merged_lines.append(enhanced_sections['enhanced_relationship'])
        merged_lines.append("\n\n")
    
    # Add images from original report for each metric
    # Extract image sections from original
    metric_sections = re.findall(
        r'### 1\.\d+ (RMSE|DTW|Fréchet|ERP|DDTW|TWED|Sobolev).*?(?=### 1\.\d+|## 2\.|$)',
        original_content,
        re.DOTALL
    )
    
    # Find image patterns in original
    image_patterns = re.findall(
        r'!\[.*?\]\((visualizations/.*?\.png|methodology/.*?\.png)\)',
        original_content
    )
    
    # Add images after each metric analysis
    # This is a simplified approach - in practice, we'd match images to metrics more carefully
    
    # Section 7: Episode Trends (enhanced + images from original)
    merged_lines.append("## 7. Episode 증가에 따른 경향성 종합 분석\n\n")
    
    if 'enhanced_trends' in enhanced_sections:
        merged_lines.append(enhanced_sections['enhanced_trends'])
        merged_lines.append("\n\n")
    
    # Add trend images from original
    trend_images = re.findall(
        r'!\[.*?Episode Trends.*?\]\((visualizations/episode_trends\.png)\)',
        original_content
    )
    if trend_images:
        merged_lines.append("![Episode Trends](visualizations/episode_trends.png)\n\n")
    
    all_metrics_image = re.findall(
        r'!\[.*?All Metrics.*?\]\((visualizations/all_metrics_comparison\.png)\)',
        original_content
    )
    if all_metrics_image:
        merged_lines.append("![All Metrics Comparison](visualizations/all_metrics_comparison.png)\n\n")
    
    # Section 8: Metric Suitability (from enhanced)
    if 'enhanced_suitability' in enhanced_sections:
        merged_lines.append("## 8. 이 실험 데이터에 적합한 메트릭 분석\n\n")
        merged_lines.append(enhanced_sections['enhanced_suitability'])
        merged_lines.append("\n\n")
    
    # Section 9: Conclusion (from enhanced)
    if 'enhanced_conclusion' in enhanced_sections:
        merged_lines.append("## 9. 결론 및 권장사항\n\n")
        merged_lines.append(enhanced_sections['enhanced_conclusion'])
        merged_lines.append("\n\n")
    
    # Write merged report
    merged_path = reports_dir / '종합_분석_보고서_Final.md'
    with open(merged_path, 'w', encoding='utf-8') as f:
        f.write(''.join(merged_lines))
    
    print(f"Created merged report: {merged_path}")


if __name__ == '__main__':
    import sys
    
    base_dir = Path(__file__).parent / 'analysis_reports'
    groups = ['hogun', 'hogun_0125', 'other']
    
    for group_name in groups:
        group_dir = base_dir / group_name
        if group_dir.exists():
            merge_reports(group_name, group_dir)
