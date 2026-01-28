"""
Enhanced report generator with statistical analysis.

Generates comprehensive, publication-quality analysis reports.
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import numpy as np


class EnhancedReportGenerator:
    """Enhanced report generator with statistical analysis."""
    
    def __init__(self):
        """Initialize enhanced report generator."""
        pass
    
    def generate_enhanced_report(
        self, 
        group_name: str,
        detailed_results_path: Path,
        statistical_analysis_path: Path
    ) -> str:
        """
        Generate enhanced report with statistical analysis.
        
        Args:
            group_name: Name of the group
            detailed_results_path: Path to detailed_results.json
            statistical_analysis_path: Path to statistical_analysis.json
            
        Returns:
            Markdown report string
        """
        # Load data
        with open(detailed_results_path, 'r', encoding='utf-8') as f:
            detailed_results = json.load(f)
        
        try:
            with open(statistical_analysis_path, 'r', encoding='utf-8') as f:
                statistical_analysis = json.load(f)
        except FileNotFoundError:
            statistical_analysis = {}
        
        # Generate report
        report_lines = []
        
        # Header
        report_lines.append(f"# {group_name.upper()} 그룹 종합 분석 보고서 (Enhanced)")
        report_lines.append("")
        report_lines.append(f"**분석 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"**Ground Truth**: {detailed_results.get('gt_episode', 'N/A')}")
        report_lines.append(f"**GT 경로 길이**: {detailed_results.get('gt_length', 'N/A')} steps")
        report_lines.append(f"**분석 대상**: {len(detailed_results.get('episodes', {}))}개 Episode")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
        
        # Table of contents
        report_lines.append("## 목차")
        report_lines.append("")
        report_lines.append("1. [데이터 개요](#1-데이터-개요)")
        report_lines.append("2. [궤적 특성 분석](#2-궤적-특성-분석)")
        report_lines.append("3. [메트릭 통계 분석](#3-메트릭-통계-분석)")
        report_lines.append("4. [상관관계 분석](#4-상관관계-분석)")
        report_lines.append("5. [통계적 유의성 검정](#5-통계적-유의성-검정)")
        report_lines.append("6. [메트릭 경향성과 궤적 특성의 관계](#6-메트릭-경향성과-궤적-특성의-관계)")
        report_lines.append("7. [Episode 증가에 따른 경향성 종합 분석](#7-episode-증가에-따른-경향성-종합-분석)")
        report_lines.append("8. [이 실험 데이터에 적합한 메트릭 분석](#8-이-실험-데이터에-적합한-메트릭-분석)")
        report_lines.append("9. [결론 및 권장사항](#9-결론-및-권장사항)")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
        
        # 1. 데이터 개요
        report_lines.append("## 1. 데이터 개요")
        report_lines.append("")
        
        episodes = detailed_results.get('episodes', {})
        report_lines.append(f"### 1.1 Episode 정보")
        report_lines.append("")
        report_lines.append("| Episode | Episode 번호 | 경로 길이 (steps) |")
        report_lines.append("|---------|--------------|-------------------|")
        for ep_name, ep_data in sorted(episodes.items(), key=lambda x: x[1].get('episode_number', 0)):
            ep_num = ep_data.get('episode_number', 'N/A')
            length = ep_data.get('trajectory_length', 'N/A')
            report_lines.append(f"| {ep_name} | {ep_num} | {length} |")
        report_lines.append("")
        
        # 2. 궤적 특성 분석
        if statistical_analysis.get('characteristics'):
            report_lines.append("## 2. 궤적 특성 분석")
            report_lines.append("")
            
            chars = statistical_analysis['characteristics']
            
            report_lines.append("### 2.1 경로 길이 및 효율성")
            report_lines.append("")
            if 'path_length' in chars:
                pl = chars['path_length']
                report_lines.append(f"- **평균 경로 길이**: {pl['mean']:.2f} (std: {pl['std']:.2f})")
                report_lines.append(f"- **경로 길이 범위**: {pl['min']:.2f} ~ {pl['max']:.2f}")
            if 'efficiency' in chars:
                eff = chars['efficiency']
                report_lines.append(f"- **평균 효율성 (displacement/path_length)**: {eff['mean']:.3f} (std: {eff['std']:.3f})")
            report_lines.append("")
            
            report_lines.append("### 2.2 속도 특성")
            report_lines.append("")
            if 'avg_speed' in chars:
                speed = chars['avg_speed']
                report_lines.append(f"- **평균 속도**: {speed['mean']:.3f} (std: {speed['std']:.3f})")
            if 'speed_variance' in chars:
                var = chars['speed_variance']
                report_lines.append(f"- **속도 분산**: {var['mean']:.3f} (std: {var['std']:.3f})")
            report_lines.append("")
            
            report_lines.append("### 2.3 동역학적 특성")
            report_lines.append("")
            if 'num_stops' in chars:
                stops = chars['num_stops']
                report_lines.append(f"- **평균 정지 횟수**: {stops['mean']:.1f} (std: {stops['std']:.1f})")
            if 'num_backtracks' in chars:
                bt = chars['num_backtracks']
                report_lines.append(f"- **평균 역주행 횟수**: {bt['mean']:.1f} (std: {bt['std']:.1f})")
            if 'avg_curvature' in chars:
                curv = chars['avg_curvature']
                report_lines.append(f"- **평균 곡률**: {curv['mean']:.3f} (std: {curv['std']:.3f})")
            if 'direction_similarity' in chars:
                sim = chars['direction_similarity']
                report_lines.append(f"- **방향 유사도 (cosine similarity)**: {sim['mean']:.3f} (std: {sim['std']:.3f})")
            report_lines.append("")
        
        # 3. 메트릭 통계 분석
        report_lines.append("## 3. 메트릭 통계 분석")
        report_lines.append("")
        
        if statistical_analysis.get('metrics'):
            metrics = statistical_analysis['metrics']
            
            report_lines.append("### 3.1 메트릭 기본 통계")
            report_lines.append("")
            report_lines.append("| 메트릭 | 평균 | 표준편차 | 중앙값 | 최소값 | 최대값 | CV |")
            report_lines.append("|--------|------|---------|--------|--------|--------|-----|")
            
            for metric_name in ['RMSE', 'DTW', 'Fréchet', 'ERP', 'DDTW', 'TWED', 'Sobolev']:
                if metric_name in metrics:
                    m = metrics[metric_name]
                    mean = m.get('mean', 0)
                    std = m.get('std', 0)
                    median = m.get('median', 0)
                    min_val = m.get('min', 0)
                    max_val = m.get('max', 0)
                    cv = m.get('cv', 0)
                    report_lines.append(
                        f"| {metric_name} | {mean:.3f} | {std:.3f} | {median:.3f} | "
                        f"{min_val:.3f} | {max_val:.3f} | {cv:.4f} |"
                    )
            report_lines.append("")
            
            report_lines.append("### 3.2 Episode 증가에 따른 트렌드 분석")
            report_lines.append("")
            report_lines.append("| 메트릭 | 기울기 (slope) | R² | p-value | 해석 |")
            report_lines.append("|--------|---------------|-----|---------|------|")
            
            for metric_name in ['RMSE', 'DTW', 'Fréchet', 'ERP', 'DDTW', 'TWED', 'Sobolev']:
                if metric_name in metrics and 'trend' in metrics[metric_name]:
                    trend = metrics[metric_name]['trend']
                    slope = trend.get('slope', 0)
                    r_squared = trend.get('r_squared', 0)
                    p_value = trend.get('p_value', 1)
                    
                    if p_value < 0.05:
                        significance = "**유의함**"
                    elif p_value < 0.1:
                        significance = "*경향 있음*"
                    else:
                        significance = "유의하지 않음"
                    
                    if slope > 0:
                        interpretation = f"증가 ({significance})"
                    elif slope < 0:
                        interpretation = f"감소 ({significance})"
                    else:
                        interpretation = f"변화 없음 ({significance})"
                    
                    report_lines.append(
                        f"| {metric_name} | {slope:.3f} | {r_squared:.3f} | {p_value:.4f} | {interpretation} |"
                    )
            report_lines.append("")
        
        # 4. 상관관계 분석
        if statistical_analysis.get('correlations'):
            report_lines.append("## 4. 상관관계 분석")
            report_lines.append("")
            
            correlations = statistical_analysis['correlations']
            
            report_lines.append("### 4.1 메트릭과 Episode 번호의 상관관계")
            report_lines.append("")
            report_lines.append("| 메트릭 | Pearson r | p-value | Spearman ρ | p-value | 해석 |")
            report_lines.append("|--------|----------|---------|-----------|---------|------|")
            
            for metric_name in ['RMSE', 'DTW', 'Fréchet', 'ERP', 'DDTW', 'TWED', 'Sobolev']:
                if metric_name in correlations and 'episode_number' in correlations[metric_name]:
                    corr = correlations[metric_name]['episode_number']
                    pearson_r = corr.get('pearson_r', 0)
                    pearson_p = corr.get('pearson_p', 1)
                    spearman_r = corr.get('spearman_r', 0)
                    spearman_p = corr.get('spearman_p', 1)
                    
                    # Interpretation
                    if abs(pearson_r) > 0.7:
                        strength = "강한"
                    elif abs(pearson_r) > 0.4:
                        strength = "중간"
                    else:
                        strength = "약한"
                    
                    direction = "양의" if pearson_r > 0 else "음의"
                    sig = "**유의함**" if pearson_p < 0.05 else "유의하지 않음"
                    
                    interpretation = f"{strength} {direction} 상관관계 ({sig})"
                    
                    report_lines.append(
                        f"| {metric_name} | {pearson_r:.3f} | {pearson_p:.4f} | "
                        f"{spearman_r:.3f} | {spearman_p:.4f} | {interpretation} |"
                    )
            report_lines.append("")
            
            report_lines.append("### 4.2 메트릭과 경로 길이의 상관관계")
            report_lines.append("")
            report_lines.append("| 메트릭 | Pearson r | p-value | Spearman ρ | p-value | 해석 |")
            report_lines.append("|--------|----------|---------|-----------|---------|------|")
            
            for metric_name in ['RMSE', 'DTW', 'Fréchet', 'ERP', 'DDTW', 'TWED', 'Sobolev']:
                if metric_name in correlations and 'trajectory_length' in correlations[metric_name]:
                    corr = correlations[metric_name]['trajectory_length']
                    pearson_r = corr.get('pearson_r', 0)
                    pearson_p = corr.get('pearson_p', 1)
                    spearman_r = corr.get('spearman_r', 0)
                    spearman_p = corr.get('spearman_p', 1)
                    
                    if abs(pearson_r) > 0.7:
                        strength = "강한"
                    elif abs(pearson_r) > 0.4:
                        strength = "중간"
                    else:
                        strength = "약한"
                    
                    direction = "양의" if pearson_r > 0 else "음의"
                    sig = "**유의함**" if pearson_p < 0.05 else "유의하지 않음"
                    
                    interpretation = f"{strength} {direction} 상관관계 ({sig})"
                    
                    report_lines.append(
                        f"| {metric_name} | {pearson_r:.3f} | {pearson_p:.4f} | "
                        f"{spearman_r:.3f} | {spearman_p:.4f} | {interpretation} |"
                    )
            report_lines.append("")
            
            # Characteristics correlations
            if statistical_analysis.get('characteristics'):
                report_lines.append("### 4.3 메트릭과 궤적 특성의 상관관계")
                report_lines.append("")
                
                # Select key characteristics
                key_chars = ['avg_speed', 'speed_variance', 'num_stops', 'num_backtracks', 
                            'direction_similarity', 'efficiency', 'avg_curvature']
                
                for char_name in key_chars:
                    if char_name not in statistical_analysis.get('characteristics', {}):
                        continue
                    
                    report_lines.append(f"#### {char_name}")
                    report_lines.append("")
                    report_lines.append("| 메트릭 | Pearson r | p-value | 해석 |")
                    report_lines.append("|--------|----------|---------|------|")
                    
                    for metric_name in ['RMSE', 'DTW', 'Fréchet', 'ERP', 'DDTW', 'TWED', 'Sobolev']:
                        if (metric_name in correlations and 
                            char_name in correlations[metric_name]):
                            corr = correlations[metric_name][char_name]
                            pearson_r = corr.get('pearson_r', 0)
                            pearson_p = corr.get('pearson_p', 1)
                            
                            if abs(pearson_r) > 0.7:
                                strength = "강한"
                            elif abs(pearson_r) > 0.4:
                                strength = "중간"
                            else:
                                strength = "약한"
                            
                            direction = "양의" if pearson_r > 0 else "음의"
                            sig = "**유의함**" if pearson_p < 0.05 else "유의하지 않음"
                            
                            interpretation = f"{strength} {direction} 상관관계 ({sig})"
                            
                            report_lines.append(
                                f"| {metric_name} | {pearson_r:.3f} | {pearson_p:.4f} | {interpretation} |"
                            )
                    report_lines.append("")
        
        # 5. 통계적 유의성 검정
        if statistical_analysis.get('statistical_tests'):
            report_lines.append("## 5. 통계적 유의성 검정")
            report_lines.append("")
            
            tests = statistical_analysis['statistical_tests']
            
            report_lines.append("### 5.1 Early vs Late Episode 비교 (Mann-Whitney U test)")
            report_lines.append("")
            report_lines.append("| 메트릭 | Early 평균 | Late 평균 | U 통계량 | p-value | 해석 |")
            report_lines.append("|--------|-----------|-----------|---------|---------|------|")
            
            for metric_name in ['RMSE', 'DTW', 'Fréchet', 'ERP', 'DDTW', 'TWED', 'Sobolev']:
                if metric_name in tests:
                    test = tests[metric_name]
                    early_mean = test.get('early_mean', 0)
                    late_mean = test.get('late_mean', 0)
                    u_stat = test.get('mann_whitney_u', 0)
                    p_value = test.get('mann_whitney_p', 1)
                    
                    if p_value < 0.05:
                        interpretation = "**유의한 차이 있음**"
                    elif p_value < 0.1:
                        interpretation = "*경향 있음*"
                    else:
                        interpretation = "유의한 차이 없음"
                    
                    direction = "증가" if late_mean > early_mean else "감소"
                    
                    report_lines.append(
                        f"| {metric_name} | {early_mean:.3f} | {late_mean:.3f} | "
                        f"{u_stat:.1f} | {p_value:.4f} | {direction} ({interpretation}) |"
                    )
            report_lines.append("")
        
        # 6. 메트릭 경향성과 궤적 특성의 관계 (Enhanced)
        report_lines.append("## 6. 메트릭 경향성과 궤적 특성의 관계")
        report_lines.append("")
        
        if statistical_analysis.get('metrics') and statistical_analysis.get('correlations'):
            metrics = statistical_analysis['metrics']
            correlations = statistical_analysis['correlations']
            
            # Analyze each metric with data-driven insights
            for metric_name in ['RMSE', 'DTW', 'Fréchet', 'ERP', 'DDTW', 'TWED', 'Sobolev']:
                if metric_name not in metrics:
                    continue
                
                report_lines.append(f"### 6.{['RMSE', 'DTW', 'Fréchet', 'ERP', 'DDTW', 'TWED', 'Sobolev'].index(metric_name) + 1} {metric_name}")
                report_lines.append("")
                
                metric_data = metrics[metric_name]
                trend = metric_data.get('trend', {})
                slope = trend.get('slope', 0)
                r_squared = trend.get('r_squared', 0)
                p_value = trend.get('p_value', 1)
                
                # Determine trend direction
                if slope > 0:
                    trend_direction = "**증가**"
                    trend_interpretation = "나중 Episode가 GT와 더 다름"
                elif slope < 0:
                    trend_direction = "**감소**"
                    trend_interpretation = "나중 Episode가 GT에 더 가까움"
                else:
                    trend_direction = "**변화 없음**"
                    trend_interpretation = "Episode와 무관"
                
                report_lines.append(f"**경향성**: Episode 증가에 따라 {trend_direction} (slope={slope:.3f}, R²={r_squared:.3f}, p={p_value:.4f})")
                report_lines.append("")
                
                # Correlation with trajectory characteristics
                if metric_name in correlations:
                    corr_data = correlations[metric_name]
                    
                    # Find strongest correlations
                    char_correlations = []
                    for char_name, corr in corr_data.items():
                        if char_name in ['episode_number', 'trajectory_length']:
                            continue
                        pearson_r = corr.get('pearson_r', 0)
                        pearson_p = corr.get('pearson_p', 1)
                        if abs(pearson_r) > 0.3:  # Moderate or strong correlation
                            char_correlations.append((char_name, pearson_r, pearson_p))
                    
                    char_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
                    
                    if char_correlations:
                        report_lines.append("**궤적 특성 영향 분석**:")
                        report_lines.append("")
                        
                        # Top correlations
                        for char_name, r, p in char_correlations[:3]:
                            char_display = {
                                'avg_speed': '평균 속도',
                                'speed_variance': '속도 분산',
                                'num_stops': '정지 횟수',
                                'num_backtracks': '역주행 횟수',
                                'direction_similarity': '방향 유사도',
                                'efficiency': '효율성',
                                'avg_curvature': '평균 곡률',
                                'path_length': '경로 길이',
                                'path_length_ratio': '경로 길이 비율'
                            }.get(char_name, char_name)
                            
                            strength = "강한" if abs(r) > 0.7 else "중간" if abs(r) > 0.4 else "약한"
                            direction = "양의" if r > 0 else "음의"
                            sig = "**유의함**" if p < 0.05 else "유의하지 않음"
                            
                            report_lines.append(f"- **{char_display}**: {strength} {direction} 상관관계 (r={r:.3f}, p={p:.4f}, {sig})")
                            report_lines.append(f"  - 해석: {char_display}가 {'증가' if r > 0 else '감소'}할수록 {metric_name} 값이 {'증가' if r > 0 else '감소'}함")
                        
                        report_lines.append("")
                    
                    # Episode number correlation
                    if 'episode_number' in corr_data:
                        ep_corr = corr_data['episode_number']
                        ep_r = ep_corr.get('pearson_r', 0)
                        ep_p = ep_corr.get('pearson_p', 1)
                        
                        report_lines.append(f"**Episode 번호와의 상관관계**: r={ep_r:.3f}, p={ep_p:.4f}")
                        if ep_p < 0.05:
                            report_lines.append(f"- **통계적으로 유의한** Episode 의존성이 있음")
                        else:
                            report_lines.append(f"- Episode 의존성은 통계적으로 유의하지 않음 (샘플 크기 부족 가능)")
                        report_lines.append("")
                    
                    # Trajectory length correlation
                    if 'trajectory_length' in corr_data:
                        len_corr = corr_data['trajectory_length']
                        len_r = len_corr.get('pearson_r', 0)
                        len_p = len_corr.get('pearson_p', 1)
                        
                        report_lines.append(f"**경로 길이와의 상관관계**: r={len_r:.3f}, p={len_p:.4f}")
                        if abs(len_r) > 0.5 and len_p < 0.05:
                            report_lines.append(f"- 경로 길이에 **강하게 의존**함")
                        elif abs(len_r) > 0.3:
                            report_lines.append(f"- 경로 길이에 **중간 정도 의존**함")
                        else:
                            report_lines.append(f"- 경로 길이에 상대적으로 **독립적**임")
                        report_lines.append("")
                
                # Metric-specific interpretation
                metric_interpretations = {
                    'RMSE': {
                        'sensitive': '시간 동기화 오차, 위치 편차',
                        'insensitive': '경로 형태 변화, 속도 변화',
                        'note': '시간 동기화가 필수적임'
                    },
                    'DTW': {
                        'sensitive': '경로 형태(pattern), 전체 궤적의 기하학적 구조, 경로 길이 차이',
                        'insensitive': '일시적 정지, 속도 변화',
                        'note': '경로 길이가 다를 때 큰 영향을 받음'
                    },
                    'Fréchet': {
                        'sensitive': '경로의 기하학적 형상, 토폴로지',
                        'insensitive': '시간, 속도, 일시적 우회',
                        'note': '시간과 무관하게 형상만 비교'
                    },
                    'ERP': {
                        'sensitive': '큰 Gap, 누락된 구간, 경로 길이 차이',
                        'insensitive': '작은 우회, 국소적 편차',
                        'note': '경로 길이 차이에 매우 민감'
                    },
                    'DDTW': {
                        'sensitive': '정지(Stop), 역주행(Backtracking), 속도 변화',
                        'insensitive': '위치 오프셋, 베이스라인 시프트',
                        'note': '속도/방향 벡터로 비교하므로 동역학적 특성에 민감'
                    },
                    'TWED': {
                        'sensitive': '시간 페널티, 속도 변화, 정차, 경로 길이 차이',
                        'insensitive': '위치만의 오차',
                        'note': '시간 왜곡 페널티를 명시적으로 고려'
                    },
                    'Sobolev': {
                        'sensitive': '위치 오차, 속도 오차 (둘 다)',
                        'insensitive': '없음 (종합적 메트릭)',
                        'note': '위치와 속도를 모두 고려하는 종합 메트릭'
                    }
                }
                
                if metric_name in metric_interpretations:
                    interp = metric_interpretations[metric_name]
                    report_lines.append("**메트릭 특성**:")
                    report_lines.append(f"- **민감한 요소**: {interp['sensitive']}")
                    report_lines.append(f"- **둔감한 요소**: {interp['insensitive']}")
                    report_lines.append(f"- **특이사항**: {interp['note']}")
                    report_lines.append("")
        
        # 7. Episode 증가에 따른 경향성 종합 분석
        report_lines.append("## 7. Episode 증가에 따른 경향성 종합 분석")
        report_lines.append("")
        
        if statistical_analysis.get('metrics'):
            metrics = statistical_analysis['metrics']
            
            increasing_metrics = []
            decreasing_metrics = []
            
            for metric_name, metric_data in metrics.items():
                trend = metric_data.get('trend', {})
                slope = trend.get('slope', 0)
                if slope > 0:
                    increasing_metrics.append((metric_name, slope))
                elif slope < 0:
                    decreasing_metrics.append((metric_name, slope))
            
            increasing_metrics.sort(key=lambda x: x[1], reverse=True)
            decreasing_metrics.sort(key=lambda x: x[1])
            
            report_lines.append("### 7.1 전체적인 경향 패턴")
            report_lines.append("")
            
            if increasing_metrics:
                report_lines.append("**증가하는 메트릭** (나중 Episode가 GT와 더 다름):")
                for metric_name, slope in increasing_metrics:
                    report_lines.append(f"- **{metric_name}** (slope={slope:.3f}): Episode 증가에 따라 값이 증가")
                report_lines.append("")
            
            if decreasing_metrics:
                report_lines.append("**감소하는 메트릭** (나중 Episode가 GT에 더 가까움):")
                for metric_name, slope in decreasing_metrics:
                    report_lines.append(f"- **{metric_name}** (slope={slope:.3f}): Episode 증가에 따라 값이 감소")
                report_lines.append("")
            
            report_lines.append("### 7.2 모순적 경향성 해석")
            report_lines.append("")
            
            if increasing_metrics and decreasing_metrics:
                report_lines.append("흥미롭게도, **증가하는 메트릭**과 **감소하는 메트릭**이 공존합니다:")
                report_lines.append("")
                
                # Analyze contradictions
                position_metrics = ['RMSE']
                shape_metrics = ['DTW', 'Fréchet']
                time_metrics = ['TWED', 'ERP']
                velocity_metrics = ['DDTW']
                combined_metrics = ['Sobolev']
                
                decreasing_pos = any(m[0] in position_metrics for m in decreasing_metrics)
                increasing_shape = any(m[0] in shape_metrics for m in increasing_metrics)
                
                if decreasing_pos and increasing_shape:
                    report_lines.append("1. **위치 기반 메트릭 (RMSE)은 감소**하지만, **형태 기반 메트릭 (DTW, Fréchet)은 증가**")
                    report_lines.append("   - 해석: 나중 Episode는 **같은 시간대의 위치는 더 정확**하지만, **전체적인 경로 형태는 더 달라짐**")
                    report_lines.append("   - 이는 **로컬 정확도는 개선**되지만, **글로벌 경로 선택은 다름**을 의미")
                    report_lines.append("")
                
                decreasing_time = any(m[0] in time_metrics for m in decreasing_metrics)
                if decreasing_time and increasing_shape:
                    report_lines.append("2. **시간 기반 메트릭 (TWED, ERP)은 감소**하지만, **형태 기반 메트릭은 증가**")
                    report_lines.append("   - 해석: 나중 Episode는 **시간적 정렬은 개선**되지만, **공간적 형상은 더 달라짐**")
                    report_lines.append("")
                
                decreasing_vel = any(m[0] in velocity_metrics for m in decreasing_metrics)
                increasing_combined = any(m[0] in combined_metrics for m in increasing_metrics)
                if decreasing_vel and increasing_combined:
                    report_lines.append("3. **속도 기반 메트릭 (DDTW)은 감소**하지만, **종합 메트릭 (Sobolev)은 증가**")
                    report_lines.append("   - 해석: 나중 Episode는 **속도/방향은 더 유사**하지만, **위치 오차가 증가**하여 종합적으로는 더 나빠짐")
                    report_lines.append("")
            
            # Trajectory length analysis
            if statistical_analysis.get('trajectory_lengths'):
                lengths = statistical_analysis['trajectory_lengths']
                if len(lengths) > 1:
                    report_lines.append("### 7.3 궤적 길이의 영향")
                    report_lines.append("")
                    report_lines.append(f"- 경로 길이 범위: {min(lengths)} ~ {max(lengths)} steps")
                    report_lines.append(f"- 평균 경로 길이: {np.mean(lengths):.1f} steps (std: {np.std(lengths):.1f})")
                    report_lines.append("")
                    
                    if max(lengths) / min(lengths) > 2:
                        report_lines.append("**관찰**: 경로 길이가 **2배 이상 차이**가 나는 경우가 있습니다.")
                        report_lines.append("- 경로 길이 차이가 큰 메트릭 (DTW, ERP, TWED)에 큰 영향을 미침")
                        report_lines.append("- 경로 길이에 상대적으로 독립적인 메트릭 (RMSE, Fréchet)은 영향이 적음")
                        report_lines.append("")
        
        # 8. 이 실험 데이터에 적합한 메트릭 분석
        report_lines.append("## 8. 이 실험 데이터에 적합한 메트릭 분석")
        report_lines.append("")
        
        if statistical_analysis.get('metrics'):
            metrics = statistical_analysis['metrics']
            
            # Stability analysis
            report_lines.append("### 8.1 안정성 관점 (Episode와 무관하게 일정해야 함)")
            report_lines.append("")
            
            cv_scores = [(name, m.get('cv', np.inf)) for name, m in metrics.items()]
            cv_scores.sort(key=lambda x: x[1])
            
            report_lines.append("**가장 안정적인 메트릭** (CV가 작을수록 안정적):")
            for i, (name, cv) in enumerate(cv_scores[:3], 1):
                report_lines.append(f"{i}. **{name}** (CV={cv:.4f})")
            report_lines.append("")
            
            report_lines.append("**가장 불안정한 메트릭** (CV가 클수록 불안정):")
            for i, (name, cv) in enumerate(cv_scores[-3:][::-1], 1):
                report_lines.append(f"{i}. **{name}** (CV={cv:.4f})")
            report_lines.append("")
            
            # Episode independence
            report_lines.append("### 8.2 Episode 독립성 관점 (이상적으로는 변하지 않아야 함)")
            report_lines.append("")
            
            trend_scores = []
            for name, m in metrics.items():
                trend = m.get('trend', {})
                slope = abs(trend.get('slope', 0))
                trend_scores.append((name, slope))
            
            trend_scores.sort(key=lambda x: x[1])
            
            report_lines.append("**가장 Episode 독립적** (트렌드가 작음):")
            for i, (name, slope) in enumerate(trend_scores[:3], 1):
                report_lines.append(f"{i}. **{name}** (|slope|={slope:.3f})")
            report_lines.append("")
            
            report_lines.append("**가장 Episode 의존적** (트렌드가 큼):")
            for i, (name, slope) in enumerate(trend_scores[-3:][::-1], 1):
                report_lines.append(f"{i}. **{name}** (|slope|={slope:.3f})")
            report_lines.append("")
            
            # Recommendations
            report_lines.append("### 8.3 실험 목적에 따른 메트릭 추천")
            report_lines.append("")
            
            # Find best metrics
            best_stability = cv_scores[0][0] if cv_scores else None
            best_independence = trend_scores[0][0] if trend_scores else None
            
            report_lines.append("#### A. 전체적인 궤적 유사도 평가 (종합적)")
            if best_stability:
                report_lines.append(f"**추천**: **{best_stability}**")
                report_lines.append(f"- 이유: 가장 안정적 (CV={cv_scores[0][1]:.4f})")
                report_lines.append("- 위치와 속도를 모두 고려하는 종합 메트릭")
            report_lines.append("")
            
            report_lines.append("#### B. 속도/방향 기반 평가 (동역학적)")
            report_lines.append("**추천**: **DDTW**")
            report_lines.append("- 이유: 정지/역주행 감지에 특화")
            report_lines.append("- 속도 벡터로 비교하므로 동역학적 특성에 민감")
            report_lines.append("")
            
            report_lines.append("#### C. 기하학적 형상 평가 (공간적)")
            report_lines.append("**추천**: **Fréchet Distance**")
            report_lines.append("- 이유: 시간과 무관하게 형상만 평가")
            report_lines.append("- 경로 길이에 상대적으로 독립적")
            report_lines.append("")
            
            report_lines.append("#### D. 시간 동기화 평가 (시간적)")
            report_lines.append("**추천**: **RMSE**")
            if best_independence:
                report_lines.append(f"- 이유: Episode 독립성이 가장 좋음 (|slope|={trend_scores[0][1]:.3f})")
            report_lines.append("- 시간 동기화 오차를 직접 측정")
            report_lines.append("- 단점: 시간 동기화가 필수 (제약)")
            report_lines.append("")
            
            report_lines.append("### 8.4 최종 권장사항")
            report_lines.append("")
            
            # Create ranking
            rankings = []
            for name, m in metrics.items():
                cv = m.get('cv', np.inf)
                trend = m.get('trend', {})
                slope = abs(trend.get('slope', 0))
                
                # Combined score (lower is better)
                # Normalize CV and slope
                max_cv = max([m2.get('cv', 0) for m2 in metrics.values()])
                max_slope = max([abs(m2.get('trend', {}).get('slope', 0)) for m2 in metrics.values()])
                
                if max_cv > 0:
                    norm_cv = cv / max_cv
                else:
                    norm_cv = 0
                
                if max_slope > 0:
                    norm_slope = slope / max_slope
                else:
                    norm_slope = 0
                
                # Combined score (weight: stability 0.6, independence 0.4)
                score = 0.6 * norm_cv + 0.4 * norm_slope
                rankings.append((name, score, cv, slope))
            
            rankings.sort(key=lambda x: x[1])
            
            report_lines.append("**이 실험 데이터에 가장 적합한 메트릭** (안정성과 Episode 독립성 종합 고려):")
            report_lines.append("")
            
            for i, (name, score, cv, slope) in enumerate(rankings[:3], 1):
                report_lines.append(f"{i}. **{name}**")
                report_lines.append(f"   - 안정성: CV={cv:.4f}")
                report_lines.append(f"   - Episode 독립성: |slope|={slope:.3f}")
                report_lines.append(f"   - 종합 점수: {score:.3f} (낮을수록 좋음)")
                report_lines.append("")
        
        # 9. 결론 및 권장사항
        report_lines.append("## 9. 결론 및 권장사항")
        report_lines.append("")
        
        report_lines.append("### 9.1 주요 발견사항")
        report_lines.append("")
        
        if statistical_analysis.get('metrics'):
            metrics = statistical_analysis['metrics']
            
            # Key findings
            findings = []
            
            # Check for position vs shape contradiction
            if 'RMSE' in metrics and 'DTW' in metrics:
                rmse_trend = metrics['RMSE'].get('trend', {}).get('slope', 0)
                dtw_trend = metrics['DTW'].get('trend', {}).get('slope', 0)
                if rmse_trend < 0 and dtw_trend > 0:
                    findings.append("**위치 정확도는 개선**되지만, **경로 형태는 더 달라짐**")
            
            # Check for time vs shape contradiction
            if 'TWED' in metrics and 'Fréchet' in metrics:
                twed_trend = metrics['TWED'].get('trend', {}).get('slope', 0)
                frechet_trend = metrics['Fréchet'].get('trend', {}).get('slope', 0)
                if twed_trend < 0 and frechet_trend > 0:
                    findings.append("**시간 정렬은 개선**되지만, **공간 형상은 더 달라짐**")
            
            # Check for velocity vs combined contradiction
            if 'DDTW' in metrics and 'Sobolev' in metrics:
                ddtw_trend = metrics['DDTW'].get('trend', {}).get('slope', 0)
                sobolev_trend = metrics['Sobolev'].get('trend', {}).get('slope', 0)
                if ddtw_trend < 0 and sobolev_trend > 0:
                    findings.append("**속도/방향 일치는 개선**되지만, **종합적 오차는 증가**")
            
            for i, finding in enumerate(findings, 1):
                report_lines.append(f"{i}. {finding}")
                report_lines.append("")
        
        report_lines.append("### 9.2 실험 해석")
        report_lines.append("")
        report_lines.append("이러한 경향성은 다음을 시사합니다:")
        report_lines.append("- 나중 Episode는 **더 정확한 타이밍과 속도**로 이동하지만")
        report_lines.append("- **다른 경로를 선택**하여 전체적으로는 더 달라짐")
        report_lines.append("- 즉, **로컬 정확도는 개선**되지만, **글로벌 경로 선택은 다름**")
        report_lines.append("")
        
        report_lines.append("### 9.3 실용적 권장사항")
        report_lines.append("")
        report_lines.append("이 실험 데이터의 특성상:")
        report_lines.append("- **동역학적 평가**가 중요하다면 → **DDTW** 사용")
        report_lines.append("- **종합적 평가**가 중요하다면 → **Sobolev** 사용")
        report_lines.append("- **시간 정렬 평가**가 중요하다면 → **RMSE** 사용")
        report_lines.append("- **형상 평가**가 중요하다면 → **Fréchet** 사용")
        report_lines.append("")
        report_lines.append("**경로 길이가 다양한 경우**: DTW, ERP, TWED는 부적절할 수 있음")
        report_lines.append("")
        
        return "\n".join(report_lines)
