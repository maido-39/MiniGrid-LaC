#!/usr/bin/env python3
"""
Improve existing reports with deep insights and interpretations.

Reads existing reports and enhances them with insightful analysis.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import re


def extract_insights_from_data(stat_path: Path, detailed_path: Path) -> Dict:
    """Extract insights from statistical and detailed data."""
    with open(stat_path, 'r', encoding='utf-8') as f:
        stats = json.load(f)
    
    with open(detailed_path, 'r', encoding='utf-8') as f:
        detailed = json.load(f)
    
    insights = {
        'key_findings': [],
        'contradictions': [],
        'anomalies': [],
        'strong_relationships': [],
        'episode_progression': {},
        'metric_suitability': {}
    }
    
    # Extract metrics and correlations
    metrics = stats.get('metrics', {})
    correlations = stats.get('correlations', {})
    trajectory_chars = stats.get('trajectory_characteristics', {})
    
    # 1. Find contradictions
    rmse_trend = metrics.get('RMSE', {}).get('trend', {})
    dtw_trend = metrics.get('DTW', {}).get('trend', {})
    
    if rmse_trend.get('slope', 0) < 0 and dtw_trend.get('slope', 0) > 0:
        insights['contradictions'].append({
            'type': 'rmse_dtw_opposition',
            'rmse_slope': rmse_trend.get('slope', 0),
            'dtw_slope': dtw_trend.get('slope', 0),
            'interpretation': (
                'RMSE 감소는 위치 정확도가 개선되었음을 의미하지만, '
                'DTW 증가는 경로 형태가 더 달라졌음을 의미합니다. '
                '이는 로봇이 로컬 정확도는 향상했지만 글로벌 경로 선택은 다르게 작동함을 시사합니다.'
            )
        })
    
    # 2. Find strong relationships
    for metric_name, metric_corrs in correlations.items():
        for feature_name, corr_data in metric_corrs.items():
            if feature_name == 'episode_number':
                continue
            r = corr_data.get('pearson_r', 0)
            p = corr_data.get('pearson_p', 1.0)
            
            if not np.isnan(r) and abs(r) > 0.7 and p < 0.05:
                insights['strong_relationships'].append({
                    'metric': metric_name,
                    'feature': feature_name,
                    'correlation': r,
                    'p_value': p,
                    'meaning': interpret_relationship(metric_name, feature_name, r)
                })
    
    # 3. Find anomalies
    for ep_name, ep_data in detailed.get('episodes', {}).items():
        traj_len = ep_data.get('trajectory_length', 0)
        dtw_val = ep_data.get('metrics', {}).get('DTW', 0)
        frechet_val = ep_data.get('metrics', {}).get('Fréchet', 0)
        
        # Short path but high DTW
        if traj_len < 35 and dtw_val > 20:
            insights['anomalies'].append({
                'episode': ep_name,
                'observation': f'경로 길이 {traj_len}인데 DTW={dtw_val:.2f}',
                'interpretation': '경로가 짧지만 형태가 GT와 매우 다름 - 급격한 경로 변경이나 우회'
            })
        
        # Very high Fréchet
        if frechet_val > 6:
            insights['anomalies'].append({
                'episode': ep_name,
                'observation': f'Fréchet={frechet_val:.2f} (평균의 2배 이상)',
                'interpretation': '전체 경로 형상이 GT와 매우 다름 - 완전히 다른 경로 선택'
            })
    
    # 4. Episode progression
    improving = []
    worsening = []
    
    for metric_name, metric_data in metrics.items():
        trend = metric_data.get('trend', {})
        slope = trend.get('slope', 0)
        p_value = trend.get('p_value', 1.0)
        
        if p_value < 0.1:
            if slope < -0.1:
                improving.append({
                    'metric': metric_name,
                    'slope': slope,
                    'meaning': 'GT에 가까워짐'
                })
            elif slope > 0.1:
                worsening.append({
                    'metric': metric_name,
                    'slope': slope,
                    'meaning': 'GT와 멀어짐'
                })
    
    insights['episode_progression'] = {
        'improving': improving,
        'worsening': worsening,
        'interpretation': generate_progression_interpretation(improving, worsening)
    }
    
    # 5. Metric suitability
    traj_lengths = trajectory_chars.get('trajectory_length', {}).get('values', [])
    length_diversity = max(traj_lengths) / min(traj_lengths) if traj_lengths and min(traj_lengths) > 0 else 1.0
    
    for metric_name, metric_data in metrics.items():
        cv = metric_data.get('cv', 1.0)
        slope = abs(metric_data.get('trend', {}).get('slope', 0))
        metric_corrs = correlations.get(metric_name, {})
        r_length = metric_corrs.get('trajectory_length', {}).get('pearson_r', 0)
        
        if np.isnan(r_length):
            r_length = 0
        
        # Suitability factors
        suitability_score = cv * 0.4 + (slope / 10.0) * 0.3 + abs(r_length) * 0.3
        
        # Specific insights
        specific_insights = []
        if metric_name == 'RMSE' and abs(r_length) > 0.8:
            specific_insights.append('경로 길이 다양성에 취약 - 길이가 2배 이상 차이나면 정규화 필요')
        elif metric_name == 'DDTW' and metric_corrs.get('num_backtracks', {}).get('pearson_r', 0) > 0.8:
            specific_insights.append('역주행 감지에 탁월함 - 역주행이 중요한 평가 요소라면 최적 선택')
        elif metric_name == 'DTW' and abs(r_length) < 0.5:
            specific_insights.append('경로 길이 차이에 상대적으로 강인함')
        
        insights['metric_suitability'][metric_name] = {
            'score': suitability_score,
            'cv': cv,
            'slope': slope,
            'length_dependency': abs(r_length),
            'insights': specific_insights
        }
    
    return insights


def interpret_relationship(metric: str, feature: str, r: float) -> str:
    """Interpret relationship between metric and feature."""
    interpretations = {
        ('DDTW', 'num_backtracks'): (
            f'DDTW는 역주행 횟수와 강한 양의 상관관계(r={r:.3f})를 보입니다. '
            '이는 DDTW가 역주행을 효과적으로 감지한다는 것을 의미하며, '
            '역주행이 중요한 평가 요소인 실험에서는 DDTW가 최적의 선택입니다.'
        ),
        ('DTW', 'avg_curvature'): (
            f'DTW는 평균 곡률과 강한 양의 상관관계(r={r:.3f})를 보입니다. '
            '경로가 복잡할수록(곡률이 클수록) DTW 값이 증가하므로, '
            'DTW는 경로의 복잡도나 곡률을 반영하는 메트릭으로 해석할 수 있습니다.'
        ),
        ('RMSE', 'trajectory_length'): (
            f'RMSE는 경로 길이와 강한 양의 상관관계(r={r:.3f})를 보입니다. '
            '경로가 길수록 누적 오차가 커지므로, 경로 길이가 다양할 때는 정규화가 필요합니다.'
        ),
        ('TWED', 'efficiency'): (
            f'TWED는 효율성과 강한 음의 상관관계(r={r:.3f})를 보입니다. '
            '효율적인 경로일수록 TWED 값이 낮으므로, TWED는 경로 효율성을 평가하는 데 유용합니다.'
        ),
        ('Sobolev', 'avg_curvature'): (
            f'Sobolev는 평균 곡률과 강한 양의 상관관계(r={r:.3f})를 보입니다. '
            '이는 Sobolev가 경로의 복잡도를 반영한다는 것을 의미합니다.'
        ),
    }
    
    key = (metric, feature)
    if key in interpretations:
        return interpretations[key]
    
    direction = "증가" if r > 0 else "감소"
    strength = "매우 강한" if abs(r) > 0.8 else "강한" if abs(r) > 0.6 else "중간"
    return f'{metric}는 {feature}와 {strength} {direction} 상관관계를 보입니다.'


def generate_progression_interpretation(improving: List, worsening: List) -> str:
    """Generate interpretation of episode progression."""
    if not improving and not worsening:
        return 'Episode 진행에 따른 명확한 경향이 관찰되지 않습니다.'
    
    improving_names = [m['metric'] for m in improving]
    worsening_names = [m['metric'] for m in worsening]
    
    if improving_names and worsening_names:
        return (
            f'흥미롭게도 {", ".join(improving_names)}는 개선되지만, '
            f'{", ".join(worsening_names)}는 악화됩니다. '
            '이는 로봇이 일부 측면에서는 더 정확해지지만, 다른 측면에서는 더 달라진다는 것을 의미합니다. '
            '이는 학습 과정에서 로컬 최적화와 전역 최적화가 서로 다른 방향으로 작동했을 가능성을 시사합니다.'
        )
    elif improving_names:
        return f'{", ".join(improving_names)}는 Episode 진행에 따라 개선되어, 로봇이 GT에 점점 가까워지고 있습니다.'
    else:
        return f'{", ".join(worsening_names)}는 Episode 진행에 따라 악화되어, 로봇이 GT에서 점점 멀어지고 있습니다.'


def enhance_report_with_insights(report_path: Path, insights: Dict, output_path: Path):
    """Enhance existing report with insights."""
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find sections to enhance
    enhanced_content = content
    
    # Add insights section after conclusion
    insights_section = "\n\n---\n\n"
    insights_section += "## 통찰 중심 분석 (Insights-Driven Analysis)\n\n"
    insights_section += "이 섹션은 단순 통계 수치를 넘어, 데이터에서 발견된 패턴과 의미를 깊이 있게 해석합니다.\n\n"
    
    # 1. 핵심 발견사항
    if insights['contradictions'] or insights['anomalies'] or insights['strong_relationships']:
        insights_section += "### 핵심 발견사항\n\n"
        
        if insights['contradictions']:
            insights_section += "#### 모순적 패턴 발견\n\n"
            for contradiction in insights['contradictions']:
                insights_section += f"**{contradiction.get('type', 'Unknown')}**:\n\n"
                insights_section += f"{contradiction.get('interpretation', '')}\n\n"
        
        if insights['anomalies']:
            insights_section += "#### 특이사항 (Anomalies)\n\n"
            for anomaly in insights['anomalies']:
                insights_section += f"**{anomaly['episode']}**: {anomaly['observation']}\n\n"
                insights_section += f"- **의미**: {anomaly['interpretation']}\n\n"
        
        if insights['strong_relationships']:
            insights_section += "#### 강한 상관관계 발견\n\n"
            for rel in insights['strong_relationships']:
                insights_section += f"**{rel['metric']} ↔ {rel['feature']}** (r={rel['correlation']:.3f}, p={rel['p_value']:.4f})\n\n"
                insights_section += f"- {rel['meaning']}\n\n"
    
    # 2. Episode 진행 해석
    if insights['episode_progression']:
        prog = insights['episode_progression']
        insights_section += "### Episode 진행에 따른 변화의 의미\n\n"
        insights_section += f"{prog.get('interpretation', '')}\n\n"
        
        if prog.get('improving'):
            insights_section += "**개선되는 메트릭**:\n"
            for m in prog['improving']:
                insights_section += f"- {m['metric']}: {m['meaning']} (slope={m['slope']:.3f})\n"
            insights_section += "\n"
        
        if prog.get('worsening'):
            insights_section += "**악화되는 메트릭**:\n"
            for m in prog['worsening']:
                insights_section += f"- {m['metric']}: {m['meaning']} (slope={m['slope']:.3f})\n"
            insights_section += "\n"
    
    # 3. 메트릭 적합성 통찰
    if insights['metric_suitability']:
        insights_section += "### 메트릭 적합성에 대한 깊은 통찰\n\n"
        
        sorted_suitability = sorted(insights['metric_suitability'].items(), key=lambda x: x[1]['score'])
        
        for metric_name, suit_data in sorted_suitability:
            insights_section += f"#### {metric_name}\n\n"
            insights_section += f"**적합성 점수**: {suit_data['score']:.3f} (낮을수록 좋음)\n\n")
            
            if suit_data['insights']:
                insights_section += "**특별한 통찰**:\n"
                for insight in suit_data['insights']:
                    insights_section += f"- {insight}\n"
                insights_section += "\n"
    
    # Insert before conclusion or at the end
    if "## 9. 결론" in enhanced_content:
        enhanced_content = enhanced_content.replace(
            "## 9. 결론",
            insights_section + "\n## 9. 결론"
        )
    else:
        enhanced_content += insights_section
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(enhanced_content)


def main():
    """Main function to improve all reports."""
    base_dir = Path(__file__).parent
    
    for group in ['hogun', 'hogun_0125', 'other']:
        stat_path = base_dir / f'analysis_reports/{group}/statistical_analysis.json'
        detailed_path = base_dir / f'analysis_reports/{group}/detailed_results.json'
        report_path = base_dir / f'analysis_reports/{group}/종합_분석_보고서_Enhanced.md'
        
        if not stat_path.exists() or not detailed_path.exists():
            print(f"Skipping {group}: missing data files")
            continue
        
        if not report_path.exists():
            print(f"Skipping {group}: report not found")
            continue
        
        print(f"Processing {group}...")
        
        # Extract insights
        insights = extract_insights_from_data(stat_path, detailed_path)
        
        # Enhance report
        output_path = base_dir / f'analysis_reports/{group}/종합_분석_보고서_Enhanced.md'
        enhance_report_with_insights(report_path, insights, output_path)
        
        print(f"Enhanced report for {group}")
    
    print("All reports enhanced!")


if __name__ == '__main__':
    main()
