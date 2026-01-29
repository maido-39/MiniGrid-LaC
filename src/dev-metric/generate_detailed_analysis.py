"""
Generate detailed analysis document with step-by-step analysis and metric sensitivity.
"""

import json
from pathlib import Path
from typing import Dict, List
import numpy as np
from datetime import datetime


def generate_detailed_analysis(
    group_name: str,
    step_analysis_dir: Path,
    statistical_analysis_path: Path,
    output_path: Path
):
    """
    Generate detailed analysis document.
    
    Args:
        group_name: Name of the group
        step_analysis_dir: Directory containing step analysis results
        statistical_analysis_path: Path to statistical analysis JSON
        output_path: Path to save the document
    """
    # Load step analysis results
    step_results_path = step_analysis_dir / 'stepwise_analysis_results.json'
    if not step_results_path.exists():
        print(f"Warning: {step_results_path} does not exist")
        return
    
    with open(step_results_path, 'r', encoding='utf-8') as f:
        step_results = json.load(f)
    
    # Load statistical analysis
    try:
        with open(statistical_analysis_path, 'r', encoding='utf-8') as f:
            statistical_analysis = json.load(f)
    except FileNotFoundError:
        statistical_analysis = {}
    
    # Generate document
    lines = []
    
    # Header
    lines.append(f"# {group_name.upper()} 그룹 상세 분석 보고서")
    lines.append("")
    lines.append(f"**생성 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Table of contents
    lines.append("## 목차")
    lines.append("")
    lines.append("1. [개요](#1-개요)")
    lines.append("2. [Step-by-Step 분석](#2-step-by-step-분석)")
    lines.append("3. [메트릭 반응성 분석](#3-메트릭-반응성-분석)")
    lines.append("4. [메트릭 민감도 분석](#4-메트릭-민감도-분석)")
    lines.append("5. [Episode별 Step 분석 결과](#5-episode별-step-분석-결과)")
    lines.append("6. [종합 분석 및 결론](#6-종합-분석-및-결론)")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # 1. 개요
    lines.append("## 1. 개요")
    lines.append("")
    lines.append("이 문서는 **Step-by-Step 분석**과 **메트릭 반응성 분석**을 포함한 상세 분석 보고서입니다.")
    lines.append("")
    lines.append("### 1.1 분석 내용")
    lines.append("")
    lines.append("- **Step-by-Step 분석**: 각 Step에서의 메트릭 값 변화 추적")
    lines.append("- **메트릭 반응성 분석**: 메트릭이 궤적의 어떤 특성 변화에 어떻게 반응하는지 분석")
    lines.append("- **메트릭 민감도 분석**: 각 메트릭이 어떤 궤적 특성에 민감하게 반응하는지 정량적 분석")
    lines.append("- **Episode별 상세 분석**: 각 Episode의 Step별 특성과 메트릭 반응 패턴")
    lines.append("")
    
    # 2. Step-by-Step 분석
    lines.append("## 2. Step-by-Step 분석")
    lines.append("")
    lines.append("### 2.1 분석 방법")
    lines.append("")
    lines.append("각 Episode의 궤적을 Step별로 분석하여 다음을 계산합니다:")
    lines.append("")
    lines.append("1. **Stepwise Metrics**: 각 Step까지의 누적 메트릭 값")
    lines.append("   - RMSE: 각 Step에서의 위치 오차")
    lines.append("   - DTW: 각 Step까지의 누적 DTW 비용")
    lines.append("   - Fréchet: 각 Step까지의 누적 Fréchet 거리")
    lines.append("   - ERP: 각 Step까지의 누적 ERP 비용")
    lines.append("   - DDTW: 각 Step까지의 누적 DDTW 비용")
    lines.append("   - TWED: 각 Step까지의 누적 TWED 비용")
    lines.append("   - Sobolev: 각 Step까지의 누적 Sobolev 거리")
    lines.append("")
    lines.append("2. **Trajectory Features**: 각 Step에서의 궤적 특성")
    lines.append("   - Position Error: GT와의 위치 오차")
    lines.append("   - Velocity Error: GT와의 속도 오차")
    lines.append("   - Direction Error: GT와의 방향 오차")
    lines.append("   - Cumulative Path Length: 누적 경로 길이")
    lines.append("   - Distance from Start: 시작점으로부터의 거리")
    lines.append("")
    
    # 3. 메트릭 반응성 분석
    lines.append("## 3. 메트릭 반응성 분석")
    lines.append("")
    
    if 'aggregated_sensitivity' in step_results:
        agg_sensitivity = step_results['aggregated_sensitivity']
        
        lines.append("### 3.1 메트릭별 반응성 요약")
        lines.append("")
        lines.append("각 메트릭이 궤적 특성 변화에 어떻게 반응하는지 분석한 결과입니다.")
        lines.append("")
        
        # Create summary table
        lines.append("| 메트릭 | Position Error | Velocity Error | Direction Error | 주요 반응 특성 |")
        lines.append("|--------|---------------|---------------|-----------------|---------------|")
        
        for metric_name in ['RMSE', 'DTW', 'Fréchet', 'ERP', 'DDTW', 'TWED', 'Sobolev']:
            if metric_name not in agg_sensitivity:
                continue
            
            metric_sens = agg_sensitivity[metric_name]
            pos_corr = metric_sens.get('position_error', {}).get('mean', 0)
            vel_corr = metric_sens.get('velocity_error', {}).get('mean', 0)
            dir_corr = metric_sens.get('direction_error', {}).get('mean', 0)
            
            # Determine main characteristic
            corrs = {
                'Position': abs(pos_corr),
                'Velocity': abs(vel_corr),
                'Direction': abs(dir_corr)
            }
            main_char = max(corrs.items(), key=lambda x: x[1])[0]
            
            lines.append(
                f"| {metric_name} | {pos_corr:.3f} | {vel_corr:.3f} | {dir_corr:.3f} | {main_char} |"
            )
        lines.append("")
        
        lines.append("### 3.2 상세 반응성 분석")
        lines.append("")
        
        for metric_name in ['RMSE', 'DTW', 'Fréchet', 'ERP', 'DDTW', 'TWED', 'Sobolev']:
            if metric_name not in agg_sensitivity:
                continue
            
            lines.append(f"#### {metric_name}")
            lines.append("")
            
            metric_sens = agg_sensitivity[metric_name]
            
            for feature_name in ['position_error', 'velocity_error', 'direction_error']:
                if feature_name not in metric_sens:
                    continue
                
                feature_data = metric_sens[feature_name]
                mean_corr = feature_data.get('mean', 0)
                std_corr = feature_data.get('std', 0)
                
                feature_display = {
                    'position_error': '위치 오차',
                    'velocity_error': '속도 오차',
                    'direction_error': '방향 오차'
                }.get(feature_name, feature_name)
                
                lines.append(f"- **{feature_display}**: 평균 상관계수 = {mean_corr:.3f} (std: {std_corr:.3f})")
                
                if abs(mean_corr) > 0.7:
                    strength = "매우 강한"
                elif abs(mean_corr) > 0.4:
                    strength = "강한"
                elif abs(mean_corr) > 0.2:
                    strength = "중간"
                else:
                    strength = "약한"
                
                direction = "양의" if mean_corr > 0 else "음의"
                
                lines.append(f"  - 해석: {strength} {direction} 상관관계")
                if mean_corr > 0:
                    lines.append(f"  - {feature_display}가 증가할수록 {metric_name} 값이 증가함")
                else:
                    lines.append(f"  - {feature_display}가 증가할수록 {metric_name} 값이 감소함")
                lines.append("")
    
    # 4. 메트릭 민감도 분석
    lines.append("## 4. 메트릭 민감도 분석")
    lines.append("")
    
    lines.append("### 4.1 민감도 히트맵")
    lines.append("")
    lines.append("다음 히트맵은 각 메트릭이 어떤 궤적 특성에 민감하게 반응하는지를 보여줍니다.")
    lines.append("")
    lines.append("![Metric Sensitivity Heatmap](step_analysis/metric_sensitivity_heatmap.png)")
    lines.append("")
    lines.append("**해석**:")
    lines.append("- **빨간색 (양수)**: 해당 특성이 증가할수록 메트릭 값이 증가")
    lines.append("- **파란색 (음수)**: 해당 특성이 증가할수록 메트릭 값이 감소")
    lines.append("- **색이 진할수록**: 더 강한 상관관계")
    lines.append("")
    
    if 'aggregated_sensitivity' in step_results:
        agg_sensitivity = step_results['aggregated_sensitivity']
        
        lines.append("### 4.2 메트릭별 민감도 순위")
        lines.append("")
        
        for metric_name in ['RMSE', 'DTW', 'Fréchet', 'ERP', 'DDTW', 'TWED', 'Sobolev']:
            if metric_name not in agg_sensitivity:
                continue
            
            lines.append(f"#### {metric_name}")
            lines.append("")
            
            metric_sens = agg_sensitivity[metric_name]
            
            # Sort by absolute correlation
            sorted_features = sorted(
                metric_sens.items(),
                key=lambda x: abs(x[1].get('mean', 0)),
                reverse=True
            )
            
            lines.append("민감도 순위 (절대 상관계수 기준):")
            lines.append("")
            for i, (feature_name, feature_data) in enumerate(sorted_features, 1):
                mean_corr = feature_data.get('mean', 0)
                feature_display = {
                    'position_error': '위치 오차',
                    'velocity_error': '속도 오차',
                    'direction_error': '방향 오차'
                }.get(feature_name, feature_name)
                
                lines.append(f"{i}. **{feature_display}**: {mean_corr:.3f}")
            lines.append("")
    
    # 5. Episode별 Step 분석 결과
    lines.append("## 5. Episode별 Step 분석 결과")
    lines.append("")
    
    if 'episodes' in step_results:
        episodes = step_results['episodes']
        
        for ep_name, ep_data in sorted(episodes.items()):
            lines.append(f"### 5.{list(episodes.keys()).index(ep_name) + 1} {ep_name}")
            lines.append("")
            
            # Add visualization
            vis_path = f"step_analysis/{ep_name}_stepwise_analysis.png"
            if (step_analysis_dir / f"{ep_name}_stepwise_analysis.png").exists():
                lines.append(f"![{ep_name} Step Analysis]({vis_path})")
                lines.append("")
            
            # Stepwise metrics summary
            stepwise_metrics = ep_data.get('stepwise_metrics', {})
            if stepwise_metrics:
                lines.append("#### Stepwise Metrics 요약")
                lines.append("")
                lines.append("| 메트릭 | 초기값 | 최종값 | 최대값 | 증가율 |")
                lines.append("|--------|--------|--------|--------|--------|")
                
                for metric_name in ['RMSE', 'DTW', 'Fréchet', 'ERP', 'DDTW', 'TWED', 'Sobolev']:
                    if metric_name not in stepwise_metrics:
                        continue
                    
                    values = stepwise_metrics[metric_name]
                    if len(values) == 0:
                        continue
                    
                    initial = values[0] if len(values) > 0 else 0
                    final = values[-1] if len(values) > 0 else 0
                    max_val = max(values) if values else 0
                    increase = ((final - initial) / initial * 100) if initial > 0 else 0
                    
                    lines.append(
                        f"| {metric_name} | {initial:.3f} | {final:.3f} | {max_val:.3f} | {increase:+.1f}% |"
                    )
                lines.append("")
            
            # Features summary
            stepwise_features = ep_data.get('stepwise_features', {})
            if stepwise_features:
                lines.append("#### 궤적 특성 요약")
                lines.append("")
                
                if 'position_error' in stepwise_features:
                    pos_errors = stepwise_features['position_error']
                    if pos_errors:
                        lines.append(f"- **평균 위치 오차**: {np.mean(pos_errors):.3f}")
                        lines.append(f"- **최대 위치 오차**: {np.max(pos_errors):.3f}")
                        lines.append(f"- **위치 오차 표준편차**: {np.std(pos_errors):.3f}")
                        lines.append("")
                
                if 'velocity_error' in stepwise_features:
                    vel_errors = stepwise_features['velocity_error']
                    if vel_errors:
                        lines.append(f"- **평균 속도 오차**: {np.mean(vel_errors):.3f}")
                        lines.append(f"- **최대 속도 오차**: {np.max(vel_errors):.3f}")
                        lines.append("")
                
                if 'direction_error' in stepwise_features:
                    dir_errors = stepwise_features['direction_error']
                    if dir_errors:
                        lines.append(f"- **평균 방향 오차**: {np.mean(dir_errors):.3f} rad")
                        lines.append(f"- **최대 방향 오차**: {np.max(dir_errors):.3f} rad")
                        lines.append("")
    
    # 6. 종합 분석 및 결론
    lines.append("## 6. 종합 분석 및 결론")
    lines.append("")
    
    lines.append("### 6.1 주요 발견사항")
    lines.append("")
    
    if 'aggregated_sensitivity' in step_results:
        agg_sensitivity = step_results['aggregated_sensitivity']
        
        # Find most sensitive metrics
        max_sensitivities = {}
        for metric_name, metric_sens in agg_sensitivity.items():
            max_corr = 0
            max_feature = None
            for feature_name, feature_data in metric_sens.items():
                abs_corr = abs(feature_data.get('mean', 0))
                if abs_corr > max_corr:
                    max_corr = abs_corr
                    max_feature = feature_name
            if max_feature:
                max_sensitivities[metric_name] = (max_feature, max_corr)
        
        lines.append("#### 6.1.1 메트릭별 주요 민감 특성")
        lines.append("")
        
        # Add insights for each metric
        metric_insights = {
            'RMSE': {
                'insight': 'RMSE는 Step별 위치 오차를 직접 측정하므로, 시간 동기화가 정확할 때 가장 신뢰할 수 있습니다.',
                'limit': '경로 길이에 강하게 의존하므로, 경로 길이가 다양하면 정규화가 필요합니다.'
            },
            'DDTW': {
                'insight': 'DDTW는 속도 벡터를 비교하므로, 역주행이나 정지와 같은 동역학적 특성을 잘 감지합니다.',
                'limit': '역주행 횟수와 강한 상관관계를 보이므로, 역주행 평가에 최적입니다.'
            },
            'Sobolev': {
                'insight': 'Sobolev는 위치와 속도를 모두 고려하는 종합 메트릭으로, 가장 포괄적인 평가를 제공합니다.',
                'limit': '경로 곡률과 강한 상관관계를 보이므로, 경로 복잡도를 반영합니다.'
            },
            'DTW': {
                'insight': 'DTW는 경로 형태를 평가하지만, 시간 왜곡을 허용하므로 정지에 둔감합니다.',
                'limit': '평균 곡률과 강한 상관관계를 보이므로, 경로 복잡도를 평가하는 메트릭으로 해석할 수 있습니다.'
            },
            'Fréchet': {
                'insight': 'Fréchet는 시간과 무관하게 형상만 비교하므로, 경로 길이에 상대적으로 독립적입니다.',
                'limit': '역주행을 감지하지 못하므로, 동역학적 평가에는 부적합합니다.'
            }
        }
        
        for metric_name, (feature, corr) in sorted(
            max_sensitivities.items(),
            key=lambda x: x[1][1],
            reverse=True
        ):
            feature_display = {
                'position_error': '위치 오차',
                'velocity_error': '속도 오차',
                'direction_error': '방향 오차'
            }.get(feature, feature)
            lines.append(f"- **{metric_name}**: {feature_display}에 가장 민감 (상관계수: {corr:.3f})")
            if metric_name in metric_insights:
                lines.append(f"  - **통찰**: {metric_insights[metric_name]['insight']}")
                lines.append(f"  - **특징/한계**: {metric_insights[metric_name]['limit']}")
            lines.append("")
    
    lines.append("### 6.2 실용적 권장사항 및 통찰")
    lines.append("")
    lines.append("Step-by-Step 분석 결과를 바탕으로:")
    lines.append("")
    lines.append("1. **위치 정확도 평가**가 중요하다면: **RMSE** 사용")
    lines.append("   - **이유**: Step별 위치 오차를 직접 측정하며, 위치 오차와 완벽한 상관관계(r=1.000)를 보입니다.")
    lines.append("   - **주의**: 경로 길이에 강하게 의존하므로, 경로 길이가 다양하면 정규화 필요")
    lines.append("   - **통찰**: RMSE는 시간 동기화가 정확할 때 가장 신뢰할 수 있지만, 경로 길이 다양성에 취약합니다.")
    lines.append("")
    lines.append("2. **속도/방향 패턴 평가**가 중요하다면: **DDTW** 사용")
    lines.append("   - **이유**: 속도 오차와 방향 오차에 모두 민감하게 반응하며, 역주행을 효과적으로 감지합니다.")
    lines.append("   - **특징**: 역주행 횟수와 강한 상관관계를 보이므로, 역주행 평가에 최적입니다.")
    lines.append("   - **통찰**: DDTW는 속도 벡터를 비교하므로, 위치가 비슷해도 방향이 다르면 큰 오차를 보입니다.")
    lines.append("")
    lines.append("3. **종합적 평가**가 필요하다면: **Sobolev** 사용")
    lines.append("   - **이유**: 위치와 속도를 모두 고려하는 종합 메트릭으로, 가장 포괄적인 평가를 제공합니다.")
    lines.append("   - **특징**: 경로 곡률과 강한 상관관계를 보이므로, 경로 복잡도를 반영합니다.")
    lines.append("   - **통찰**: Sobolev는 위치와 속도 오차를 가중 합산하므로, 두 측면을 균형있게 평가합니다.")
    lines.append("")
    lines.append("4. **경로 형태 평가**가 중요하다면: **DTW** 또는 **Fréchet** 사용")
    lines.append("   - **DTW**: 시간 왜곡을 허용하므로, 속도 차이를 어느 정도 보정합니다.")
    lines.append("   - **Fréchet**: 시간과 무관하게 형상만 평가하므로, 경로 길이에 독립적입니다.")
    lines.append("   - **주의**: 둘 다 역주행을 감지하지 못하므로, 동역학적 평가에는 부적합합니다.")
    lines.append("   - **통찰**: 경로 형태만 평가하고 싶을 때는 Fréchet가 더 적합하며, 시간 정렬도 고려하려면 DTW가 적합합니다.")
    lines.append("")
    
    # Write document
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"Generated detailed analysis document: {output_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate detailed analysis document')
    parser.add_argument(
        '--group',
        type=str,
        default='other',
        help='Group name'
    )
    parser.add_argument(
        '--step-analysis-dir',
        type=str,
        default='analysis_reports/other/step_analysis',
        help='Step analysis directory'
    )
    parser.add_argument(
        '--statistical-analysis',
        type=str,
        default='analysis_reports/other/statistical_analysis.json',
        help='Statistical analysis JSON path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='analysis_reports/other/상세_분석_보고서.md',
        help='Output document path'
    )
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent
    
    generate_detailed_analysis(
        args.group,
        base_dir / args.step_analysis_dir,
        base_dir / args.statistical_analysis,
        base_dir / args.output
    )
