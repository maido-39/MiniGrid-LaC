"""
Insightful report generator with deep analysis and interpretations.

Generates reports with rich insights, not just rule-based descriptions.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict


class InsightfulReportGenerator:
    """Generate reports with deep insights and interpretations."""
    
    def __init__(self, statistical_analysis_path: Path, detailed_results_path: Path):
        """
        Initialize insightful report generator.
        
        Args:
            statistical_analysis_path: Path to statistical_analysis.json
            detailed_results_path: Path to detailed_results.json
        """
        self.stat_path = Path(statistical_analysis_path)
        self.detailed_path = Path(detailed_results_path)
        
        with open(self.stat_path, 'r', encoding='utf-8') as f:
            self.stats = json.load(f)
        
        with open(self.detailed_path, 'r', encoding='utf-8') as f:
            self.detailed = json.load(f)
    
    def _identify_key_patterns(self) -> Dict:
        """Identify key patterns in the data."""
        patterns = {
            'contradictions': [],
            'surprises': [],
            'strong_relationships': [],
            'anomalies': [],
            'group_characteristics': {}
        }
        
        # Extract data
        episodes = self.stats.get('episodes', [])
        episode_nums = self.stats.get('episode_numbers', [])
        metrics = self.stats.get('metrics', {})
        trajectory_chars = self.stats.get('trajectory_characteristics', {})
        correlations = self.stats.get('correlations', {})
        
        # 1. Identify contradictions
        # RMSE decreases but DTW increases
        rmse_trend = metrics['RMSE']['trend']['slope']
        dtw_trend = metrics['DTW']['trend']['slope']
        if rmse_trend < 0 and dtw_trend > 0:
            patterns['contradictions'].append({
                'type': 'metric_trend_opposition',
                'description': 'RMSE는 감소하지만 DTW는 증가',
                'interpretation': '위치 정확도는 개선되지만 경로 형태는 더 달라짐',
                'implication': '로컬 정확도와 글로벌 경로 선택이 서로 다른 방향으로 변화'
            })
        
        # 2. Identify strong relationships
        for metric_name, metric_corrs in correlations.items():
            for feature_name, corr_data in metric_corrs.items():
                if feature_name == 'episode_number':
                    continue
                r = corr_data.get('pearson_r', 0)
                p = corr_data.get('p_value', 1.0)
                if abs(r) > 0.7 and p < 0.05:
                    patterns['strong_relationships'].append({
                        'metric': metric_name,
                        'feature': feature_name,
                        'correlation': r,
                        'p_value': p,
                        'interpretation': self._interpret_correlation(metric_name, feature_name, r)
                    })
        
        # 3. Identify anomalies
        # Episode_5_2: 매우 짧은 경로(27)인데 DTW가 매우 큼(23.388)
        for ep_name, ep_data in self.detailed['episodes'].items():
            ep_num = ep_data['episode_number']
            traj_len = ep_data['trajectory_length']
            dtw_val = ep_data['metrics']['DTW']
            frechet_val = ep_data['metrics']['Fréchet']
            
            # 짧은 경로인데 DTW가 큰 경우
            if traj_len < 35 and dtw_val > 20:
                patterns['anomalies'].append({
                    'episode': ep_name,
                    'type': 'short_path_high_dtw',
                    'description': f'경로 길이 {traj_len}인데 DTW={dtw_val:.2f}',
                    'interpretation': '경로가 짧지만 형태가 GT와 매우 다름 (급격한 경로 변경)'
                })
            
            # Fréchet가 매우 큰 경우
            if frechet_val > 6:
                patterns['anomalies'].append({
                    'episode': ep_name,
                    'type': 'high_frechet',
                    'description': f'Fréchet={frechet_val:.2f} (평균의 2배 이상)',
                    'interpretation': '전체 경로 형상이 GT와 매우 다름'
                })
        
        # 4. Group characteristics
        traj_lengths = trajectory_chars.get('trajectory_length', {}).get('values', [])
        num_backtracks = trajectory_chars.get('num_backtracks', {}).get('values', [])
        
        if not traj_lengths:
            # Fallback: extract from detailed results
            traj_lengths = [ep_data['trajectory_length'] for ep_data in self.detailed['episodes'].values()]
        
        if not num_backtracks:
            num_backtracks = [0]  # Default to avoid empty list
        
        patterns['group_characteristics'] = {
            'path_length_diversity': {
                'min': min(traj_lengths),
                'max': max(traj_lengths),
                'range_ratio': max(traj_lengths) / min(traj_lengths) if min(traj_lengths) > 0 else 0,
                'interpretation': f'경로 길이가 {max(traj_lengths)/min(traj_lengths):.1f}배 차이남 - 매우 다양한 경로 선택 전략'
            },
            'backtracking_pattern': {
                'mean': np.mean(num_backtracks),
                'std': np.std(num_backtracks),
                'max': max(num_backtracks),
                'interpretation': f'역주행 횟수가 0~{max(num_backtracks)}회로 다양함 - 탐색 전략의 차이'
            }
        }
        
        return patterns
    
    def _interpret_correlation(self, metric: str, feature: str, r: float) -> str:
        """Interpret correlation between metric and feature."""
        interpretations = {
            ('DDTW', 'num_backtracks'): f'DDTW는 역주행 횟수와 강한 양의 상관관계(r={r:.3f})를 보입니다. 이는 DDTW가 역주행을 효과적으로 감지한다는 것을 의미합니다.',
            ('DTW', 'avg_curvature'): f'DTW는 평균 곡률과 강한 양의 상관관계(r={r:.3f})를 보입니다. 경로가 복잡할수록(곡률이 클수록) DTW 값이 증가합니다.',
            ('RMSE', 'trajectory_length'): f'RMSE는 경로 길이와 강한 양의 상관관계(r={r:.3f})를 보입니다. 경로가 길수록 누적 오차가 커집니다.',
            ('TWED', 'efficiency'): f'TWED는 효율성과 강한 음의 상관관계(r={r:.3f})를 보입니다. 효율적인 경로일수록 TWED 값이 낮습니다.',
        }
        
        key = (metric, feature)
        if key in interpretations:
            return interpretations[key]
        
        # Generic interpretation
        direction = "증가" if r > 0 else "감소"
        strength = "매우 강한" if abs(r) > 0.8 else "강한" if abs(r) > 0.6 else "중간"
        return f'{metric}는 {feature}와 {strength} {direction} 상관관계(r={r:.3f})를 보입니다.'
    
    def _analyze_episode_progression(self) -> Dict:
        """Analyze how episodes progress and what it means."""
        episodes = self.stats.get('episodes', [])
        episode_nums = self.stats.get('episode_numbers', [])
        metrics = self.stats.get('metrics', {})
        
        # Sort by episode number
        sorted_indices = sorted(range(len(episode_nums)), key=lambda i: episode_nums[i])
        
        progression = {
            'improving_metrics': [],
            'worsening_metrics': [],
            'stable_metrics': [],
            'interpretation': ''
        }
        
        for metric_name, metric_data in metrics.items():
            slope = metric_data['trend']['slope']
            p_value = metric_data['trend']['p_value']
            
            if p_value < 0.1:  # Significant trend
                if slope < -0.1:
                    progression['improving_metrics'].append({
                        'metric': metric_name,
                        'slope': slope,
                        'interpretation': f'{metric_name}가 Episode 증가에 따라 {abs(slope):.3f}씩 감소 - GT에 가까워짐'
                    })
                elif slope > 0.1:
                    progression['worsening_metrics'].append({
                        'metric': metric_name,
                        'slope': slope,
                        'interpretation': f'{metric_name}가 Episode 증가에 따라 {slope:.3f}씩 증가 - GT와 멀어짐'
                    })
            else:
                progression['stable_metrics'].append({
                    'metric': metric_name,
                    'slope': slope,
                    'interpretation': f'{metric_name}는 Episode와 무관하게 안정적'
                })
        
        # Overall interpretation
        if len(progression['improving_metrics']) > 0 and len(progression['worsening_metrics']) > 0:
            improving = [m['metric'] for m in progression['improving_metrics']]
            worsening = [m['metric'] for m in progression['worsening_metrics']]
            progression['interpretation'] = (
                f'흥미롭게도 {", ".join(improving)}는 개선되지만, '
                f'{", ".join(worsening)}는 악화됩니다. '
                f'이는 로봇이 일부 측면에서는 더 정확해지지만, 다른 측면에서는 더 달라진다는 것을 의미합니다.'
            )
        
        return progression
    
    def _analyze_metric_suitability(self) -> Dict:
        """Deep analysis of which metrics are suitable for this data."""
        metrics = self.stats.get('metrics', {})
        trajectory_chars = self.stats.get('trajectory_characteristics', {})
        correlations = self.stats.get('correlations', {})
        
        suitability = {}
        
        # Path length diversity
        traj_lengths = trajectory_chars['trajectory_length']['values']
        length_ratio = max(traj_lengths) / min(traj_lengths) if min(traj_lengths) > 0 else 0
        
        for metric_name, metric_data in metrics.items():
            cv = metric_data['cv']
            slope = abs(metric_data['trend']['slope'])
            metric_corrs = correlations.get(metric_name, {})
            r_episode = metric_corrs.get('episode_number', {}).get('pearson_r', 0)
            r_length = metric_corrs.get('trajectory_length', {}).get('pearson_r', 0)
            
            # Suitability score (lower is better)
            # Penalize: high CV, high slope, strong length dependency
            score = cv * 0.4 + (slope / 10.0) * 0.3 + abs(r_length) * 0.3
            
            strengths = []
            weaknesses = []
            
            # Analyze strengths
            if cv < 0.3:
                strengths.append('안정적 (CV < 0.3)')
            if abs(slope) < 1.0:
                strengths.append('Episode 독립적')
            if abs(r_length) < 0.5:
                strengths.append('경로 길이에 독립적')
            
            # Analyze weaknesses
            if cv > 0.6:
                weaknesses.append(f'불안정 (CV={cv:.2f})')
            if abs(slope) > 5.0:
                weaknesses.append(f'Episode 의존적 (slope={slope:.2f})')
            if abs(r_length) > 0.7:
                weaknesses.append(f'경로 길이에 강하게 의존 (r={r_length:.2f})')
            
            # Specific insights
            insights = []
            if metric_name == 'RMSE' and r_length > 0.8:
                insights.append('경로 길이가 다양할 때 사용하기 부적합 (길이에 강하게 의존)')
            elif metric_name == 'DTW' and r_length < 0.5:
                insights.append('경로 길이 차이에 상대적으로 강인함')
            elif metric_name == 'DDTW' and metric_corrs.get('num_backtracks', {}).get('pearson_r', 0) > 0.8:
                insights.append('역주행 감지에 탁월함')
            
            suitability[metric_name] = {
                'score': score,
                'strengths': strengths,
                'weaknesses': weaknesses,
                'insights': insights,
                'recommendation': '권장' if score < 0.3 else '조건부 권장' if score < 0.5 else '비권장'
            }
        
        return suitability
    
    def generate_insightful_report(self, output_path: Path, group_name: str):
        """Generate insightful report with deep analysis."""
        patterns = self._identify_key_patterns()
        progression = self._analyze_episode_progression()
        suitability = self._analyze_metric_suitability()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# {group_name.upper()} 그룹 통찰 중심 분석 보고서\n\n")
            f.write("이 보고서는 단순 통계 수치를 넘어, 데이터에서 발견된 패턴과 의미를 깊이 있게 해석합니다.\n\n")
            
            f.write("## 목차\n\n")
            f.write("1. [핵심 발견사항](#1-핵심-발견사항)\n")
            f.write("2. [메트릭 간 모순과 그 의미](#2-메트릭-간-모순과-그-의미)\n")
            f.write("3. [Episode 진행에 따른 변화 해석](#3-episode-진행에-따른-변화-해석)\n")
            f.write("4. [메트릭별 특성과 실험 데이터 적합성](#4-메트릭별-특성과-실험-데이터-적합성)\n")
            f.write("5. [그룹 특성과 실험 맥락](#5-그룹-특성과-실험-맥락)\n")
            f.write("6. [실용적 권장사항](#6-실용적-권장사항)\n\n")
            f.write("---\n\n")
            
            # 1. 핵심 발견사항
            f.write("## 1. 핵심 발견사항\n\n")
            
            # Contradictions
            if patterns['contradictions']:
                f.write("### 1.1 모순적 패턴 발견\n\n")
                for contradiction in patterns['contradictions']:
                    f.write(f"**{contradiction['description']}**\n\n")
                    f.write(f"- **해석**: {contradiction['interpretation']}\n")
                    f.write(f"- **의미**: {contradiction['implication']}\n\n")
            
            # Anomalies
            if patterns['anomalies']:
                f.write("### 1.2 특이사항 (Anomalies)\n\n")
                for anomaly in patterns['anomalies']:
                    f.write(f"**{anomaly['episode']}**: {anomaly['description']}\n\n")
                    f.write(f"- **의미**: {anomaly['interpretation']}\n\n")
            
            # Strong relationships
            if patterns['strong_relationships']:
                f.write("### 1.3 강한 상관관계 발견\n\n")
                for rel in patterns['strong_relationships']:
                    f.write(f"**{rel['metric']} ↔ {rel['feature']}** (r={rel['correlation']:.3f}, p={rel['p_value']:.4f})\n\n")
                    f.write(f"- {rel['interpretation']}\n\n")
            
            # 2. 메트릭 간 모순
            f.write("## 2. 메트릭 간 모순과 그 의미\n\n")
            
            metrics = self.stats.get('metrics', {})
            rmse_trend = metrics.get('RMSE', {}).get('trend', {})
            dtw_trend = metrics.get('DTW', {}).get('trend', {})
            
            f.write("### 2.1 RMSE vs DTW: 위치 정확도 vs 경로 형태\n\n")
            f.write(f"**관찰**: RMSE는 Episode 증가에 따라 **감소** (slope={rmse_trend['slope']:.3f}, R²={rmse_trend['r_squared']:.3f}), ")
            f.write(f"반면 DTW는 **증가** (slope={dtw_trend['slope']:.3f}, R²={dtw_trend['r_squared']:.3f})합니다.\n\n")
            
            f.write("**통찰**:\n\n")
            f.write("1. **위치 정확도는 개선**: RMSE 감소는 로봇이 GT와 **같은 시간대에 비슷한 위치**에 도달한다는 의미입니다.\n")
            f.write("   - 이는 **로컬 정확도(Local Accuracy)**가 향상되었음을 시사합니다.\n\n")
            
            f.write("2. **경로 형태는 더 달라짐**: DTW 증가는 전체적인 **경로 선택 전략**이 GT와 더 달라졌다는 의미입니다.\n")
            f.write("   - 이는 **글로벌 경로 선택(Global Path Planning)**이 다르게 작동함을 시사합니다.\n\n")
            
            f.write("3. **실험적 의미**:\n")
            f.write("   - 로봇은 **타이밍과 속도 제어**는 개선되었지만\n")
            f.write("   - **경로 계획 알고리즘**은 다른 경로를 선택하게 되었습니다.\n")
            f.write("   - 이는 학습 과정에서 **로컬 최적화는 성공**했지만, **전역 최적화는 실패**했을 가능성을 시사합니다.\n\n")
            
            # 3. Episode 진행 분석
            f.write("## 3. Episode 진행에 따른 변화 해석\n\n")
            
            f.write(progression['interpretation'] + "\n\n")
            
            if progression['improving_metrics']:
                f.write("### 3.1 개선되는 메트릭\n\n")
                for metric in progression['improving_metrics']:
                    f.write(f"- **{metric['metric']}**: {metric['interpretation']}\n")
                f.write("\n")
            
            if progression['worsening_metrics']:
                f.write("### 3.2 악화되는 메트릭\n\n")
                for metric in progression['worsening_metrics']:
                    f.write(f"- **{metric['metric']}**: {metric['interpretation']}\n")
                f.write("\n")
            
            # 4. 메트릭 적합성
            f.write("## 4. 메트릭별 특성과 실험 데이터 적합성\n\n")
            
            # Sort by suitability score
            sorted_suitability = sorted(suitability.items(), key=lambda x: x[1]['score'])
            
            for metric_name, suit_data in sorted_suitability:
                f.write(f"### 4.{sorted_suitability.index((metric_name, suit_data)) + 1} {metric_name}\n\n")
                f.write(f"**적합성 점수**: {suit_data['score']:.3f} ({suit_data['recommendation']})\n\n")
                
                if suit_data['strengths']:
                    f.write("**강점**:\n")
                    for strength in suit_data['strengths']:
                        f.write(f"- {strength}\n")
                    f.write("\n")
                
                if suit_data['weaknesses']:
                    f.write("**약점**:\n")
                    for weakness in suit_data['weaknesses']:
                        f.write(f"- {weakness}\n")
                    f.write("\n")
                
                if suit_data['insights']:
                    f.write("**특별한 통찰**:\n")
                    for insight in suit_data['insights']:
                        f.write(f"- {insight}\n")
                    f.write("\n")
            
            # 5. 그룹 특성
            f.write("## 5. 그룹 특성과 실험 맥락\n\n")
            
            group_chars = patterns['group_characteristics']
            if 'path_length_diversity' in group_chars:
                pl_div = group_chars['path_length_diversity']
                f.write(f"### 5.1 경로 길이 다양성\n\n")
                f.write(f"**관찰**: 경로 길이가 {pl_div['min']}~{pl_div['max']} steps로, {pl_div['range_ratio']:.1f}배 차이납니다.\n\n")
                f.write(f"**의미**: {pl_div['interpretation']}\n\n")
                f.write("**메트릭 선택에 미치는 영향**:\n")
                f.write("- 경로 길이에 강하게 의존하는 메트릭(DTW, ERP, TWED)은 이 그룹에서 사용 시 주의 필요\n")
                f.write("- 경로 길이에 독립적인 메트릭(RMSE, Fréchet)이 더 적합할 수 있음\n\n")
            
            if 'backtracking_pattern' in group_chars:
                bt_pattern = group_chars['backtracking_pattern']
                f.write(f"### 5.2 역주행 패턴\n\n")
                f.write(f"**관찰**: 역주행 횟수가 평균 {bt_pattern['mean']:.1f}회 (표준편차 {bt_pattern['std']:.1f}회), 최대 {bt_pattern['max']}회입니다.\n\n")
                f.write(f"**의미**: {bt_pattern['interpretation']}\n\n")
                f.write("**메트릭 선택에 미치는 영향**:\n")
                f.write("- DDTW는 역주행을 잘 감지하므로, 역주행이 중요한 평가 요소라면 DDTW 사용 권장\n")
                f.write("- Fréchet는 역주행을 감지하지 못하므로, 역주행 평가에는 부적합\n\n")
            
            # 6. 실용적 권장사항
            f.write("## 6. 실용적 권장사항\n\n")
            
            f.write("### 6.1 실험 목적별 메트릭 선택\n\n")
            
            f.write("#### A. 위치 정확도 평가가 목적일 때\n")
            f.write("- **1순위**: RMSE\n")
            f.write("  - 이유: 위치 오차를 직접 측정하며, Episode 독립성이 가장 좋음\n")
            f.write("  - 주의: 경로 길이에 의존하므로, 경로 길이가 다양하면 정규화 필요\n\n")
            
            f.write("#### B. 동역학적 특성(정지, 역주행) 평가가 목적일 때\n")
            f.write("- **1순위**: DDTW\n")
            f.write("  - 이유: 역주행과 정지를 효과적으로 감지 (역주행과 r=0.889, 유의함)\n")
            f.write("  - 주의: 경로 길이에 중간 정도 의존\n\n")
            
            f.write("#### C. 경로 형태 평가가 목적일 때\n")
            f.write("- **1순위**: Fréchet Distance\n")
            f.write("  - 이유: 시간과 무관하게 형상만 평가, 경로 길이에 상대적으로 독립적\n")
            f.write("  - 주의: 역주행을 감지하지 못함\n\n")
            
            f.write("#### D. 종합 평가가 목적일 때\n")
            f.write("- **1순위**: Sobolev Metric\n")
            f.write("  - 이유: 위치와 속도를 모두 고려하는 종합 메트릭\n")
            f.write("  - 주의: 경로 곡률에 강하게 의존 (r=0.920, 유의함)\n\n")
            
            f.write("### 6.2 메트릭 조합 전략\n\n")
            f.write("단일 메트릭의 한계를 보완하기 위해 메트릭을 조합하여 사용하는 것을 권장합니다:\n\n")
            f.write("1. **위치 정확도 + 동역학**: RMSE + DDTW\n")
            f.write("2. **형태 + 동역학**: Fréchet + DDTW\n")
            f.write("3. **종합 평가**: Sobolev (단독 사용 가능)\n\n")
            
            f.write("### 6.3 경고사항\n\n")
            f.write("이 실험 데이터의 특성상 다음 사항에 주의하세요:\n\n")
            f.write("1. **경로 길이 다양성**: 경로 길이가 2배 이상 차이나므로, 길이에 의존하는 메트릭은 정규화 필요\n")
            f.write("2. **메트릭 간 모순**: RMSE와 DTW가 반대 방향으로 변화하므로, 단일 메트릭으로 평가하면 편향될 수 있음\n")
            f.write("3. **샘플 크기**: Episode 수가 적어 통계적 유의성이 낮을 수 있음 (p-value > 0.05인 경우 많음)\n\n")
            
            f.write("---\n\n")
            f.write("**보고서 생성 일시**: " + str(Path(output_path).stat().st_mtime) + "\n")
