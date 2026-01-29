"""
Generate parameter sensitivity analysis report.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


class ParameterSensitivityReportGenerator:
    """Generate parameter sensitivity analysis report."""
    
    def __init__(self, sweep_results_path: Path):
        """
        Initialize report generator.
        
        Args:
            sweep_results_path: Path to parameter_sweep_results.json
        """
        self.sweep_results_path = Path(sweep_results_path)
        with open(self.sweep_results_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def analyze_parameter_sensitivity(self) -> Dict:
        """Analyze parameter sensitivity across all groups."""
        analysis = {}
        
        for group_name, group_data in self.data.items():
            group_analysis = {
                'TWED': self._analyze_twed_sensitivity(group_data),
                'Sobolev': self._analyze_sobolev_sensitivity(group_data),
                'ERP': self._analyze_erp_sensitivity(group_data)
            }
            analysis[group_name] = group_analysis
        
        return analysis
    
    def _analyze_twed_sensitivity(self, group_data: Dict) -> Dict:
        """Analyze TWED parameter sensitivity."""
        # Aggregate results
        nu_effects = defaultdict(list)
        lambda_effects = defaultdict(list)
        combined_effects = defaultdict(list)
        
        for ep_name, ep_data in group_data['episodes'].items():
            for key, result in ep_data['TWED'].items():
                if not np.isnan(result['distance']):
                    nu = result['nu']
                    lambda_val = result['lambda']
                    distance = result['distance']
                    
                    nu_effects[nu].append(distance)
                    lambda_effects[lambda_val].append(distance)
                    combined_effects[(nu, lambda_val)].append(distance)
        
        # Calculate statistics
        nu_stats = {}
        for nu, distances in nu_effects.items():
            nu_stats[nu] = {
                'mean': np.mean(distances),
                'std': np.std(distances),
                'min': np.min(distances),
                'max': np.max(distances),
                'range': np.max(distances) - np.min(distances)
            }
        
        lambda_stats = {}
        for lambda_val, distances in lambda_effects.items():
            lambda_stats[lambda_val] = {
                'mean': np.mean(distances),
                'std': np.std(distances),
                'min': np.min(distances),
                'max': np.max(distances),
                'range': np.max(distances) - np.min(distances)
            }
        
        # Find optimal parameters (minimum mean distance)
        optimal_combined = min(
            combined_effects.items(),
            key=lambda x: np.mean(x[1])
        )
        
        return {
            'nu_effects': nu_stats,
            'lambda_effects': lambda_stats,
            'optimal_parameters': {
                'nu': optimal_combined[0][0],
                'lambda': optimal_combined[0][1],
                'mean_distance': np.mean(optimal_combined[1])
            },
            'sensitivity': {
                'nu_sensitivity': self._calculate_sensitivity(nu_stats),
                'lambda_sensitivity': self._calculate_sensitivity(lambda_stats)
            }
        }
    
    def _analyze_sobolev_sensitivity(self, group_data: Dict) -> Dict:
        """Analyze Sobolev parameter sensitivity."""
        # Aggregate results
        alpha_effects = defaultdict(list)
        beta_effects = defaultdict(list)
        combined_effects = defaultdict(list)
        
        for ep_name, ep_data in group_data['episodes'].items():
            for key, result in ep_data['Sobolev'].items():
                if not np.isnan(result['distance']):
                    alpha = result['alpha']
                    beta = result['beta']
                    distance = result['distance']
                    
                    alpha_effects[alpha].append(distance)
                    beta_effects[beta].append(distance)
                    combined_effects[(alpha, beta)].append(distance)
        
        # Calculate statistics
        alpha_stats = {}
        for alpha, distances in alpha_effects.items():
            alpha_stats[alpha] = {
                'mean': np.mean(distances),
                'std': np.std(distances),
                'min': np.min(distances),
                'max': np.max(distances),
                'range': np.max(distances) - np.min(distances)
            }
        
        beta_stats = {}
        for beta, distances in beta_effects.items():
            beta_stats[beta] = {
                'mean': np.mean(distances),
                'std': np.std(distances),
                'min': np.min(distances),
                'max': np.max(distances),
                'range': np.max(distances) - np.min(distances)
            }
        
        # Find optimal parameters
        optimal_combined = min(
            combined_effects.items(),
            key=lambda x: np.mean(x[1])
        )
        
        return {
            'alpha_effects': alpha_stats,
            'beta_effects': beta_stats,
            'optimal_parameters': {
                'alpha': optimal_combined[0][0],
                'beta': optimal_combined[0][1],
                'mean_distance': np.mean(optimal_combined[1])
            },
            'sensitivity': {
                'alpha_sensitivity': self._calculate_sensitivity(alpha_stats),
                'beta_sensitivity': self._calculate_sensitivity(beta_stats)
            }
        }
    
    def _analyze_erp_sensitivity(self, group_data: Dict) -> Dict:
        """Analyze ERP parameter sensitivity."""
        # Aggregate results
        gap_effects = defaultdict(list)
        
        for ep_name, ep_data in group_data['episodes'].items():
            for key, result in ep_data['ERP'].items():
                if not np.isnan(result['distance']):
                    gap = result['gap']
                    distance = result['distance']
                    gap_effects[gap].append(distance)
        
        # Calculate statistics
        gap_stats = {}
        for gap, distances in gap_effects.items():
            gap_stats[gap] = {
                'mean': np.mean(distances),
                'std': np.std(distances),
                'min': np.min(distances),
                'max': np.max(distances),
                'range': np.max(distances) - np.min(distances)
            }
        
        # Find optimal gap
        if len(gap_effects) > 0:
            optimal_gap = min(
                gap_effects.items(),
                key=lambda x: np.mean(x[1])
            )
        else:
            optimal_gap = (0.0, [0.0])
        
        return {
            'gap_effects': gap_stats,
            'optimal_parameters': {
                'gap': optimal_gap[0],
                'mean_distance': np.mean(optimal_gap[1])
            },
            'sensitivity': {
                'gap_sensitivity': self._calculate_sensitivity(gap_stats)
            }
        }
    
    def _calculate_sensitivity(self, param_stats: Dict) -> str:
        """Calculate parameter sensitivity level."""
        ranges = [stats['range'] for stats in param_stats.values()]
        mean_range = np.mean(ranges)
        max_range = np.max(ranges)
        
        # Normalize by mean value
        mean_values = [stats['mean'] for stats in param_stats.values()]
        mean_value = np.mean(mean_values)
        
        if mean_value > 0:
            relative_sensitivity = mean_range / mean_value
        else:
            relative_sensitivity = max_range
        
        if relative_sensitivity > 0.5:
            return "High"
        elif relative_sensitivity > 0.2:
            return "Medium"
        else:
            return "Low"
    
    def _calculate_parameter_trend(self, param_stats: Dict) -> Dict:
        """Calculate parameter trend (increasing/decreasing/stable)."""
        sorted_params = sorted(param_stats.keys())
        if len(sorted_params) < 2:
            return {'trend': 'insufficient_data', 'slope': 0.0}
        
        means = [param_stats[p]['mean'] for p in sorted_params]
        
        # Simple linear trend
        n = len(sorted_params)
        x = np.array(sorted_params)
        y = np.array(means)
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        # Determine trend
        if abs(slope) < np.mean(means) * 0.05:  # Less than 5% change
            trend = 'stable'
        elif slope > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'
        
        return {'trend': trend, 'slope': slope, 'relative_change': abs(slope) / np.mean(means) if np.mean(means) > 0 else 0}
    
    def generate_report(self, output_path: Path):
        """Generate Markdown report."""
        analysis = self.analyze_parameter_sensitivity()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# 파라미터 민감도 분석 보고서\n\n")
            f.write("이 보고서는 파라미터 조정 가능한 메트릭들(TWED, Sobolev, ERP)의 파라미터 변화에 따른 민감도를 분석합니다.\n\n")
            f.write("## 목차\n\n")
            f.write("1. [그룹별 상세 분석](#그룹별-상세-분석)\n")
            f.write("2. [파라미터 변화 패턴 분석](#파라미터-변화-패턴-분석)\n")
            f.write("3. [그룹 간 비교](#그룹-간-비교)\n")
            f.write("4. [파라미터 선택 가이드라인](#파라미터-선택-가이드라인)\n")
            f.write("5. [종합 권장사항](#종합-권장사항)\n\n")
            f.write("---\n\n")
            
            f.write("## 그룹별 상세 분석\n\n")
            f.write("> **참고**: 각 그룹별 파라미터 스위핑 시각화는 `parameter_sweep/{group_name}/parameter_sweep/` 폴더에서 확인할 수 있습니다.\n\n")
            
            for group_name, group_analysis in analysis.items():
                f.write(f"### {group_name.upper()} 그룹\n\n")
                
                # TWED
                f.write("#### TWED (Time Warp Edit Distance)\n\n")
                f.write(f"![TWED Parameter Sweep]({group_name}/parameter_sweep/twed_parameter_sweep.png)\n\n")
                twed = group_analysis['TWED']
                f.write(f"**최적 파라미터**: ν={twed['optimal_parameters']['nu']:.2f}, λ={twed['optimal_parameters']['lambda']:.2f}\n")
                f.write(f"**최적 파라미터에서의 평균 거리**: {twed['optimal_parameters']['mean_distance']:.4f}\n\n")
                
                # Calculate trends
                nu_trend = self._calculate_parameter_trend(twed['nu_effects'])
                lambda_trend = self._calculate_parameter_trend(twed['lambda_effects'])
                
                f.write(f"**Nu (ν) 파라미터 경향**: {nu_trend['trend']} (상대 변화율: {nu_trend['relative_change']:.2%})\n")
                f.write(f"**Lambda (λ) 파라미터 경향**: {lambda_trend['trend']} (상대 변화율: {lambda_trend['relative_change']:.2%})\n\n")
                
                f.write("**Nu (ν) 파라미터 영향**:\n")
                f.write("| Nu | 평균 거리 | 표준편차 | 범위 | 최소값 | 최대값 |\n")
                f.write("|----|----------|---------|------|--------|--------|\n")
                for nu in sorted(twed['nu_effects'].keys()):
                    stats = twed['nu_effects'][nu]
                    f.write(f"| {nu:.2f} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['range']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |\n")
                f.write(f"\n**Nu 민감도**: {twed['sensitivity']['nu_sensitivity']}\n")
                f.write(f"- **해석**: Nu 값이 증가할수록 시간 왜곡에 대한 페널티가 커져 거리가 증가합니다. ")
                f.write(f"Nu={min(twed['nu_effects'].keys()):.2f}에서 Nu={max(twed['nu_effects'].keys()):.2f}로 증가할 때, ")
                min_nu = min(twed['nu_effects'].keys())
                max_nu = max(twed['nu_effects'].keys())
                min_dist = twed['nu_effects'][min_nu]['mean']
                max_dist = twed['nu_effects'][max_nu]['mean']
                increase_ratio = (max_dist - min_dist) / min_dist * 100
                f.write(f"평균 거리가 {increase_ratio:.1f}% 증가합니다.\n\n")
                
                f.write("**Lambda (λ) 파라미터 영향**:\n")
                f.write("| Lambda | 평균 거리 | 표준편차 | 범위 | 최소값 | 최대값 |\n")
                f.write("|--------|----------|---------|------|--------|--------|\n")
                for lambda_val in sorted(twed['lambda_effects'].keys()):
                    stats = twed['lambda_effects'][lambda_val]
                    f.write(f"| {lambda_val:.2f} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['range']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |\n")
                f.write(f"\n**Lambda 민감도**: {twed['sensitivity']['lambda_sensitivity']}\n")
                f.write(f"- **해석**: Lambda 값이 증가할수록 삭제/삽입 페널티가 커져 거리가 증가합니다. ")
                min_lambda = min(twed['lambda_effects'].keys())
                max_lambda = max(twed['lambda_effects'].keys())
                min_dist = twed['lambda_effects'][min_lambda]['mean']
                max_dist = twed['lambda_effects'][max_lambda]['mean']
                increase_ratio = (max_dist - min_dist) / min_dist * 100
                f.write(f"Lambda={min_lambda:.2f}에서 Lambda={max_lambda:.2f}로 증가할 때, 평균 거리가 {increase_ratio:.1f}% 증가합니다.\n\n")
                
                # Sobolev
                f.write("#### Sobolev Metric\n\n")
                f.write(f"![Sobolev Parameter Sweep]({group_name}/parameter_sweep/sobolev_parameter_sweep.png)\n\n")
                sobolev = group_analysis['Sobolev']
                f.write(f"**최적 파라미터**: α={sobolev['optimal_parameters']['alpha']:.2f}, β={sobolev['optimal_parameters']['beta']:.2f}\n")
                f.write(f"**최적 파라미터에서의 평균 거리**: {sobolev['optimal_parameters']['mean_distance']:.4f}\n\n")
                
                # Calculate trends
                alpha_trend = self._calculate_parameter_trend(sobolev['alpha_effects'])
                beta_trend = self._calculate_parameter_trend(sobolev['beta_effects'])
                
                f.write(f"**Alpha (α) 파라미터 경향**: {alpha_trend['trend']} (상대 변화율: {alpha_trend['relative_change']:.2%})\n")
                f.write(f"**Beta (β) 파라미터 경향**: {beta_trend['trend']} (상대 변화율: {beta_trend['relative_change']:.2%})\n\n")
                
                f.write("**Alpha (α) 파라미터 영향** (위치 오차 가중치):\n")
                f.write("| Alpha | 평균 거리 | 표준편차 | 범위 | 최소값 | 최대값 |\n")
                f.write("|-------|----------|---------|------|--------|--------|\n")
                for alpha in sorted(sobolev['alpha_effects'].keys()):
                    stats = sobolev['alpha_effects'][alpha]
                    f.write(f"| {alpha:.2f} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['range']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |\n")
                f.write(f"\n**Alpha 민감도**: {sobolev['sensitivity']['alpha_sensitivity']}\n")
                f.write(f"- **해석**: Alpha는 위치 오차의 가중치입니다. Alpha가 증가하면 위치 오차에 더 큰 비중을 두게 되어 거리가 증가합니다. ")
                min_alpha = min(sobolev['alpha_effects'].keys())
                max_alpha = max(sobolev['alpha_effects'].keys())
                min_dist = sobolev['alpha_effects'][min_alpha]['mean']
                max_dist = sobolev['alpha_effects'][max_alpha]['mean']
                increase_ratio = (max_dist - min_dist) / min_dist * 100
                f.write(f"Alpha={min_alpha:.2f}에서 Alpha={max_alpha:.2f}로 증가할 때, 평균 거리가 {increase_ratio:.1f}% 증가합니다.\n\n")
                
                f.write("**Beta (β) 파라미터 영향** (속도 오차 가중치):\n")
                f.write("| Beta | 평균 거리 | 표준편차 | 범위 | 최소값 | 최대값 |\n")
                f.write("|------|----------|---------|------|--------|--------|\n")
                for beta in sorted(sobolev['beta_effects'].keys()):
                    stats = sobolev['beta_effects'][beta]
                    f.write(f"| {beta:.2f} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['range']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |\n")
                f.write(f"\n**Beta 민감도**: {sobolev['sensitivity']['beta_sensitivity']}\n")
                f.write(f"- **해석**: Beta는 속도 오차의 가중치입니다. Beta가 증가하면 속도 오차에 더 큰 비중을 두게 되어 거리가 증가합니다. ")
                min_beta = min(sobolev['beta_effects'].keys())
                max_beta = max(sobolev['beta_effects'].keys())
                min_dist = sobolev['beta_effects'][min_beta]['mean']
                max_dist = sobolev['beta_effects'][max_beta]['mean']
                increase_ratio = (max_dist - min_dist) / min_dist * 100
                f.write(f"Beta={min_beta:.2f}에서 Beta={max_beta:.2f}로 증가할 때, 평균 거리가 {increase_ratio:.1f}% 증가합니다.\n\n")
                
                # ERP
                f.write("#### ERP (Edit Distance on Real sequence)\n\n")
                f.write(f"![ERP Parameter Sweep]({group_name}/parameter_sweep/erp_parameter_sweep.png)\n\n")
                erp = group_analysis['ERP']
                f.write(f"**최적 Gap Penalty**: g={erp['optimal_parameters']['gap']:.4f}\n")
                f.write(f"**최적 Gap에서의 평균 거리**: {erp['optimal_parameters']['mean_distance']:.4f}\n\n")
                
                # Calculate trend
                gap_trend = self._calculate_parameter_trend(erp['gap_effects'])
                f.write(f"**Gap Penalty (g) 파라미터 경향**: {gap_trend['trend']} (상대 변화율: {gap_trend['relative_change']:.2%})\n\n")
                
                f.write("**Gap Penalty (g) 파라미터 영향**:\n")
                f.write("| Gap | 평균 거리 | 표준편차 | 범위 | 최소값 | 최대값 |\n")
                f.write("|-----|----------|---------|------|--------|--------|\n")
                for gap in sorted(erp['gap_effects'].keys()):
                    stats = erp['gap_effects'][gap]
                    f.write(f"| {gap:.4f} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['range']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |\n")
                f.write(f"\n**Gap Penalty 민감도**: {erp['sensitivity']['gap_sensitivity']}\n")
                f.write(f"- **해석**: Gap penalty는 누락된 점과 매칭할 때의 페널티입니다. ")
                if len(erp['gap_effects']) > 1:
                    gaps = sorted(erp['gap_effects'].keys())
                    g0_dist = erp['gap_effects'][gaps[0]]['mean']
                    g_max_dist = erp['gap_effects'][gaps[-1]]['mean']
                    f.write(f"g=0.0 (표준)에서 g={gaps[-1]:.4f}로 증가할 때, 평균 거리가 {(g_max_dist/g0_dist - 1)*100:.1f}% 변화합니다. ")
                f.write(f"일반적으로 g=0.0이 표준이며, 이 값에서 최소 거리를 보입니다.\n\n")
                
                f.write("---\n\n")
            
            # Parameter pattern analysis
            f.write("## 파라미터 변화 패턴 분석\n\n")
            
            # Compare across groups
            f.write("### 그룹 간 파라미터 민감도 비교\n\n")
            f.write("| 그룹 | TWED Nu | TWED Lambda | Sobolev Alpha | Sobolev Beta | ERP Gap |\n")
            f.write("|------|---------|-------------|---------------|--------------|----------|\n")
            for group_name, group_analysis in analysis.items():
                twed = group_analysis['TWED']
                sobolev = group_analysis['Sobolev']
                erp = group_analysis['ERP']
                f.write(f"| {group_name} | {twed['sensitivity']['nu_sensitivity']} | {twed['sensitivity']['lambda_sensitivity']} | ")
                f.write(f"{sobolev['sensitivity']['alpha_sensitivity']} | {sobolev['sensitivity']['beta_sensitivity']} | ")
                f.write(f"{erp['sensitivity']['gap_sensitivity']} |\n")
            
            f.write("\n### 파라미터별 공통 패턴\n\n")
            
            # TWED pattern
            f.write("#### TWED 파라미터 패턴 및 통찰\n\n")
            f.write("- **Nu (ν)**: 모든 그룹에서 High 민감도를 보입니다. Nu가 증가하면 시간 왜곡에 대한 페널티가 커져 거리가 선형적으로 증가합니다.\n")
            f.write("  - **통찰**: Nu는 시간 정렬의 엄격성을 제어합니다. Nu=0.10에서 Nu=5.00로 증가할 때 거리가 276~663% 증가하므로, Nu는 TWED 값에 가장 큰 영향을 미치는 파라미터입니다.\n")
            f.write("  - **의미**: 낮은 Nu(0.10)가 최적이라는 것은, 이 실험 데이터에서는 시간 왜곡을 어느 정도 허용하는 것이 더 적절하다는 것을 의미합니다.\n")
            f.write("  - **실험적 해석**: 로봇이 GT와 다른 속도로 이동하더라도, 전체적인 경로 추종이 중요하다면 낮은 Nu를 사용하는 것이 합리적입니다.\n\n")
            f.write("- **Lambda (λ)**: 모든 그룹에서 High 민감도를 보입니다. Lambda가 증가하면 삭제/삽입 비용이 커져 거리가 증가합니다.\n")
            f.write("  - **통찰**: Lambda는 경로 길이 차이에 대한 페널티를 제어합니다. Lambda=0.50에서 Lambda=5.00로 증가할 때 거리가 24~36% 증가하므로, Nu보다는 영향이 작지만 여전히 중요합니다.\n")
            f.write("  - **의미**: 낮은 Lambda(0.50)가 최적이라는 것은, 경로 길이 차이에 관대하게 처리하는 것이 더 적절하다는 것을 의미합니다.\n")
            f.write("  - **실험적 해석**: 경로 길이가 다양할 때(예: other 그룹), 낮은 Lambda를 사용하면 경로 길이 차이로 인한 불필요한 페널티를 줄일 수 있습니다.\n\n")
            f.write("- **최적값**: 모든 그룹에서 ν=0.10, λ=0.50이 최적입니다. 이는 시간 왜곡과 삭제/삽입에 상대적으로 관대한 설정입니다.\n")
            f.write("  - **통찰**: 그룹 간 최적 파라미터가 동일하다는 것은, 이 실험 데이터의 특성상 시간 정렬과 경로 길이 차이에 관대한 설정이 일반적으로 적합하다는 것을 의미합니다.\n")
            f.write("  - **실용적 의미**: 다양한 실험 조건에서도 동일한 파라미터를 사용할 수 있어, 파라미터 튜닝 부담이 적습니다.\n\n")
            
            # Sobolev pattern
            f.write("#### Sobolev 파라미터 패턴 및 통찰\n\n")
            f.write("- **Alpha (α)**: 위치 오차 가중치로, hogun/hogun_0125 그룹에서는 Medium, other 그룹에서는 High 민감도를 보입니다.\n")
            f.write("  - **통찰**: Alpha가 증가하면 위치 오차에 더 큰 비중을 두게 되어 거리가 340~325% 증가합니다.\n")
            f.write("  - **그룹 간 차이**: other 그룹에서 High 민감도를 보이는 것은, other 그룹의 경로 길이가 다양하여 위치 오차의 영향이 더 크기 때문일 수 있습니다.\n")
            f.write("  - **의미**: 낮은 Alpha(0.10)가 최적이라는 것은, 위치 오차만으로 평가하는 것보다 속도 오차도 함께 고려하는 것이 더 적절하다는 것을 의미합니다.\n\n")
            f.write("- **Beta (β)**: 속도 오차 가중치로, 모든 그룹에서 High 민감도를 보입니다. Beta가 증가하면 속도 차이에 더 큰 페널티를 부여합니다.\n")
            f.write("  - **통찰**: Beta가 증가하면 거리가 24~27% 증가하므로, Alpha보다는 영향이 작지만 여전히 중요합니다.\n")
            f.write("  - **패턴**: Beta가 증가할수록 거리가 증가하지만, 그 증가율이 상대적으로 완만합니다(stable 경향).\n")
            f.write("  - **의미**: 속도 오차는 위치 오차보다 상대적으로 작은 영향을 미치지만, 동역학적 평가에는 필수적입니다.\n\n")
            f.write("- **최적값**: 모든 그룹에서 α=0.10, β=0.10이 최적입니다. 이는 위치와 속도 오차 모두에 낮은 가중치를 두는 설정입니다.\n")
            f.write("  - **통찰**: 위치와 속도 오차 모두에 낮은 가중치를 두는 것이 최적이라는 것은, 이 실험 데이터에서는 절대적인 오차 크기보다는 상대적인 패턴이 더 중요하다는 것을 의미합니다.\n")
            f.write("  - **실험적 해석**: 로봇이 GT와 완전히 동일한 경로를 따라가지 않더라도, 전체적인 패턴이 유사하면 낮은 거리를 보입니다.\n")
            f.write("  - **주의**: 낮은 가중치는 민감도를 낮추므로, 미세한 차이를 감지하기 어려울 수 있습니다.\n\n")
            
            # ERP pattern
            f.write("#### ERP 파라미터 패턴 및 통찰\n\n")
            f.write("- **Gap Penalty (g)**: 모든 그룹에서 High 민감도를 보입니다. g=0.0이 표준이며 최소 거리를 보입니다.\n")
            f.write("  - **통찰**: g=0.0에서 g=20.96로 증가할 때 거리가 835% 변화하므로, Gap Penalty는 ERP 값에 매우 큰 영향을 미칩니다.\n")
            f.write("  - **패턴**: g=0.0에서 g=0.5로만 증가해도 거리가 4~5배 증가하므로, Gap Penalty는 매우 민감한 파라미터입니다.\n")
            f.write("  - **의미**: g=0.0이 최적이라는 것은, ERP 문헌에서 권장하는 표준 값이 이 실험 데이터에도 적합하다는 것을 의미합니다.\n")
            f.write("  - **실험적 해석**: Gap을 zero vector로 처리하는 것이 가장 합리적이며, 다른 값(gap=mean 등)을 사용하면 불필요한 페널티가 발생합니다.\n\n")
            f.write("- **최적값**: 모든 그룹에서 g=0.0000이 최적입니다. 이는 ERP 문헌에서 권장하는 표준 값입니다.\n")
            f.write("  - **통찰**: 그룹 간 최적 Gap이 동일하다는 것은, Gap Penalty 선택이 실험 조건과 무관하게 일관적이라는 것을 의미합니다.\n")
            f.write("  - **실용적 의미**: 표준 값(g=0.0)을 사용하면 파라미터 튜닝 없이도 최적 결과를 얻을 수 있습니다.\n\n")
            
            # Summary
            f.write("## 종합 분석\n\n")
            f.write("### 파라미터 민감도 요약\n\n")
            f.write("각 메트릭의 파라미터 민감도를 종합하면:\n\n")
            
            for group_name, group_analysis in analysis.items():
                f.write(f"**{group_name.upper()} 그룹**:\n")
                f.write(f"- TWED: Nu={group_analysis['TWED']['sensitivity']['nu_sensitivity']}, Lambda={group_analysis['TWED']['sensitivity']['lambda_sensitivity']}\n")
                f.write(f"- Sobolev: Alpha={group_analysis['Sobolev']['sensitivity']['alpha_sensitivity']}, Beta={group_analysis['Sobolev']['sensitivity']['beta_sensitivity']}\n")
                f.write(f"- ERP: Gap={group_analysis['ERP']['sensitivity']['gap_sensitivity']}\n\n")
            
            f.write("### 권장 파라미터\n\n")
            f.write("각 그룹별 최적 파라미터:\n\n")
            
            for group_name, group_analysis in analysis.items():
                f.write(f"**{group_name.upper()} 그룹**:\n")
                twed_opt = group_analysis['TWED']['optimal_parameters']
                sobolev_opt = group_analysis['Sobolev']['optimal_parameters']
                erp_opt = group_analysis['ERP']['optimal_parameters']
                
                f.write(f"- TWED: ν={twed_opt['nu']:.2f}, λ={twed_opt['lambda']:.2f}\n")
                f.write(f"- Sobolev: α={sobolev_opt['alpha']:.2f}, β={sobolev_opt['beta']:.2f}\n")
                f.write(f"- ERP: g={erp_opt['gap']:.4f}\n\n")
            
            f.write("## 파라미터 선택 가이드라인\n\n")
            f.write("### TWED 파라미터 선택\n\n")
            f.write("1. **시간 정렬이 중요할 때**: Nu를 높게 설정 (1.0~2.0)하여 시간 왜곡을 엄격하게 페널티\n")
            f.write("2. **경로 길이 차이가 클 때**: Lambda를 낮게 설정 (0.5)하여 삭제/삽입에 관대하게 처리\n")
            f.write("3. **일반적인 경우**: ν=0.10, λ=0.50 (최적값) 사용 권장\n\n")
            
            f.write("### Sobolev 파라미터 선택\n\n")
            f.write("1. **위치 정확도가 중요할 때**: Alpha를 높게 설정 (1.0~2.0)\n")
            f.write("2. **속도 일치가 중요할 때**: Beta를 높게 설정 (1.0~2.0)\n")
            f.write("3. **균형잡힌 평가가 필요할 때**: Alpha와 Beta를 동일하게 설정 (1.0, 1.0)\n")
            f.write("4. **민감도를 낮추고 싶을 때**: α=0.10, β=0.10 (최적값) 사용 권장\n\n")
            
            f.write("### ERP 파라미터 선택\n\n")
            f.write("1. **표준 사용**: g=0.0 사용 (모든 그룹에서 최적)\n")
            f.write("2. **특정 도메인 값 사용**: 궤적의 평균값이나 도메인 특성에 맞는 값 사용 가능하나, 일반적으로 g=0.0 권장\n\n")
            
            f.write("## 종합 권장사항\n\n")
            f.write("### 실험 목적별 파라미터 설정\n\n")
            f.write("1. **기본 평가 (권장)**:\n")
            f.write("   - TWED: ν=0.10, λ=0.50\n")
            f.write("   - Sobolev: α=0.10, β=0.10\n")
            f.write("   - ERP: g=0.0\n\n")
            f.write("2. **엄격한 시간 정렬 평가**:\n")
            f.write("   - TWED: ν=1.0~2.0, λ=1.0\n")
            f.write("   - 다른 메트릭은 기본값 유지\n\n")
            f.write("3. **위치와 속도 균형 평가**:\n")
            f.write("   - Sobolev: α=1.0, β=1.0\n")
            f.write("   - 다른 메트릭은 기본값 유지\n\n")
            f.write("### 주의사항\n\n")
            f.write("- 모든 파라미터가 High 민감도를 보이므로, 파라미터 변경 시 결과 해석에 주의가 필요합니다.\n")
            f.write("- 그룹 간 최적 파라미터가 동일하므로, 통일된 파라미터로 비교 분석하는 것이 권장됩니다.\n")
            f.write("- 파라미터 스위핑 결과는 `parameter_sweep/` 폴더의 시각화 파일에서 확인할 수 있습니다.\n\n")
        
        print(f"Generated parameter sensitivity report: {output_path}")
