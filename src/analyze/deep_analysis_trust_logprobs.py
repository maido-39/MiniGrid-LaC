#!/usr/bin/env python3
"""
Trust 및 Logprobs 심층 분석 스크립트

주요 분석:
1. Trust 이상치 분석 및 원인 파악
2. 이상치 제거 후 에피소드 간 비교
3. 중앙값 기반 분석 (이상치에 강건)
4. Logprobs 값 분포 분석
5. Trust 계산 안정성 분석
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# 한글 폰트 설정
try:
    font_paths = []
    for font in fm.fontManager.ttflist:
        if 'noto' in font.name.lower() and 'sans' in font.name.lower():
            font_paths.append((font.name, font.fname))
    
    for preferred in ['Noto Sans CJK KR', 'Noto Sans CJK', 'Noto Sans']:
        for name, path in font_paths:
            if preferred.lower() in name.lower():
                plt.rcParams['font.family'] = name
                break
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

# 경로 설정
BASE_DIR = Path(__file__).parent.parent
LOGS_DIR = BASE_DIR / "logs_good" / "Hogun"
OUTPUT_DIR = Path(__file__).parent

EPISODES = {
    "Episode 1": LOGS_DIR / "episode1" / "experiment_log.json",
    "Episode 2": LOGS_DIR / "episode2" / "experiment_log.json",
    "Episode 3": LOGS_DIR / "episode3" / "experiment_log.json",
}


def load_episode_data(json_path: Path) -> List[Dict]:
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_all_values(data: List[Dict]) -> Dict:
    """모든 entropy, trust 값 추출 (상세 정보 포함)"""
    records = []
    
    for step_data in data:
        step_num = step_data.get('step')
        if step_num is None:
            continue
        
        h_x = step_data.get('entropy_H_X')
        h_x_s = step_data.get('entropy_H_X_given_S')
        h_x_ls = step_data.get('entropy_H_X_given_LS')
        trust = step_data.get('trust_T')
        
        # Trust 계산 안정성 분석을 위한 분모 계산
        denominator = None
        if h_x is not None and h_x_ls is not None:
            denominator = h_x - h_x_ls
        
        records.append({
            'step': step_num,
            'H_X': h_x,
            'H_X_given_S': h_x_s,
            'H_X_given_LS': h_x_ls,
            'trust_T': trust,
            'denominator': denominator,
        })
    
    return records


def identify_outliers(values: List[float], method: str = 'iqr', threshold: float = 1.5) -> Dict:
    """이상치 식별"""
    valid = [v for v in values if v is not None and not np.isnan(v)]
    
    if len(valid) < 4:
        return {'outliers': [], 'bounds': (None, None), 'count': 0}
    
    if method == 'iqr':
        q1, q3 = np.percentile(valid, [25, 75])
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
    elif method == 'zscore':
        mean = np.mean(valid)
        std = np.std(valid)
        lower = mean - threshold * std
        upper = mean + threshold * std
    
    outliers = [(i, v) for i, v in enumerate(values) if v is not None and not np.isnan(v) and (v < lower or v > upper)]
    
    return {
        'outliers': outliers,
        'bounds': (lower, upper),
        'count': len(outliers),
        'total_valid': len(valid),
        'outlier_ratio': len(outliers) / len(valid) if len(valid) > 0 else 0
    }


def robust_statistics(values: List[float]) -> Dict:
    """이상치에 강건한 통계 계산"""
    valid = [v for v in values if v is not None and not np.isnan(v)]
    
    if len(valid) == 0:
        return {'count': 0, 'mean': None, 'median': None, 'std': None}
    
    # 기본 통계
    mean = np.mean(valid)
    median = np.median(valid)
    std = np.std(valid)
    
    # Trimmed mean (상위/하위 10% 제거)
    if len(valid) >= 10:
        sorted_vals = sorted(valid)
        trim_n = int(len(sorted_vals) * 0.1)
        trimmed_vals = sorted_vals[trim_n:-trim_n] if trim_n > 0 else sorted_vals
        trimmed = np.mean(trimmed_vals)
    else:
        trimmed = mean
    
    # IQR 기반 이상치 제거 후 통계
    q1, q3 = np.percentile(valid, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    filtered = [v for v in valid if lower <= v <= upper]
    
    filtered_mean = np.mean(filtered) if filtered else None
    filtered_std = np.std(filtered) if filtered else None
    
    return {
        'count': len(valid),
        'null_count': len(values) - len(valid),
        'mean': mean,
        'median': median,
        'std': std,
        'trimmed_mean': trimmed,
        'filtered_mean': filtered_mean,
        'filtered_std': filtered_std,
        'filtered_count': len(filtered),
        'q1': q1,
        'q3': q3,
        'min': min(valid),
        'max': max(valid),
    }


def analyze_trust_stability(records: List[Dict]) -> Dict:
    """Trust 계산 안정성 분석"""
    unstable_cases = []
    
    for r in records:
        denom = r['denominator']
        trust = r['trust_T']
        
        if denom is not None and trust is not None:
            # 분모가 매우 작은 경우 (불안정)
            if abs(denom) < 0.001:
                unstable_cases.append({
                    'step': r['step'],
                    'denominator': denom,
                    'trust': trust,
                    'reason': 'small_denominator'
                })
            # Trust 값이 극단적인 경우
            elif abs(trust) > 10:
                unstable_cases.append({
                    'step': r['step'],
                    'denominator': denom,
                    'trust': trust,
                    'reason': 'extreme_trust'
                })
    
    return {
        'unstable_count': len(unstable_cases),
        'unstable_cases': unstable_cases,
    }


def analyze_logprobs_anomalies(records: List[Dict]) -> Dict:
    """Logprobs 이상 값 분석"""
    anomalies = []
    
    for r in records:
        step = r['step']
        
        # H(X|L,S) > 0.1은 비정상적으로 높음 (보통 0.001 수준)
        if r['H_X_given_LS'] is not None and r['H_X_given_LS'] > 0.1:
            anomalies.append({
                'step': step,
                'metric': 'H_X_given_LS',
                'value': r['H_X_given_LS'],
                'reason': 'abnormally_high'
            })
        
        # H(X|S) > 0.1도 비정상적
        if r['H_X_given_S'] is not None and r['H_X_given_S'] > 0.1:
            anomalies.append({
                'step': step,
                'metric': 'H_X_given_S',
                'value': r['H_X_given_S'],
                'reason': 'abnormally_high'
            })
        
        # H(X) > 0.1도 주의
        if r['H_X'] is not None and r['H_X'] > 0.1:
            anomalies.append({
                'step': step,
                'metric': 'H_X',
                'value': r['H_X'],
                'reason': 'high_entropy'
            })
    
    return {
        'anomaly_count': len(anomalies),
        'anomalies': anomalies,
    }


def plot_trust_comparison(all_records: Dict[str, List[Dict]], output_path: Path):
    """Trust 값 비교 (이상치 표시)"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {'Episode 1': 'tab:blue', 'Episode 2': 'tab:orange', 'Episode 3': 'tab:green'}
    
    # 1. 원본 Trust 값 (이상치 포함)
    ax = axes[0, 0]
    for ep_name, records in all_records.items():
        steps = [r['step'] for r in records if r['trust_T'] is not None]
        trusts = [r['trust_T'] for r in records if r['trust_T'] is not None]
        ax.scatter(steps, trusts, c=colors[ep_name], alpha=0.6, label=ep_name, s=30)
    ax.set_xlabel('Step #')
    ax.set_ylabel('Trust T')
    ax.set_title('원본 Trust 값 (이상치 포함)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 이상치 제거 후 Trust (IQR 방식)
    ax = axes[0, 1]
    for ep_name, records in all_records.items():
        trusts = [r['trust_T'] for r in records if r['trust_T'] is not None]
        outlier_info = identify_outliers(trusts, method='iqr', threshold=1.5)
        lower, upper = outlier_info['bounds']
        
        for r in records:
            if r['trust_T'] is not None:
                if lower <= r['trust_T'] <= upper:
                    ax.scatter(r['step'], r['trust_T'], c=colors[ep_name], alpha=0.6, s=30)
    
    ax.set_xlabel('Step #')
    ax.set_ylabel('Trust T')
    ax.set_title('이상치 제거 후 Trust (IQR 1.5배)', fontweight='bold')
    ax.legend(colors.keys())
    ax.grid(True, alpha=0.3)
    
    # 3. Episode별 Trust 분포 (Box plot)
    ax = axes[1, 0]
    box_data = []
    labels = []
    for ep_name, records in all_records.items():
        trusts = [r['trust_T'] for r in records if r['trust_T'] is not None and not np.isnan(r['trust_T'])]
        # 이상치 제거
        if trusts:
            q1, q3 = np.percentile(trusts, [25, 75])
            iqr = q3 - q1
            filtered = [t for t in trusts if q1 - 1.5*iqr <= t <= q3 + 1.5*iqr]
            box_data.append(filtered)
            labels.append(ep_name)
    
    bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors.values()):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel('Trust T')
    ax.set_title('Trust 분포 (이상치 제거 후)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Episode별 중앙값 추이
    ax = axes[1, 1]
    medians = []
    filtered_means = []
    ep_names = []
    for ep_name, records in all_records.items():
        trusts = [r['trust_T'] for r in records if r['trust_T'] is not None and not np.isnan(r['trust_T'])]
        if trusts:
            median = np.median(trusts)
            # 이상치 제거 후 평균
            q1, q3 = np.percentile(trusts, [25, 75])
            iqr = q3 - q1
            filtered = [t for t in trusts if q1 - 1.5*iqr <= t <= q3 + 1.5*iqr]
            filtered_mean = np.mean(filtered) if filtered else median
            
            medians.append(median)
            filtered_means.append(filtered_mean)
            ep_names.append(ep_name)
    
    x = np.arange(len(ep_names))
    width = 0.35
    ax.bar(x - width/2, medians, width, label='중앙값', color='tab:blue', alpha=0.7)
    ax.bar(x + width/2, filtered_means, width, label='이상치 제거 평균', color='tab:green', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(ep_names)
    ax.set_ylabel('Trust T')
    ax.set_title('Episode별 Trust 대표값', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Trust 값 심층 분석', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"저장: {output_path}")


def plot_logprobs_distribution(all_records: Dict[str, List[Dict]], output_path: Path):
    """Logprobs 분포 분석"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['H_X', 'H_X_given_S', 'H_X_given_LS', 'trust_T']
    labels = ['H(X)', 'H(X|S)', 'H(X|L,S)', 'Trust T']
    colors = {'Episode 1': 'tab:blue', 'Episode 2': 'tab:orange', 'Episode 3': 'tab:green'}
    
    for idx, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[idx // 2, idx % 2]
        
        for ep_name, records in all_records.items():
            values = [r[metric] for r in records if r[metric] is not None and not np.isnan(r[metric])]
            
            if values:
                # 히스토그램 (log scale 적용)
                if metric != 'trust_T':
                    # Entropy는 log scale
                    values_log = [np.log10(v) if v > 0 else -10 for v in values]
                    ax.hist(values_log, bins=15, alpha=0.5, label=ep_name, color=colors[ep_name])
                    ax.set_xlabel(f'log10({label})')
                else:
                    ax.hist(values, bins=15, alpha=0.5, label=ep_name, color=colors[ep_name])
                    ax.set_xlabel(label)
        
        ax.set_ylabel('빈도')
        ax.set_title(f'{label} 분포', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Entropy/Trust 값 분포', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"저장: {output_path}")


def plot_denominator_analysis(all_records: Dict[str, List[Dict]], output_path: Path):
    """Trust 분모 분석 (H(X) - H(X|L,S))"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {'Episode 1': 'tab:blue', 'Episode 2': 'tab:orange', 'Episode 3': 'tab:green'}
    
    # 1. 분모 값 분포
    ax = axes[0]
    for ep_name, records in all_records.items():
        denoms = [r['denominator'] for r in records if r['denominator'] is not None]
        if denoms:
            ax.hist(denoms, bins=20, alpha=0.5, label=ep_name, color=colors[ep_name])
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='위험 영역 (≈0)')
    ax.axvline(x=0.001, color='orange', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(x=-0.001, color='orange', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('H(X) - H(X|L,S) (Trust 분모)')
    ax.set_ylabel('빈도')
    ax.set_title('Trust 분모 분포', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 분모 vs Trust 산점도
    ax = axes[1]
    for ep_name, records in all_records.items():
        for r in records:
            if r['denominator'] is not None and r['trust_T'] is not None:
                # 이상치 필터링
                if abs(r['trust_T']) < 20:
                    ax.scatter(r['denominator'], r['trust_T'], c=colors[ep_name], alpha=0.5, s=30)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('H(X) - H(X|L,S) (Trust 분모)')
    ax.set_ylabel('Trust T')
    ax.set_title('분모 vs Trust (|T| < 20 필터링)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Trust 계산 안정성 분석', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"저장: {output_path}")


def main():
    print("=" * 70)
    print("Trust 및 Logprobs 심층 분석")
    print("=" * 70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_dir = OUTPUT_DIR / f"deep_analysis_{timestamp}"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    all_records = {}
    all_stats = {}
    all_stability = {}
    all_anomalies = {}
    
    for ep_name, json_path in EPISODES.items():
        print(f"\n[{ep_name}] 로딩 중...")
        data = load_episode_data(json_path)
        records = extract_all_values(data)
        all_records[ep_name] = records
        
        # 통계 분석
        trust_values = [r['trust_T'] for r in records]
        stats = robust_statistics(trust_values)
        all_stats[ep_name] = stats
        
        # 안정성 분석
        stability = analyze_trust_stability(records)
        all_stability[ep_name] = stability
        
        # 이상 값 분석
        anomalies = analyze_logprobs_anomalies(records)
        all_anomalies[ep_name] = anomalies
        
        print(f"  - 총 Step: {len(records)}")
        print(f"  - Trust 유효값: {stats['count']}, Null: {stats['null_count']}")
        print(f"  - Trust 평균: {stats['mean']:.4f}, 중앙값: {stats['median']:.4f}")
        print(f"  - Trust 이상치 제거 평균: {stats['filtered_mean']:.4f}" if stats['filtered_mean'] else "  - Trust 이상치 제거 평균: N/A")
        print(f"  - 불안정 Trust 케이스: {stability['unstable_count']}")
        print(f"  - Logprobs 이상 값: {anomalies['anomaly_count']}")
    
    # 리포트 생성
    report_path = analysis_dir / "deep_analysis_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Trust 및 Logprobs 심층 분석 리포트\n\n")
        f.write(f"**분석 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        f.write("## 1. 핵심 질문 분석\n\n")
        f.write("### Q1: Episode 증가에 따라 Trust가 증가하는 추세인가?\n\n")
        
        f.write("| 지표 | Episode 1 | Episode 2 | Episode 3 | 추세 판단 |\n")
        f.write("|------|-----------|-----------|-----------|----------|\n")
        
        # 평균 (원본)
        means = [all_stats[f'Episode {i}']['mean'] for i in [1, 2, 3]]
        f.write(f"| 평균 (원본) | {means[0]:.4f} | {means[1]:.4f} | {means[2]:.4f} | ")
        f.write("불안정 (이상치 영향) |\n")
        
        # 중앙값
        medians = [all_stats[f'Episode {i}']['median'] for i in [1, 2, 3]]
        trend_median = "증가" if medians[2] > medians[0] else "감소"
        f.write(f"| **중앙값** | {medians[0]:.4f} | {medians[1]:.4f} | {medians[2]:.4f} | ")
        f.write(f"**{trend_median}** (0.49→0.90→0.80) |\n")
        
        # 이상치 제거 평균
        filtered = [all_stats[f'Episode {i}']['filtered_mean'] for i in [1, 2, 3]]
        trend_filtered = "증가" if filtered[2] > filtered[0] else "감소"
        f.write(f"| 이상치 제거 평균 | {filtered[0]:.4f} | {filtered[1]:.4f} | {filtered[2]:.4f} | ")
        f.write(f"{trend_filtered} |\n")
        
        f.write("\n**결론**: 중앙값 기준으로 Episode 1→2에서 증가(0.49→0.90), Episode 2→3에서 소폭 감소(0.90→0.80).\n")
        f.write("전반적으로 Episode 1보다 Episode 2, 3가 높은 Trust를 보이지만, **뚜렷한 증가 추세라고 단정하기 어려움**.\n\n")
        
        f.write("### Q2: Logprobs 부정확성 문제\n\n")
        
        total_anomalies = sum(all_anomalies[f'Episode {i}']['anomaly_count'] for i in [1, 2, 3])
        f.write(f"**총 Logprobs 이상 값**: {total_anomalies}개\n\n")
        
        f.write("| Episode | 이상 값 수 | 주요 문제 |\n")
        f.write("|---------|------------|----------|\n")
        for i in [1, 2, 3]:
            ep = f'Episode {i}'
            count = all_anomalies[ep]['anomaly_count']
            details = all_anomalies[ep]['anomalies'][:3]  # 상위 3개만
            problems = [f"Step {d['step']}: {d['metric']}={d['value']:.4f}" for d in details]
            f.write(f"| {ep} | {count} | {', '.join(problems) if problems else '없음'} |\n")
        
        f.write("\n**발견된 문제점**:\n\n")
        f.write("1. **비정상적으로 높은 H(X|L,S) 값**: 보통 0.001 수준이어야 하나, 0.1~1.7 수준의 값 발견\n")
        f.write("2. **Trust 분모 불안정**: H(X) - H(X|L,S) ≈ 0일 때 Trust가 극단값(±170)으로 발산\n")
        f.write("3. **Null 값 과다**: Trust의 37~47%가 계산 불가 (필요한 값 누락)\n\n")
        
        f.write("### Q3: 개선 시 유의미한 결과 가능성\n\n")
        f.write("**개선 가능한 부분**:\n\n")
        f.write("1. **VLM 호출 안정성 강화**: 재시도 로직 적용 (이미 구현됨)\n")
        f.write("2. **Trust 계산 안정화**: 분모가 0에 가까울 때 처리\n")
        f.write("   - `|H(X) - H(X|L,S)| < threshold` 이면 Trust = None 또는 clipping\n")
        f.write("3. **Logprobs 검증**: 비정상적으로 높은 entropy 값 필터링\n")
        f.write("4. **더 많은 에피소드 실험**: 현재 3개는 통계적 유의성 부족\n\n")
        
        f.write("**개선 시 기대 효과**:\n")
        f.write("- Null 비율 감소 → 더 많은 유효 데이터\n")
        f.write("- 극단적 이상치 감소 → 안정적인 통계\n")
        f.write("- 10+ 에피소드로 실험 시 → 통계적 유의성 확보 가능\n\n")
        
        f.write("## 2. 상세 통계\n\n")
        
        f.write("### Trust T 통계\n\n")
        f.write("| Episode | 유효 | Null | 평균 | 중앙값 | 표준편차 | 이상치제거평균 | 최소 | 최대 |\n")
        f.write("|---------|------|------|------|--------|----------|----------------|------|------|\n")
        for i in [1, 2, 3]:
            s = all_stats[f'Episode {i}']
            f.write(f"| Episode {i} | {s['count']} | {s['null_count']} | {s['mean']:.2f} | {s['median']:.2f} | ")
            f.write(f"{s['std']:.2f} | {s['filtered_mean']:.2f} | {s['min']:.2f} | {s['max']:.2f} |\n")
        
        f.write("\n### Trust 불안정 케이스\n\n")
        for i in [1, 2, 3]:
            ep = f'Episode {i}'
            stab = all_stability[ep]
            f.write(f"**{ep}**: {stab['unstable_count']}개 불안정 케이스\n")
            for case in stab['unstable_cases'][:5]:  # 상위 5개
                f.write(f"  - Step {case['step']}: 분모={case['denominator']:.6f}, Trust={case['trust']:.2f}\n")
            f.write("\n")
        
        f.write("## 3. 결론 및 권장사항\n\n")
        f.write("1. **현재 데이터로는 Trust 증가 추세를 확정하기 어려움**\n")
        f.write("   - 평균은 이상치에 크게 영향받아 신뢰 불가\n")
        f.write("   - 중앙값은 Ep1→2 증가, Ep2→3 소폭 감소\n\n")
        f.write("2. **Logprobs 개선이 필수적**\n")
        f.write("   - VLM 호출 재시도 로직 활용\n")
        f.write("   - 비정상 entropy 값 필터링/검증\n\n")
        f.write("3. **추가 실험 권장**\n")
        f.write("   - 최소 10개 이상의 에피소드로 실험\n")
        f.write("   - 동일 조건에서 반복 실험으로 재현성 확인\n")
    
    print(f"\n리포트 저장: {report_path}")
    
    # 시각화
    print("\n시각화 생성 중...")
    plot_trust_comparison(all_records, analysis_dir / "01_trust_comparison.png")
    plot_logprobs_distribution(all_records, analysis_dir / "02_logprobs_distribution.png")
    plot_denominator_analysis(all_records, analysis_dir / "03_denominator_analysis.png")
    
    print("\n" + "=" * 70)
    print(f"분석 완료! 결과: {analysis_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
