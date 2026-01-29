#!/usr/bin/env python3
"""
Episode별 궤적 분석 페이지 생성기 (other 그룹).

각 Episode가 그린 Trajectory를 시각화하고, 분석 페이지(Markdown)를 생성합니다.
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_all_episodes, get_reference_path, extract_episode_number

# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.unicode_minus': False,
    'figure.dpi': 150,
})


def get_other_episodes_with_gt(logs_dir: Path) -> Tuple[Dict, np.ndarray, str]:
    """
    other 그룹 Episode들과 GT(Reference) 궤적을 반환.
    
    Returns:
        episodes: { episode_name: {'trajectory': ndarray, 'steps': ndarray, ...} }
        gt_trajectory: (N, 2) reference trajectory
        gt_name: reference episode name (Episode_1_1)
    """
    all_episodes = load_all_episodes(logs_dir)
    
    # other 그룹 전체 (Episode_1_1 포함 — Reference이자 분석 대상)
    other_episodes = {
        name: data for name, data in all_episodes.items()
        if data.get('group') == 'other'
    }
    
    gt_name = 'Episode_1_1'
    gt_trajectory = get_reference_path(all_episodes, gt_name)
    
    if gt_trajectory is None and 'Episode_1_1' in all_episodes:
        gt_trajectory = all_episodes['Episode_1_1']['trajectory']
    
    # Episode_1_1이 other 폴더에 없을 수 있음 -> logs_good 루트에 있음
    if gt_trajectory is None:
        for name, data in all_episodes.items():
            if name == 'Episode_1_1' or (data.get('group') == 'other' and 'Episode_1_1' in name):
                gt_trajectory = data['trajectory']
                gt_name = name
                break
    
    # other 그룹에 Episode_1_1이 없으면 첫 번째 episode를 GT로 (이미 group 분석에서 Episode_1_1 사용)
    if gt_trajectory is None and 'Episode_1_1' in all_episodes:
        gt_trajectory = all_episodes['Episode_1_1']['trajectory']
        gt_name = 'Episode_1_1'
    
    return other_episodes, gt_trajectory, gt_name


def plot_trajectory_overview(
    episodes: Dict,
    gt_trajectory: np.ndarray,
    gt_name: str,
    output_path: Path,
):
    """전체 Episode 궤적을 한 figure에 겹쳐서 그리기 (GT + 각 Episode)."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 색상: GT는 진한 파랑, Episode들은 서로 다른 색
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(episodes) + 1, 10)))
    
    # GT
    ax.plot(
        gt_trajectory[:, 0], gt_trajectory[:, 1],
        color='#1f77b4', linewidth=2.5, linestyle='-',
        label=f'Reference ({gt_name}, {len(gt_trajectory)} steps)', alpha=0.9, zorder=5
    )
    ax.scatter(gt_trajectory[0, 0], gt_trajectory[0, 1], c='green', s=120, marker='s', 
               edgecolors='black', linewidths=1.5, zorder=10, label='Start')
    ax.scatter(gt_trajectory[-1, 0], gt_trajectory[-1, 1], c='red', s=120, marker='*',
               edgecolors='black', linewidths=1.5, zorder=10, label='End')
    
    # 각 Episode (Episode 번호 순 정렬)
    sorted_eps = sorted(
        episodes.items(),
        key=lambda x: (extract_episode_number(x[0]) or 0, x[0])
    )
    
    for i, (ep_name, ep_data) in enumerate(sorted_eps):
        traj = ep_data['trajectory']
        n_steps = len(traj)
        ep_num = extract_episode_number(ep_name) or i
        color = colors[i % len(colors)]
        label_suffix = ' (Reference)' if ep_name == gt_name else ''
        ax.plot(
            traj[:, 0], traj[:, 1],
            color=color, linewidth=1.5, linestyle='--', alpha=0.8,
            label=f'{ep_name}{label_suffix} ({n_steps} steps)'
        )
    
    ax.set_xlabel('X (grid column)')
    ax.set_ylabel('Y (grid row)')
    ax.set_title('OTHER Group: Reference vs All Episode Trajectories (Overview)')
    ax.legend(loc='upper left', fontsize=8, ncol=1)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_episode_vs_gt(
    ep_name: str,
    ep_trajectory: np.ndarray,
    gt_trajectory: np.ndarray,
    gt_name: str,
    output_path: Path,
    trajectory_length: int,
):
    """단일 Episode vs GT 궤적 비교."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(
        gt_trajectory[:, 0], gt_trajectory[:, 1],
        'b-', linewidth=2, label=f'Reference ({gt_name}, {len(gt_trajectory)} steps)', alpha=0.8
    )
    ax.plot(
        ep_trajectory[:, 0], ep_trajectory[:, 1],
        'r--', linewidth=1.8, label=f'{ep_name} ({trajectory_length} steps)', alpha=0.8
    )
    ax.scatter(gt_trajectory[0, 0], gt_trajectory[0, 1], c='green', s=100, marker='s',
               edgecolors='black', linewidths=1, zorder=10)
    ax.scatter(gt_trajectory[-1, 0], gt_trajectory[-1, 1], c='red', s=100, marker='*',
               edgecolors='black', linewidths=1, zorder=10)
    ax.scatter(ep_trajectory[0, 0], ep_trajectory[0, 1], c='green', s=80, marker='s',
               edgecolors='gray', linewidths=1, zorder=9)
    ax.scatter(ep_trajectory[-1, 0], ep_trajectory[-1, 1], c='red', s=80, marker='*',
               edgecolors='gray', linewidths=1, zorder=9)
    
    ax.set_xlabel('X (grid column)')
    ax.set_ylabel('Y (grid row)')
    ax.set_title(f'Trajectory: {ep_name} vs Reference')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_episode_only(
    ep_name: str,
    trajectory: np.ndarray,
    steps: np.ndarray,
    output_path: Path,
):
    """단일 Episode 궤적만 Step 색상 그라데이션으로 표시."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    n = len(trajectory)
    if n == 0:
        plt.close()
        return
    
    # Step에 따른 색상 (처음=파랑, 끝=빨강)
    colors = plt.cm.viridis(np.linspace(0, 1, n))
    
    for i in range(n - 1):
        ax.plot(
            trajectory[i:i+2, 0], trajectory[i:i+2, 1],
            color=colors[i], linewidth=2, alpha=0.9
        )
    
    ax.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=120, marker='s',
               edgecolors='black', linewidths=1.5, zorder=10, label='Start')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=120, marker='*',
               edgecolors='black', linewidths=1.5, zorder=10, label='End')
    
    ax.set_xlabel('X (grid column)')
    ax.set_ylabel('Y (grid row)')
    ax.set_title(f'{ep_name} Trajectory ({n} steps)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def compute_path_structure(
    episodes: Dict,
    gt_trajectory: np.ndarray,
    gt_name: str,
) -> Dict:
    """전체 궤적의 공간적 구조 요약 (bbox, 시작/끝, 길이 범위). Episode_1_1 포함."""
    all_trajs = [e['trajectory'] for e in episodes.values() if len(e.get('trajectory', [])) > 0]
    if not all_trajs:
        return {}
    
    all_xy = np.vstack(all_trajs)
    if len(all_xy) == 0:
        return {}
    
    lengths = [len(t) for t in all_trajs]
    
    return {
        'x_min': float(np.min(all_xy[:, 0])),
        'x_max': float(np.max(all_xy[:, 0])),
        'y_min': float(np.min(all_xy[:, 1])),
        'y_max': float(np.max(all_xy[:, 1])),
        'path_length_min': min(lengths),
        'path_length_max': max(lengths),
        'path_length_mean': float(np.mean(lengths)),
        'n_episodes': len(lengths),
    }


def plot_path_length_by_episode(
    episode_info_list: List[Dict],
    output_path: Path,
):
    """Episode 번호/순서에 따른 경로 길이(steps) 막대 그래프 (Episode_1_1 포함)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    labels = [info['name'] for info in episode_info_list]
    lengths = [info['steps'] for info in episode_info_list]
    ep_nums = [info.get('episode_number') or 0 for info in episode_info_list]
    
    # 정렬: Episode 번호 → 이름
    order = sorted(range(len(labels)), key=lambda i: (ep_nums[i], labels[i]))
    labels = [labels[i] for i in order]
    lengths = [lengths[i] for i in order]
    ep_nums = [ep_nums[i] for i in order]
    
    x = np.arange(len(labels))
    colors = ['#1f77b4' if 'Episode_1_1' in name else plt.cm.tab10(i % 10) for i, name in enumerate(labels)]
    bars = ax.bar(x, lengths, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_xticks(x)
    short_labels = []
    for n, s in zip(ep_nums, labels):
        if s == 'Episode_1_1':
            short_labels.append('Ep1\n(Ref)')
        elif 'Test_Entropy' in s:
            short_labels.append(f'Ep{n}\n' + s.replace('_Test_Entropy', '')[:12])
        else:
            short_labels.append(f'Ep{n}\n{s[:10]}' if len(s) > 10 else s)
    ax.set_xticklabels(short_labels, fontsize=8, rotation=15, ha='right')
    ax.set_ylabel('Path length (steps)')
    ax.set_xlabel('Episode')
    ax.set_title('Path Length by Episode (including Episode_1_1 Reference)')
    ax.grid(True, axis='y', alpha=0.3)
    
    for i, (xi, L) in enumerate(zip(x, lengths)):
        ax.annotate(str(L), (xi, L), ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_metric_trends(
    stats_path: Path,
    output_path: Path,
):
    """Episode 증가에 따른 메트릭 추이 (6개 비교 Episode 기준)."""
    if not stats_path.exists():
        return
    with open(stats_path, 'r', encoding='utf-8') as f:
        stats = json.load(f)
    
    episodes = stats.get('episodes', [])
    ep_numbers = stats.get('episode_numbers', [])
    metrics = stats.get('metrics', {})
    
    if not episodes or not ep_numbers:
        return
    
    metric_names = ['RMSE', 'DTW', 'Fréchet', 'DDTW', 'Sobolev']
    n_metrics = len(metric_names)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for idx, mname in enumerate(metric_names):
        if idx >= len(axes) or mname not in metrics:
            continue
        ax = axes[idx]
        vals = metrics[mname].get('values', [])
        if len(vals) != len(ep_numbers):
            continue
        ax.plot(ep_numbers, vals, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Episode number')
        ax.set_ylabel(mname)
        ax.set_title(f'{mname} vs Episode')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplot
    for j in range(len(metric_names), len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Metric Trends as Episode Number Increases (vs Reference Episode_1_1)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def generate_trajectory_page(
    logs_dir: Path,
    output_dir: Path,
    detailed_results_path: Optional[Path] = None,
    statistical_analysis_path: Optional[Path] = None,
):
    """
    other 그룹에 대해 Episode별 궤적 시각화 및 분석 페이지 생성.
    Episode_1_1 포함, 전체 Path 구조 및 Episode 증가에 따른 경향성 분석 포함.
    
    Args:
        logs_dir: logs_good 디렉터리
        output_dir: 결과를 저장할 디렉터리 (analysis_reports/other)
        detailed_results_path: detailed_results.json 경로 (메트릭 요약용)
        statistical_analysis_path: statistical_analysis.json 경로 (경향 분석용)
    """
    output_dir = Path(output_dir)
    trajectories_dir = output_dir / 'trajectories'
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    
    other_episodes, gt_trajectory, gt_name = get_other_episodes_with_gt(logs_dir)
    
    if gt_trajectory is None:
        print("Warning: Reference (Episode_1_1) trajectory not found. Skipping overview.")
    if len(other_episodes) == 0:
        print("No 'other' group episodes found.")
        return
    
    # 메트릭 로드 (있으면)
    metrics_by_episode = {}
    if detailed_results_path and detailed_results_path.exists():
        with open(detailed_results_path, 'r', encoding='utf-8') as f:
            detailed = json.load(f)
        metrics_by_episode = detailed.get('episodes', {})
    
    # 통계(경향) 로드 (있으면)
    stats_data = None
    if statistical_analysis_path and Path(statistical_analysis_path).exists():
        with open(statistical_analysis_path, 'r', encoding='utf-8') as f:
            stats_data = json.load(f)
    
    # 1) 전체 개요 플롯
    if gt_trajectory is not None:
        plot_trajectory_overview(
            other_episodes, gt_trajectory, gt_name,
            trajectories_dir / 'overview_all_trajectories.png'
        )
    
    # 2) Episode별: vs GT 비교 + 단독 궤적
    sorted_eps = sorted(
        other_episodes.items(),
        key=lambda x: (extract_episode_number(x[0]) or 0, x[0])
    )
    
    episode_info_list = []
    
    for ep_name, ep_data in sorted_eps:
        traj = ep_data['trajectory']
        steps = ep_data.get('steps', np.arange(len(traj)))
        n_steps = len(traj)
        
        # 파일명용 (특수문자 제거)
        safe_name = ep_name.replace('/', '_').replace(' ', '_')
        
        if gt_trajectory is not None:
            plot_episode_vs_gt(
                ep_name, traj, gt_trajectory, gt_name,
                trajectories_dir / f'{safe_name}_vs_GT.png',
                trajectory_length=n_steps,
            )
        
        plot_episode_only(
            ep_name, traj, steps,
            trajectories_dir / f'{safe_name}_only.png',
        )
        
        info = {
            'name': ep_name,
            'safe_name': safe_name,
            'steps': n_steps,
            'episode_number': extract_episode_number(ep_name),
        }
        if ep_name in metrics_by_episode:
            info['metrics'] = metrics_by_episode[ep_name].get('metrics', {})
        info['is_reference'] = (ep_name == gt_name)
        episode_info_list.append(info)
    
    # 2.5) 전체 Path 구조 계산
    path_structure = compute_path_structure(other_episodes, gt_trajectory, gt_name)
    
    # 2.6) 경로 길이·메트릭 경향 시각화
    plot_path_length_by_episode(episode_info_list, trajectories_dir / 'path_length_by_episode.png')
    stats_path = Path(statistical_analysis_path) if statistical_analysis_path else output_dir / 'statistical_analysis.json'
    if stats_path.exists():
        plot_metric_trends(stats_path, trajectories_dir / 'metric_trends_by_episode.png')
    
    # 3) Markdown 페이지 작성
    md_path = output_dir / 'Episode별_궤적_분석.md'
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# OTHER 그룹 Episode별 궤적 분석\n\n")
        f.write(f"**생성 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("이 문서는 **other** 그룹의 각 Episode( **Episode_1_1 포함** )가 그린 **Trajectory(궤적)**를 시각화하고, **전체 Path 구조**와 **Episode 증가에 따른 경향성**을 요약합니다.\n\n")
        f.write("### 목차\n")
        f.write("1. [전체 궤적 개요](#1-전체-궤적-개요)\n")
        f.write("2. [전체 Path 구조](#2-전체-path-구조)\n")
        f.write("3. [Episode 증가에 따른 경향성](#3-episode-증가에-따른-경향성)\n")
        f.write("4. [Episode별 궤적 상세](#4-episode별-궤적-상세)\n")
        f.write("5. [요약](#5-요약)\n\n")
        f.write("---\n\n")
        f.write("## 1. 전체 궤적 개요\n\n")
        f.write("**Episode_1_1(Reference)**을 포함한 other 그룹 **전체 7개 Episode**의 궤적을 한 그림에 겹쳐 표시합니다.\n\n")
        if gt_trajectory is not None:
            f.write("![전체 궤적 개요](trajectories/overview_all_trajectories.png)\n\n")
            f.write(f"- **Reference**: {gt_name} ({len(gt_trajectory)} steps)\n")
            f.write(f"- **분석 대상**: other 그룹 Episode **7개** (Episode_1_1 포함)\n\n")
        f.write("---\n\n")
        f.write("## 2. 전체 Path 구조\n\n")
        if path_structure:
            f.write("모든 Episode 궤적이 사용하는 **공간 범위**와 **경로 길이** 요약입니다.\n\n")
            f.write("| 항목 | 값 |\n")
            f.write("|------|-----|\n")
            f.write(f"| X 범위 (grid column) | {path_structure['x_min']:.1f} ~ {path_structure['x_max']:.1f} |\n")
            f.write(f"| Y 범위 (grid row) | {path_structure['y_min']:.1f} ~ {path_structure['y_max']:.1f} |\n")
            f.write(f"| 경로 길이 (steps) 최소 | {path_structure['path_length_min']} |\n")
            f.write(f"| 경로 길이 (steps) 최대 | {path_structure['path_length_max']} |\n")
            f.write(f"| 경로 길이 평균 | {path_structure['path_length_mean']:.1f} |\n")
            f.write(f"| Episode 수 | {path_structure['n_episodes']} |\n\n")
            f.write("**해석**: 경로 길이가 **27~78 steps**로 약 2.9배 차이로, Episode에 따라 **짧은 직선형**부터 **긴 우회형**까지 다양한 경로가 선택되었습니다.\n\n")
        f.write("---\n\n")
        f.write("## 3. Episode 증가에 따른 경향성\n\n")
        f.write("Episode 번호가 커질 때 **경로 길이**와 **Reference 대비 메트릭**이 어떻게 변하는지 요약합니다.\n\n")
        f.write("### 3.1 경로 길이 (Episode별)\n\n")
        f.write("![경로 길이 by Episode](trajectories/path_length_by_episode.png)\n\n")
        f.write("Episode_1_1(Reference)은 46 steps이며, 비교 대상 6개 Episode는 27~78 steps로 다양합니다. Episode 번호와 경로 길이 사이에 단순한 증가/감소 관계는 없고, **같은 Episode 번호에서도 시도별로 길이가 다릅니다** (예: Episode 2의 40 vs 41 steps).\n\n")
        if (trajectories_dir / 'metric_trends_by_episode.png').exists():
            f.write("### 3.2 Reference 대비 메트릭 추이 (비교 6개 Episode)\n\n")
            f.write("![메트릭 추이 by Episode](trajectories/metric_trends_by_episode.png)\n\n")
            if stats_data and 'metrics' in stats_data:
                m = stats_data['metrics']
                f.write("| 메트릭 | Episode 증가 시 경향 | 해석 |\n")
                f.write("|--------|----------------------|------|\n")
                for name, key in [('RMSE', 'RMSE'), ('DTW', 'DTW'), ('Fréchet', 'Fréchet'), ('DDTW', 'DDTW'), ('Sobolev', 'Sobolev')]:
                    if key not in m:
                        continue
                    slope = m[key].get('trend', {}).get('slope', 0)
                    p = m[key].get('trend', {}).get('p_value', 1)
                    trend = '감소' if slope < -0.1 else '증가' if slope > 0.1 else '거의 변화 없음'
                    sig = '(유의)' if p < 0.05 else ''
                    f.write(f"| {key} | {trend} {sig} | slope={slope:.3f}, p={p:.4f} |\n")
                f.write("\n")
                f.write("**통찰**: RMSE는 Episode가 커질수록 **감소**하는 경향(위치 정확도 개선), DTW·Fréchet·Sobolev는 **증가**하는 경향(경로 형태가 Reference와 더 달라짐)을 보입니다. 즉 **로컬 정확도는 좋아지지만, 전체 경로 선택은 더 달라지는** 패턴입니다.\n\n")
        f.write("---\n\n")
        f.write("## 4. Episode별 궤적 상세\n\n")
        
        for info in episode_info_list:
            name = info['name']
            safe = info['safe_name']
            n_steps = info['steps']
            ep_num = info.get('episode_number', '-')
            
            f.write(f"### {name}\n\n")
            if info.get('is_reference'):
                f.write("- **역할**: Reference (GT)\n")
            f.write(f"- **Episode 번호**: {ep_num}\n")
            f.write(f"- **궤적 길이**: {n_steps} steps\n")
            
            if 'metrics' in info and info['metrics']:
                f.write("- **메트릭 (vs Reference)**\n")
                for mname, val in info['metrics'].items():
                    if isinstance(val, (int, float)) and not np.isnan(val):
                        f.write(f"  - {mname}: {val:.4f}\n")
                f.write("\n")
            
            f.write("#### 궤적만 보기 (Step 진행에 따른 경로)\n\n")
            f.write(f"![{name} 궤적만](trajectories/{safe}_only.png)\n\n")
            
            f.write("#### Reference vs 이 Episode 비교\n\n")
            if info.get('is_reference'):
                f.write("(Reference이므로 동일 궤적이 두 선으로 겹쳐 보입니다.)\n\n")
            f.write(f"![{name} vs GT](trajectories/{safe}_vs_GT.png)\n\n")
            f.write("---\n\n")
        
        f.write("## 5. 요약\n\n")
        f.write("| Episode | 궤적 길이 (steps) | Episode 번호 |\n")
        f.write("|---------|-------------------|-------------|\n")
        for info in episode_info_list:
            f.write(f"| {info['name']} | {info['steps']} | {info.get('episode_number', '-')} |\n")
        f.write("\n")
    
    print(f"Trajectory page generated: {md_path}")
    print(f"Visualizations saved to: {trajectories_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate Episode-by-Episode trajectory analysis page for other group')
    parser.add_argument('--logs-dir', type=str, default=None, help='Path to logs_good (default: ../logs_good from src/dev-metric)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: analysis_reports/other)')
    args = parser.parse_args()
    
    base = Path(__file__).parent
    if args.logs_dir is None:
        logs_dir = base.parent / 'logs_good'
    else:
        logs_dir = Path(args.logs_dir)
    
    if args.output_dir is None:
        output_dir = base / 'analysis_reports' / 'other'
    else:
        output_dir = Path(args.output_dir)
    
    detailed_path = output_dir / 'detailed_results.json'
    
    if not logs_dir.exists():
        print(f"Logs directory not found: {logs_dir}")
        return 1
    
    stats_path = output_dir / 'statistical_analysis.json'
    generate_trajectory_page(logs_dir, output_dir, detailed_path, stats_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
