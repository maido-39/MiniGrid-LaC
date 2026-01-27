#!/usr/bin/env python3
"""
Verbalized Entropy 기반 Step별 Entropy와 Trust 값을 기록하고 시각화하는 스크립트

새로운 로깅 구조 (verbalized_entropy 섹션)를 지원합니다:
- H_X, H_X_given_S, H_X_given_LS (backward compatibility)
- H_X_details, H_X_given_S_details, H_X_given_LS_details (상세 정보)
  - weighted_entropy, step_probs, step_entropies 포함

사용법:
    python analyze_verbalized_entropy_trust.py <experiment_log.json 경로>
    
예시:
    python analyze_verbalized_entropy_trust.py "src/logs/.../experiment_log.json"
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# 한글 폰트 설정 (Noto Sans CJK)
try:
    # 시스템에 설치된 폰트 찾기
    font_paths = []
    for font in fm.fontManager.ttflist:
        if 'noto' in font.name.lower() and 'sans' in font.name.lower():
            font_paths.append((font.name, font.fname))
    
    # Noto Sans CJK KR 우선, 없으면 다른 Noto Sans 사용
    preferred_names = ['Noto Sans CJK KR', 'Noto Sans CJK', 'Noto Sans']
    font_name = None
    font_path = None
    
    for preferred in preferred_names:
        for name, path in font_paths:
            if preferred.lower() in name.lower():
                font_name = name
                font_path = path
                break
        if font_name:
            break
    
    if font_name:
        # 폰트 직접 설정
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_name
        # 한글 깨짐 방지를 위한 추가 설정
        plt.rcParams['axes.unicode_minus'] = False
        print(f"폰트 설정: {font_name}")
    else:
        # 기본 폰트 사용
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        print("Noto Sans 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")
except Exception as e:
    print(f"폰트 설정 중 오류: {e}")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False


def extract_verbalized_entropy_trust_values(step_data: Dict) -> Tuple[
    Optional[float], Optional[float], Optional[float], Optional[float],
    Optional[Dict], Optional[Dict], Optional[Dict]
]:
    """
    Step 데이터에서 Verbalized Entropy와 Trust 값을 추출
    
    Args:
        step_data: step의 JSON 데이터
        
    Returns:
        (entropy_H_X, entropy_H_X_given_S, entropy_H_X_given_LS, trust_T,
         H_X_details, H_X_given_S_details, H_X_given_LS_details) 튜플
    """
    verbalized_entropy = step_data.get('verbalized_entropy', {})
    
    if not verbalized_entropy:
        return None, None, None, None, None, None, None
    
    # Backward compatibility: 직접 접근
    entropy_H_X = verbalized_entropy.get('H_X')
    entropy_H_X_given_S = verbalized_entropy.get('H_X_given_S')
    entropy_H_X_given_LS = verbalized_entropy.get('H_X_given_LS')
    trust_T = verbalized_entropy.get('trust_T')
    
    # 상세 정보 추출
    H_X_details = verbalized_entropy.get('H_X_details')
    H_X_given_S_details = verbalized_entropy.get('H_X_given_S_details')
    H_X_given_LS_details = verbalized_entropy.get('H_X_given_LS_details')
    
    # None 문자열이나 null을 None으로 변환
    if entropy_H_X == 'null' or entropy_H_X == 'None':
        entropy_H_X = None
    if entropy_H_X_given_S == 'null' or entropy_H_X_given_S == 'None':
        entropy_H_X_given_S = None
    if entropy_H_X_given_LS == 'null' or entropy_H_X_given_LS == 'None':
        entropy_H_X_given_LS = None
    if trust_T == 'null' or trust_T == 'None':
        trust_T = None
    
    return entropy_H_X, entropy_H_X_given_S, entropy_H_X_given_LS, trust_T, \
           H_X_details, H_X_given_S_details, H_X_given_LS_details


def extract_step_entropies(details: Optional[Dict]) -> List[Optional[float]]:
    """
    Details에서 step별 entropy 추출
    
    Args:
        details: H_X_details, H_X_given_S_details, 또는 H_X_given_LS_details
        
    Returns:
        [step1_entropy, step2_entropy, step3_entropy] 리스트
    """
    if not details or not isinstance(details, dict):
        return [None, None, None]
    
    step_entropies = details.get('step_entropies', [])
    if isinstance(step_entropies, list) and len(step_entropies) >= 3:
        return [
            step_entropies[0] if step_entropies[0] is not None else None,
            step_entropies[1] if len(step_entropies) > 1 and step_entropies[1] is not None else None,
            step_entropies[2] if len(step_entropies) > 2 and step_entropies[2] is not None else None
        ]
    return [None, None, None]


def filter_outliers(values: List[Optional[float]], n_std: float = 2.0) -> List[Optional[float]]:
    """
    이상치를 필터링 (None 값은 유지)
    
    Args:
        values: 값들의 리스트 (None 포함 가능)
        n_std: 표준편차 배수 (기본값: 2.0)
        
    Returns:
        필터링된 값들의 리스트
    """
    # None이 아닌 값들만 추출
    valid_values = [v for v in values if v is not None]
    
    if len(valid_values) == 0:
        return values
    
    mean = np.mean(valid_values)
    std = np.std(valid_values)
    
    # 이상치 범위 설정
    lower_bound = mean - n_std * std
    upper_bound = mean + n_std * std
    
    # 필터링된 값들 생성
    filtered = []
    for v in values:
        if v is None:
            filtered.append(None)
        elif v < lower_bound or v > upper_bound:
            # 이상치는 None으로 처리 (시각화에서 제외)
            filtered.append(None)
        else:
            filtered.append(v)
    
    return filtered


def analyze_experiment_log(json_path: str) -> Dict[str, List]:
    """
    Experiment log JSON 파일을 분석하여 각 step의 entropy와 trust 값을 추출
    
    Args:
        json_path: experiment_log.json 파일 경로
        
    Returns:
        분석된 데이터 딕셔너리
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {json_path}")
    
    print(f"JSON 파일 로딩 중: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("JSON 데이터는 리스트 형식이어야 합니다.")
    
    steps = []
    H_X = []
    H_X_given_S = []
    H_X_given_LS = []
    trust_T = []
    
    # Step별 entropy (상세 정보)
    H_X_step1 = []
    H_X_step2 = []
    H_X_step3 = []
    H_X_given_S_step1 = []
    H_X_given_S_step2 = []
    H_X_given_S_step3 = []
    H_X_given_LS_step1 = []
    H_X_given_LS_step2 = []
    H_X_given_LS_step3 = []
    
    print(f"총 {len(data)}개의 step을 분석 중...")
    
    for step_data in data:
        step_num = step_data.get('step')
        if step_num is None:
            continue
        
        # Entropy와 Trust 값 추출
        h_x, h_x_s, h_x_ls, trust, h_x_details, h_x_s_details, h_x_ls_details = \
            extract_verbalized_entropy_trust_values(step_data)
        
        steps.append(step_num)
        H_X.append(h_x)
        H_X_given_S.append(h_x_s)
        H_X_given_LS.append(h_x_ls)
        trust_T.append(trust)
        
        # Step별 entropy 추출
        h_x_steps = extract_step_entropies(h_x_details)
        h_x_s_steps = extract_step_entropies(h_x_s_details)
        h_x_ls_steps = extract_step_entropies(h_x_ls_details)
        
        H_X_step1.append(h_x_steps[0])
        H_X_step2.append(h_x_steps[1])
        H_X_step3.append(h_x_steps[2])
        H_X_given_S_step1.append(h_x_s_steps[0])
        H_X_given_S_step2.append(h_x_s_steps[1])
        H_X_given_S_step3.append(h_x_s_steps[2])
        H_X_given_LS_step1.append(h_x_ls_steps[0])
        H_X_given_LS_step2.append(h_x_ls_steps[1])
        H_X_given_LS_step3.append(h_x_ls_steps[2])
        
        # 출력
        status = []
        if h_x is not None:
            status.append(f"H(X)={h_x:.6f}")
        if h_x_s is not None:
            status.append(f"H(X|S)={h_x_s:.6f}")
        if h_x_ls is not None:
            status.append(f"H(X|L,S)={h_x_ls:.6f}")
        if trust is not None:
            status.append(f"T={trust:.6f}")
        
        if status:
            print(f"  Step {step_num}: {', '.join(status)}")
        else:
            print(f"  Step {step_num}: 모든 값이 null")
    
    return {
        'steps': steps,
        'H_X': H_X,
        'H_X_given_S': H_X_given_S,
        'H_X_given_LS': H_X_given_LS,
        'trust_T': trust_T,
        'H_X_step1': H_X_step1,
        'H_X_step2': H_X_step2,
        'H_X_step3': H_X_step3,
        'H_X_given_S_step1': H_X_given_S_step1,
        'H_X_given_S_step2': H_X_given_S_step2,
        'H_X_given_S_step3': H_X_given_S_step3,
        'H_X_given_LS_step1': H_X_given_LS_step1,
        'H_X_given_LS_step2': H_X_given_LS_step2,
        'H_X_given_LS_step3': H_X_given_LS_step3
    }


def visualize_entropy_trust(data: Dict[str, List], output_path: Optional[str] = None):
    """
    Entropy와 Trust를 이중 Y축으로 시각화
    
    Args:
        data: 분석된 데이터 딕셔너리
        output_path: 출력 파일 경로 (None이면 표시만)
    """
    steps = np.array(data['steps'])
    
    # None 값을 NaN으로 변환
    def convert_none_to_nan(values):
        return np.array([np.nan if v is None else v for v in values])
    
    H_X = convert_none_to_nan(data['H_X'])
    H_X_given_S = convert_none_to_nan(data['H_X_given_S'])
    H_X_given_LS = convert_none_to_nan(data['H_X_given_LS'])
    trust_T = convert_none_to_nan(data['trust_T'])
    
    # 이상치 필터링
    H_X_filtered = filter_outliers(H_X.tolist(), n_std=2.0)
    H_X_given_S_filtered = filter_outliers(H_X_given_S.tolist(), n_std=2.0)
    H_X_given_LS_filtered = filter_outliers(H_X_given_LS.tolist(), n_std=2.0)
    trust_T_filtered = filter_outliers(trust_T.tolist(), n_std=2.0)
    
    # None을 NaN으로 변환하여 numpy 배열로 변환
    H_X_filtered = convert_none_to_nan(H_X_filtered)
    H_X_given_S_filtered = convert_none_to_nan(H_X_given_S_filtered)
    H_X_given_LS_filtered = convert_none_to_nan(H_X_given_LS_filtered)
    trust_T_filtered = convert_none_to_nan(trust_T_filtered)
    
    # 유효한 값들의 범위 계산 (이상치 제외)
    valid_entropy = []
    for arr in [H_X_filtered, H_X_given_S_filtered, H_X_given_LS_filtered]:
        valid = arr[~np.isnan(arr)]
        if len(valid) > 0:
            valid_entropy.extend(valid)
    
    valid_trust = trust_T_filtered[~np.isnan(trust_T_filtered)]
    
    # 평균값 계산
    mean_H_X = np.nanmean(H_X_filtered) if not np.isnan(H_X_filtered).all() else None
    mean_H_X_given_S = np.nanmean(H_X_given_S_filtered) if not np.isnan(H_X_given_S_filtered).all() else None
    mean_H_X_given_LS = np.nanmean(H_X_given_LS_filtered) if not np.isnan(H_X_given_LS_filtered).all() else None
    mean_trust_T = np.nanmean(trust_T_filtered) if not np.isnan(trust_T_filtered).all() else None
    
    # Y축 범위 설정 (유효한 값들의 5%와 95% 백분위수 사용)
    if len(valid_entropy) > 0:
        entropy_min = np.percentile(valid_entropy, 5)
        entropy_max = np.percentile(valid_entropy, 95)
        entropy_range = entropy_max - entropy_min
        entropy_ylim = [max(0, entropy_min - 0.1 * entropy_range), entropy_max + 0.1 * entropy_range]
    else:
        entropy_ylim = [0, 1]
    
    if len(valid_trust) > 0:
        trust_min = np.percentile(valid_trust, 5)
        trust_max = np.percentile(valid_trust, 95)
        trust_range = trust_max - trust_min
        trust_ylim = [max(0, trust_min - 0.1 * trust_range), trust_max + 0.1 * trust_range]
    else:
        trust_ylim = [0, 1]
    
    # 그래프 생성
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # 왼쪽 Y축: Trust
    ax1.set_xlabel('Step #', fontsize=12)
    ax1.set_ylabel('Trust T', fontsize=12, color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.set_ylim(trust_ylim)
    
    # Trust 플롯 (null 값 처리)
    def plot_with_nan_connections(ax, steps, values, style, color, label, markersize=4, linewidth=1.5):
        """null 값 사이를 회색 점선으로 연결하면서 플롯"""
        valid_mask = ~np.isnan(values)
        if not np.any(valid_mask):
            return
        
        # 유효한 값들 플롯
        ax.plot(steps[valid_mask], values[valid_mask], 
                style, color=color, label=label, markersize=markersize, linewidth=linewidth)
        
        # null 값 사이를 회색 점선으로 연결
        nan_mask = np.isnan(values)
        if np.any(nan_mask):
            i = 0
            while i < len(values):
                if nan_mask[i]:
                    start_idx = i
                    while i < len(values) and nan_mask[i]:
                        i += 1
                    end_idx = i - 1
                    
                    # 전후 유효한 값이 있으면 점선으로 연결
                    if start_idx > 0 and end_idx < len(values) - 1:
                        if not nan_mask[start_idx - 1] and not nan_mask[end_idx + 1]:
                            ax.plot([steps[start_idx - 1], steps[end_idx + 1]], 
                                    [values[start_idx - 1], values[end_idx + 1]],
                                    '--', color='gray', alpha=0.5, linewidth=1)
                else:
                    i += 1
    
    plot_with_nan_connections(ax1, steps, trust_T_filtered, 'o-', 'tab:red', 'Trust T', markersize=4, linewidth=1.5)
    
    # Trust 평균 가로선
    if mean_trust_T is not None and not np.isnan(mean_trust_T):
        ax1.axhline(y=mean_trust_T, color='tab:red', linestyle='--', alpha=0.3, linewidth=1.5, 
                   label=f'Trust T 평균: {mean_trust_T:.4f}')
    
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 오른쪽 Y축: Entropy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Entropy (H)', fontsize=12, color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.set_ylim(entropy_ylim)
    
    # Entropy 플롯들 (null 값 처리)
    plot_with_nan_connections(ax2, steps, H_X_filtered, 'o-', 'tab:blue', 'H(X)', markersize=3, linewidth=1.5)
    plot_with_nan_connections(ax2, steps, H_X_given_S_filtered, 's-', 'tab:cyan', 'H(X|S)', markersize=3, linewidth=1.5)
    plot_with_nan_connections(ax2, steps, H_X_given_LS_filtered, '^-', 'tab:purple', 'H(X|L,S)', markersize=3, linewidth=1.5)
    
    # Entropy 평균 가로선
    if mean_H_X is not None and not np.isnan(mean_H_X):
        ax2.axhline(y=mean_H_X, color='tab:blue', linestyle='--', alpha=0.3, linewidth=1.5,
                   label=f'H(X) 평균: {mean_H_X:.6f}')
    if mean_H_X_given_S is not None and not np.isnan(mean_H_X_given_S):
        ax2.axhline(y=mean_H_X_given_S, color='tab:cyan', linestyle='--', alpha=0.3, linewidth=1.5,
                   label=f'H(X|S) 평균: {mean_H_X_given_S:.6f}')
    if mean_H_X_given_LS is not None and not np.isnan(mean_H_X_given_LS):
        ax2.axhline(y=mean_H_X_given_LS, color='tab:purple', linestyle='--', alpha=0.3, linewidth=1.5,
                   label=f'H(X|L,S) 평균: {mean_H_X_given_LS:.6f}')
    
    ax2.legend(loc='upper right')
    
    plt.title('Step별 Entropy와 Trust (Verbalized Entropy)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n시각화 결과 저장: {output_path}")
    else:
        plt.show()


def visualize_step_entropies(data: Dict[str, List], output_path: Optional[str] = None):
    """
    Step별 entropy를 별도로 시각화 (step1, step2, step3)
    
    Args:
        data: 분석된 데이터 딕셔너리
        output_path: 출력 파일 경로 (None이면 표시만)
    """
    steps = np.array(data['steps'])
    
    # None 값을 NaN으로 변환
    def convert_none_to_nan(values):
        return np.array([np.nan if v is None else v for v in values])
    
    # Step별 entropy 추출
    H_X_step1 = convert_none_to_nan(data['H_X_step1'])
    H_X_step2 = convert_none_to_nan(data['H_X_step2'])
    H_X_step3 = convert_none_to_nan(data['H_X_step3'])
    H_X_given_S_step1 = convert_none_to_nan(data['H_X_given_S_step1'])
    H_X_given_S_step2 = convert_none_to_nan(data['H_X_given_S_step2'])
    H_X_given_S_step3 = convert_none_to_nan(data['H_X_given_S_step3'])
    H_X_given_LS_step1 = convert_none_to_nan(data['H_X_given_LS_step1'])
    H_X_given_LS_step2 = convert_none_to_nan(data['H_X_given_LS_step2'])
    H_X_given_LS_step3 = convert_none_to_nan(data['H_X_given_LS_step3'])
    
    # 그래프 생성 (3개 subplot: step1, step2, step3)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    step_names = ['Step 1', 'Step 2', 'Step 3']
    step_data_list = [
        (H_X_step1, H_X_given_S_step1, H_X_given_LS_step1),
        (H_X_step2, H_X_given_S_step2, H_X_given_LS_step2),
        (H_X_step3, H_X_given_S_step3, H_X_given_LS_step3)
    ]
    
    for idx, (ax, step_name, (h_x, h_x_s, h_x_ls)) in enumerate(zip(axes, step_names, step_data_list)):
        # 유효한 값들의 범위 계산
        valid_values = []
        for arr in [h_x, h_x_s, h_x_ls]:
            valid = arr[~np.isnan(arr)]
            if len(valid) > 0:
                valid_values.extend(valid)
        
        if len(valid_values) > 0:
            y_min = np.percentile(valid_values, 5)
            y_max = np.percentile(valid_values, 95)
            y_range = y_max - y_min
            ylim = [max(0, y_min - 0.1 * y_range), y_max + 0.1 * y_range]
        else:
            ylim = [0, 2]
        
        ax.set_ylim(ylim)
        ax.set_xlabel('Step #', fontsize=10)
        ax.set_ylabel(f'{step_name} Entropy', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 플롯
        def plot_with_nan(ax, steps, values, style, color, label):
            valid_mask = ~np.isnan(values)
            if np.any(valid_mask):
                ax.plot(steps[valid_mask], values[valid_mask], 
                       style, color=color, label=label, markersize=3, linewidth=1.5)
        
        plot_with_nan(ax, steps, h_x, 'o-', 'tab:blue', 'H(X)')
        plot_with_nan(ax, steps, h_x_s, 's-', 'tab:cyan', 'H(X|S)')
        plot_with_nan(ax, steps, h_x_ls, '^-', 'tab:purple', 'H(X|L,S)')
        
        ax.legend(loc='upper right', fontsize=9)
        ax.set_title(f'{step_name} Entropy Comparison', fontsize=11, fontweight='bold')
    
    plt.suptitle('Step별 Entropy 비교 (Step 1, 2, 3)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nStep별 Entropy 시각화 결과 저장: {output_path}")
    else:
        plt.show()


def main():
    if len(sys.argv) < 2:
        print("사용법: python analyze_verbalized_entropy_trust.py <experiment_log.json 경로>")
        sys.exit(1)
    
    json_path = sys.argv[1]
    
    try:
        # 분석 수행
        data = analyze_experiment_log(json_path)
        
        if len(data['steps']) == 0:
            print("분석할 데이터가 없습니다.")
            return
        
        # 통계 출력
        print(f"\n총 {len(data['steps'])}개의 step 분석 완료")
        
        for name, values in [('H(X)', data['H_X']), ('H(X|S)', data['H_X_given_S']), 
                            ('H(X|L,S)', data['H_X_given_LS']), ('Trust T', data['trust_T'])]:
            valid_values = [v for v in values if v is not None]
            if len(valid_values) > 0:
                print(f"{name}: 범위 {min(valid_values):.6f} ~ {max(valid_values):.6f}, 평균 {np.mean(valid_values):.6f}, 유효값 {len(valid_values)}개")
            else:
                print(f"{name}: 유효값 없음")
        
        # Step별 entropy 통계
        print("\nStep별 Entropy 통계:")
        for step_idx, step_name in enumerate(['Step1', 'Step2', 'Step3'], 1):
            for entropy_type, prefix in [('H_X', 'H(X)'), ('H_X_given_S', 'H(X|S)'), ('H_X_given_LS', 'H(X|L,S)')]:
                key = f'{entropy_type}_step{step_idx}'
                values = data.get(key, [])
                valid_values = [v for v in values if v is not None]
                if len(valid_values) > 0:
                    print(f"  {prefix} {step_name}: 평균 {np.mean(valid_values):.6f}, 유효값 {len(valid_values)}개")
        
        # 결과를 텍스트 파일로 저장
        json_file = Path(json_path)
        output_dir = json_file.parent
        output_txt = output_dir / f"verbalized_entropy_trust_{json_file.stem}.txt"
        
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write("Step #\tH(X)\tH(X|S)\tH(X|L,S)\tTrust T\t")
            f.write("H(X)_step1\tH(X)_step2\tH(X)_step3\t")
            f.write("H(X|S)_step1\tH(X|S)_step2\tH(X|S)_step3\t")
            f.write("H(X|L,S)_step1\tH(X|L,S)_step2\tH(X|L,S)_step3\n")
            
            for i, step in enumerate(data['steps']):
                h_x = data['H_X'][i] if data['H_X'][i] is not None else 'null'
                h_x_s = data['H_X_given_S'][i] if data['H_X_given_S'][i] is not None else 'null'
                h_x_ls = data['H_X_given_LS'][i] if data['H_X_given_LS'][i] is not None else 'null'
                trust = data['trust_T'][i] if data['trust_T'][i] is not None else 'null'
                
                h_x_s1 = data['H_X_step1'][i] if data['H_X_step1'][i] is not None else 'null'
                h_x_s2 = data['H_X_step2'][i] if data['H_X_step2'][i] is not None else 'null'
                h_x_s3 = data['H_X_step3'][i] if data['H_X_step3'][i] is not None else 'null'
                
                h_x_s_s1 = data['H_X_given_S_step1'][i] if data['H_X_given_S_step1'][i] is not None else 'null'
                h_x_s_s2 = data['H_X_given_S_step2'][i] if data['H_X_given_S_step2'][i] is not None else 'null'
                h_x_s_s3 = data['H_X_given_S_step3'][i] if data['H_X_given_S_step3'][i] is not None else 'null'
                
                h_x_ls_s1 = data['H_X_given_LS_step1'][i] if data['H_X_given_LS_step1'][i] is not None else 'null'
                h_x_ls_s2 = data['H_X_given_LS_step2'][i] if data['H_X_given_LS_step2'][i] is not None else 'null'
                h_x_ls_s3 = data['H_X_given_LS_step3'][i] if data['H_X_given_LS_step3'][i] is not None else 'null'
                
                f.write(f"{step}\t{h_x}\t{h_x_s}\t{h_x_ls}\t{trust}\t")
                f.write(f"{h_x_s1}\t{h_x_s2}\t{h_x_s3}\t")
                f.write(f"{h_x_s_s1}\t{h_x_s_s2}\t{h_x_s_s3}\t")
                f.write(f"{h_x_ls_s1}\t{h_x_ls_s2}\t{h_x_ls_s3}\n")
        
        print(f"\n결과 저장: {output_txt}")
        
        # 시각화
        output_png = output_dir / f"verbalized_entropy_trust_{json_file.stem}.png"
        visualize_entropy_trust(data, str(output_png))
        
        # Step별 entropy 시각화
        output_step_png = output_dir / f"verbalized_entropy_steps_{json_file.stem}.png"
        visualize_step_entropies(data, str(output_step_png))
        
    except Exception as e:
        print(f"오류 발생: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
