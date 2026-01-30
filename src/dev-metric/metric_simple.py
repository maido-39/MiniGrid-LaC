import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.cm as cm # 색상 자동 생성을 위해 추가

# ==========================================
# 1. 경로 데이터 정의 (Baselines & CSV Loading)
# ==========================================

# (1) 기준 경로들 (Ideal & Human - 기존 유지)
ideal_path = [[3, 11], [4, 11], [5, 11], [5, 10], [5, 9], [4, 9], [3, 9], [3, 8], [3, 7], [3, 6], [3, 5], [3, 6], [3, 7], [3, 6], [4, 6], [4, 5], [4, 4], [4, 3], [5, 3], [6, 3], [7, 3], [8, 3], [8, 2]]
human_path = [[3, 11], [4, 11], [5, 11], [6, 11], [7, 11], [8, 11], [9, 11], [10, 11], [10, 10], [10, 9], [10, 8], [10, 7], [11, 7], [12, 7], [12, 6], [12, 7], [12, 8], [12, 7], [11, 7], [10, 7], [9, 7], [8, 7], [7, 7], [6, 7], [5, 7], [4, 7], [3, 7], [3, 6], [3, 5], [3, 4], [3, 3], [4, 3], [5, 3], [6, 3], [7, 3], [8, 3], [8, 2]]

# (2) CSV 파일에서 VLM 경로 불러오기
# 분석할 CSV 파일들의 경로를 리스트에 담아주세요.
csv_files = [
    # '/Path/to/another/experiment_log_2.csv', # 추가 파일이 있다면 주석 해제 후 입력
    # '/Path/to/another/experiment_log_3.csv',
]

vlm_paths = []
vlm_labels = [] # 그래프 범례용 파일 이름 저장

print("======== CSV Loading Start ========")
for i, file_path in enumerate(csv_files):
    try:
        df = pd.read_csv(file_path)
        
        # agent_x, agent_y 컬럼만 선택하여 리스트로 변환
        # (제공해주신 코드 로직 적용)
        path_data = df[['agent_x', 'agent_y']].values.tolist()
        
        if len(path_data) > 0:
            vlm_paths.append(path_data)
            # 파일명만 추출하여 라벨로 사용 (예: experiment_log.csv)
            file_name = file_path.split('/')[-1]
            vlm_labels.append(f"VLM {i+1} ({file_name})")
            
            print(f"✅ [{i+1}] 로드 성공: {file_name} (Steps: {len(path_data)})")
        else:
            print(f"⚠️ [{i+1}] 데이터 없음: {file_path}")

    except FileNotFoundError:
        print(f"❌ [{i+1}] 오류: 파일을 찾을 수 없습니다 -> {file_path}")
    except KeyError:
        print(f"❌ [{i+1}] 오류: 'agent_x' 또는 'agent_y' 컬럼 누락 -> {file_path}")
    except Exception as e:
        print(f"❌ [{i+1}] 알 수 없는 오류: {e}")

print("======== CSV Loading End ========\n")

# 데이터가 하나도 로드되지 않았으면 예제 데이터로 대체 (에러 방지용)
if not vlm_paths:
    print("!!! 주의: 로드된 CSV가 없습니다. 빈 리스트로 진행 시 오류가 발생할 수 있습니다.")
    # 필요시 여기에 기본 더미 데이터를 넣을 수도 있음

# ==========================================
# 2. 계산 함수 (Sobolev & DTW - 기존 유지)
# ==========================================
def compute_velocity(trajectory):
    if len(trajectory) == 0: return np.array([]).reshape(0, 2)
    if len(trajectory) == 1: return np.array([[0.0, 0.0]])
    velocities = np.zeros_like(trajectory, dtype=float)
    for dim in range(trajectory.shape[1]):
        velocities[:, dim] = np.gradient(trajectory[:, dim])
    return velocities

def interpolate_to_same_length(traj1, traj2):
    traj1, traj2 = np.array(traj1), np.array(traj2)
    target_len = max(len(traj1), len(traj2))
    # Handle edge case where length is 0 or 1
    if len(traj1) < 2 or len(traj2) < 2:
        return traj1, traj2 
        
    indices1 = np.linspace(0, len(traj1) - 1, target_len)
    indices2 = np.linspace(0, len(traj2) - 1, target_len)
    
    interp_traj1 = np.array([np.interp(indices1, np.arange(len(traj1)), traj1[:, dim]) for dim in range(traj1.shape[1])]).T
    interp_traj2 = np.array([np.interp(indices2, np.arange(len(traj2)), traj2[:, dim]) for dim in range(traj2.shape[1])]).T
    return interp_traj1, interp_traj2

def sobolev_distance(t1, t2):
    if len(t1) < 2 or len(t2) < 2: return np.inf
    t1_i, t2_i = interpolate_to_same_length(t1, t2)
    vel1, vel2 = compute_velocity(t1_i), compute_velocity(t2_i)
    pos_err = np.sum((t1_i - t2_i)**2)
    vel_err = np.sum((vel1 - vel2)**2)
    return float(np.sqrt(pos_err + vel_err))

def compute_manhattan_dtw(path_a, path_b):
    path_a, path_b = np.array(path_a), np.array(path_b)
    if len(path_a) == 0 or len(path_b) == 0: return np.inf
    d_mat = cdist(path_a, path_b, metric='cityblock')
    n, m = len(path_a), len(path_b)
    dtw_mat = np.zeros((n + 1, m + 1))
    dtw_mat[0, 1:] = np.inf; dtw_mat[1:, 0] = np.inf
    dtw_mat[1:, 1:] = d_mat
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dtw_mat[i, j] += min(dtw_mat[i-1, j], dtw_mat[i, j-1], dtw_mat[i-1, j-1])
    return dtw_mat[-1, -1]

# ==========================================
# 3. 점수 계산
# ==========================================
dtw_scores = []
sobolev_scores = []
dtw_human = compute_manhattan_dtw(human_path, ideal_path)
sobolev_human = sobolev_distance(human_path, ideal_path)

for v_path in vlm_paths:
    dtw_scores.append(compute_manhattan_dtw(v_path, ideal_path))
    sobolev_scores.append(sobolev_distance(v_path, ideal_path))

# ==========================================
# 4. 시각화 (동적 색상 할당 적용)
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)

# 색상 및 마커 자동 생성 (파일 개수에 맞춰서)
num_vlms = len(vlm_paths)
colors = cm.rainbow(np.linspace(0, 1, num_vlms)) # VLM 개수에 맞춰 무지개색 분할
markers = ['^', 's', 'D', 'o', '*', 'P'] * (num_vlms // 6 + 1) # 마커 리스트 반복

# [Graph 1] Trajectory Map
ax1 = axes[0, 0]
i_arr = np.array(ideal_path)
h_arr = np.array(human_path)
ax1.plot(i_arr[:, 0], i_arr[:, 1], 'k--', linewidth=3, label='Ideal', alpha=0.3)
ax1.plot(h_arr[:, 0], h_arr[:, 1], 'k-o', linewidth=1, label='Human', alpha=0.3)

for i, v_path in enumerate(vlm_paths):
    v_arr = np.array(v_path)
    ax1.plot(v_arr[:, 0], v_arr[:, 1], color=colors[i], marker=markers[i], 
             linewidth=2, label=vlm_labels[i], alpha=0.7)

ax1.set_title("1. Trajectory Map", fontsize=14, fontweight='bold')
ax1.legend(); ax1.grid(True, linestyle='--'); ax1.invert_yaxis()

# [Graph 2] Velocity Profile
ax2 = axes[0, 1]
def get_speed(traj):
    if len(traj) < 2: return np.zeros(50) # 예외 처리
    interp, _ = interpolate_to_same_length(traj, traj)
    return np.linalg.norm(compute_velocity(interp), axis=1)

target_len = 50
x_axis = np.linspace(0, 1, target_len)
def resample(data): 
    if len(data) == 0: return np.zeros(target_len)
    return np.interp(np.linspace(0, len(data)-1, target_len), np.arange(len(data)), data)

ax2.plot(x_axis, resample(get_speed(ideal_path)), 'k--', label='Ideal', alpha=0.5)
for i, v_path in enumerate(vlm_paths):
    ax2.plot(x_axis, resample(get_speed(v_path)), color=colors[i], linewidth=2, label=vlm_labels[i])

ax2.set_title("2. Velocity Profile (Smoothness)", fontsize=14, fontweight='bold')
ax2.set_ylabel("Speed Magnitude"); ax2.legend(); ax2.grid(True, alpha=0.3)

# [Graph 3] DTW Comparison
ax3 = axes[1, 0]
x = np.arange(len(vlm_paths))
if len(vlm_paths) > 0:
    bars1 = ax3.bar(x, dtw_scores, width=0.5, color=colors, edgecolor='black', alpha=0.7)
    ax3.bar_label(bars1, fmt='%.1f', padding=3, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'VLM {i+1}' for i in range(len(vlm_paths))])

ax3.axhline(y=dtw_human, color='blue', linestyle='--', linewidth=2, label=f'Human ({dtw_human:.1f})')
ax3.set_title("3. DTW Score (Position Only)", fontsize=14, fontweight='bold')
ax3.set_ylabel("Lower is Better"); ax3.legend(); ax3.grid(axis='y', alpha=0.3)
if dtw_scores: ax3.set_ylim(0, max(dtw_scores + [dtw_human]) * 1.2)

# [Graph 4] Sobolev Comparison
ax4 = axes[1, 1]
if len(vlm_paths) > 0:
    bars2 = ax4.bar(x, sobolev_scores, width=0.5, color=colors, edgecolor='black', alpha=0.7)
    ax4.bar_label(bars2, fmt='%.1f', padding=3, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'VLM {i+1}' for i in range(len(vlm_paths))])

ax4.axhline(y=sobolev_human, color='blue', linestyle='--', linewidth=2, label=f'Human ({sobolev_human:.1f})')
ax4.set_title("4. Sobolev Score (Pos + Vel)", fontsize=14, fontweight='bold')
ax4.set_ylabel("Lower is Better"); ax4.legend(); ax4.grid(axis='y', alpha=0.3)
if sobolev_scores: ax4.set_ylim(0, max(sobolev_scores + [sobolev_human]) * 1.2)

plt.show()
