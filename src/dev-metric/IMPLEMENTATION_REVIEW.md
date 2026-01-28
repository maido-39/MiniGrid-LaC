# 메트릭 구현 검토 보고서

온라인 검색을 통해 표준 라이브러리 및 구현체와 비교한 결과입니다.

## 검토 방법

각 메트릭에 대해 최소 2개 이상의 신뢰할 수 있는 구현체를 찾아 비교했습니다:
- **dtaidistance** (⭐ 1.1k+): DTW, DDTW의 표준 구현
- **similaritymeasures** (⭐ 262+): Fréchet, ERP의 표준 구현
- **tslearn** (⭐ 3.1k+): 시계열 머신러닝 표준 라이브러리
- **pytwed**: TWED의 공식 구현

---

## 1. DDTW (Derivative Dynamic Time Warping)

### 표준 구현 (dtaidistance)
```python
from dtaidistance.preprocessing import derivative
from dtaidistance import dtw

# Keogh & Pazzani의 공식 사용
deriv1 = derivative(series1, smooth=None)  # len(series) - 1 반환
deriv2 = derivative(series2, smooth=None)
distance = dtw.distance_fast(deriv1, deriv2)
```

**Keogh & Pazzani 공식:**
```
D(x_i) = ((x_{i+1} - x_{i-1}) + (x_i - x_{i-1})/2) / 2
```

### 현재 구현의 문제점

1. **미분 계산 방식이 표준과 다름**
   - 현재: Forward/Backward/Central difference 혼합
   - 표준: Keogh & Pazzani의 특수 공식 사용
   - 결과: 미분값의 길이가 원본과 같음 (표준은 len-1)

2. **경계 처리**
   - 현재: 첫/마지막 점도 미분값 계산
   - 표준: 미분 결과는 len-1 (경계 처리 방식 다름)

### 수정 필요 사항

```python
def compute_derivatives_keogh(trajectory: np.ndarray) -> np.ndarray:
    """
    Keogh & Pazzani의 DDTW 공식 사용
    """
    if len(trajectory) < 2:
        return np.array([]).reshape(0, 2)
    
    n = len(trajectory)
    derivatives = np.zeros((n - 1, trajectory.shape[1]))
    
    for i in range(n - 1):
        if i == 0:
            # First point: forward difference
            derivatives[i] = trajectory[i + 1] - trajectory[i]
        else:
            # Keogh & Pazzani formula
            forward = trajectory[i + 1] - trajectory[i - 1]
            backward = (trajectory[i] - trajectory[i - 1]) / 2.0
            derivatives[i] = (forward + backward) / 2.0
    
    return derivatives
```

---

## 2. TWED (Time Warp Edit Distance)

### 표준 구현 (pytwed, distancia)
- **pytwed**: Marteau의 C 구현을 Python으로 래핑
- **distancia**: Python 구현체

### 현재 구현 검토

**TWED 수식 (Marteau, 2009):**
```
TWED(A, B) = min {
    TWED(A[1:i-1], B[1:j-1]) + d(A[i], B[j]) + nu * |i - j|^p,
    TWED(A[1:i-1], B[1:j]) + d(A[i], A[i-1]) + lambda + nu * |i-1 - j|^p,
    TWED(A[1:i], B[1:j-1]) + d(B[j], B[j-1]) + lambda + nu * |i - (j-1)|^p
}
```

### 현재 구현의 문제점

1. **시간 페널티 계산 오류**
   - 현재: `nu * abs(i - j)` (인덱스 차이)
   - 표준: `nu * |i - j|^p` (일반적으로 p=2, 시간 차이의 제곱)
   - **시간 차이는 실제 시간 간격이어야 함** (step 간격 고려 필요)

2. **Delete/Insert 비용 계산**
   - 현재: 이전 점과의 거리 + lambda + 시간 페널티
   - 표준: 이전 점과의 거리 + lambda + 시간 페널티 (맞음)
   - 하지만 시간 페널티 계산이 잘못됨

### 수정 필요 사항

시간 페널티는 실제 시간 간격을 사용해야 합니다:
```python
# 시간 간격을 고려한 TWED
time_penalty = nu * (abs(time1[i] - time2[j]) ** p)
```

---

## 3. Fréchet Distance

### 표준 구현 (similaritymeasures)
- 재귀적 구현 + 메모이제이션
- SciPy의 `cdist` 사용으로 최적화 (v0.7.0+)

### 현재 구현 검토

✅ **올바른 부분:**
- 재귀적 구현 방식 맞음
- 메모이제이션 사용 맞음
- 기본 알고리즘 구조 올바름

⚠️ **개선 가능 사항:**
- 대용량 데이터(1000+ 점)에서 재귀 깊이 제한 문제
- Dynamic Programming 방식으로 변경 고려 (similaritymeasures v0.7.0+)

### 권장 사항

현재 구현은 **기본적으로 올바르지만**, 성능 최적화를 위해 DP 방식 고려:
```python
# DP 방식 (재귀 대신)
def frechet_distance_dp(traj1, traj2):
    n, m = len(traj1), len(traj2)
    dp = np.full((n, m), np.inf)
    
    # Base case
    dp[0, 0] = euclidean_distance(traj1[0], traj2[0])
    
    # Fill DP table
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue
            candidates = []
            if i > 0:
                candidates.append(dp[i-1, j])
            if j > 0:
                candidates.append(dp[i, j-1])
            if i > 0 and j > 0:
                candidates.append(dp[i-1, j-1])
            dp[i, j] = max(min(candidates), euclidean_distance(traj1[i], traj2[j]))
    
    return dp[n-1, m-1]
```

---

## 4. ERP (Edit Distance on Real sequence)

### 표준 구현 (similaritymeasures, sktime)
- Gap element `g`는 일반적으로 **0 벡터** 또는 **원점** 사용
- 현재 구현은 `mean(trajectory1)`을 사용 → **표준과 다름**

### 현재 구현의 문제점

1. **Gap element 선택**
   - 현재: `np.mean(trajectory1, axis=0)` (트레이젝토리 평균)
   - 표준: `np.zeros(2)` 또는 사용자 지정 값
   - 문제: Gap element가 트레이젝토리에 의존적이면 비대칭적

2. **ERP 수식**
   ```
   ERP(i, j) = min {
       ERP(i-1, j-1) + d(x_i, y_j),
       ERP(i-1, j) + d(x_i, g),
       ERP(i, j-1) + d(y_j, g)
   }
   ```
   - 현재 구현의 로직은 맞지만, gap element만 수정 필요

### 수정 필요 사항

```python
def erp_distance(
    trajectory1: np.ndarray,
    trajectory2: np.ndarray,
    gap_penalty: Optional[np.ndarray] = None
) -> float:
    # Default gap element: zero vector (표준)
    if gap_penalty is None:
        gap_element = np.zeros(trajectory1.shape[1])  # [0, 0] for 2D
    else:
        gap_element = np.array(gap_penalty)
    # ... 나머지 로직 동일
```

---

## 5. Sobolev Metric

### 표준 구현
- 직접 구현이 일반적 (라이브러리 없음)
- `np.gradient` 사용 권장

### 현재 구현 검토

⚠️ **개선 가능 사항:**

1. **속도 계산**
   - 현재: 단순 차분 `traj[i+1] - traj[i]`
   - 표준: `np.gradient` 사용 (더 정확한 미분)
   - 마지막 점 처리: 현재는 이전 속도 복사, 표준은 backward difference

2. **보간 방식**
   - 현재: 선형 보간 사용 (적절함)
   - 대안: 스플라인 보간 고려 가능

### 권장 수정

```python
def compute_velocity(trajectory: np.ndarray) -> np.ndarray:
    """
    np.gradient를 사용한 더 정확한 속도 계산
    """
    if len(trajectory) == 0:
        return np.array([]).reshape(0, 2)
    
    # np.gradient는 각 차원별로 미분 계산
    velocities = np.zeros_like(trajectory)
    for dim in range(trajectory.shape[1]):
        velocities[:, dim] = np.gradient(trajectory[:, dim])
    
    return velocities
```

---

## 6. DTW (Dynamic Time Warping)

### 표준 구현 (dtaidistance)
- C 기반 고속 구현
- 다양한 최적화 옵션 (pruning, window 등)

### 현재 구현 검토

✅ **올바른 부분:**
- 기본 DP 알고리즘 구조 맞음
- Cost matrix 초기화 올바름

⚠️ **개선 가능 사항:**
- 성능: C 구현체(dtaidistance) 사용 권장
- 최적화: Pruning, Window 제약 추가 가능

### 권장 사항

현재 구현은 **기본적으로 올바르지만**, 성능을 위해 dtaidistance 사용 고려:
```python
from dtaidistance import dtw
distance = dtw.distance_fast(traj1, traj2, use_pruning=True)
```

---

## 7. RMSE

### 표준 구현 (scikit-learn, numpy)
```python
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(traj1, traj2))
```

### 현재 구현 검토

✅ **올바름:**
- 기본 로직 정확
- 길이 불일치 처리 적절 (최소 길이 사용)

---

## 종합 평가 및 수정 우선순위

### 🔴 높은 우선순위 (수정 필요)

1. **DDTW**: Keogh & Pazzani 공식으로 변경
2. **ERP**: Gap element를 0 벡터로 변경
3. **TWED**: 시간 페널티 계산 수정 (시간 간격 사용)

### 🟡 중간 우선순위 (개선 권장)

4. **Sobolev**: `np.gradient` 사용
5. **Fréchet**: DP 방식으로 최적화 (대용량 데이터)

### 🟢 낮은 우선순위 (선택적)

6. **DTW**: dtaidistance 라이브러리 사용 고려 (성능)
7. **RMSE**: 현재 구현 유지

---

## 참고 자료

1. **DDTW**: Keogh, E. J., & Pazzani, M. J. (2001). Derivative dynamic time warping.
2. **TWED**: Marteau, P. F. (2009). Time warp edit distance with stiffness adjustment.
3. **Fréchet**: Eiter, T., & Mannila, H. (1994). Computing discrete Fréchet distance.
4. **ERP**: Chen, L., & Ng, R. (2004). On the marriage of Lp-norms and edit distance.
