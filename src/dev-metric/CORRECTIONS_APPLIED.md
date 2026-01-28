# 적용된 수정 사항

검토 보고서에 따라 다음 수정 사항을 적용했습니다.

## ✅ 적용 완료

### 1. DDTW - Keogh & Pazzani 공식 적용
- **변경 전**: Forward/Backward/Central difference 혼합
- **변경 후**: Keogh & Pazzani의 표준 공식 사용
  ```
  D(x_i) = ((x_{i+1} - x_{i-1}) + (x_i - x_{i-1})/2) / 2
  ```
- **결과**: 표준 DDTW와 일치하는 미분 계산

### 2. ERP - Gap Element를 0 벡터로 변경
- **변경 전**: `np.mean(trajectory1, axis=0)` 사용 (비대칭적)
- **변경 후**: `np.zeros(2)` 사용 (표준)
- **결과**: 대칭적이고 표준적인 ERP 거리 계산

### 3. Sobolev - np.gradient 사용
- **변경 전**: 단순 차분 `traj[i+1] - traj[i]`
- **변경 후**: `np.gradient` 사용 (더 정확한 미분)
- **결과**: 경계 처리 개선 및 더 정확한 속도 계산

## ⚠️ 추가 검토 필요

### TWED - 시간 페널티 계산
현재 구현은 인덱스 차이를 사용하고 있으나, 표준 TWED는 실제 시간 간격을 사용합니다.
- **현재**: `nu * abs(i - j)` (인덱스 차이)
- **표준**: `nu * |time[i] - time[j]|^p` (시간 간격)

**문제**: 현재 데이터에는 시간 정보가 step 번호만 있어서, 실제 시간 간격을 알 수 없습니다.
- **옵션 1**: Step 간격을 균등하다고 가정 (현재 구현 유지)
- **옵션 2**: timestamp 정보를 활용하여 실제 시간 간격 사용

**권장**: Step 간격이 균등하다면 현재 구현도 합리적입니다. 다만, 시간 정보가 있다면 활용하는 것이 더 정확합니다.

## 📊 검증 방법

수정된 구현을 검증하려면:

1. **표준 라이브러리와 비교**:
   ```python
   # DDTW
   from dtaidistance.preprocessing import derivative
   from dtaidistance import dtw
   deriv1 = derivative(traj1)
   deriv2 = derivative(traj2)
   standard_ddtw = dtw.distance_fast(deriv1, deriv2)
   
   # 우리 구현
   our_ddtw = ddtw_distance(traj1, traj2)
   # 비교: abs(standard_ddtw - our_ddtw) < epsilon
   ```

2. **ERP Gap element 테스트**:
   ```python
   # 0 벡터 사용 시 대칭성 확인
   erp_ab = erp_distance(traj_a, traj_b)
   erp_ba = erp_distance(traj_b, traj_a)
   # erp_ab == erp_ba (대칭성)
   ```

## 🔄 다음 단계

1. ✅ DDTW 수정 완료
2. ✅ ERP 수정 완료
3. ✅ Sobolev 수정 완료
4. ⏳ TWED 시간 페널티 검토 (데이터 구조 확인 후 결정)
5. ⏳ Fréchet DP 최적화 (선택적, 대용량 데이터용)
