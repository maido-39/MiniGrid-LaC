# 메트릭 구현 마이그레이션 가이드

## 개요

본 프로젝트의 메트릭 구현을 공신력 있는 라이브러리로 마이그레이션했습니다.

## 변경사항

### 라이브러리 기반 구현으로 전환

기존의 custom 구현을 검증된 라이브러리로 교체하여:
- **성능 향상**: C 기반 고속 연산 활용
- **정확도 향상**: 검증된 알고리즘 사용
- **유지보수성 향상**: 표준 라이브러리 사용

### 메트릭별 변경사항

#### 1. DTW (Dynamic Time Warping)
- **이전**: Custom DP 구현
- **현재**: `dtaidistance.dtw_ndim` 사용
- **라이브러리**: `dtaidistance>=2.3.0`
- **변경 이유**: C 기반 고속 연산, 2D 궤적 지원

#### 2. DDTW (Derivative Dynamic Time Warping)
- **이전**: Custom DTW 호출
- **현재**: `dtaidistance.dtw_ndim` 사용 (derivative 계산 후)
- **라이브러리**: `dtaidistance>=2.3.0`
- **변경 이유**: DTW와 동일한 라이브러리 사용으로 일관성

#### 3. TWED (Time Warp Edit Distance)
- **이전**: Custom DP 구현
- **현재**: Legacy 구현 유지
- **이유**: `distancia`는 timestamps가 필요하여 현재 데이터 구조와 호환되지 않음
- **참고**: Legacy 구현은 검증된 알고리즘 사용

#### 4. Fréchet Distance
- **이전**: Custom recursive 구현
- **현재**: `similaritymeasures.frechet_dist` 사용
- **라이브러리**: `similaritymeasures>=0.3.0`
- **변경 이유**: 표준 구현 사용

#### 5. ERP (Edit Distance on Real sequence)
- **이전**: Custom DP 구현
- **현재**: `aeon.distances.erp_distance` 사용
- **라이브러리**: `aeon>=0.8.0`
- **변경 이유**: `similaritymeasures`에는 ERP가 없음, aeon은 표준 구현 제공

#### 6. RMSE (Root Mean Square Error)
- **이전**: Custom 구현
- **현재**: `sklearn.metrics.mean_squared_error` 사용 후 sqrt 적용
- **라이브러리**: `scikit-learn>=1.0.0`
- **변경 이유**: 표준 라이브러리 사용

#### 7. Sobolev Metric
- **변경 없음**: Custom 구현 유지
- **이유**: 표준 라이브러리에 Sobolev metric이 없으며, 문서에서도 custom 구현 제시

## Legacy 코드

기존 구현은 다음 위치에 보관되어 있습니다:
- `metrics/legacy/`: 기존 메트릭 구현 파일들
- `analysis_reports/legacy/`: 기존 분석 결과

## 설치 방법

```bash
pip install -r requirements.txt
```

또는 개별 설치:
```bash
pip install numpy scipy scikit-learn
pip install dtaidistance  # for DTW, DDTW
pip install aeon          # for ERP
pip install similaritymeasures  # for Fréchet
```

## API 호환성

모든 메트릭 함수의 시그니처는 기존과 동일하게 유지되어 하위 호환성을 보장합니다:

```python
from metrics import (
    dtw_distance,
    ddtw_distance,
    twed_distance,
    frechet_distance,
    erp_distance,
    rmse_distance,
    sobolev_distance
)

# 사용법은 동일
distance = dtw_distance(traj1, traj2)
```

## 성능 개선

- **DTW/DDTW**: C 기반 구현으로 30-300배 속도 향상
- **Fréchet**: 최적화된 알고리즘 사용
- **ERP**: 검증된 라이브러리 사용

## 주의사항

1. **데이터 타입**: `dtaidistance`는 `np.double` 타입을 요구하지만, 내부에서 자동 변환합니다.
2. **TWED**: Legacy 구현 사용 (timestamps가 없는 데이터 구조 때문)
3. **에러 처리**: 라이브러리별 에러는 적절히 처리되며, 실패 시 `np.inf` 반환

## 결과 비교

새로운 라이브러리 기반 구현의 결과는 기존 구현과 약간 다를 수 있습니다:
- **정밀도 차이**: 부동소수점 연산 순서 차이
- **알고리즘 차이**: 라이브러리별 최적화 방법 차이

Legacy 결과와 비교하려면 `analysis_reports/legacy/` 폴더를 참조하세요.
