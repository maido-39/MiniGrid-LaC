# Trajectory Metric Analysis

로봇 궤적(Robot Trajectory)과 참조 궤적(Reference Path)을 비교하기 위한 다양한 메트릭 구현 및 분석 도구입니다.

## 개요

이 프로젝트는 **사용자의 주행 의도(Intention)가 포함된 시공간 궤적(Spatio-Temporal Trajectory)**을 평가하기 위한 메트릭들을 제공합니다. 단순 위치 비교를 넘어서, 이동 방향, 속도, 정지, 역주행 등의 동역학적 특성을 고려합니다.

## 주요 특징

1. **Topology 보존**: 위치뿐만 아니라 이동 방향과 순서를 고려
2. **Kinematic Constraints**: 정지(Stop)와 역주행(Backtracking)을 페널티로 반영
3. **Local Deviation**: 미세한 경로 이탈을 오차 누적으로 계산

## 구현된 메트릭

### 고우선순위 메트릭 (Proposed Methods)

#### 1. DDTW (Derivative Dynamic Time Warping)
- **파일**: `metrics/ddtw.py`
- **핵심 개념**: 위치 좌표 대신 미분값(속도/방향 벡터)을 사용한 DTW
- **장점**:
  - 정지(Stop) 감지: 로봇이 멈추면 미분값이 0이 되어 큰 오차 발생
  - 역주행(Backtracking) 감지: 미분 벡터의 방향이 반대가 되면 큰 오차 기록
- **수학적 정의**: 
  - 궤적 $P(t) = (x(t), y(t))$의 미분값 $P'(t) = (x'(t), y'(t))$ 계산
  - 미분값 시퀀스에 DTW 적용

#### 2. TWED (Time Warp Edit Distance)
- **파일**: `metrics/twed.py`
- **핵심 개념**: 시간 축 왜곡에 명시적 페널티를 부과하는 Edit Distance
- **파라미터**:
  - `nu` (stiffness parameter): 시간 왜곡 페널티 (기본값: 0.5)
  - `lambda_param`: 삭제/삽입 페널티 (기본값: 1.0)
- **장점**: 속도 변화나 정차를 비용으로 환산하여 '과정의 충실도' 평가

#### 3. Sobolev Metric (H¹ Norm)
- **파일**: `metrics/sobolev.py`
- **핵심 개념**: 위치 오차와 속도 오차를 가중 합산
- **수학적 정의**:
  $$
  d_{Sobolev}(P_1, P_2) = \sqrt{\alpha \|P_1 - P_2\|^2 + \beta \|P_1' - P_2'\|^2}
  $$
  - $\alpha$: 위치 오차 가중치 (기본값: 1.0)
  - $\beta$: 속도 오차 가중치 (기본값: 1.0)
- **장점**: 수학적으로 엄밀하게 "위치도 비슷하고 속도도 비슷한가"를 정의

### Baseline 메트릭

#### 4. DTW (Dynamic Time Warping)
- **파일**: `metrics/dtw.py`
- **설명**: 시계열 비교의 표준 알고리즘
- **특징**: 시간 축을 비선형적으로 늘려서 두 경로의 형태를 최대한 매칭
- **한계점**: 로봇이 멈춰 있어도 시간 축을 무한정 늘려서 매칭시켜 버리므로, "멈춤"에 대한 페널티가 거의 없음

#### 5. Fréchet Distance
- **파일**: `metrics/frechet.py`
- **설명**: 기하학적 형상 비교의 표준 (강아지 산책 거리)
- **특징**: 시간 속도와 상관없이 "경로의 모양(Shape)"이 얼마나 일치하는지 평가
- **한계점**: 로봇이 제자리에 있거나 되돌아가도, 전체적인 궤적의 '그림'이 같으면 거리가 0이 나올 수 있음

#### 6. ERP (Edit Distance on Real sequence)
- **파일**: `metrics/erp.py`
- **설명**: 문자열 편집 거리를 실수 값에 적용
- **특징**: Gap(누락) 처리에 특화
- **한계점**: 로봇의 미세한 우회나 경로 변경을 '오차'가 아닌 'Gap'으로 처리해버리는 맹점

#### 7. RMSE (Root Mean Square Error)
- **파일**: `metrics/rmse.py`
- **설명**: 가장 단순한 물리적 거리 평균
- **특징**: 시간이 딱딱 맞는다는 전제하에 같은 시간의 점끼리 유클리드 거리 계산
- **한계점**: 조금만 밀려도 값이 폭발하므로, 시퀀스 정렬 알고리즘이 필요한 이유를 설명하는 용도

## 설치 및 사용법

### 요구사항

```bash
pip install numpy matplotlib seaborn
```

### 기본 사용법

```python
from pathlib import Path
from dev_metric.analyzer import TrajectoryAnalyzer
from dev_metric.visualizer import TrajectoryVisualizer

# 분석기 초기화
logs_dir = Path('logs_good')
analyzer = TrajectoryAnalyzer(logs_dir, reference_name='Episode_1_1')

# 데이터 로드
analyzer.load_data()

# 모든 에피소드 분석
analyzer.analyze_all_episodes()

# 결과 저장
output_dir = Path('analysis_results')
analyzer.save_results(output_dir)

# 시각화 생성
visualizer = TrajectoryVisualizer(analyzer)
visualizer.generate_all_visualizations(output_dir)
```

### 명령줄 사용법

```bash
cd src/dev-metric
python main.py --logs-dir ../../logs_good --reference Episode_1_1 --output results
```

## 데이터 형식

입력 데이터는 `experiment_log.json` 파일에서 다음 형식으로 추출됩니다:

```json
{
  "step": 1,
  "state": {
    "agent_pos": [3, 11]
  }
}
```

- `step`: 시간 스텝 번호
- `agent_pos`: 로봇의 [x, y] 좌표

## 출력 결과

분석 실행 후 다음 파일들이 생성됩니다:

1. **detailed_results.json**: 모든 에피소드의 상세 메트릭 값
2. **metrics_summary.csv**: 에피소드별 메트릭 요약 (CSV 형식)
3. **statistics.json**: 메트릭별 통계 (평균, 표준편차, 최소, 최대 등)
4. **시각화 파일들**:
   - `01_metrics_comparison.png`: 에피소드별 메트릭 비교
   - `02_metrics_boxplot_by_group.png`: 그룹별 메트릭 분포
   - `03_metrics_heatmap.png`: 메트릭 값 히트맵
   - `04_statistics_summary.png`: 통계 요약
   - `trajectory_comparison_*.png`: 참조 궤적 vs. 에피소드 궤적 비교

## Episode 클러스터링

에피소드는 자동으로 다음 그룹으로 분류됩니다:

1. **hogun**: `hogun/` 또는 `Hogun/` 폴더의 에피소드
2. **hogun_0125**: `hogun_0125/` 폴더의 에피소드
3. **other**: 그 외 에피소드 (Episode_1_1, Episode_2_1 등)

## 메트릭 선택 가이드

### 정지(Stop)와 역주행(Backtracking)을 감지하려면
- **DDTW** 또는 **TWED** 사용 권장
- DDTW는 미분값 기반으로 자연스럽게 정지/역주행을 감지
- TWED는 시간 왜곡 페널티로 정차를 비용으로 환산

### 위치와 속도를 모두 고려하려면
- **Sobolev Metric** 사용 권장
- 위치 오차와 속도 오차를 가중 합산하여 평가

### 기하학적 형상만 비교하려면
- **Fréchet Distance** 사용
- 시간 속도와 무관하게 경로의 모양만 평가

### 표준 비교 기준이 필요하면
- **DTW** 사용
- 가장 널리 쓰이는 시계열 비교 알고리즘

## 참고 문헌

- **DDTW**: Keogh, E. J., & Pazzani, M. J. (2001). Derivative dynamic time warping.
- **TWED**: Marteau, P. F. (2009). Time warp edit distance with stiffness adjustment for time series matching.
- **Sobolev Metric**: 제어 이론에서 궤적 추종 성능 평가에 사용되는 표준 척도

## 라이선스

이 프로젝트는 연구 및 교육 목적으로 자유롭게 사용할 수 있습니다.

## 문의

문제가 발생하거나 개선 사항이 있으면 이슈를 등록해주세요.
