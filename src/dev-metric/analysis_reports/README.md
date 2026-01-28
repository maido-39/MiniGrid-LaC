# 궤적 메트릭 분석 보고서

이 폴더에는 각 그룹별 종합 분석 보고서와 시각화 결과가 포함되어 있습니다.

## 폴더 구조

```
analysis_reports/
├── hogun/
│   ├── 종합_분석_보고서.md          # 이미지 포함 종합 보고서
│   ├── visualizations/              # Episode 트렌드 플롯
│   │   ├── episode_trends.png
│   │   ├── all_metrics_comparison.png
│   │   ├── rmse_trend.png
│   │   ├── dtw_trend.png
│   │   ├── frechet_trend.png
│   │   ├── erp_trend.png
│   │   ├── ddtw_trend.png
│   │   ├── twed_trend.png
│   │   └── sobolev_trend.png
│   └── methodology/                 # 각 메트릭 방법론 시각화
│       ├── rmse_comparison.png
│       ├── dtw_warping.png
│       ├── frechet_matching.png
│       ├── ddtw_derivatives.png
│       └── sobolev_components.png
├── hogun_0125/
│   └── (동일 구조)
└── other/
    └── (동일 구조)
```

## 보고서 내용

각 그룹별 `종합_분석_보고서.md`에는 다음이 포함됩니다:

1. **메트릭 경향성과 궤적 특성의 관계 분석**
   - 각 메트릭이 Episode 증가에 따라 어떻게 변화하는지
   - 궤적의 어떤 특성(위치, 속도, 형상, 시간 등)이 메트릭 값에 영향을 주는지
   - 각 메트릭이 민감/둔감한 요소

2. **Episode 증가에 따른 경향성 종합 분석**
   - 전체적인 경향 패턴
   - 모순적 경향성 해석
   - 경로 길이, 시간 정렬 등의 영향

3. **이 실험 데이터에 적합한 메트릭 분석**
   - 안정성 관점 평가
   - Episode 독립성 관점 평가
   - 실험 목적에 따른 메트릭 추천
   - 최종 권장사항

4. **이미지 및 시각화 분석**
   - 각 메트릭별 트렌드 플롯
   - 방법론 시각화 (각 메트릭이 어떻게 비교하는지)
   - 이미지에 대한 상세 분석

## 주요 발견사항

### hogun / hogun_0125 그룹
- 위치 정확도는 개선되지만 경로 형태는 더 달라짐
- 추천 메트릭: **DDTW** (1순위), **Sobolev** (2순위)

### other 그룹
- 경로 길이가 매우 다양함 (27~78 steps)
- RMSE는 유일하게 감소하지만 다른 메트릭은 증가
- 추천 메트릭: **RMSE** (1순위), **Sobolev** (2순위)
- 경로 길이에 민감한 DTW, ERP, TWED는 비추천

## 보고서 읽는 방법

1. 각 그룹 폴더의 `종합_분석_보고서.md`를 열어보세요.
2. 이미지는 보고서 내에 markdown 형식으로 포함되어 있습니다.
3. 각 이미지 아래에 해당 이미지에 대한 상세 분석이 포함되어 있습니다.
4. 방법론 시각화는 각 메트릭이 어떻게 두 궤적을 비교하는지 직관적으로 보여줍니다.
