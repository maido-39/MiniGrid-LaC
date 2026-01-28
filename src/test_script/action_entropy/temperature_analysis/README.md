# Temperature vs Entropy Analysis

## 개요
Temperature 변화에 따른 Action Entropy 분포 및 답변 경향성 분석

## 실험 설계

### Temperature 범위
- **범위**: 0.2 ~ 1.5
- **간격**: 0.15
- **값**: [0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.1, 1.25, 1.4]

### Language Prompts (3종류)

| 유형 | 프롬프트 | 기대 결과 |
|------|----------|----------|
| **불확실 (uncertain)** | "move toward toilet, usually colored room is toilet." | 높은 entropy, 분산된 확률 |
| **확실 (certain)** | "i'm hungry. grab some apple" | 낮은 entropy, 집중된 확률 |
| **이상함 (strange)** | "find and move toward desktop pc" | 매우 높은 entropy (이미지에 없음) |

### 반복 횟수
- 각 (Temperature × Prompt) 조합당 **5회** 반복
- 총 실험 횟수: 9 × 3 × 5 = **135회**

## 실행 방법

```bash
# 방법 1: 셸 스크립트
./run_analysis.sh

# 방법 2: Python 직접 실행
python temperature_entropy_analysis.py
```

## 출력 파일

```
results/
└── run_YYYYMMDD_HHMMSS/
    ├── raw_results.json              # 전체 원본 데이터
    ├── temperature_entropy_analysis.png   # 메인 분석 플롯
    ├── direction_distribution_heatmap.png # 방향별 분포 히트맵
    └── entropy_boxplot_by_temperature.png # Box plot
```

## 분석 지표

1. **Entropy Mean ± STD**: Temperature에 따른 entropy 변화
2. **Entropy Variance**: Temperature가 높을수록 분산 증가 예상
3. **Executability**: 명령 수행 가능성 평가
4. **Direction Bias**: 특정 방향(특히 east)으로의 편향

## 예상 결과

| Temperature | Entropy 경향 | 분산 경향 |
|-------------|-------------|----------|
| 낮음 (0.2~0.5) | 낮음 (결정적) | 낮음 |
| 중간 (0.5~1.0) | 중간 | 중간 |
| 높음 (1.0~1.5) | 높음 (랜덤) | 높음 |

## 참고
- Shannon Entropy 계산: H = -Σ p(x) log₂(p(x))
- 가중 평균: Step1×0.5 + Step2×0.3 + Step3×0.2
- 최대 entropy (uniform): 2.0 bits (4방향)
