# Action Entropy 분석 문서

이 폴더는 Gemini 모델을 사용한 로봇 제어 시스템에서 Action Entropy를 추출하고 분석한 전체 과정을 문서화합니다.

## 문서 구조

```
docs/entropy/
├── README.md                          # 이 파일
├── action-entropy-analysis.md         # 전체 분석 보고서
└── images/                            # 분석 결과 플롯 이미지
    ├── minigrid_debug.png             # MiniGrid 환경 이미지
    ├── temperature_entropy_analysis.png
    ├── direction_distribution_heatmap.png
    └── entropy_boxplot_by_temperature.png
```

## 주요 문서

### [Action Entropy 분석 보고서](./action-entropy-analysis.md)

전체 분석 과정을 상세히 기록한 문서입니다. 다음 내용을 포함합니다:

- **개요**: 연구 배경 및 목적
- **방법론**: Verbalized Confidence, Step-wise 확률 분포, AFCE
- **실험 설계**: Temperature 범위, Language Prompts, System Prompt 구조
- **초기 결과 및 문제점**: 높은 Entropy, East Bias, 명령어 무반응 등
- **개선 과정**: 시나리오 기반 → Step-wise, AFCE 도입, 프롬프트 최적화
- **Temperature 분석**: 상세한 결과 분석 및 시각화
- **결론 및 향후 과제**: 성과, 남은 문제점, 향후 연구 방향

## 핵심 개념

### Action Entropy
로봇 행동의 불확실성을 측정하는 지표로, Shannon Entropy를 기반으로 계산됩니다.

```
H(Step_i) = -Σ p(direction) × log₂(p(direction))
Final Entropy = 0.5 × H(Step1) + 0.3 × H(Step2) + 0.2 × H(Step3)
```

### Verbalized Confidence
LLM이 직접 텍스트로 출력하는 확률값을 사용하는 방식으로, 내부 log-probability보다 더 정확하게 교정된 값을 제공합니다.

### Answer-Free Confidence Estimation (AFCE)
답변 생성과 신뢰도 평가를 분리하여 모델의 과잉 확신 문제를 완화하는 기법입니다.

## 실험 결과 요약

### Temperature 분석 결과

| 프롬프트 | Entropy 범위 | Executability | 특징 |
|---------|-------------|---------------|------|
| **uncertain** | 0.85~1.24 | 0.62~0.72 | 동작함, East bias 강함 |
| **certain** | 대부분 2.0 (실패) | 0.5 (실패 시) | 간헐적 성공 |
| **strange** | 항상 2.0 | 0.5 | 완전 실패, 균일 분포 |

### 주요 발견

1. **Temperature 효과**: 낮은 Temperature에서 더 결정적, 높은 Temperature에서 더 다양한 응답
2. **East Bias**: uncertain 프롬프트에서 매우 강함 (특히 낮은 Temperature)
3. **객체 인식 한계**: MiniGrid 환경에서 일반 객체("apple", "desktop pc") 인식 실패

## 관련 코드

- **노트북**: `src/test_script/action_entropy/gemini_action_export.ipynb`
- **Temperature 분석**: `src/test_script/action_entropy/temperature_analysis/temperature_entropy_analysis.py`
- **결과 데이터**: `src/test_script/action_entropy/temperature_analysis/results/run_20260127_004251/`

## 참고 문헌

- Tian et al. (2023) - Verbalized Confidence 논문

---

**최종 업데이트**: 2026-01-27
