# Entropy 분석 스크립트 사용 가이드

실험 로그에서 Entropy와 Trust 값을 분석하고 시각화하는 스크립트들의 사용법을 설명합니다.

---

## 스크립트 위치

```
src/test_script/action_entropy/
├── analyze_step_entropy.py      # Action Logprobs Shannon Entropy 분석
└── analyze_entropy_trust.py     # Entropy/Trust 종합 분석
```

---

## 1. analyze_step_entropy.py

### 개요

`experiment_log.json` 파일에서 각 step의 `action_logprobs` 첫 번째 항목의 logprobs 값을 사용하여 Shannon Entropy를 계산하고 시각화합니다.

### 사용법

```bash
# 프로젝트 루트에서 실행
cd /home/syaro/DeepL_WS/multigrid-LaC

# 기본 사용법
python src/test_script/action_entropy/analyze_step_entropy.py <experiment_log.json 경로>

# 예시 1: 상대 경로 사용
python src/test_script/action_entropy/analyze_step_entropy.py "logs_good/EPISODE 1 - scenario2_absolute_example_map_20260123_173946/experiment_log.json"

# 예시 2: 절대 경로 사용
python src/test_script/action_entropy/analyze_step_entropy.py "/home/syaro/DeepL_WS/multigrid-LaC/logs_good/EPISODE 1 - scenario2_absolute_example_map_20260123_173946/experiment_log.json"
```

### 입력 데이터 구조

스크립트는 JSON 파일에서 다음 구조를 읽습니다:

```json
{
  "step": 1,
  "vlm_response": {
    "action_logprobs_info": {
      "action_logprobs": [
        [
          "east",
          ["east:-0.0000", "north:-13.9009", "eat:-15.6525", ...],
          0.00006148,
          13
        ],
        ...
      ]
    }
  }
}
```

- `action_logprobs[0][1]`: 첫 번째 action의 top-k logprobs 리스트 (예: `"east:-0.0000"`)

### Shannon Entropy 계산

```
H(X) = -Σ p(x) * log2(p(x))

where p(x) = exp(logprob) / Σ exp(logprobs)
```

### 출력 파일

입력 JSON 파일과 **같은 디렉토리**에 저장됩니다:

| 파일명 | 내용 |
|--------|------|
| `step_entropy_experiment_log.txt` | Step별 Entropy 값 (TSV 형식) |
| `step_entropy_experiment_log.png` | Entropy 시각화 그래프 |

### 출력 예시 (txt)

```
Step #	H(Step#)
1	0.000030
2	0.000054
3	0.000045
...
```

---

## 2. analyze_entropy_trust.py

### 개요

`experiment_log.json` 파일에서 각 step의 Entropy 값들(`entropy_H_X`, `entropy_H_X_given_S`, `entropy_H_X_given_LS`)과 Trust 값(`trust_T`)을 추출하여 시각화합니다.

### 사용법

```bash
# 프로젝트 루트에서 실행
cd /home/syaro/DeepL_WS/multigrid-LaC

# 기본 사용법
python src/test_script/action_entropy/analyze_entropy_trust.py <experiment_log.json 경로>

# 예시 1: src/logs_good 내의 파일
python src/test_script/action_entropy/analyze_entropy_trust.py "src/logs_good/Episode_2_2_Test_Entropy/experiment_log.json"

# 예시 2: logs_good 내의 파일
python src/test_script/action_entropy/analyze_entropy_trust.py "logs_good/EPISODE 1 - scenario2_absolute_example_map_20260123_173946/experiment_log.json"
```

### 입력 데이터 구조

스크립트는 JSON 파일에서 다음 필드를 읽습니다:

```json
{
  "step": 1,
  "entropy_H_X": 0.0003487322124026907,
  "entropy_H_X_given_S": 0.00033772235066927827,
  "entropy_H_X_given_LS": 0.00011363915141295733,
  "trust_T": 0.04683192981988192
}
```

- `null` 값은 자동으로 처리됩니다 (그래프에서 건너뜀, 전후 유효값은 회색 점선으로 연결)

### 그래프 특징

1. **이중 Y축**:
   - 왼쪽 Y축: Trust T (빨간색)
   - 오른쪽 Y축: Entropy (파란색, 청록색, 보라색)

2. **null 값 처리**:
   - null 값이 있는 step은 플롯에서 제외
   - 전후 유효값 사이를 회색 점선으로 연결

3. **이상치 필터링**:
   - 표준편차 2배를 넘는 값은 시각화에서 제외
   - Y축 범위는 유효 값의 5%~95% 백분위수 기준으로 자동 설정

4. **평균선**:
   - 각 값의 평균을 해당 색상의 점선으로 표시
   - 범례에 평균값 포함

5. **한글 폰트**:
   - Noto Sans CJK 자동 감지 및 사용

### 출력 파일

입력 JSON 파일과 **같은 디렉토리**에 저장됩니다:

| 파일명 | 내용 |
|--------|------|
| `entropy_trust_experiment_log.txt` | Step별 Entropy/Trust 값 (TSV 형식) |
| `entropy_trust_experiment_log.png` | Entropy/Trust 시각화 그래프 |

### 출력 예시 (txt)

```
Step #	H(X)	H(X|S)	H(X|L,S)	Trust T
1	0.000349	0.000338	0.000114	0.046832
2	0.000527	0.000360	0.000348	0.929567
3	0.000407	0.000143	0.000160	1.067522
4	0.054553	0.000017	null	null
...
```

---

## 경로 지정 팁

### 상대 경로 vs 절대 경로

| 경로 유형 | 예시 | 실행 위치 |
|-----------|------|-----------|
| 상대 경로 | `logs_good/EPISODE 1.../experiment_log.json` | 프로젝트 루트에서 실행 필요 |
| 절대 경로 | `/home/syaro/.../experiment_log.json` | 어디서든 실행 가능 |

### 공백이 포함된 경로

경로에 공백이 포함된 경우 **반드시 따옴표로 감싸야 합니다**:

```bash
# 올바른 사용
python script.py "logs_good/EPISODE 1 - scenario2/experiment_log.json"

# 잘못된 사용 (오류 발생)
python script.py logs_good/EPISODE 1 - scenario2/experiment_log.json
```

### 일반적인 로그 경로 패턴

```bash
# 프로젝트 루트의 logs_good
logs_good/<에피소드명>/experiment_log.json

# src 내의 logs_good
src/logs_good/<에피소드명>/experiment_log.json

# src 내의 logs
src/logs/<에피소드명>/experiment_log.json
```

---

## 의존성

두 스크립트 모두 다음 Python 패키지가 필요합니다:

```
numpy
matplotlib
```

한글 표시를 위해 시스템에 Noto Sans CJK 폰트가 설치되어 있어야 합니다:

```bash
# Ubuntu/Debian
sudo apt install fonts-noto-cjk
```

---

## 문제 해결

### 한글이 깨지는 경우

Noto Sans CJK 폰트가 설치되어 있는지 확인하세요:

```bash
fc-list | grep -i "noto.*cjk"
```

### JSON 로딩 오류

- 파일 경로가 올바른지 확인
- JSON 파일이 유효한 형식인지 확인
- 파일 인코딩이 UTF-8인지 확인

### 데이터 없음 오류

- `experiment_log.json` 파일에 step 데이터가 있는지 확인
- `action_logprobs` 또는 `entropy_H_X` 등의 필드가 존재하는지 확인

---

## Box Plot 해석 가이드

### Box Plot 구조

Box Plot(상자 그림)은 데이터 분포를 요약해서 보여주는 시각화입니다.

```
    ○  ← Outlier (이상치): 1.5×IQR 밖의 데이터
    |
   ─┬─  ← Upper Whisker: Q3 + 1.5×IQR 이내 최대값
    │
   ┌┴┐
   │ │  ← Q3 (75th percentile): 상위 25% 경계
   ├─┤  ← Median (50th percentile): 중앙값
   │ │  ← Q1 (25th percentile): 하위 25% 경계
   └┬┘
    │
   ─┴─  ← Lower Whisker: Q1 - 1.5×IQR 이내 최소값
    |
    ○  ← Outlier (이상치)
```

### 구성 요소 설명

| 구성 요소 | 설명 |
|-----------|------|
| **박스 (Box)** | 데이터의 50%가 포함된 범위 (Q1~Q3) |
| **박스 높이 (IQR)** | Interquartile Range = Q3 - Q1, 데이터 분산의 척도 |
| **중앙선** | Median (중앙값), 데이터의 중심 경향 |
| **수염 (Whisker)** | Q1 - 1.5×IQR ~ Q3 + 1.5×IQR 범위 |
| **흰색 동그라미** | Outlier (이상치), 수염 범위를 벗어난 극단값 |

### Trust T Box Plot 해석 예시

Trust 값의 Box Plot에서 에피소드 간 비교:

| Episode | 박스 크기 | 해석 |
|---------|-----------|------|
| Episode 1 | 매우 작음 | Trust 값이 일관됨 (대부분 0 근처) |
| Episode 2 | 커짐 | Trust 값의 변동성 증가 |
| Episode 3 | 가장 큼 | Trust 값이 매우 불안정함 |

### 분산 증가의 의미

에피소드가 진행됨에 따라 **박스가 커지는 현상**은:

1. **긍정적 해석**:
   - VLM이 더 다양한 상황을 경험
   - 일부 상황에서 매우 높은 Trust 달성 (이상치 상승)

2. **부정적 해석**:
   - Grounding의 효과가 일관되지 않음
   - 어떤 상황에서는 도움이 되고 (T > 0), 다른 상황에서는 해가 됨 (T < 0)
   - "안정적인 개선"보다 "불안정한 확산" 패턴

3. **개선 방향**:
   - 이상치가 발생하는 상황 분석
   - 음수 Trust가 나오는 케이스 식별
   - Grounding 전략의 일관성 향상 필요

### 이상적인 Trust 변화 패턴

```
이상적인 패턴:
Episode 1 → 2 → 3
- 중앙값: 상승
- 박스 크기: 유지 또는 감소
- 이상치: 양의 방향만

실제 관찰 패턴:
Episode 1 → 2 → 3
- 중앙값: 약간 상승
- 박스 크기: 증가
- 이상치: 양/음 방향 모두 증가
```

이 패턴은 Grounding이 **평균적으로** 도움이 되지만, **일관성** 있게 도움이 되지는 않음을 나타냅니다.
