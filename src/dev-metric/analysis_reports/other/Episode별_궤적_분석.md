# OTHER 그룹 Episode별 궤적 분석

**생성 일시**: 2026-01-29 16:23:10

이 문서는 **other** 그룹의 각 Episode( **Episode_1_1 포함** )가 그린 **Trajectory(궤적)**를 시각화하고, **전체 Path 구조**와 **Episode 증가에 따른 경향성**을 요약합니다.

### 목차
1. [전체 궤적 개요](#1-전체-궤적-개요)
2. [전체 Path 구조](#2-전체-path-구조)
3. [Episode 증가에 따른 경향성](#3-episode-증가에-따른-경향성)
4. [Episode별 궤적 상세](#4-episode별-궤적-상세)
5. [요약](#5-요약)

---

## 1. 전체 궤적 개요

**Episode_1_1(Reference)**을 포함한 other 그룹 **전체 7개 Episode**의 궤적을 한 그림에 겹쳐 표시합니다.

![전체 궤적 개요](trajectories/overview_all_trajectories.png)

- **Reference**: Episode_1_1 (46 steps)
- **분석 대상**: other 그룹 Episode **7개** (Episode_1_1 포함)

---

## 2. 전체 Path 구조

모든 Episode 궤적이 사용하는 **공간 범위**와 **경로 길이** 요약입니다.

| 항목 | 값 |
|------|-----|
| X 범위 (grid column) | 2.0 ~ 12.0 |
| Y 범위 (grid row) | 1.0 ~ 11.0 |
| 경로 길이 (steps) 최소 | 27 |
| 경로 길이 (steps) 최대 | 78 |
| 경로 길이 평균 | 49.7 |
| Episode 수 | 7 |

**해석**: 경로 길이가 **27~78 steps**로 약 2.9배 차이로, Episode에 따라 **짧은 직선형**부터 **긴 우회형**까지 다양한 경로가 선택되었습니다.

---

## 3. Episode 증가에 따른 경향성

Episode 번호가 커질 때 **경로 길이**와 **Reference 대비 메트릭**이 어떻게 변하는지 요약합니다.

### 3.1 경로 길이 (Episode별)

![경로 길이 by Episode](trajectories/path_length_by_episode.png)

Episode_1_1(Reference)은 46 steps이며, 비교 대상 6개 Episode는 27~78 steps로 다양합니다. Episode 번호와 경로 길이 사이에 단순한 증가/감소 관계는 없고, **같은 Episode 번호에서도 시도별로 길이가 다릅니다** (예: Episode 2의 40 vs 41 steps).

### 3.2 Reference 대비 메트릭 추이 (비교 6개 Episode)

![메트릭 추이 by Episode](trajectories/metric_trends_by_episode.png)

| 메트릭 | Episode 증가 시 경향 | 해석 |
|--------|----------------------|------|
| RMSE | 감소  | slope=-0.324, p=0.0802 |
| DTW | 증가 (유의) | slope=4.193, p=0.0281 |
| Fréchet | 증가  | slope=1.007, p=0.0719 |
| DDTW | 증가  | slope=0.508, p=0.3085 |
| Sobolev | 증가  | slope=3.323, p=0.0993 |

**통찰**: RMSE는 Episode가 커질수록 **감소**하는 경향(위치 정확도 개선), DTW·Fréchet·Sobolev는 **증가**하는 경향(경로 형태가 Reference와 더 달라짐)을 보입니다. 즉 **로컬 정확도는 좋아지지만, 전체 경로 선택은 더 달라지는** 패턴입니다.

---

## 4. Episode별 궤적 상세

### Episode_1_1

- **역할**: Reference (GT)
- **Episode 번호**: 1
- **궤적 길이**: 46 steps
#### 궤적만 보기 (Step 진행에 따른 경로)

![Episode_1_1 궤적만](trajectories/Episode_1_1_only.png)

#### Reference vs 이 Episode 비교

(Reference이므로 동일 궤적이 두 선으로 겹쳐 보입니다.)

![Episode_1_1 vs GT](trajectories/Episode_1_1_vs_GT.png)

---

### Episode_1_2_Test_Entropy

- **Episode 번호**: 1
- **궤적 길이**: 60 steps
- **메트릭 (vs Reference)**
  - RMSE: 3.2471
  - DTW: 5.6569
  - Fréchet: 2.0000
  - ERP: 42.2339
  - DDTW: 4.6637
  - TWED: 194.4573
  - Sobolev: 17.1204

#### 궤적만 보기 (Step 진행에 따른 경로)

![Episode_1_2_Test_Entropy 궤적만](trajectories/Episode_1_2_Test_Entropy_only.png)

#### Reference vs 이 Episode 비교

![Episode_1_2_Test_Entropy vs GT](trajectories/Episode_1_2_Test_Entropy_vs_GT.png)

---

### Episode_2_1_Test_Entropy

- **Episode 번호**: 2
- **궤적 길이**: 40 steps
- **메트릭 (vs Reference)**
  - RMSE: 2.5495
  - DTW: 4.1231
  - Fréchet: 1.4142
  - ERP: 29.9204
  - DDTW: 3.2977
  - TWED: 99.6317
  - Sobolev: 11.9033

#### 궤적만 보기 (Step 진행에 따른 경로)

![Episode_2_1_Test_Entropy 궤적만](trajectories/Episode_2_1_Test_Entropy_only.png)

#### Reference vs 이 Episode 비교

![Episode_2_1_Test_Entropy vs GT](trajectories/Episode_2_1_Test_Entropy_vs_GT.png)

---

### Episode_2_2_Test_Entropy

- **Episode 번호**: 2
- **궤적 길이**: 41 steps
- **메트릭 (vs Reference)**
  - RMSE: 2.5231
  - DTW: 5.1962
  - Fréchet: 2.2361
  - ERP: 30.2135
  - DDTW: 3.4187
  - TWED: 105.9601
  - Sobolev: 14.2454

#### 궤적만 보기 (Step 진행에 따른 경로)

![Episode_2_2_Test_Entropy 궤적만](trajectories/Episode_2_2_Test_Entropy_only.png)

#### Reference vs 이 Episode 비교

![Episode_2_2_Test_Entropy vs GT](trajectories/Episode_2_2_Test_Entropy_vs_GT.png)

---

### Episode_3_2_Test_Entropy

- **Episode 번호**: 3
- **궤적 길이**: 78 steps
- **메트릭 (vs Reference)**
  - RMSE: 3.2033
  - DTW: 11.3137
  - Fréchet: 2.8284
  - ERP: 42.6699
  - DDTW: 7.2414
  - TWED: 463.5634
  - Sobolev: 17.5407

#### 궤적만 보기 (Step 진행에 따른 경로)

![Episode_3_2_Test_Entropy 궤적만](trajectories/Episode_3_2_Test_Entropy_only.png)

#### Reference vs 이 Episode 비교

![Episode_3_2_Test_Entropy vs GT](trajectories/Episode_3_2_Test_Entropy_vs_GT.png)

---

### Episode_4_1_Test_Entropy

- **Episode 번호**: 4
- **궤적 길이**: 56 steps
- **메트릭 (vs Reference)**
  - RMSE: 2.4272
  - DTW: 9.4340
  - Fréchet: 2.2361
  - ERP: 32.3860
  - DDTW: 5.3561
  - TWED: 161.6042
  - Sobolev: 16.3408

#### 궤적만 보기 (Step 진행에 따른 경로)

![Episode_4_1_Test_Entropy 궤적만](trajectories/Episode_4_1_Test_Entropy_only.png)

#### Reference vs 이 Episode 비교

![Episode_4_1_Test_Entropy vs GT](trajectories/Episode_4_1_Test_Entropy_vs_GT.png)

---

### Episode_5_2_Test_Entropy

- **Episode 번호**: 5
- **궤적 길이**: 27 steps
- **메트릭 (vs Reference)**
  - RMSE: 1.5275
  - DTW: 23.3880
  - Fréchet: 6.7082
  - ERP: 15.8725
  - DDTW: 5.6292
  - TWED: 173.1104
  - Sobolev: 31.0095

#### 궤적만 보기 (Step 진행에 따른 경로)

![Episode_5_2_Test_Entropy 궤적만](trajectories/Episode_5_2_Test_Entropy_only.png)

#### Reference vs 이 Episode 비교

![Episode_5_2_Test_Entropy vs GT](trajectories/Episode_5_2_Test_Entropy_vs_GT.png)

---

## 5. 요약

| Episode | 궤적 길이 (steps) | Episode 번호 |
|---------|-------------------|-------------|
| Episode_1_1 | 46 | 1 |
| Episode_1_2_Test_Entropy | 60 | 1 |
| Episode_2_1_Test_Entropy | 40 | 2 |
| Episode_2_2_Test_Entropy | 41 | 2 |
| Episode_3_2_Test_Entropy | 78 | 3 |
| Episode_4_1_Test_Entropy | 56 | 4 |
| Episode_5_2_Test_Entropy | 27 | 5 |

