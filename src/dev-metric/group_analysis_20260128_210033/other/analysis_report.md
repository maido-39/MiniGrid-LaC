# OTHER Group Analysis Report

**Analysis Date**: 2026-01-28 21:00:46

**Ground Truth**: Episode_1_1 (Episode 1)

**Total Episodes Analyzed**: 6

---

## 1. Executive Summary

This report analyzes trajectory similarity metrics across different episodes, using Episode 1 as the Ground Truth (GT). Ideally, all metrics should remain constant regardless of episode number, indicating consistent performance. However, in practice, metrics may vary, revealing which aspects of trajectory following are most stable or sensitive.

## 2. Metric Characteristics Analysis

### RMSE

**Description**: Measures point-to-point Euclidean distance assuming perfect time synchronization. Most sensitive to timing misalignment.

**Sensitive to**: Position errors, time synchronization

**Insensitive to**: Time warping, path shape variations

**Statistics**:
- Mean: 3.648113
- Std: 0.810074
- Coefficient of Variation: 0.2221 (Medium stability)
- Range: [2.160247, 4.592054]
- Episode Trend: -0.457534 per episode (decreasing)

### DTW

**Description**: Allows non-linear time warping to match similar shapes. Can match trajectories even if robot stops temporarily.

**Sensitive to**: Path shape, overall trajectory pattern

**Insensitive to**: Temporary stops, speed variations

**Statistics**:
- Mean: 55.952591
- Std: 42.379112
- Coefficient of Variation: 0.7574 (Low stability)
- Range: [13.899495, 134.018874]
- Episode Trend: +27.980940 per episode (increasing)

### Fréchet

**Description**: Measures geometric similarity independent of time. Focuses on the "shape" of the path, like a dog leash analogy.

**Sensitive to**: Geometric shape, path topology

**Insensitive to**: Time, speed, temporary detours

**Statistics**:
- Mean: 2.903830
- Std: 1.751505
- Coefficient of Variation: 0.6032 (Low stability)
- Range: [1.414214, 6.708204]
- Episode Trend: +1.006710 per episode (increasing)

### ERP

**Description**: Handles gaps by matching points with a gap element. May treat small detours as gaps rather than errors.

**Sensitive to**: Large gaps, missing segments

**Insensitive to**: Small detours, local deviations

**Statistics**:
- Mean: 178.932471
- Std: 87.566737
- Coefficient of Variation: 0.4894 (Low stability)
- Range: [88.661716, 349.853645]
- Episode Trend: +14.978026 per episode (increasing)

### DDTW

**Description**: Compares derivatives (velocity vectors) instead of positions. Naturally penalizes stops and reverse movements.

**Sensitive to**: Stops, backtracking, velocity changes

**Insensitive to**: Position offsets, baseline shifts

**Statistics**:
- Mean: 24.191775
- Std: 11.661616
- Coefficient of Variation: 0.4820 (Low stability)
- Range: [10.060798, 44.564119]
- Episode Trend: +4.598868 per episode (increasing)

### TWED

**Description**: Adds explicit penalty for time warping. More sensitive to temporal aspects than pure DTW.

**Sensitive to**: Time penalties, speed changes, stops

**Insensitive to**: Position-only errors

**Statistics**:
- Mean: 199.721181
- Std: 122.889178
- Coefficient of Variation: 0.6153 (Low stability)
- Range: [99.631691, 463.563389]
- Episode Trend: +10.434435 per episode (increasing)

### Sobolev

**Description**: Combines position error and velocity error in a weighted sum. Most comprehensive but may be sensitive to both aspects.

**Sensitive to**: Both position and velocity errors

**Insensitive to**: Neither (comprehensive metric)

**Statistics**:
- Mean: 18.026675
- Std: 6.113456
- Coefficient of Variation: 0.3391 (Low stability)
- Range: [11.903314, 31.009465]
- Episode Trend: +3.322789 per episode (increasing)

## 3. Episode Trend Analysis

The following analysis examines how each metric changes as episode number increases.

**Ideal Behavior**: All metrics should remain constant (horizontal line) regardless of episode number.

**Observed Behavior**:

### Most Stable Metrics (Lowest Episode Dependency)

- **RMSE**: Trend = -0.457534, CV = 0.2221
- **Fréchet**: Trend = +1.006710, CV = 0.6032
- **Sobolev**: Trend = +3.322789, CV = 0.3391

### Least Stable Metrics (Highest Episode Dependency)

- **TWED**: Trend = +10.434435, CV = 0.6153
- **ERP**: Trend = +14.978026, CV = 0.4894
- **DTW**: Trend = +27.980940, CV = 0.7574

## 4. Key Insights

### Stability Analysis

- **Most Stable Metric**: RMSE (trend: -0.457534)
  - This metric shows the least variation across episodes, suggesting it measures aspects that are consistent regardless of episode number.

- **Least Stable Metric**: DTW (trend: +27.980940)
  - This metric shows the most variation, indicating it is sensitive to aspects that change between episodes.

### Episode Dependency

- **Increasing with Episode**: DTW, Fréchet, ERP, DDTW, TWED, Sobolev
  - These metrics suggest that later episodes deviate more from GT.

- **Decreasing with Episode**: RMSE
  - These metrics suggest that later episodes are closer to GT.

## 5. Recommendations

### For Trajectory Evaluation

Based on the analysis:

1. **For Overall Similarity**: Use metrics that show high stability (low CV) and low episode dependency.

2. **For Detecting Specific Issues**:
   - Use **DDTW** or **TWED** to detect stops and backtracking
   - Use **Fréchet** to evaluate geometric path similarity
   - Use **Sobolev** for comprehensive position+velocity evaluation

3. **For Time-Sensitive Analysis**: Use **RMSE** or **DTW** to evaluate temporal alignment.

### For Episode Comparison

Metrics DTW, Fréchet, ERP, DDTW, TWED, Sobolev increase with episode number, which may indicate:
- Performance degradation in later episodes
- Accumulated errors
- Different experimental conditions

