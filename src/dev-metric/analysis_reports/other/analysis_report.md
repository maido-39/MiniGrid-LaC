# OTHER Group Analysis Report

**Analysis Date**: 2026-01-28 23:59:09

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
- Mean: 2.579605
- Std: 0.572809
- Coefficient of Variation: 0.2221 (Medium stability)
- Range: [1.527525, 3.247072]
- Episode Trend: -0.323526 per episode (decreasing)

### DTW

**Description**: Allows non-linear time warping to match similar shapes. Can match trajectories even if robot stops temporarily.

**Sensitive to**: Path shape, overall trajectory pattern

**Insensitive to**: Temporary stops, speed variations

**Statistics**:
- Mean: 9.851972
- Std: 6.552759
- Coefficient of Variation: 0.6651 (Low stability)
- Range: [4.123106, 23.388031]
- Episode Trend: +4.193451 per episode (increasing)

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
- Mean: 32.216045
- Std: 9.000564
- Coefficient of Variation: 0.2794 (Medium stability)
- Range: [15.872507, 42.669883]
- Episode Trend: -4.454285 per episode (decreasing)

### DDTW

**Description**: Compares derivatives (velocity vectors) instead of positions. Naturally penalizes stops and reverse movements.

**Sensitive to**: Stops, backtracking, velocity changes

**Insensitive to**: Position offsets, baseline shifts

**Statistics**:
- Mean: 4.934454
- Std: 1.355973
- Coefficient of Variation: 0.2748 (Medium stability)
- Range: [3.297726, 7.241374]
- Episode Trend: +0.508159 per episode (increasing)

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

- **RMSE**: Trend = -0.323526, CV = 0.2221
- **DDTW**: Trend = +0.508159, CV = 0.2748
- **Fréchet**: Trend = +1.006710, CV = 0.6032

### Least Stable Metrics (Highest Episode Dependency)

- **DTW**: Trend = +4.193451, CV = 0.6651
- **ERP**: Trend = -4.454285, CV = 0.2794
- **TWED**: Trend = +10.434435, CV = 0.6153

## 4. Key Insights

### Stability Analysis

- **Most Stable Metric**: RMSE (trend: -0.323526)
  - This metric shows the least variation across episodes, suggesting it measures aspects that are consistent regardless of episode number.

- **Least Stable Metric**: TWED (trend: +10.434435)
  - This metric shows the most variation, indicating it is sensitive to aspects that change between episodes.

### Episode Dependency

- **Increasing with Episode**: DTW, Fréchet, DDTW, TWED, Sobolev
  - These metrics suggest that later episodes deviate more from GT.

- **Decreasing with Episode**: RMSE, ERP
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

Metrics DTW, Fréchet, DDTW, TWED, Sobolev increase with episode number, which may indicate:
- Performance degradation in later episodes
- Accumulated errors
- Different experimental conditions

