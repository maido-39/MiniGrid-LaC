# HOGUN_0125 Group Analysis Report

**Analysis Date**: 2026-01-28 23:59:09

**Ground Truth**: hogun_0125_episode1 (Episode 1)

**Total Episodes Analyzed**: 2

---

## 1. Executive Summary

This report analyzes trajectory similarity metrics across different episodes, using Episode 1 as the Ground Truth (GT). Ideally, all metrics should remain constant regardless of episode number, indicating consistent performance. However, in practice, metrics may vary, revealing which aspects of trajectory following are most stable or sensitive.

## 2. Metric Characteristics Analysis

### RMSE

**Description**: Measures point-to-point Euclidean distance assuming perfect time synchronization. Most sensitive to timing misalignment.

**Sensitive to**: Position errors, time synchronization

**Insensitive to**: Time warping, path shape variations

**Statistics**:
- Mean: 2.080772
- Std: 0.631634
- Coefficient of Variation: 0.3036 (Low stability)
- Range: [1.449138, 2.712405]
- Episode Trend: -1.263268 per episode (decreasing)

### DTW

**Description**: Allows non-linear time warping to match similar shapes. Can match trajectories even if robot stops temporarily.

**Sensitive to**: Path shape, overall trajectory pattern

**Insensitive to**: Temporary stops, speed variations

**Statistics**:
- Mean: 8.122421
- Std: 2.554657
- Coefficient of Variation: 0.3145 (Low stability)
- Range: [5.567764, 10.677078]
- Episode Trend: +5.109314 per episode (increasing)

### Fréchet

**Description**: Measures geometric similarity independent of time. Focuses on the "shape" of the path, like a dog leash analogy.

**Sensitive to**: Geometric shape, path topology

**Insensitive to**: Time, speed, temporary detours

**Statistics**:
- Mean: 2.509882
- Std: 1.095669
- Coefficient of Variation: 0.4365 (Low stability)
- Range: [1.414214, 3.605551]
- Episode Trend: +2.191338 per episode (increasing)

### ERP

**Description**: Handles gaps by matching points with a gap element. May treat small detours as gaps rather than errors.

**Sensitive to**: Large gaps, missing segments

**Insensitive to**: Small detours, local deviations

**Statistics**:
- Mean: 23.170278
- Std: 6.362569
- Coefficient of Variation: 0.2746 (Medium stability)
- Range: [16.807709, 29.532847]
- Episode Trend: -12.725138 per episode (decreasing)

### DDTW

**Description**: Compares derivatives (velocity vectors) instead of positions. Naturally penalizes stops and reverse movements.

**Sensitive to**: Stops, backtracking, velocity changes

**Insensitive to**: Position offsets, baseline shifts

**Statistics**:
- Mean: 3.913688
- Std: 0.163688
- Coefficient of Variation: 0.0418 (High stability)
- Range: [3.750000, 4.077377]
- Episode Trend: -0.327377 per episode (decreasing)

### TWED

**Description**: Adds explicit penalty for time warping. More sensitive to temporal aspects than pure DTW.

**Sensitive to**: Time penalties, speed changes, stops

**Insensitive to**: Position-only errors

**Statistics**:
- Mean: 92.243637
- Std: 34.753310
- Coefficient of Variation: 0.3768 (Low stability)
- Range: [57.490328, 126.996947]
- Episode Trend: -69.506619 per episode (decreasing)

### Sobolev

**Description**: Combines position error and velocity error in a weighted sum. Most comprehensive but may be sensitive to both aspects.

**Sensitive to**: Both position and velocity errors

**Insensitive to**: Neither (comprehensive metric)

**Statistics**:
- Mean: 15.780576
- Std: 1.578551
- Coefficient of Variation: 0.1000 (Medium stability)
- Range: [14.202026, 17.359127]
- Episode Trend: +3.157102 per episode (increasing)

## 3. Episode Trend Analysis

The following analysis examines how each metric changes as episode number increases.

**Ideal Behavior**: All metrics should remain constant (horizontal line) regardless of episode number.

**Observed Behavior**:

### Most Stable Metrics (Lowest Episode Dependency)

- **DDTW**: Trend = -0.327377, CV = 0.0418
- **RMSE**: Trend = -1.263268, CV = 0.3036
- **Fréchet**: Trend = +2.191338, CV = 0.4365

### Least Stable Metrics (Highest Episode Dependency)

- **DTW**: Trend = +5.109314, CV = 0.3145
- **ERP**: Trend = -12.725138, CV = 0.2746
- **TWED**: Trend = -69.506619, CV = 0.3768

## 4. Key Insights

### Stability Analysis

- **Most Stable Metric**: DDTW (trend: -0.327377)
  - This metric shows the least variation across episodes, suggesting it measures aspects that are consistent regardless of episode number.

- **Least Stable Metric**: TWED (trend: -69.506619)
  - This metric shows the most variation, indicating it is sensitive to aspects that change between episodes.

### Episode Dependency

- **Increasing with Episode**: DTW, Fréchet, Sobolev
  - These metrics suggest that later episodes deviate more from GT.

- **Decreasing with Episode**: RMSE, ERP, DDTW, TWED
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

Metrics DTW, Fréchet, Sobolev increase with episode number, which may indicate:
- Performance degradation in later episodes
- Accumulated errors
- Different experimental conditions

