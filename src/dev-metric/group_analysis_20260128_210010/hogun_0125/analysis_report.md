# HOGUN_0125 Group Analysis Report

**Analysis Date**: 2026-01-28 21:00:23

**Ground Truth**: episode1 (Episode 1)

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
- Mean: 2.942655
- Std: 0.893265
- Coefficient of Variation: 0.3036 (Low stability)
- Range: [2.049390, 3.835920]
- Episode Trend: -1.786530 per episode (decreasing)

### DTW

**Description**: Allows non-linear time warping to match similar shapes. Can match trajectories even if robot stops temporarily.

**Sensitive to**: Path shape, overall trajectory pattern

**Insensitive to**: Temporary stops, speed variations

**Statistics**:
- Mean: 36.895733
- Std: 11.345956
- Coefficient of Variation: 0.3075 (Low stability)
- Range: [25.549776, 48.241689]
- Episode Trend: +22.691912 per episode (increasing)

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
- Mean: 119.827868
- Std: 34.985041
- Coefficient of Variation: 0.2920 (Medium stability)
- Range: [84.842827, 154.812909]
- Episode Trend: -69.970082 per episode (decreasing)

### DDTW

**Description**: Compares derivatives (velocity vectors) instead of positions. Naturally penalizes stops and reverse movements.

**Sensitive to**: Stops, backtracking, velocity changes

**Insensitive to**: Position offsets, baseline shifts

**Statistics**:
- Mean: 16.311747
- Std: 1.638248
- Coefficient of Variation: 0.1004 (Medium stability)
- Range: [14.673499, 17.949994]
- Episode Trend: -3.276495 per episode (decreasing)

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

- **RMSE**: Trend = -1.786530, CV = 0.3036
- **Fréchet**: Trend = +2.191338, CV = 0.4365
- **Sobolev**: Trend = +3.157102, CV = 0.1000

### Least Stable Metrics (Highest Episode Dependency)

- **DTW**: Trend = +22.691912, CV = 0.3075
- **TWED**: Trend = -69.506619, CV = 0.3768
- **ERP**: Trend = -69.970082, CV = 0.2920

## 4. Key Insights

### Stability Analysis

- **Most Stable Metric**: RMSE (trend: -1.786530)
  - This metric shows the least variation across episodes, suggesting it measures aspects that are consistent regardless of episode number.

- **Least Stable Metric**: ERP (trend: -69.970082)
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

