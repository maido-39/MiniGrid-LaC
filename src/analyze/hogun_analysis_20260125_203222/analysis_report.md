# Hogun Episodes Analysis Report

**Analysis Date**: 2026-01-26 00:12:42

---

## 1. Episode Overview

| Episode | Total Steps |
|---------|-------------|
| Episode 1 | 35 |
| Episode 2 | 46 |
| Episode 3 | 38 |

## 2. Statistical Summary

### H(X)

| Episode | Valid Count | Null Ratio | Mean | Std | Min | Max | Median |
|---------|-------------|------------|------|-----|-----|-----|--------|
| Episode 1 | 28 | 20.0% | 0.005473 | 0.013200 | 0.000022 | 0.062465 | 0.000829 |
| Episode 2 | 35 | 23.9% | 0.026426 | 0.101893 | 0.000022 | 0.590242 | 0.001041 |
| Episode 3 | 31 | 18.4% | 0.003105 | 0.008319 | 0.000012 | 0.035876 | 0.000477 |

### H(X|S)

| Episode | Valid Count | Null Ratio | Mean | Std | Min | Max | Median |
|---------|-------------|------------|------|-----|-----|-----|--------|
| Episode 1 | 27 | 22.9% | 0.004963 | 0.014212 | 0.000014 | 0.074782 | 0.000738 |
| Episode 2 | 37 | 19.6% | 0.050988 | 0.230361 | 0.000020 | 1.364914 | 0.000520 |
| Episode 3 | 28 | 26.3% | 0.015538 | 0.037427 | 0.000040 | 0.164584 | 0.000907 |

### H(X|L,S)

| Episode | Valid Count | Null Ratio | Mean | Std | Min | Max | Median |
|---------|-------------|------------|------|-----|-----|-----|--------|
| Episode 1 | 31 | 11.4% | 0.001084 | 0.001466 | 0.000048 | 0.007434 | 0.000475 |
| Episode 2 | 38 | 17.4% | 0.047704 | 0.284222 | 0.000140 | 1.776544 | 0.000572 |
| Episode 3 | 29 | 23.7% | 0.007557 | 0.032402 | 0.000032 | 0.177925 | 0.000598 |

### Trust T

| Episode | Valid Count | Null Ratio | Mean | Std | Min | Max | Median |
|---------|-------------|------------|------|-----|-----|-----|--------|
| Episode 1 | 22 | 37.1% | -1.833298 | 7.082041 | -29.954272 | 2.064893 | 0.492790 |
| Episode 2 | 25 | 45.7% | -5.334973 | 33.739728 | -169.790405 | 14.837056 | 0.899636 |
| Episode 3 | 20 | 47.4% | 12.479493 | 37.596620 | -7.221516 | 170.432361 | 0.804934 |

## 3. Analysis Results

### H(X) - Overall Entropy

- **Lowest mean H(X)**: Episode 3 (0.003105)
- Lower H(X) indicates higher confidence in VLM's first action token.

### H(X|L,S) - Conditional Entropy based on Logprobs

- **Lowest mean H(X|L,S)**: Episode 1 (0.001084)
- H(X|L,S) reflects action decision certainty from actual logprobs distribution.

### Trust T

- Episode 1: Mean Trust = -1.8333 (±7.0820)
- Episode 2: Mean Trust = -5.3350 (±33.7397)
- Episode 3: Mean Trust = 12.4795 (±37.5966)

### Null Value Ratio Analysis

- **H(X)**: Average Null Ratio = 20.8%
- **H(X|S)**: Average Null Ratio = 22.9%
- **H(X|L,S)**: Average Null Ratio = 17.5%
- **Trust T**: Average Null Ratio = 43.4%

## 4. Trajectory Analysis

| Episode | Start Position | End Position | Total Distance | Direct Distance | Efficiency | Unique Positions | Revisit Count |
|---------|----------------|--------------|----------------|-----------------|------------|------------------|---------------|
| Episode 1 | (3, 11) | (8, 3) | 33 | 13 | 39.4% | 33 | 2 |
| Episode 2 | (3, 11) | (8, 3) | 45 | 13 | 28.9% | 36 | 10 |
| Episode 3 | (3, 11) | (10, 5) | 37 | 13 | 35.1% | 27 | 11 |

### Path Efficiency Analysis

- **Efficiency** = Direct Distance / Total Distance × 100%
- Higher efficiency indicates a more direct path to the goal.
- Higher revisit count suggests inefficient path planning.

## 5. Conclusions

Based on the analysis:

1. **Entropy Comparison**: Compare mean entropy values across episodes to evaluate VLM's decision confidence.
2. **Trust Patterns**: Higher Trust variance indicates more situation-dependent reliability changes.
3. **Data Completeness**: Higher null ratio indicates more missing data for metric calculation.
4. **Path Efficiency**: Episodes with higher efficiency and lower revisit count achieved better path planning.
