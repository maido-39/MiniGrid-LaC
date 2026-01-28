# Trust Analysis Report

**Analysis Date**: 2026-01-25 21:24:01

---

## 1. Trust Formula

```
T = (H(X) - H(X|S)) / (H(X) - H(X|L,S))
```

Where:
- **H(X)**: Entropy without any context (maximum uncertainty)
- **H(X|S)**: Entropy with Grounding only
- **H(X|L,S)**: Entropy with Language Instruction + Grounding (minimum uncertainty)
- **Numerator**: Uncertainty reduction from Grounding alone
- **Denominator**: Total uncertainty reduction

**Interpretation**:
- T ≈ 1: Grounding alone is very effective
- T ≈ 0: Language Instruction is more important than Grounding
- T < 0: Grounding increases uncertainty (problematic)
- T > 1: Adding Language Instruction increases uncertainty (problematic)

---

## 2. Statistical Summary

| Episode | Valid Trust | Mean | Median | Std | % in [0,1] | % Negative |
|---------|-------------|------|--------|-----|------------|------------|
| Episode 1 | 22 | -1.833 | 0.493 | 7.082 | 63.6% | 27.3% |
| Episode 2 | 25 | -5.335 | 0.900 | 33.740 | 32.0% | 28.0% |
| Episode 3 | 20 | 12.479 | 0.805 | 37.597 | 20.0% | 35.0% |

---

## 3. Issues Identified

### Issue 1: Extreme Trust Values

Trust values range from very negative to very positive, indicating formula instability.

### Issue 2: Small Denominator Problem

When H(X) ≈ H(X|L,S), the denominator approaches zero, causing Trust to explode.

**Solution**: Apply minimum threshold to denominator or use alternative formulation.

### Issue 3: Negative Numerator

When H(X|S) > H(X), grounding actually increases uncertainty.

**Possible causes**:
- Grounding information is misleading or confusing
- VLM interpretation of grounding varies
- Sampling variance in entropy estimation

### Issue 4: Entropy Ordering Violations

Theoretically: H(X) ≥ H(X|S) ≥ H(X|L,S)

Violations indicate:
- Stochastic VLM behavior
- Context sometimes confuses rather than helps
- Need for more robust entropy estimation

---

## 4. Positive Evidence

- **35/67** (52.2%) Trust values are in well-behaved range [0, 2]
- Positive cumulative Trust shows grounding has net positive effect
- Trend analysis shows slight positive improvement over steps

---

## 5. Recommendations

1. **Apply denominator threshold**: Set minimum denominator to 0.001 to avoid division instability
2. **Filter extreme values**: Focus analysis on Trust in [-2, 3] range
3. **Use robust statistics**: Report median instead of mean for Trust
4. **Increase sample size**: More episodes needed for statistically significant trends
5. **Investigate violations**: Analyze specific steps where entropy ordering is violated
