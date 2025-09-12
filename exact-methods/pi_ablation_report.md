# PI Optimization Ablation Study - Research Report

## Executive Summary

**Key Finding:** Greedy Init Only provides the most significant improvement over baseline, reducing iterations by 1.0 (7.7%) with large effect size.

**Statistical Summary:** 2 out of 3 optimization components showed statistically significant improvements (p < 0.05).

## Detailed Component Analysis

### 1. Greedy Init Only

- **Performance:** 12.0 ± 4.2 iterations (vs baseline: 13.0 ± 4.1)

- **Improvement:** 1.0 iterations (7.7% reduction)

- **Statistical Significance:** Significant (p = 0.000)

- **Effect Size:** Cohen's d = 1.378 (large effect)

- **95% CI:** [10.1, 13.9] iterations

- **Interpretation:** **Recommended** - Shows both statistical and practical significance


### 2. Full Optimized

- **Performance:** 13.8 ± 2.3 iterations (vs baseline: 13.0 ± 4.1)

- **Performance Change:** 0.8 iterations worse (5.8% increase)

- **Statistical Significance:** Not significant (p = 0.057)

- **Effect Size:** Cohen's d = -0.409 (small effect)

- **95% CI:** [12.7, 14.8] iterations

- **Interpretation:** Neither statistically nor practically significant


### 3. Howards Improvement Only

- **Performance:** 14.5 ± 2.6 iterations (vs baseline: 13.0 ± 4.1)

- **Performance Change:** 1.5 iterations worse (11.5% increase)

- **Statistical Significance:** Significant (p = 0.002)

- **Effect Size:** Cohen's d = -0.975 (large effect)

- **95% CI:** [13.3, 15.7] iterations

- **Interpretation:** **Recommended** - Shows both statistical and practical significance


## Component Interaction Analysis

The combined optimization shows **antagonistic** interaction.

- Expected additive improvement: -0.5 iterations

- Actual combined improvement: -0.8 iterations

- Interpretation: Components interfere with each other


## Practical Recommendations

### High Priority (Implement First)

- **Greedy Init Only**: 1.0 iteration improvement (large effect, p = 0.000)

- **Howards Improvement Only**: -1.5 iteration improvement (large effect, p = 0.002)


### Low Priority (Not Recommended)

- **Full Optimized**: No significant improvement over baseline


## Methodology Summary

- **Sample Sizes:** [20, 20, 20, 20] experiments per configuration

- **Statistical Tests:** Mann-Whitney U test (non-parametric)

- **Effect Size:** Cohen's d with pooled standard deviation

- **Significance Level:** α = 0.05

- **Confidence Intervals:** 95% CI using t-distribution

- **Effect Size Thresholds:** Small (0.2), Medium (0.5), Large (0.8)


## Suggestions for Future Work

- Test combinations of significant components for synergistic effects

- Evaluate performance across different environment types (gridworld variants)

- Investigate computational cost vs. performance trade-offs

- Replicate findings with larger sample sizes for marginal effects
