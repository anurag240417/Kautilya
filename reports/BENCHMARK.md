# Kautilya Benchmark Report

- Dataset: **Elliptic++ (txs_features/txs_classes)**
- Features: M1 (blockchain) (182 columns)
- Split: train steps 1-34, test steps after (strictly out-of-time)
- Train rows: 29,894; test rows: 16,670
- Seed: 42

## 1. Detector comparison (same test rows)

Test positives: 1,083 of 16,670 (base rate 6.497%). PR-AUC of a random detector equals the base rate.

| Detector | PR-AUC | PR-AUC 95% CI | ROC-AUC | P@100 | P@100 95% CI | Recall@500 | Best-F1 | FPR at best-F1 |
|---|---|---|---|---|---|---|---|---|
| Random (chance) | 0.066 | [0.061, 0.072] | 0.501 | 0.050 | [0.010, 0.100] | 0.032 | 0.123 | 0.8132 |
| Rules only | 0.058 | [0.054, 0.061] | 0.263 | 0.000 | [0.000, 0.010] | 0.001 | 0.122 | 1.0000 |
| Logistic regression | 0.300 | [0.279, 0.322] | 0.884 | 0.290 | [0.200, 0.390] | 0.152 | 0.447 | 0.0647 |
| Isolation Forest only | 0.036 | [0.034, 0.038] | 0.154 | 0.000 | [0.000, 0.000] | 0.000 | 0.122 | 0.9999 |
| Random Forest (ML only) | 0.794 | [0.774, 0.815] | 0.932 | 1.000 | [1.000, 1.000] | 0.462 | 0.825 | 0.0012 |
| Fused: RF + anomaly | 0.777 | [0.755, 0.800] | 0.892 | 1.000 | [1.000, 1.000] | 0.462 | 0.819 | 0.0022 |
| Fused: RF + anomaly + rules | 0.724 | [0.700, 0.748] | 0.865 | 0.990 | [0.970, 1.000] | 0.453 | 0.713 | 0.0058 |

Best-F1 picks its threshold on the test data, so read it as a ceiling. PR-AUC and precision@k are the headline numbers.

### Precision@k and lift over chance

| Detector | P@50 | P@100 | P@200 | P@500 | Lift@100 |
|---|---|---|---|---|---|
| Random (chance) | 0.020 | 0.050 | 0.065 | 0.070 | 0.8x |
| Rules only | 0.000 | 0.000 | 0.000 | 0.002 | 0.0x |
| Logistic regression | 0.300 | 0.290 | 0.315 | 0.330 | 4.5x |
| Isolation Forest only | 0.000 | 0.000 | 0.000 | 0.000 | 0.0x |
| Random Forest (ML only) | 1.000 | 1.000 | 1.000 | 1.000 | 15.4x |
| Fused: RF + anomaly | 1.000 | 1.000 | 1.000 | 1.000 | 15.4x |
| Fused: RF + anomaly + rules | 0.980 | 0.990 | 0.995 | 0.982 | 15.2x |

## 2. Stability across time windows (rolling origin)

Retrained at each cutoff, tested on the next unseen window.

| Train steps | Test steps | Test n | Positives | Base rate | ML PR-AUC | Rules PR-AUC | ML P@100 |
|---|---|---|---|---|---|---|---|
| 1-20 | 21-25 | 5,311 | 566 | 10.657% | 0.840 | 0.094 | 1.000 |
| 1-25 | 26-30 | 2,705 | 617 | 22.810% | 0.977 | 0.203 | 1.000 |
| 1-30 | 31-35 | 4,330 | 690 | 15.935% | 0.991 | 0.137 | 1.000 |
| 1-34 | 35-39 | 5,486 | 447 | 8.148% | 0.958 | 0.076 | 1.000 |
| 1-39 | 40-44 | 7,458 | 515 | 6.905% | 0.828 | 0.062 | 1.000 |
| 1-44 | 45-49 | 3,726 | 121 | 3.247% | 0.324 | 0.033 | 0.320 |

## 3. Robustness to noisy or missing features

Deployed model, degraded test features (no retraining).

**Gaussian noise (sd = level x training std)**

| Noise level | RF PR-AUC | Fused PR-AUC | Anomaly PR-AUC |
|---|---|---|---|
| 0.0 | 0.794 | 0.777 | 0.036 |
| 0.1 | 0.189 | 0.057 | 0.037 |
| 0.25 | 0.122 | 0.049 | 0.037 |
| 0.5 | 0.095 | 0.047 | 0.039 |
| 1.0 | 0.079 | 0.048 | 0.042 |

**Feature dropout (cells replaced by training mean)**

| Dropout rate | RF PR-AUC | Fused PR-AUC | Anomaly PR-AUC |
|---|---|---|---|
| 0.0 | 0.794 | 0.777 | 0.036 |
| 0.1 | 0.777 | 0.731 | 0.036 |
| 0.25 | 0.723 | 0.458 | 0.037 |
| 0.5 | 0.435 | 0.096 | 0.038 |

## Notes and limits

- Rules baseline uses five fixed heuristics (fan-out, fan-in, peel shape, pass-through hub, large value); thresholds were set before looking at test data.
- The fused rows call the production `synthesize_risk_score`, so they measure what the dashboard ranks by.
- Elliptic++ labels are a known, partial ground truth. Unlabelled (class 3) rows are excluded.
- Held-out *scenario-type* generalisation is not covered here. Elliptic++ has no scenario labels; the rolling-origin test is the out-of-distribution check.
