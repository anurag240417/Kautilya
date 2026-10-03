# ChainTrace Forensics Benchmark (entity level, synthetic ground truth)

- Dataset: synthetic, seed 11
- Transactions: 58,193
- Entities evaluated: 34,355 (575 illicit)
- Illicit entities per scenario: {'darknet_market': 172, 'dust_attack': 36, 'layering': 204, 'ransomware': 163}
- All numbers are out-of-fold; folds are grouped by campaign so no campaign is in both train and test.
- Data is synthetic. Treat results as evidence the method works on the modelled typologies, not as real-world accuracy.

## 1. Ablation: what each feature family adds

| Feature set | PR-AUC | 95% CI | P@10 | P@50 | P@100 |
|---|---|---|---|---|---|
| behavioural_only | 0.964 | [0.947, 0.978] | 1.000 | 0.980 | 0.990 |
| +network | 0.974 | [0.961, 0.985] | 1.000 | 0.980 | 0.990 |
| +heuristics | 0.981 | [0.970, 0.989] | 1.000 | 1.000 | 1.000 |
| +graph_propagation | 0.988 | [0.979, 0.996] | 1.000 | 1.000 | 1.000 |
| +graph_embeddings (full) | 0.993 | [0.987, 0.998] | 1.000 | 1.000 | 1.000 |

## 2. Baselines

| Detector | PR-AUC | 95% CI | ROC-AUC | P@10 | P@50 | P@100 |
|---|---|---|---|---|---|---|
| random | 0.017 | [0.016, 0.020] | 0.509 | 0.000 | 0.020 | 0.040 |
| rules_only (structural heuristics) | 0.882 | [0.860, 0.904] | 0.958 | 1.000 | 1.000 | 1.000 |
| anomaly_only (IsolationForest) | 0.276 | [0.250, 0.316] | 0.940 | 0.000 | 0.260 | 0.280 |
| logistic_regression | 0.989 | [0.983, 0.994] | 1.000 | 1.000 | 1.000 | 1.000 |
| RF_behavioural_only | 0.964 | [0.947, 0.978] | 0.988 | 1.000 | 0.980 | 0.990 |
| RF_full (ours, model only) | 0.993 | [0.987, 0.998] | 1.000 | 1.000 | 1.000 | 1.000 |
| FUSED (ours) | 0.984 | [0.975, 0.992] | 0.998 | 1.000 | 1.000 | 1.000 |

## 3. Unseen scenarios (leave-one-scenario-out)

Each row trains with that scenario removed entirely, then tests on it (plus held-out normal entities).

| Held-out scenario | Test pos / neg | Rules only | RF behavioural | RF full | FUSED |
|---|---|---|---|---|---|
| darknet_market | 172 / 10081 | 0.737 | 0.454 | 0.627 | 0.785 |
| dust_attack | 36 / 10081 | 0.993 | 0.493 | 0.904 | 0.974 |
| layering | 204 / 10081 | 1.000 | 0.996 | 1.000 | 1.000 |
| ransomware | 163 / 10081 | 0.873 | 0.760 | 0.983 | 0.982 |

(PR-AUC; base rate of positives in each test set is low, so chance is near 0.)

## 4. Robustness (RF full, noisy or missing test features)

| Perturbation | Level | PR-AUC | P@25 |
|---|---|---|---|
| noise | 0.0 | 0.993 | 1.000 |
| noise | 0.25 | 0.972 | 1.000 |
| noise | 0.5 | 0.944 | 1.000 |
| noise | 1.0 | 0.896 | 1.000 |
| dropout | 0.0 | 0.993 | 1.000 |
| dropout | 0.25 | 0.965 | 1.000 |
| dropout | 0.5 | 0.926 | 1.000 |

## 5. Heuristic validation against planted truth

```json
{
  "coinjoin": {
    "flagged": 49,
    "true_positives": 49,
    "precision": 1.0,
    "recall": 1.0
  },
  "dust_spray": {
    "flagged": 35,
    "true_positives": 35,
    "precision": 1.0,
    "recall": 0.7291666666666666
  },
  "peel_chains": {
    "chains_detected": 15,
    "chains_mostly_illicit_sender": 13,
    "chain_precision": 0.8666666666666667,
    "ransomware_operators_with_detected_chain": 12,
    "ransomware_operators_total": 12,
    "operator_recall": 1.0
  },
  "layering_chains": {
    "chains_detected": 49,
    "chain_precision": 1.0,
    "mule_or_source_recall": 1.0
  },
  "change_detection": {
    "detected": 38234,
    "correct": 37123,
    "precision": 0.9709420934247005,
    "coverage_of_2_output_txs": 0.855289354182046
  }
}
```

## 6. Entity resolution quality

```json
{
  "n_clusters": 79341,
  "n_true_entities": 10074,
  "address_weighted_purity": 1.0,
  "mean_clusters_per_multi_address_entity": 8.112331861587432,
  "share_true_entities_fully_merged": 0.0013348393058835609
}
```
