# Kautilya Scale Test

- Machine: Windows-11-10.0.26200-SP0, Python 3.14.2
- Transactions: 200,127; addresses: 464,303; network observations: 399,975
- Entities resolved: 357,728; active (>=2 events): 125,833; alerts raised: 12,090
- Peak memory: 1060 MB
- Data generation: 14 s (not part of the pipeline)
- **Pipeline total: 20 s** (9,779 tx/s)

| Stage | Seconds |
|---|---|
| heuristics | 0.3 |
| entities | 2.1 |
| graph_signals | 11.0 |
| model | 5.3 |
| fusion | 0.4 |
| peer_groups | 0.6 |
| total | 20.5 |

Fused-score PR-AUC on this run: 0.994 (synthetic ground truth, 2-fold cross-fit).

Notes: model training caps negatives at 40,000 entities; Random Forest trees: 100; folds: 2. Fusion runs on candidate entities only.
