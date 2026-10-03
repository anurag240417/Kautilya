# ChainTrace Scale Test

- Machine: Windows-11-10.0.26200-SP0, Python 3.14.2
- Transactions: 509,504; addresses: 1,637,091; network observations: 1,019,231
- Entities resolved: 1,374,409; active (>=2 events): 332,446; alerts raised: 30,102
- Peak memory: 1836 MB
- Data generation: 39 s (not part of the pipeline)
- **Pipeline total: 67 s** (7,647 tx/s)

| Stage | Seconds |
|---|---|
| heuristics | 0.7 |
| entities | 8.8 |
| graph_signals | 42.6 |
| model | 9.2 |
| fusion | 1.1 |
| peer_groups | 1.6 |
| total | 66.6 |

Fused-score PR-AUC on this run: 0.995 (synthetic ground truth, 2-fold cross-fit).

Notes: model training caps negatives at 40,000 entities; Random Forest trees: 100; folds: 2. Fusion runs on candidate entities only.
