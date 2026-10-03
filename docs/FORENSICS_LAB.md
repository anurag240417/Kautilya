# Forensics Lab: Technical Write-Up

The Forensics Lab turns raw Bitcoin transaction and network metadata (the problem statement's schema) into
ranked, explainable investigative leads. It runs fully offline and sits beside the original Elliptic++ pipeline.

## 1. Pipeline

```
CSV / JSON / XML  ->  ingest + audit  ->  heuristics  ->  entity resolution  ->  features
   (timestamp, IPs, ports, txid,           (CoinJoin, dust, change,   (common-input +       (structure, timing,
    input/output addresses+amounts,         peel, layering, batch)     change-link union)    pass-through, network)
    fee, script, geo, asn)
                                                   |
   graph signals  ->  cross-fitted model  ->  fusion  ->  ranked alerts + evidence + summaries
   (PPMI-SVD embeddings,   (Random Forest +      (production risk    (HTML report, link graph,
    neighbour propagation)   Isolation Forest)    synthesizer)        analyst feedback loop)
```

| Stage | Module | What it does |
|---|---|---|
| Ingest | `backend/forensics/ingest.py` | Tolerant about form (aliases, list encodings, satoshi/BTC, epoch/ISO times), strict about content. Drops and counts bad rows in an audit report. |
| Dataset | `backend/forensics/dataset.py` | Flattens list columns into integer-indexed tables (`tx`, `inputs`, `outputs`, `obs`). Needed for scale. |
| IP intelligence | `backend/forensics/geo.py` | Country/ASN from real open databases if present (DB-IP Lite, MaxMind ASN CSV), else a built-in table. Tor exit list and hosting/VPN ASN list. |
| Heuristics | `backend/forensics/heuristics.py` | Vectorised CoinJoin, dust spray, change address, peel chain, rapid-hop layering, consolidation, batch payout. |
| Entities | `backend/forensics/entities.py` | Common-input ownership plus confident change links -> entity ids. 46 features per entity. |
| Graph ML | `backend/forensics/graphml.py` | Sparse PPMI-SVD node embeddings, neighbour propagation, peer-group clustering. NumPy/SciPy only. |
| Model | `backend/forensics/model.py` | Random Forest (calibrated, with intervals) + Isolation Forest + occlusion explanations. |
| Pipeline | `backend/forensics/pipeline.py` | Orchestrates, cross-fits, fuses through `synthesize_risk_score`. |
| Evidence | `backend/forensics/evidence.py` | Per-entity evidence package and plain-language summary. |
| Cases | `backend/forensics/cases.py` | SQLite feedback + cases, active-learning queue, retraining. |
| Report | `backend/forensics/report.py` | Printable HTML with SHA-256 chain of custody and offline verification. |

## 2. Model choices and why

* **Random Forest for the classifier.** Tabular entity features, strong on small/imbalanced data, no tuning fragility,
  per-tree spread gives an uncertainty interval for free, out-of-bag predictions give honest calibration data.
  (Note: the older Elliptic++ documents say "GBM"; the code in `backend/ml/classifier.py` is a Random Forest.)
* **Isotonic calibration on out-of-bag predictions.** "0.8" then means roughly 80% of similar training entities were illicit.
* **Isolation Forest for anomaly.** Unsupervised, reported as a percentile rank, and capped during fusion when it is the
  only strong signal: novelty is not guilt.
* **Graph embeddings without a deep-learning stack.** The PPMI of the 1+2-hop transition matrix factorised by SVD is the
  closed form that DeepWalk/node2vec approximate. It is sparse, deterministic, scales linearly, and needs no GPU.
  Neighbour propagation is the fixed-weight message-passing step of a GraphSAGE layer over label-free indicators.
* **Cross-fitting.** Entities are split into folds by *campaign*; every score shown is from a model that never saw that
  entity or its campaign. Dashboard numbers are therefore not in-sample.

## 3. Explainability

Three layers, each tagged with its provenance so observations, inferences and predictions are never conflated:

1. **Observation:** facts in the data (dust outputs sent, first-seen origin IP, Tor exit match).
2. **Inference:** heuristic conclusions with their thresholds and supporting TXIDs (peel chain of N hops, CoinJoin).
3. **Model prediction:** calibrated probability with a 90% interval, plus **occlusion** contributions: each factor is
   replaced by the training median and the change in probability is reported (`+x pts raises risk`). This is exact for
   the deployed model, model-agnostic, and verified in `tests/test_forensics_pipeline.py`.

Every alert also gets a deterministic, template-based plain-language summary. No external LLM is used.

## 4. Network-layer correlation

* First-seen origin per transaction (earliest observation), origin IP, country, ASN.
* Tor exit match and hosting/VPN ASN match; rapid country changes (consecutive sends from different countries within 10 minutes).
* **Wallet-to-IP link with uncertainty:** the share of an entity's sends from its most common IP with a Wilson 95%
  interval, always shown with a correlation-is-not-attribution note.

## 5. Evaluation (synthetic ground truth)

Full tables: `reports/FORENSICS_BENCHMARK.md` (58k transactions, 34,355 entities, 575 illicit; reproduce with
`python -m scripts.run_forensic_benchmark`). All figures are entity-level, out-of-fold, campaign-grouped.

| Detector | PR-AUC (95% CI) |
|---|---|
| Random | 0.017 |
| Anomaly only (Isolation Forest) | 0.276 |
| Rules only (structural heuristics) | 0.882 |
| Random Forest, behavioural features only | 0.964 |
| Logistic regression | 0.989 |
| Random Forest, all features | 0.993 |
| Fused risk score | 0.984 |

**Unseen scenarios (leave-one-scenario-out, PR-AUC)**

| Held out | Rules only | RF behavioural | RF full | Fused |
|---|---|---|---|---|
| darknet market | 0.737 | 0.454 | 0.627 | **0.785** |
| dust attack | 0.993 | 0.493 | 0.904 | 0.974 |
| layering | 1.000 | 0.996 | 1.000 | 1.000 |
| ransomware | 0.873 | 0.760 | 0.983 | 0.982 |

**Heuristic validation against planted truth:** CoinJoin 49/49; dust-spray precision 100% (recall 73%); peel chains caught
for 12/12 ransomware operators (chain precision 87%); layering chains precision and recall 100%; change detection 97.1%
precision at 85.5% coverage.

**Robustness:** PR-AUC stays above 0.89 with noise at one training standard deviation and 0.93 with half of the
features missing.

### What these numbers do and do not say

* The data is **synthetic and generated by us**. The heuristics target patterns the generator plants. Treat the results as
  proof the method works on the modelled typologies, not as real-world accuracy.
* **The problem is easy in-distribution:** plain logistic regression matches the Random Forest. We report it rather than hide it.
* **Fusion does not raise in-distribution PR-AUC** (0.984 vs 0.993). Its value is elsewhere: explainable tiers, the
  anomaly safety cap, and the hardest unseen scenario (darknet market: fused 0.785 vs RF 0.627).
* **Graph embeddings add little.** Ablation gains (0.964 -> 0.993) are within overlapping confidence intervals, and on
  another random seed the embeddings did not help. The honest summary is that heuristics and neighbour propagation do most of the work.
* **Entity resolution is pure but fragmented:** 100% purity, yet a true multi-address owner is split into ~8 clusters on average.
  Receive-only addresses cannot be linked without further evidence. Treat an entity as a "wallet fragment".
* The darknet-market typology is the weakest on unseen data; a real deployment needs labelled examples of each typology.

### Real labelled data: Elliptic++ (transactions)

Full tables: `reports/BENCHMARK.md` (`python -m scripts.run_benchmark`). Real Bitcoin transactions with anonymised features,
strict out-of-time split (train steps 1-34, test 35-49): 29,894 train rows, 16,670 test rows, 1,083 illicit (6.5%).

| Detector | PR-AUC (95% CI) | P@100 |
|---|---|---|
| Random | 0.066 | 0.05 |
| Rules only | 0.058 (ROC-AUC 0.26) | 0.00 |
| Isolation Forest only | 0.036 (ROC-AUC 0.15) | 0.00 |
| Logistic regression | 0.300 [0.278, 0.321] | 0.29 |
| Random Forest | **0.794 [0.775, 0.815]** | **1.00** |
| Fused RF + anomaly | 0.777 | 1.00 |
| Fused RF + anomaly + rules | 0.724 | 0.99 |

What this says, plainly:

* **The supervised Random Forest is what works on real data**: 4.5x the lift of logistic regression and 12x chance.
* **Rules and anomaly detection do not help here and are worse than chance** (ROC-AUC below 0.5): illicit transactions in
  Elliptic++ look *more* typical than the population, not less. Fusing them in lowers PR-AUC (0.794 -> 0.777 -> 0.724).
  The multi-signal design earns its place through explanation and safety caps, not through accuracy on this dataset.
* **Concept drift is real:** retraining at each cutoff and testing on the next five steps gives PR-AUC 0.84, 0.98, 0.99, 0.96,
  0.83 and then **0.32** on the last window (steps 45-49, P@100 0.32), after the known dark-market shutdown. A deployed model
  needs retraining and monitoring.
* **Fragile to noise:** noise of 0.1 training standard deviations on every feature drops PR-AUC from 0.79 to 0.19 (heavy-tailed
  anonymised features make this an extreme test); 25% missing features costs about 0.07. Half missing: 0.44.
* The rules baseline here is a naive set of five fixed heuristics, not the laundering heuristics used on raw transactions.
  Elliptic++ has no addresses, amounts per address or IPs, so the Forensics Lab pipeline cannot run on it.

## 6. Scale

`python -m scripts.scale_test --n-tx N` writes `reports/SCALE_TEST.md`. Measured on an 11.7 GB Windows laptop
(about 2 GB free at the time), synthetic data, peak memory includes data generation:

| Transactions | Addresses | Pipeline time | Throughput | Peak memory |
|---|---|---|---|---|
| 200,127 | 464,303 | 20 s | ~9,800 tx/s | 1.06 GB |
| 509,504 | 1,637,091 | 67 s | ~7,600 tx/s | 1.84 GB |

**A 1,000,000-transaction run was not executed on this machine** (extrapolated need is about 3.7 GB, more than was free).
Expect roughly 2.5 to 3 minutes of pipeline time, dominated by graph embeddings, which grow faster than linearly.
Run `python -m scripts.scale_test --n-tx 1000000` on a machine with 8 GB or more free to confirm.
All heavy stages are vectorised (pandas/NumPy/SciPy sparse); model training caps negatives at 40,000 entities.

## 7. Analyst workflow

1. Open **Forensics Lab**, review ranked leads (tier, probability, reason chips).
2. Select a lead: summary, fused signals, structural evidence with TXIDs, network evidence, key transactions.
3. **Link analysis:** timed flow graph with replay slider; amber = onward money trail to an exchange-style hub, purple = inbound trail.
4. Give a verdict (confirm / false positive / unsure). The active-learning list shows which leads are most useful to label next.
5. **Retrain with feedback:** before/after metrics are measured on entities that were not in the training labels.
6. Tick leads and **download the report**: standalone HTML (print to PDF), with evidence digest, dataset fingerprint and
   model/config hashes. `python -m backend.forensics.report verify report.html` re-checks the digest offline.

## 8. API

See `backend/api/forensics_routes.py` (`/forensics/status|load|alerts|entities/{id}|graph/{id}|feedback|uncertain|retrain|reset-model|report|cases|benchmark`).
File loading is confined to the data directory (`CHAINTRACE_DATA_DIR`); path traversal is rejected and tested.
