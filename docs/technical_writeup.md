# ChainTrace — Technical Write-Up

## Problem Statement

Bitcoin's pseudonymous nature makes it a preferred channel for illicit financial flows including money laundering, ransomware payments, and darknet marketplace transactions. Law enforcement and financial intelligence units need tools to identify, analyze, and prioritize suspicious blockchain entities for investigation.

**Challenge**: Build a blockchain forensics platform that combines multiple evidence sources to assist investigators in identifying suspicious transactions, wallets, and relationships — while maintaining transparency, explainability, and respect for the presumption of innocence.

---

## Approach

ChainTrace is an **investigative decision-support system** that synthesizes six distinct signal categories into actionable investigative priorities:

1. **ML Behavioral** — Supervised classification (illicit/licit prediction)
2. **Anomaly** — Unsupervised statistical deviance detection
3. **Graph Structural** — Transaction graph topology analysis
4. **Temporal** — Time-based correlation of events
5. **Correlation** — Network/IP observation linkage
6. **Known Indicator** — Heuristic pattern matching

These signals are architecturally separate — each represents a distinct analytical concept and is never treated as a synonym for another.

### Key Design Principles

- **Evidence vs. Inference separation**: Raw observations and derived conclusions remain distinguishable at the data-model level
- **Non-accusation principle**: Transaction-level predictions never become blanket accusations against entire wallets or entities
- **Synthetic data transparency**: Every synthetic record carries `is_synthetic=True` and is clearly marked in all layers
- **Offline operation**: Zero runtime network dependencies

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Vite + React)                  │
│   Dashboard │ Alerts │ Investigation │ Entity Profile │ Sources │
└──────────────────────────┬──────────────────────────────────────┘
                           │ REST API
┌──────────────────────────┴──────────────────────────────────────┐
│                    API Layer (WSGI + Pydantic)                   │
│              Routes → Service → Domain Models                   │
└───┬──────────┬──────────┬──────────┬──────────┬─────────────────┘
    │          │          │          │          │
┌───┴───┐ ┌───┴───┐ ┌───┴───┐ ┌───┴───┐ ┌───┴───┐
│Ingest │ │  ML   │ │ Risk  │ │ Graph │ │Correl.│
│  +    │ │Engine │ │Engine │ │Engine │ │Engine │
│Validate│ │       │ │       │ │       │ │       │
└───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘
    │         │         │         │         │
┌───┴─────────┴─────────┴─────────┴─────────┴─────────────────────┐
│                     Domain Models (Pydantic v2)                  │
│  Transaction │ Wallet │ RiskScore │ Evidence │ Alert │ Graph     │
└──────────────────────────────────────────────────────────────────┘
```

---

## Model Choice

### Supervised Classification: Gradient Boosted Machine (GBM)

**Why GBM over Deep Learning?**

| Factor | GBM | Deep Learning |
|--------|-----|---------------|
| Tabular data performance | State of the art | Often worse (Grinsztajn et al., 2022) |
| Training time | Seconds | Minutes to hours |
| Interpretability | Native feature importance | Requires post-hoc methods |
| Hardware requirements | CPU only | GPU preferred |
| Sample efficiency | Good with ~200K samples | Needs larger datasets |
| Offline deployment | Lightweight | Heavier dependencies |

**Output**: `predicted_label` (categorical) + `illicit_probability` (continuous [0,1]) + `model_version` + `classifier_explanation`

### Anomaly Detection: Isolation Forest

**Why Isolation Forest?**

- **Unsupervised**: No labels required — critical when ~70% of Elliptic++ transactions are unlabeled
- **Complementary**: Independent signal from the supervised classifier
- **Efficient**: O(n log n) training complexity
- **Interpretable**: Anomaly score directly measures isolation depth

**Output**: `anomaly_score` (deviance from baseline) + `anomaly_explanation`

**Critical distinction**: Anomaly explanations explain why a record is statistically unusual. They are NEVER presented as evidence of illicitness unless independently corroborated by classifier evidence.

---

## Multi-Signal Risk Synthesis

Instead of producing a single opaque "risk score," ChainTrace maintains 6 distinct signals and synthesizes them through a transparent, configurable pipeline:

```
┌─────────────────────────────────────────────────────┐
│              Signal Inputs (0.0 – 1.0)              │
│  illicit_probability  │  anomaly_score              │
│  graph_signal         │  correlation_confidence     │
│  known_indicator      │  custom_signals...          │
└───────────────────────┬─────────────────────────────┘
                        │
                ┌───────┴───────┐
                │  Synthesis    │
                │  Engine       │
                │               │
                │  • Dynamic    │
                │    weighting  │
                │  • Corrobora- │
                │    tion boost │
                │  • Anomaly    │
                │    safety cap │
                └───────┬───────┘
                        │
                ┌───────┴───────┐
                │  RiskScore    │
                │  (0 – 100)    │
                │               │
                │  + tier       │
                │  + evidence   │
                │  + narrative  │
                └───────────────┘
```

**Key features**:
- **Dynamic normalization**: Missing signals don't artificially suppress the score
- **Corroboration analysis**: Multiple independent signals boost confidence
- **Anomaly safety cap**: Isolated anomaly signals are capped at MEDIUM tier to prevent false escalation
- **Configurable thresholds**: Priority tiers (Critical ≥85, High ≥65, Medium ≥40, Low <40)

---

## Explainability Method

### Structured Evidence Ledger

Every risk score is accompanied by a structured evidence ledger containing records across 6 categories:

| Category | Source | Example |
|----------|--------|---------|
| ML Behavioral | Supervised classifier | "Predicted illicit (p=0.92). Key features: high volume, unusual fee ratio" |
| Anomaly | Isolation Forest | "Statistically unusual relative to baseline (score=0.78)" |
| Graph Structural | NetworkX topology | "Hub node with 5+ connections — typical mixer pattern" |
| Temporal | Timestamp analysis | "Transaction within 30min of known suspicious event" |
| Correlation | Network observations | "Same IP observed in 3 transactions (synthetic)" |
| Known Indicator | Heuristic rules | "Address pattern matches darknet marketplace" |

### Evidence Record Structure

Each evidence record contains:
- **Category**: One of the 6 categories above
- **Evidence type**: OBSERVATION (raw data) or INFERENCE (derived conclusion)
- **Explanation type**: CLASSIFIER_EXPLANATION, ANOMALY_EXPLANATION, or NARRATIVE
- **Description**: Human-readable explanation
- **Confidence**: Numeric confidence in the evidence quality
- **is_synthetic**: Whether the evidence derives from synthetic data
- **Caveats**: Known limitations or disclaimers

### What we DON'T do:
- Present anonymized Elliptic++ feature names to users
- Claim anomaly detection proves illicitness
- Use a generic "confidence" field that could be confused with risk
- Assign blanket risk to an entire wallet from a single transaction prediction

---

## Entity-Level Aggregation

Transaction predictions roll up to wallet/entity level through explicit, documented aggregation:

- **Methods**: Maximum, Mean, Volume-Weighted, Frequency-Based
- **Traceability**: Every aggregated score links back to its contributing transactions
- **Non-accusation**: The aggregation method and contributing evidence are always stored and reproducible

---

## Evaluation

### Temporal Evaluation (No Data Leakage)
- Training: Earlier Elliptic++ time steps
- Testing: Later time steps
- Random shuffling across time steps is explicitly prohibited as a data-leakage defect

### Metrics for Imbalanced Data
- Precision, Recall, F1 Score
- PR-AUC (more informative than ROC-AUC for ~2% illicit rate)
- Calibration analysis for probability outputs

---

## Key Innovations

1. **Six-signal synthesis**: Not a single ML score — structured combination of independent analytical dimensions
2. **Evidence-inference separation**: Architectural enforcement at the data-model level
3. **Anomaly safety capping**: Prevents isolated statistical anomalies from triggering false critical alerts
4. **Synthetic data traceability**: `is_synthetic` flag survives normalization, correlation, scoring, export, and UI
5. **Non-accusation principle**: Transaction findings never automatically become wallet-level accusations
6. **Zero network dependency**: Fully offline operation including GeoIP resolution

---

## Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Frontend | React + Vite | Fast builds, modern DX |
| API | Python WSGI + Pydantic | Zero-dependency, offline |
| ML | scikit-learn (GBM, IF) | Tabular data, interpretable |
| Graph | NetworkX | Pure Python, no DB needed |
| Validation | Pydantic v2 | Type-safe domain models |
| Testing | pytest (516 tests) | Comprehensive coverage |
