# Kautilya — Anticipated Judge Questions & Answers

Prepared Q&A for the Smart India Hackathon (SIH) panel covering architecture, ML, data handling, security, and scalability.

---

## Architecture & Design

### Q: Why did you build your own WSGI framework instead of using Flask/FastAPI?

**A**: We chose a zero-dependency WSGI approach for several reasons:
1. **Offline requirement**: SIH mandates fully offline operation. Minimizing dependencies reduces installation complexity on air-gapped systems.
2. **Transparency**: Judges can inspect every line of our HTTP handling — no magic framework behavior.
3. **Lightweight**: Our API has ~15 endpoints. A full framework would be unnecessary overhead.
4. **Learning demonstration**: Shows understanding of HTTP fundamentals, not just framework APIs.

We still use Pydantic for data validation, which provides production-grade input sanitization.

### Q: How does your system scale beyond the demo dataset?

**A**: The architecture supports scaling through:
1. **Modular design**: Each component (ingestion, ML, risk, graph, API) is independently replaceable
2. **Graph engine**: Built on NetworkX for demo; can swap to Neo4j/JanusGraph for production
3. **Storage**: In-memory dicts for demo; designed for database migration (PostgreSQL/Redis)
4. **Batch processing**: ML training is separated from inference; models are pre-trained and serialized
5. **Stateless API**: Service layer can be replicated behind a load balancer

### Q: Why a monolith instead of microservices?

**A**: For an investigation platform at SIH demo scale, a monolith is the right choice:
- **Simplicity**: One process, one deployment, zero network latency between components
- **Debugging**: Stack traces are continuous, not distributed
- **Offline**: No service mesh or message broker needed
- **Principle**: "Implement before abstracting" — our AGENTS.md explicitly mandates this

The module boundaries are clean enough that extraction into services is straightforward if needed.

---

## ML Model & Predictions

### Q: Why a Random Forest for classification?

**A**: A Random Forest was chosen because:
1. **Tabular data performance**: Consistently outperforms deep learning on structured/tabular datasets (Grinsztajn et al., 2022)
2. **Interpretability**: Tree-based models support feature importance natively, and the spread across trees gives an uncertainty range
3. **Training speed**: Trains in seconds on Elliptic++ (~200K transactions), enabling rapid iteration
4. **Class imbalance**: Handles imbalanced classes (illicit is ~2%) with balanced class weights
5. **Offline inference**: No GPU required, sub-millisecond per prediction

### Q: Why Isolation Forest for anomaly detection?

**A**: Isolation Forest is ideal for this use case:
1. **Unsupervised**: Doesn't require labels — important when ~70% of transactions are unlabeled
2. **Interpretable**: Anomaly scores directly represent "how isolated" a data point is
3. **Efficient**: O(n log n) training, fast inference
4. **Complementary**: Provides a signal independent of the supervised classifier

### Q: How do you handle the class imbalance problem?

**A**: Multiple approaches:
1. **Evaluation metrics**: We use Precision, Recall, F1, and PR-AUC instead of accuracy
2. **Temporal evaluation**: Train on earlier time steps, test on later ones — no data leakage
3. **Multi-signal synthesis**: Classification is just one of 6 signals; a single false positive doesn't dominate
4. **Priority tiers**: Threshold-based tiers (Critical/High/Medium/Low) let investigators set their own tolerance

### Q: What is your model's accuracy/performance?

**A**: We evaluate with metrics appropriate for class imbalance:
- **Precision**: Measures false positive rate (critical for investigations — you don't want to wrongly flag legitimate users)
- **Recall**: Measures detection rate (how many illicit transactions are caught)
- **F1 Score**: Harmonic mean of precision and recall
- **PR-AUC**: Area under precision-recall curve, more informative than ROC-AUC for imbalanced data
- All metrics are computed using **temporal evaluation** (no random shuffling across time steps)

---

## Explainability

### Q: How do you explain ML predictions to investigators?

**A**: Through a **Structured Evidence Ledger** with 6 distinct categories:
1. **ML Behavioral**: "Classifier predicted illicit with 92% probability. Key contributing features: high transaction volume, unusual fee ratio, elevated input degree."
2. **Anomaly**: "Record is statistically unusual relative to baseline. Anomaly score: 0.78." (Never claims illicitness)
3. **Graph Structural**: "Wallet acts as a hub with 5+ connections — typical of mixer services"
4. **Temporal**: "Transaction occurred within 30 minutes of a known suspicious event"
5. **Correlation**: "Same IP observed in 3 transactions within a 2-hour window"
6. **Known Indicator**: "Address pattern matches darknet marketplace heuristic"

Each category is architecturally separate — an anomaly explanation is NEVER presented as evidence of illicitness.

### Q: What about the anonymized features in Elliptic++?

**A**: Elliptic++ contains anonymized features (`Local_feature_1`, `Aggregate_feature_23`, etc.) whose real-world meaning is unknown. Our approach:
- Anonymized feature names **never appear in user-facing explanations**
- If a model relies on anonymized features, the explanation honestly states: "Record is statistically unusual relative to the learned baseline"
- We focus on **interpretable features** (transaction volume, fee ratios, degree, timing) for human-readable explanations

### Q: How do you distinguish evidence from inference?

**A**: Architecturally enforced separation:
- **Evidence** (observation): "Transaction T1 occurred at timestamp X" or "IP 198.51.100.42 was observed"
- **Inference** (conclusion): "Transaction T1 is temporally correlated with network event N1"
- Evidence records carry `evidence_type` (OBSERVATION vs INFERENCE) at the data-model level
- The UI visually distinguishes observations from inferences

---

## Data & Provenance

### Q: Is your data real Bitcoin data?

**A**: Partially:
- **Blockchain layer**: Uses the **Elliptic++ dataset**, a real/public Bitcoin transaction and address dataset. Transaction IDs are anonymized numeric identifiers (not raw on-chain TXIDs).
- **Network layer**: 100% **synthetic** — Kautilya does not have real transaction-to-IP observations. Every synthetic record carries `is_synthetic=True` at the data-model level.
- The demo scenario ("Operation Shadow Mixer") uses synthetic data explicitly marked as such.

### Q: How do you handle synthetic data transparency?

**A**: Multi-layer approach:
1. **Data model**: Every record has an `is_synthetic` boolean field
2. **Risk scores**: `contains_synthetic_input` flag propagates through synthesis
3. **Evidence ledger**: Synthetic evidence records include caveats
4. **Alerts**: `contains_synthetic_input` flag on every alert
5. **API responses**: `is_synthetic` field in all entity responses
6. **UI**: Prominent synthetic data tags and forensic disclaimer
7. **Statistics**: Dashboard shows synthetic alerts percentage

### Q: What is GeoIP and how do you handle it offline?

**A**: GeoIP resolution maps IP addresses to countries/ASNs. We use a **local/bundled database file** (MaxMind GeoLite2 format) — no network calls. This is mandated by our architecture: "Kautilya MUST run fully offline."

---

## Privacy & Security

### Q: How do you handle sensitive data?

**A**: 
- No hardcoded secrets, API keys, or credentials anywhere in the codebase
- Environment variables for any configuration secrets (`.env.example` provided)
- No logging of sensitive data
- The Elliptic++ dataset uses anonymized transaction IDs — no real personal data
- Synthetic network data contains fabricated IPs from documentation ranges (RFC 5737)

### Q: Is this system admissible as legal evidence?

**A**: Kautilya is explicitly an **investigative decision-support system**, not a legal evidence tool:
- Risk scores represent **investigative triage priority**, NOT probability of criminality
- The system NEVER claims that an IP, wallet, or person is definitively responsible for criminal activity
- All outputs include forensic disclaimers
- Investigators must independently verify findings before any legal action

---

## Offline Operation

### Q: How do you verify the system works offline?

**A**: Automated verification:
1. **Import audit**: Script scans all Python files for network-calling libraries (`requests`, `urllib.request`, `httpx`, `socket.connect`, etc.)
2. **Code scan**: Grep for HTTP/network patterns in source code
3. **GeoIP check**: Verify local database usage, no external lookups
4. **Environment check**: Ensure no environment variables point to external URLs
5. **Runtime test**: Full test suite (516 tests) runs without network access

### Q: What dependencies does the system need?

**A**: Minimal:
- **Python 3.11+** with standard library
- **Pydantic**: Data validation
- **scikit-learn**: ML models (Random Forest, Isolation Forest)
- **NetworkX**: Graph operations
- **Node.js** (build-time only): Frontend build
- No database server, no message queue, no cloud services

---

## Limitations & Honesty

### Q: What are the limitations of your system?

**A**: We're transparent about limitations:
1. **Scale**: In-memory storage limits dataset size (~100K transactions practical). Production needs persistent storage.
2. **Network layer**: 100% synthetic — we can't validate real IP-to-transaction correlations
3. **Real-time**: No live blockchain ingestion. Designed for batch forensic analysis.
4. **Single-chain**: Currently Bitcoin only. Multi-chain support would need additional ingestion modules.
5. **Model generalization**: Trained on Elliptic++ which covers specific time periods. Performance on newer patterns is unverified.
6. **Anonymized features**: Some ML signals rely on features whose real-world meaning is unknown.

### Q: What would you add with more time?

**A**:
1. **Persistent storage** (PostgreSQL + Redis caching)
2. **Real-time ingestion** from blockchain nodes
3. **Multi-chain support** (Ethereum, Monero tracing)
4. **Collaborative investigation** (multi-user case management)
5. **Advanced graph algorithms** (community detection, PageRank)
6. **Model retraining pipeline** with drift detection
7. **Report generation** (PDF export of investigation findings)
