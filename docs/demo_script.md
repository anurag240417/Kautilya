# ChainTrace Live Demo Script — Operation Shadow Mixer

## Pre-Demo Checklist

- [ ] Backend dependencies installed (`pip install -e .`)
- [ ] Frontend built (`cd frontend && npm run build`)
- [ ] Demo runner ready (`./run_demo.sh`)
- [ ] Browser open to `http://localhost:8000`
- [ ] Terminal visible for API calls (optional)

---

## Act 1: The Dashboard (2 min)

### Narration
> "ChainTrace is a blockchain and network forensics platform that combines multiple sources of evidence — machine learning predictions, anomaly detection, graph topology analysis, temporal correlations, and network observations — to help investigators identify and prioritize suspicious entities."

### Actions
1. **Open the dashboard** — show the Overview page
2. Point out the **KPI counters**: total transactions, wallets, alerts, and synthetic data percentage
3. Highlight the **tier distribution**: Critical, High, Medium, Low alerts
4. Note the **forensic disclaimer**: "Risk scores represent investigative triage priority, NOT probability of criminality."

---

## Act 2: Alert Triage (3 min)

### Narration
> "Investigators start with the alert queue, which ranks entities by investigative priority using multi-signal risk synthesis."

### Actions
1. Navigate to **Alerts** page
2. Show **Critical tier** alerts — click on the source wallet alert
3. Point out the **priority tier badge** and **risk score breakdown**
4. Show the **6 evidence categories** in the evidence panel:
   - ML Behavioral (classifier prediction)
   - Anomaly (statistical deviance)
   - Graph Structural (hub topology)
   - Temporal (timestamp proximity)
   - Correlation (same-IP observation)
   - Known Indicator (darknet heuristic)
5. Highlight the **synthetic data tag** — all network evidence is clearly marked

---

## Act 3: Transaction Deep Dive (3 min)

### Narration
> "Let's drill into a specific suspicious transaction to see the full evidence chain."

### Actions
1. Search for **Transaction 1001** (Source → Mixer illicit transfer)
2. Show the **interpretable features**: total BTC, fees, input/output degree
3. Show the **risk score** with multi-signal breakdown
4. Show the **correlations**: network observations with IP, port, protocol
5. Point out: "Each correlation carries a confidence score and is clearly marked as synthetic."
6. Show **connected wallets** (inputs and outputs from graph)

---

## Act 4: Wallet Investigation (3 min)

### Narration
> "Now let's look at the mixer wallet — a key entity in this money-laundering scenario."

### Actions
1. Search for wallet **1MixServiceXjk8dqG2hP**
2. Show **wallet statistics**: transaction counts, BTC volumes, lifetime
3. Show **aggregated risk score** — explain aggregation method (MAX)
4. Highlight: "Transaction-level predictions roll up to wallet-level, but a single suspicious transaction does NOT automatically accuse the entire wallet."
5. Show **counterparty list**

---

## Act 5: Transaction Graph (3 min)

### Narration
> "The interactive graph visualizes how entities are connected through the transaction network."

### Actions
1. Navigate to the **Investigation** page
2. Load graph for **wallet 1MixServiceXjk8dqG2hP** at depth 2
3. Show the **hub topology** at the mixer wallet
4. Point out **color coding**: nodes colored by risk tier
5. Show **edge types**: addr_tx, tx_addr, tx_tx, addr_addr
6. Click on individual nodes to see details
7. Demonstrate **path finding**: find shortest path from TX 1001 → TX 1004
8. Show the complete **money flow**: Source → Mixer → Layer → Cash-out

---

## Act 6: Explainability & Provenance (2 min)

### Narration
> "Every risk score comes with a structured evidence ledger that explains WHY an entity was flagged. We separate evidence from inference."

### Actions
1. Return to a **Critical alert** evidence panel
2. Walk through the **human-readable summary**
3. Point out: "Classifier explanations tell you why the model predicted 'illicit'. Anomaly explanations tell you why a record is statistically unusual. These are different concepts."
4. Show the **Data Sources** page — Elliptic++ provenance, synthetic network layer

---

## Act 7: Architecture & Key Decisions (2 min)

### Narration
> "ChainTrace runs fully offline — no network calls, no cloud APIs. Everything from GeoIP lookups to ML inference runs locally."

### Talking Points
- **ML Models**: GBM classification + Isolation Forest anomaly detection
- **Multi-signal synthesis**: Not a single score — 6 distinct signal categories
- **Non-accusation principle**: Transaction predictions never become blanket wallet accusations
- **Synthetic data transparency**: Every synthetic record carries `is_synthetic=True` at the data-model level
- **Zero dependencies on network**: Verified by automated offline compliance checker

---

## Demo API Endpoints (for terminal demos)

```bash
# Health check
curl http://localhost:8000/api/v1/statistics

# Transaction lookup
curl http://localhost:8000/api/v1/transactions/1001

# Wallet lookup
curl http://localhost:8000/api/v1/wallets/1DrK44np3gMKuvcGeFHv

# Alert listing
curl http://localhost:8000/api/v1/alerts

# Graph subgraph
curl "http://localhost:8000/api/v1/graph/1001?depth=2"

# Shortest path
curl "http://localhost:8000/api/v1/graph/path?source=1001&target=1004"
```

---

## Timing Guide

| Act | Duration | Cumulative |
|-----|----------|------------|
| Dashboard | 2 min | 2 min |
| Alert Triage | 3 min | 5 min |
| Transaction Deep Dive | 3 min | 8 min |
| Wallet Investigation | 3 min | 11 min |
| Transaction Graph | 3 min | 14 min |
| Explainability | 2 min | 16 min |
| Architecture | 2 min | 18 min |
| Q&A Buffer | 2 min | 20 min |
