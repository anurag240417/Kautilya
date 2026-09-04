# ChainTrace

> **Blockchain & Network Forensics Platform**
> Built for Smart India Hackathon (SIH)

ChainTrace is an investigative decision-support system that synthesizes blockchain transaction data, network observations, temporal patterns, graph topology, and machine-learning signals to assist investigators in identifying and prioritizing suspicious cryptocurrency activity.

ChainTrace provides **probative triage rankings** and **structured evidence ledgers**, explicitly distinguishing raw observations from inferences and honoring the presumption of innocence.

---

## Key Capabilities

- **Multi-Signal Risk Synthesis**: Combines 6 distinct signal categories (ML Behavioral, Anomaly, Graph Structural, Temporal, Network Correlation, and Known Indicators) into a transparent, configurable investigative priority score without black-box opacity.
- **Dual ML Architecture**:
  - **Supervised Classifier (GBM)**: Predicts illicit transaction likelihood with local feature attribution (SHAP-style explanations).
  - **Unsupervised Anomaly Detector (Isolation Forest)**: Identifies statistical deviance from baseline behavior without equating anomalies to criminality.
- **Graph Forensics Engine**: Multi-hop ego-graph extraction, hub-and-spoke mixer detection, and shortest-path transaction tracing between suspicious entities.
- **Correlation & Temporal Analysis**: Links temporal bursts and network-layer observations (IP/ASN/GeoIP) with confidence scores and explicit uncertainty bounds.
- **Synthetic Data Transparency**: First-class `is_synthetic` flags tracked throughout all data models, APIs, and UI visualizations.
- **Zero Runtime Network Dependency**: 100% offline operation on Linux/Unix environments with local SQLite/in-memory caches and bundled reference data.
- **Investigative Decision Support**: Entity rollups preserve transaction-level provenance; a flagged transaction never becomes an automatic blanket accusation against an entire wallet.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Web UI (Vite + React + CSS)                       │
│    Overview Dashboard  │  Alert Triage  │  Graph Trace  │  Provenance   │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ REST / JSON
┌────────────────────────────────────┴────────────────────────────────────┐
│                    Forensics API (WSGI + Pydantic v2)                   │
│          Routes  →  Investigation Service  →  Domain Models            │
└──────┬──────────┬──────────┬──────────┬──────────┬──────────┬───────────┘
       │          │          │          │          │          │
┌──────┴───┐ ┌────┴────┐ ┌───┴────┐ ┌───┴────┐ ┌───┴────┐ ┌───┴────┐
│ Ingest   │ │ ML      │ │ Risk   │ │ Graph  │ │ Correl │ │ Evidence│
│ Pipeline │ │ Engine  │ │ Engine │ │ Engine │ │ Engine │ │ Ledger  │
│ (Audit/  │ │ (GBM +  │ │ (Multi-│ │ (Ego/  │ │ (IP/   │ │ (Audited│
│ Validate)│ │ IsoFor) │ │ Signal)│ │ Trace) │ │ Time)  │ │ Trails) │
└──────┬───┘ └────┬────┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘
       │          │          │          │          │          │
┌──────┴──────────┴──────────┴──────────┴──────────┴──────────┴───────────┐
│                    Forensic Data & Domain Store                         │
│   Transactions  │  Wallets  │  Alerts  │  Observations  │  Features     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. One-Command Live Demo

Run the end-to-end demo runner script, which verifies the virtual environment, installs dependencies, builds the frontend, and launches the server:

```bash
./run_demo.sh
```

Or skip the frontend build step if already compiled:

```bash
./run_demo.sh --skip-build
```

Access the application:
- **Investigation UI**: [http://localhost:8000/](http://localhost:8000/)
- **API Endpoints**: [http://localhost:8000/api/v1/](http://localhost:8000/api/v1/)
- **API Status**: [http://localhost:8000/api/v1/status](http://localhost:8000/api/v1/status)

---

### 2. Manual Setup

#### Backend Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package in editable mode with dev dependencies
pip install -e ".[dev]"

# Start backend server
python -m backend.main
```

#### Frontend Setup

```bash
cd frontend
npm install
npm run build      # For production build served by backend
# or
npm run dev        # For Vite HMR dev server
```

---

## Live Demo: Operation Shadow Mixer

ChainTrace includes a pre-seeded, realistic investigation scenario named **"Operation Shadow Mixer"**:

1. **Source Transaction (TX 1001)**: High-risk darknet transfer (50.0 BTC) to a known mixing hub.
2. **Mixer Entity (`1MixService...`)**: High-degree mixing service with multiple incoming and outgoing peels.
3. **Layering Path (TX 1002 → TX 1003 → TX 1004)**: Rapid multi-hop layering split across multiple intermediate hops.
4. **Cash-Out Endpoint (TX 1004)**: Liquidation attempt through an OTC exchange counterparty.
5. **Network Correlation**: Matching synthetic network observations with confidence intervals and geographic indicators.

Refer to [`docs/demo_script.md`](docs/demo_script.md) for the complete demonstration walkthrough and narration cues.

---

## Verification & Testing

### Automated Test Suite

ChainTrace maintains an extensive test suite covering unit, integration, validation, and end-to-end pipeline flows:

```bash
pytest
```

### Core Performance Benchmarks

Run the benchmark suite to evaluate latency across ingestion, query resolution, graph traversal, and multi-signal risk calculation:

```bash
python benchmarks/bench_core.py
```

### Zero Runtime Network Dependency Audit

Verify that the system executes completely air-gapped without external requests or unbundled lookups:

```bash
python scripts/verify_offline.py
```

---

## Documentation

- [`docs/technical_writeup.md`](docs/technical_writeup.md): Technical write-up covering design rationale, ML approach, explainability method, and metrics.
- [`docs/demo_script.md`](docs/demo_script.md): Complete script and action checklist for live demonstration.
- [`docs/judge_qa.md`](docs/judge_qa.md): Anticipated questions and answers for hackathon judges.
- [`docs/failure_cases.md`](docs/failure_cases.md): Edge case handling and graceful degradation documentation.
- [`ARCHITECTURE.md`](ARCHITECTURE.md): System architecture and data pipeline specification.
- [`CONTEXT.md`](CONTEXT.md): Domain context and forensic principles.
- [`TASKS.md`](TASKS.md): Project roadmap and implementation checklist.

---

## Technology Stack

- **Backend**: Python 3.11+, WSGI standard library, Pydantic v2, NetworkX, Scikit-learn, LightGBM
- **Frontend**: React 18, Vite, Lucide Icons, Vanilla CSS (modular design tokens)
- **Data & Testing**: Pytest, NumPy, Pandas
- **Packaging**: PEP 621 (`pyproject.toml`)

---

## License & Forensic Ethics

ChainTrace is developed strictly as an investigative decision-support tool. It presents probabilistic rankings, anomalies, and structural indicators with explicit uncertainty metrics. It does not replace judicial oversight or legal due process.
