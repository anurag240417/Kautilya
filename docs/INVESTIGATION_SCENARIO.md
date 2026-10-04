# Forensic Investigation Scenario: Operation Shadow Mixer

**Case Reference**: `CASE-2024-03-SM7`  
**Classification**: Investigative Decision Support — Confidential  
**Target Entities**: Unregistered Bitcoin Mixing Ring & Layered Capital Flight  
**Dataset Provenance**: Real Elliptic++ blockchain layer topology + local synthetic network propagation (`is_synthetic=True`)  
**Compliance Standard**: Zero external network requests; all inference, graph traversal, and GeoIP resolved locally.

---

## 1. Executive Summary & Objective

An automated alert was triggered by the Kautilya triage engine on transaction `TX-1001` (14.5 BTC). Preliminary machine learning behavioral signals flagged the transaction as **Critical Priority (Score: 92.5/100)** with illicit probability $p = 0.94$.

The investigative objective is to:
1. **Verify Evidence**: Decompose the risk score into its constituent analytical signals (ML Behavioral, Anomaly, Graph Topology, Temporal, and Network Correlations).
2. **Trace the Flow of Funds**: Map the upstream and downstream movement of funds through mixer splits and intermediate layering hops to the cash-out destination.
3. **Corroborate Network Observations**: Determine whether independent transactions share temporal or network propagation characteristics (e.g., candidate originator IP addresses).
4. **Preserve Forensic Integrity**: Maintain strict separation between direct evidence and derived inferences, ensuring entity-level triage does NOT make uncalibrated accusations against entire wallets.

---

## 2. Entity & Topology Matrix

| Entity Identifier | Entity Type | Role in Investigation | Priority / Label | Key Analytical Findings |
|---|---|---|---|---|
| **`1DrK44np3gMKuvcGeFHv`** | Wallet | Darknet Source | Illicit (Known) | Source wallet with 21 lifetime transactions; matched known darknet heuristic. |
| **`TX-1001`** | Transaction | Source Deposit | Critical (92.5) | 14.5 BTC transfer; $p=0.94$ illicit probability; volume anomaly ($4.2\sigma$). Direct deposit into mixer. |
| **`1MixServiceXjk8dqG2hP`** | Wallet | High-Throughput Mixer | Unknown / Triage Hub | Hub topology: 80 total transactions, 42 outbound, 38 inbound within 720 blocks. Rapid fund churn. |
| **`TX-1002`** | Transaction | Mixer Split A | Medium (55.0) | 7.2 BTC split; $1:3$ fan-out ratio ($2.8\sigma$ above baseline); $<10$ min delay from TX-1001. |
| **`TX-1003`** | Transaction | Mixer Split B | Medium (52.0) | 7.0 BTC secondary split; isolated fan-out anomaly. |
| **`1LayerHopZqfDmpv9nRy`** | Wallet | Layering Hop | Unknown | Intermediate relay wallet; lifetime 144 blocks; receives splits from mixer. |
| **`TX-1004`** | Transaction | Consolidation | High (68.0) | 13.8 BTC merge ($2 \to 1$ inputs/outputs); $3.6\sigma$ volume anomaly. Relayed by IP `203.0.113.88`. |
| **`1CashOutNn3bVx7wqFsT`** | Wallet | Cash-Out Destination | Unknown | Endpoint receiving consolidated funds; short lifetime (48 blocks). |
| **`TX-1006`** | Transaction | Second Source Deposit | Critical (85.0) | 8.3 BTC transfer; $p=0.88$ illicit probability. **Propagated by same IP `198.51.100.45`**. |
| **`TX-1007`** | Transaction | Fast-Turnaround Bypass | High (60.0) | 8.1 BTC direct mixer $\to$ cash-out bypass ($<15$ min); shared relay IP `203.0.113.88`. |
| **`1LegitExchWdRy4Pqb2K`** | Wallet | Clean Exchange (Control) | Licit (Baseline) | High volume (850 BTC), 215 transactions; baseline control entity. |
| **`TX-1005`** | Transaction | Exchange Transfer (Control)| Low (8.0) | 0.5 BTC licit withdrawal; no active anomalies; baseline control. |

---

## 3. Step-by-Step Investigator Journey (Live Demo Script)

### Step 1: Alert Triage on Overview Dashboard (`/#/`)
- **Action**: Navigate to Overview (`/#/`).
- **Observation**:
  - The KPI counters immediately show **6 Active Alerts** (2 Critical, 2 High, 2 Medium), 100% Synthetic Network Signal Share, and Sovereign Offline Status (`127.0.0.1:8000`).
  - The Top Alert card highlights `[CRITICAL] Transaction 1001 — Priority 92.5/100` with active signals `behavioral`, `graph`, `anomaly`, `correlation`.
  - The card features a clear `SYNTH` tag and dashed border marking the synthetic origin of the network correlation component.
- **Investigator Decision**: Click the `Inspect Evidence` button on TX-1001.

---

### Step 2: Evidence-Trail Reveal on Investigation Page (`/#/investigation?entity=1001`)
- **Action**: Observe the Evidence-Trail sequence panel on the right.
- **Narrative Progression**:
  1. **Seed Step**: Transaction `#1001` recorded at time step 25 (14.5 BTC volume, 2 inputs, 2 outputs).
  2. **Step 1 (ML Behavioral)**: Supervised classifier prediction $p = 0.94$. Explanation: high volume (14.5 BTC), elevated fee-to-value ratio, and multi-input consolidation pattern.
  3. **Step 2 (Graph Structural)**: Direct flow into mixer wallet `1MixServiceXjk8dqG2hP` (hub topology: 80 txs, 42 outbound).
  4. **Step 3 (Statistical Anomaly)**: Volume is $4.2\sigma$ above baseline mean ($2.1$ BTC). The interface explicitly reminds: *"This is a statistical observation and does not independently prove criminal activity."*
  5. **Step 4 (Network Correlation)**: Candidate originator IP `198.51.100.45` (AS13335, US) observed propagating TX-1001.
  6. **Final Step (Synthesis)**: Synthesized priority score **92.5 / 100 (Critical Tier)**. All 4 categories converge.
- **Forensic Principle Demonstrated**: Zero raw anonymized feature names (`Local_feature_*`) appear in the explanation. Natural language derived exclusively from interpretable dimensions.

---

### Step 3: Link Analysis & Money-Flow Path Finding
- **Action**: Inspect the Ego Subgraph on the left.
- **Observation**:
  - Node `1001` is rendered as an amber rectangle with direct edges from input wallet `1DrK44np3gMKuvcGeFHv` (steel blue) and outgoing edges to mixer wallet `1MixServiceXjk8dqG2hP`.
  - Outgoing paths from the mixer branch into splits `1002` and `1003`, which converge into layering wallet `1LayerHopZqfDmpv9nRy`, flow through consolidation transaction `1004`, and terminate at cash-out wallet `1CashOutNn3bVx7wqFsT`.
- **Action (Shortest Path Query)**:
  - Query shortest path: `source=1DrK44np3gMKuvcGeFHv` $\to$ `target=1CashOutNn3bVx7wqFsT`.
  - **Result**: Kautilya computes the 4-hop chain:
    $$\text{Source Wallet} \xrightarrow{\text{TX-1001}} \text{Mixer Wallet} \xrightarrow{\text{TX-1002}} \text{Layer Hop} \xrightarrow{\text{TX-1004}} \text{Cash-Out Wallet}$$
  - The path edges carry explicit relationship labels (`addr_tx`, `tx_addr`) and synthetic provenance markers.

---

### Step 4: Network Layer Cross-Correlation (Multi-Transaction Corroboration)
- **Action**: Switch to the **Correlated Observations** tab.
- **Observation**:
  - Candidate IP `198.51.100.45` is recorded with correlation confidence $0.82$ and role `originator`.
  - Notice the listed corroborating transaction ID: `TX-1006`.
- **Action**: Click on `TX-1006` or enter `1006` in the search bar.
- **Key Forensic Finding**:
  - TX-1006 (8.3 BTC) also originates from wallet `1DrK44np3gMKuvcGeFHv` and flows into the same mixer `1MixServiceXjk8dqG2hP`.
  - Candidate originator IP `198.51.100.45` was independently observed propagating TX-1006 with confidence $0.74$.
  - **Forensic Note**: The system highlights that while multiple independent transactions from the same IP increase correlation confidence, *temporal proximity alone does not constitute legal proof of identity or wallet ownership*.

---

### Step 5: Entity-Level Aggregation & Non-Accusation Principle (`/#/entity-profile?address=1MixServiceXjk8dqG2hP`)
- **Action**: Navigate to Entity Profile for the mixer wallet `1MixServiceXjk8dqG2hP`.
- **Observation**:
  - The wallet participates in multiple transactions with differing risk profiles (some illicit deposits, some unlabelled relay splits).
  - The entity-level prioritization panel displays the aggregation breakdown:
    - **Aggregation Method**: `max` (or `mean`, `volume_weighted`, `frequency` selectable).
    - **Contributing Transactions Table**: Explicitly lists each individual transaction ID, individual risk score, and illicit probability.
  - **Mandatory Forensic Disclaimer Displayed**:
    > *"Notice: Entity risk scores represent investigative triage prioritization. Individual transaction predictions do not automatically imply that the entire wallet or owner is engaged in illicit activity."*
- **Investigator Action**: Update alert status to `IN_REVIEW` on the Alert Queue page (`/#/alerts`), recording triage notes for the case dossier.

---

## 4. Signal Synthesis & Scoring Breakdown

Kautilya computes the priority score through dynamic multi-signal synthesis:

$$\text{Final Priority Score} = 100 \times \min\left(1.0, \, \frac{\sum_{s} w_s \cdot \text{signal}_s}{\sum_{s} w_s} + \text{corroboration\_boost}\right)$$

For `TX-1001`:
- **ML Behavioral Signal**: $0.95$ ($w = 0.35$)
- **Graph Structural Signal**: $0.85$ ($w = 0.25$)
- **Anomaly Signal**: $0.78$ ($w = 0.20$)
- **Network Correlation Signal**: $0.82$ ($w = 0.20$)
- **Corroborating Signals Active**: 4 independent categories $\to$ Multi-signal corroboration boost triggered.
- **Calculated Priority Score**: **92.5 / 100 (CRITICAL TIER)**.

For isolated anomaly `TX-1003`:
- Only the anomaly signal ($0.58$) is active without supervised classifier corroboration.
- **Safety Cap Applied**: Isolated deviance cannot trigger a Critical alert on its own, properly placing it into the **Medium Priority (52.0 / 100)** queue for routine review.

---

## 5. Judge Presentation Talking Points

1. **"Why isn't this just another dark theme dashboard?"**
   > *"Kautilya is built as an evidential decision-support system for law enforcement and financial intelligence units. Every score is a triage rank, not a verdict. Notice how our Evidence Trail decomposes ML predictions into natural language metrics rather than opaque feature vectors."*

2. **"How do you prevent false accusations against cryptocurrency exchanges or shared wallets?"**
   > *"Under our Entity Rollup Traceability principle, transactions maintain their individual scores permanently. When looking at a wallet like `1MixService...`, we never brand the whole entity illicit; we show an auditable table of contributing transactions with explicit disclaimers."*

3. **"Where does the network data come from?"**
   > *"The Bitcoin transaction graph is real data from public Elliptic++ benchmarks. The network propagation layer is generated through our local synthetic generator with reproducible seeds. Notice that every synthetic node, edge, and correlation is marked with dashed lines, `SYNTH` badges, and explicit provenance flags."*

4. **"Does this make external API calls at runtime?"**
   > *"Zero. Not even Google Fonts or CDN scripts. The frontend font files and JavaScript bundles are compiled locally; the backend runs entirely in-memory with local GeoIP resolution on offline Linux."*
