"""Demonstration dataset for SIH live demo — Operation Shadow Mixer.

Constructs a coherent money-laundering investigation scenario that exercises
every ChainTrace signal category:

    ML Behavioral → classifier predictions on illicit source transactions
    Anomaly       → statistical deviance on mixer fan-out and rapid churn
    Graph         → hub topology at mixer wallet, convergence at cash-out
    Temporal      → tight timestamp windows between source and split
    Correlation   → same IP observed near multiple source transactions
    Known Indicator → darknet-market heuristic match on source wallet

All data is explicitly synthetic (``is_synthetic=True`` at the data-model
level) and intended for demonstration purposes only.

Forensic Principle:
    Synthetic data is explicitly marked. Inferences are distinguished from
    observations. Risk scores represent investigative triage priority,
    NOT probability of criminality.
"""

from backend.domain.alert import InvestigativeAlert
from backend.domain.graph import AddrAddrEdge, AddrTxEdge, TxAddrEdge, TxTxEdge
from backend.domain.risk import (
    EvidenceCategory,
    EvidenceLedger,
    EvidenceRecord,
    RiskScore,
)
from backend.domain.transaction import Transaction
from backend.domain.types import (
    EntityClass,
    EvidenceType,
    ExplanationType,
    PriorityTier,
)
from backend.domain.wallet import StatsSummary, Wallet
from backend.risk.ranking import generate_alert

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Wallet addresses — descriptive prefixes for demo readability
WALLET_A = "1DrK44np3gMKuvcGeFHv"   # Source (darknet)
WALLET_B = "1MixServiceXjk8dqG2hP"  # Mixer
WALLET_C = "1LayerHopZqfDmpv9nRy"   # Layering hop
WALLET_D = "1CashOutNn3bVx7wqFsT"   # Cash-out
WALLET_E = "1LegitExchWdRy4Pqb2K"   # Clean exchange (control)

# Transaction IDs — simple range for demo clarity
TX_1001 = 1001  # Source → Mixer (illicit)
TX_1002 = 1002  # Mixer → Layer (split A)
TX_1003 = 1003  # Mixer → Layer (split B)
TX_1004 = 1004  # Layer → Cash-out (consolidation)
TX_1005 = 1005  # Exchange → legit user (control)
TX_1006 = 1006  # Second source → Mixer (corroboration)
TX_1007 = 1007  # Mixer → Cash-out (fast turnaround)

PROVENANCE = "demo:operation_shadow_mixer"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _build_transactions() -> dict[int, Transaction]:
    """Build 7 demonstration transactions."""
    return {
        TX_1001: Transaction(
            txid=TX_1001,
            time_step=25,
            label=EntityClass.ILLICIT,
            total_btc=14.5,
            fees=0.002,
            size=350.0,
            num_input_addresses=2.0,
            num_output_addresses=2.0,
            in_txs_degree=2.0,
            out_txs_degree=1.0,
        ),
        TX_1002: Transaction(
            txid=TX_1002,
            time_step=25,
            label=EntityClass.UNKNOWN,
            total_btc=7.2,
            fees=0.0008,
            size=225.0,
            num_input_addresses=1.0,
            num_output_addresses=3.0,
            in_txs_degree=1.0,
            out_txs_degree=3.0,
        ),
        TX_1003: Transaction(
            txid=TX_1003,
            time_step=25,
            label=EntityClass.UNKNOWN,
            total_btc=7.0,
            fees=0.0007,
            size=220.0,
            num_input_addresses=1.0,
            num_output_addresses=2.0,
            in_txs_degree=1.0,
            out_txs_degree=2.0,
        ),
        TX_1004: Transaction(
            txid=TX_1004,
            time_step=26,
            label=EntityClass.UNKNOWN,
            total_btc=13.8,
            fees=0.0015,
            size=310.0,
            num_input_addresses=2.0,
            num_output_addresses=1.0,
            in_txs_degree=2.0,
            out_txs_degree=1.0,
        ),
        TX_1005: Transaction(
            txid=TX_1005,
            time_step=26,
            label=EntityClass.LICIT,
            total_btc=0.5,
            fees=0.0001,
            size=180.0,
            num_input_addresses=1.0,
            num_output_addresses=1.0,
            in_txs_degree=1.0,
            out_txs_degree=1.0,
        ),
        TX_1006: Transaction(
            txid=TX_1006,
            time_step=25,
            label=EntityClass.ILLICIT,
            total_btc=8.3,
            fees=0.0018,
            size=290.0,
            num_input_addresses=1.0,
            num_output_addresses=1.0,
            in_txs_degree=1.0,
            out_txs_degree=1.0,
        ),
        TX_1007: Transaction(
            txid=TX_1007,
            time_step=25,
            label=EntityClass.UNKNOWN,
            total_btc=8.1,
            fees=0.0012,
            size=240.0,
            num_input_addresses=1.0,
            num_output_addresses=1.0,
            in_txs_degree=1.0,
            out_txs_degree=1.0,
        ),
    }


def _build_wallets() -> dict[str, Wallet]:
    """Build 5 demonstration wallets."""
    return {
        WALLET_A: Wallet(
            address=WALLET_A,
            time_step=25,
            label=EntityClass.ILLICIT,
            num_txs_as_sender=18.0,
            num_txs_as_receiver=3.0,
            total_txs=21.0,
            lifetime_in_blocks=2880.0,
            btc_transacted=StatsSummary(
                total=62.5, min=0.1, max=14.5, mean=2.97, median=1.8
            ),
            btc_sent=StatsSummary(
                total=58.0, min=0.1, max=14.5, mean=3.22, median=2.0
            ),
            btc_received=StatsSummary(
                total=4.5, min=0.5, max=3.0, mean=1.5, median=1.0
            ),
        ),
        WALLET_B: Wallet(
            address=WALLET_B,
            time_step=25,
            label=EntityClass.UNKNOWN,
            num_txs_as_sender=42.0,
            num_txs_as_receiver=38.0,
            total_txs=80.0,
            lifetime_in_blocks=720.0,
            btc_transacted=StatsSummary(
                total=320.0, min=0.01, max=14.5, mean=4.0, median=2.5
            ),
            btc_sent=StatsSummary(
                total=160.0, min=0.01, max=8.1, mean=3.8, median=2.2
            ),
            btc_received=StatsSummary(
                total=160.0, min=0.01, max=14.5, mean=4.2, median=2.8
            ),
        ),
        WALLET_C: Wallet(
            address=WALLET_C,
            time_step=25,
            label=EntityClass.UNKNOWN,
            num_txs_as_sender=5.0,
            num_txs_as_receiver=8.0,
            total_txs=13.0,
            lifetime_in_blocks=144.0,
            btc_transacted=StatsSummary(
                total=42.0, min=0.5, max=13.8, mean=3.23, median=2.0
            ),
        ),
        WALLET_D: Wallet(
            address=WALLET_D,
            time_step=26,
            label=EntityClass.UNKNOWN,
            num_txs_as_sender=2.0,
            num_txs_as_receiver=6.0,
            total_txs=8.0,
            lifetime_in_blocks=48.0,
            btc_transacted=StatsSummary(
                total=35.0, min=1.0, max=13.8, mean=4.37, median=3.5
            ),
        ),
        WALLET_E: Wallet(
            address=WALLET_E,
            time_step=26,
            label=EntityClass.LICIT,
            num_txs_as_sender=120.0,
            num_txs_as_receiver=95.0,
            total_txs=215.0,
            lifetime_in_blocks=14400.0,
            btc_transacted=StatsSummary(
                total=850.0, min=0.001, max=50.0, mean=3.95, median=1.2
            ),
        ),
    }


def _build_evidence_ledger_tx1001() -> EvidenceLedger:
    """Evidence ledger for TX-1001 (illicit source → mixer)."""
    ledger = EvidenceLedger(entity_id=str(TX_1001), entity_type="transaction")
    ledger.add_record(EvidenceRecord(
        evidence_id="ev-1001-ml",
        category=EvidenceCategory.ML_BEHAVIORAL,
        evidence_type=EvidenceType.MODEL_PREDICTION,
        source_entity_id=str(TX_1001),
        source_entity_type="transaction",
        headline="High illicit probability from supervised classifier",
        description=(
            "Random forest classifier assigned illicit probability 0.94. "
            "Key contributing features: unusually high transaction volume "
            "(14.5 BTC), elevated fee-to-value ratio, and multi-input "
            "address pattern consistent with consolidation behavior."
        ),
        supporting_metrics={
            "illicit_probability": 0.94,
            "total_btc": 14.5,
            "fee_ratio": 0.000138,
            "num_input_addresses": 2,
        },
        confidence=0.94,
        is_synthetic=True,
        provenance=PROVENANCE,
        explanation_type=ExplanationType.CLASSIFIER_EXPLANATION,
    ))
    ledger.add_record(EvidenceRecord(
        evidence_id="ev-1001-graph",
        category=EvidenceCategory.GRAPH_STRUCTURAL,
        evidence_type=EvidenceType.OBSERVATION,
        source_entity_id=str(TX_1001),
        source_entity_type="transaction",
        headline="Direct link to high-throughput mixer wallet",
        description=(
            "Transaction output flows directly into wallet "
            f"{WALLET_B} which exhibits hub topology: 80 total transactions, "
            "42 outbound, 38 inbound within 720 blocks. This fan-in/fan-out "
            "pattern is structurally consistent with mixing services."
        ),
        supporting_metrics={
            "target_wallet": WALLET_B,
            "target_total_txs": 80,
            "target_out_degree": 42,
            "hops_to_mixer": 1,
        },
        confidence=0.85,
        is_synthetic=True,
        provenance=PROVENANCE,
    ))
    ledger.add_record(EvidenceRecord(
        evidence_id="ev-1001-anomaly",
        category=EvidenceCategory.ANOMALY,
        evidence_type=EvidenceType.MODEL_PREDICTION,
        source_entity_id=str(TX_1001),
        source_entity_type="transaction",
        headline="Volume anomaly relative to baseline",
        description=(
            "Transaction volume (14.5 BTC) is 4.2 standard deviations "
            "above the learned baseline mean for transactions at time_step 25. "
            "This is a statistical observation and does not independently "
            "indicate illicit activity."
        ),
        supporting_metrics={
            "anomaly_score": 0.78,
            "z_score": 4.2,
            "baseline_mean_btc": 2.1,
        },
        confidence=0.78,
        is_synthetic=True,
        provenance=PROVENANCE,
        explanation_type=ExplanationType.ANOMALY_EXPLANATION,
    ))
    ledger.add_record(EvidenceRecord(
        evidence_id="ev-1001-corr",
        category=EvidenceCategory.SYNTHETIC_NETWORK,
        evidence_type=EvidenceType.CORRELATION,
        source_entity_id=str(TX_1001),
        source_entity_type="transaction",
        headline="IP correlation — originator 198.51.100.45",
        description=(
            "Synthetic network observation: IP address 198.51.100.45 "
            "(AS13335, US) was observed propagating this transaction. "
            "The same IP was independently observed near TX-1006, "
            "suggesting a possible common originator. "
            "Temporal proximity alone does not prove identity."
        ),
        supporting_metrics={
            "candidate_ip": "198.51.100.45",
            "asn": 13335,
            "country": "US",
            "corroborating_txids": [TX_1006],
        },
        confidence=0.82,
        is_synthetic=True,
        provenance=PROVENANCE,
    ))
    return ledger


def _build_evidence_ledger_tx1006() -> EvidenceLedger:
    """Evidence ledger for TX-1006 (second illicit source → mixer)."""
    ledger = EvidenceLedger(entity_id=str(TX_1006), entity_type="transaction")
    ledger.add_record(EvidenceRecord(
        evidence_id="ev-1006-ml",
        category=EvidenceCategory.ML_BEHAVIORAL,
        evidence_type=EvidenceType.MODEL_PREDICTION,
        source_entity_id=str(TX_1006),
        source_entity_type="transaction",
        headline="Illicit classification with elevated volume",
        description=(
            "Supervised classifier predicted illicit label with probability 0.88. "
            "Notable features: 8.3 BTC volume, single-input single-output "
            "pattern, and fee ratio consistent with privacy-preserving transfers."
        ),
        supporting_metrics={
            "illicit_probability": 0.88,
            "total_btc": 8.3,
            "fee_ratio": 0.000217,
        },
        confidence=0.88,
        is_synthetic=True,
        provenance=PROVENANCE,
        explanation_type=ExplanationType.CLASSIFIER_EXPLANATION,
    ))
    ledger.add_record(EvidenceRecord(
        evidence_id="ev-1006-corr",
        category=EvidenceCategory.SYNTHETIC_NETWORK,
        evidence_type=EvidenceType.CORRELATION,
        source_entity_id=str(TX_1006),
        source_entity_type="transaction",
        headline="Same IP as TX-1001 — corroborating originator",
        description=(
            "IP 198.51.100.45 (AS13335, US) was also observed propagating "
            "TX-1001. Two independent transactions originating from the same "
            "IP increases the correlation confidence, but does not constitute "
            "proof of common ownership."
        ),
        supporting_metrics={
            "candidate_ip": "198.51.100.45",
            "asn": 13335,
            "country": "US",
            "corroborating_txids": [TX_1001],
        },
        confidence=0.74,
        is_synthetic=True,
        provenance=PROVENANCE,
    ))
    return ledger


def _build_evidence_ledger_tx1002() -> EvidenceLedger:
    """Evidence ledger for TX-1002 (mixer split A)."""
    ledger = EvidenceLedger(entity_id=str(TX_1002), entity_type="transaction")
    ledger.add_record(EvidenceRecord(
        evidence_id="ev-1002-anomaly",
        category=EvidenceCategory.ANOMALY,
        evidence_type=EvidenceType.MODEL_PREDICTION,
        source_entity_id=str(TX_1002),
        source_entity_type="transaction",
        headline="Unusual fan-out ratio from mixer",
        description=(
            "Transaction has 1 input but 3 outputs — a 1:3 fan-out ratio that "
            "is 2.8σ above the baseline for this time step. This pattern is "
            "statistically unusual and consistent with value-splitting behavior."
        ),
        supporting_metrics={
            "anomaly_score": 0.62,
            "fan_out_ratio": 3.0,
            "z_score": 2.8,
        },
        confidence=0.62,
        is_synthetic=True,
        provenance=PROVENANCE,
        explanation_type=ExplanationType.ANOMALY_EXPLANATION,
    ))
    ledger.add_record(EvidenceRecord(
        evidence_id="ev-1002-temporal",
        category=EvidenceCategory.TEMPORAL,
        evidence_type=EvidenceType.CORRELATION,
        source_entity_id=str(TX_1002),
        source_entity_type="transaction",
        headline="Rapid churn — <10 min after source TX-1001",
        description=(
            "TX-1002 appeared within the same time_step as the source "
            "transaction TX-1001, suggesting rapid processing through "
            "the mixer wallet. Short dwell time is common in automated "
            "mixing services."
        ),
        supporting_metrics={
            "source_txid": TX_1001,
            "same_time_step": True,
            "estimated_delay_minutes": 8,
        },
        confidence=0.55,
        is_synthetic=True,
        provenance=PROVENANCE,
    ))
    return ledger


def _build_evidence_ledger_tx1004() -> EvidenceLedger:
    """Evidence ledger for TX-1004 (consolidation into cash-out)."""
    ledger = EvidenceLedger(entity_id=str(TX_1004), entity_type="transaction")
    ledger.add_record(EvidenceRecord(
        evidence_id="ev-1004-graph",
        category=EvidenceCategory.GRAPH_STRUCTURAL,
        evidence_type=EvidenceType.OBSERVATION,
        source_entity_id=str(TX_1004),
        source_entity_type="transaction",
        headline="Consolidation pattern — 2 inputs, 1 output",
        description=(
            "This transaction merges two input streams from the layering "
            f"hop ({WALLET_C}) into a single output directed at the "
            f"cash-out wallet ({WALLET_D}). This reconvergence pattern "
            "is notable after an upstream split through a mixer."
        ),
        supporting_metrics={
            "num_inputs": 2,
            "num_outputs": 1,
            "total_btc": 13.8,
            "upstream_mixer": WALLET_B,
        },
        confidence=0.72,
        is_synthetic=True,
        provenance=PROVENANCE,
    ))
    ledger.add_record(EvidenceRecord(
        evidence_id="ev-1004-anomaly",
        category=EvidenceCategory.ANOMALY,
        evidence_type=EvidenceType.MODEL_PREDICTION,
        source_entity_id=str(TX_1004),
        source_entity_type="transaction",
        headline="High-value consolidation deviance",
        description=(
            "13.8 BTC consolidated into a single output is 3.6σ above "
            "the baseline mean for consolidation transactions. Statistically "
            "unusual volume that warrants inspection."
        ),
        supporting_metrics={
            "anomaly_score": 0.71,
            "z_score": 3.6,
            "consolidation_btc": 13.8,
        },
        confidence=0.71,
        is_synthetic=True,
        provenance=PROVENANCE,
        explanation_type=ExplanationType.ANOMALY_EXPLANATION,
    ))
    ledger.add_record(EvidenceRecord(
        evidence_id="ev-1004-relay",
        category=EvidenceCategory.SYNTHETIC_NETWORK,
        evidence_type=EvidenceType.CORRELATION,
        source_entity_id=str(TX_1004),
        source_entity_type="transaction",
        headline="Relay IP 203.0.113.88 observed",
        description=(
            "Synthetic network observation: IP 203.0.113.88 (AS9002, DE) "
            "was observed relaying this transaction. The same IP was also "
            "observed near TX-1007, suggesting possible relay infrastructure."
        ),
        supporting_metrics={
            "candidate_ip": "203.0.113.88",
            "asn": 9002,
            "country": "DE",
            "corroborating_txids": [TX_1007],
        },
        confidence=0.61,
        is_synthetic=True,
        provenance=PROVENANCE,
    ))
    return ledger


def _build_evidence_ledger_tx1007() -> EvidenceLedger:
    """Evidence ledger for TX-1007 (mixer → cash-out fast turnaround)."""
    ledger = EvidenceLedger(entity_id=str(TX_1007), entity_type="transaction")
    ledger.add_record(EvidenceRecord(
        evidence_id="ev-1007-anomaly",
        category=EvidenceCategory.ANOMALY,
        evidence_type=EvidenceType.MODEL_PREDICTION,
        source_entity_id=str(TX_1007),
        source_entity_type="transaction",
        headline="Rapid mixer-to-cashout turnaround",
        description=(
            "TX-1007 moved 8.1 BTC from the mixer directly to the cash-out "
            "wallet within the same time_step as the source TX-1006. "
            "This rapid turnaround (est. <15 min) is 3.1σ faster than "
            "baseline dwell time in the mixer."
        ),
        supporting_metrics={
            "anomaly_score": 0.68,
            "z_score": 3.1,
            "estimated_dwell_minutes": 12,
            "total_btc": 8.1,
        },
        confidence=0.68,
        is_synthetic=True,
        provenance=PROVENANCE,
        explanation_type=ExplanationType.ANOMALY_EXPLANATION,
    ))
    ledger.add_record(EvidenceRecord(
        evidence_id="ev-1007-graph",
        category=EvidenceCategory.GRAPH_STRUCTURAL,
        evidence_type=EvidenceType.OBSERVATION,
        source_entity_id=str(TX_1007),
        source_entity_type="transaction",
        headline="Direct mixer-to-cashout link bypassing layering",
        description=(
            f"Unlike the TX-1002/1003 path which passes through {WALLET_C}, "
            f"TX-1007 goes directly from {WALLET_B} (mixer) to {WALLET_D} "
            "(cash-out). This shortcut path may indicate urgency or a "
            "different operational mode."
        ),
        supporting_metrics={
            "source_wallet": WALLET_B,
            "target_wallet": WALLET_D,
            "hops": 1,
            "bypassed_layers": 1,
        },
        confidence=0.65,
        is_synthetic=True,
        provenance=PROVENANCE,
    ))
    ledger.add_record(EvidenceRecord(
        evidence_id="ev-1007-relay",
        category=EvidenceCategory.SYNTHETIC_NETWORK,
        evidence_type=EvidenceType.CORRELATION,
        source_entity_id=str(TX_1007),
        source_entity_type="transaction",
        headline="Same relay IP as TX-1004",
        description=(
            "IP 203.0.113.88 (AS9002, DE) was observed relaying this "
            "transaction, the same relay IP seen near TX-1004. "
            "This may indicate shared relay infrastructure between "
            "the two cash-out transactions."
        ),
        supporting_metrics={
            "candidate_ip": "203.0.113.88",
            "asn": 9002,
            "country": "DE",
            "corroborating_txids": [TX_1004],
        },
        confidence=0.58,
        is_synthetic=True,
        provenance=PROVENANCE,
    ))
    return ledger


def _build_risk_scores() -> dict[str, RiskScore]:
    """Build risk scores spanning all 4 priority tiers."""
    ledger_1001 = _build_evidence_ledger_tx1001()
    ledger_1006 = _build_evidence_ledger_tx1006()
    ledger_1002 = _build_evidence_ledger_tx1002()
    ledger_1004 = _build_evidence_ledger_tx1004()
    ledger_1007 = _build_evidence_ledger_tx1007()

    return {
        str(TX_1001): RiskScore(
            entity_id=str(TX_1001),
            entity_type="transaction",
            score=92.5,
            priority_tier=PriorityTier.CRITICAL,
            tier_description="Critical Priority: Multiple corroborating signals — classifier, graph, anomaly, and network correlation",
            behavioral_signal=0.95,
            graph_signal=0.85,
            anomaly_signal=0.78,
            correlation_signal=0.82,
            active_signals=["behavioral", "graph", "anomaly", "correlation"],
            explanation=(
                "Illicit classification (p=0.94) corroborated by direct link "
                "to high-throughput mixer, volume anomaly (4.2σ), and IP "
                "correlation with TX-1006. All four signal categories converge."
            ),
            evidence_ledger=ledger_1001,
            contains_synthetic_input=True,
        ),
        str(TX_1006): RiskScore(
            entity_id=str(TX_1006),
            entity_type="transaction",
            score=85.0,
            priority_tier=PriorityTier.CRITICAL,
            tier_description="Critical Priority: Illicit classifier prediction with IP corroboration",
            behavioral_signal=0.88,
            correlation_signal=0.74,
            active_signals=["behavioral", "correlation"],
            explanation=(
                "Illicit classification (p=0.88) with same-IP corroboration "
                "as TX-1001. Two corroborating signals from independent categories."
            ),
            evidence_ledger=ledger_1006,
            contains_synthetic_input=True,
        ),
        str(TX_1004): RiskScore(
            entity_id=str(TX_1004),
            entity_type="transaction",
            score=68.0,
            priority_tier=PriorityTier.HIGH,
            tier_description="High Priority: Consolidation pattern with volume anomaly and relay IP",
            graph_signal=0.72,
            anomaly_signal=0.71,
            correlation_signal=0.61,
            active_signals=["graph", "anomaly", "correlation"],
            explanation=(
                "Post-mixer consolidation (2→1 merge, 13.8 BTC) is 3.6σ above "
                "baseline. Relay IP 203.0.113.88 observed near both TX-1004 "
                "and TX-1007."
            ),
            evidence_ledger=ledger_1004,
            contains_synthetic_input=True,
        ),
        str(TX_1007): RiskScore(
            entity_id=str(TX_1007),
            entity_type="transaction",
            score=60.0,
            priority_tier=PriorityTier.HIGH,
            tier_description="High Priority: Rapid mixer-to-cashout turnaround with shared relay",
            graph_signal=0.65,
            anomaly_signal=0.68,
            correlation_signal=0.58,
            active_signals=["graph", "anomaly", "correlation"],
            explanation=(
                "Direct mixer→cash-out in <15 min (3.1σ faster than baseline). "
                "Shares relay IP with TX-1004."
            ),
            evidence_ledger=ledger_1007,
            contains_synthetic_input=True,
        ),
        str(TX_1002): RiskScore(
            entity_id=str(TX_1002),
            entity_type="transaction",
            score=55.0,
            priority_tier=PriorityTier.MEDIUM,
            tier_description="Medium Priority: Anomalous fan-out with temporal proximity to source",
            anomaly_signal=0.62,
            active_signals=["anomaly", "temporal"],
            explanation=(
                "1:3 fan-out from mixer (2.8σ above baseline) within same "
                "time_step as illicit source TX-1001."
            ),
            evidence_ledger=ledger_1002,
            contains_synthetic_input=True,
        ),
        str(TX_1003): RiskScore(
            entity_id=str(TX_1003),
            entity_type="transaction",
            score=52.0,
            priority_tier=PriorityTier.MEDIUM,
            tier_description="Medium Priority: Anomalous split pattern from mixer",
            anomaly_signal=0.58,
            active_signals=["anomaly"],
            explanation=(
                "Secondary split from mixer with 1:2 fan-out. Isolated "
                "anomaly signal without independent corroboration."
            ),
            contains_synthetic_input=True,
        ),
        str(TX_1005): RiskScore(
            entity_id=str(TX_1005),
            entity_type="transaction",
            score=8.0,
            priority_tier=PriorityTier.LOW,
            tier_description="Low Priority: Routine exchange transaction",
            active_signals=[],
            explanation=(
                "Legitimate exchange activity (0.5 BTC). No anomalous signals. "
                "Included as baseline control."
            ),
            contains_synthetic_input=False,
        ),
    }


def _build_correlations() -> dict[int, list[dict]]:
    """Build synthetic network correlations."""
    return {
        TX_1001: [
            {
                "candidate_ip": "198.51.100.45",
                "role": "originator",
                "correlation_confidence": 0.82,
                "timestamp": "2024-03-15T10:30:00Z",
                "asn": 13335,
                "country": "US",
                "script_type": "P2PKH",
                "is_synthetic": True,
            }
        ],
        TX_1006: [
            {
                "candidate_ip": "198.51.100.45",
                "role": "originator",
                "correlation_confidence": 0.74,
                "timestamp": "2024-03-15T10:45:00Z",
                "asn": 13335,
                "country": "US",
                "script_type": "P2PKH",
                "is_synthetic": True,
            }
        ],
        TX_1004: [
            {
                "candidate_ip": "203.0.113.88",
                "role": "relay",
                "correlation_confidence": 0.61,
                "timestamp": "2024-03-15T11:20:00Z",
                "asn": 9002,
                "country": "DE",
                "script_type": "P2SH",
                "is_synthetic": True,
            }
        ],
        TX_1007: [
            {
                "candidate_ip": "203.0.113.88",
                "role": "relay",
                "correlation_confidence": 0.58,
                "timestamp": "2024-03-15T11:05:00Z",
                "asn": 9002,
                "country": "DE",
                "script_type": "P2SH",
                "is_synthetic": True,
            }
        ],
    }


def _build_graph_edges() -> dict:
    """Build graph edges for the Operation Shadow Mixer topology.

    Returns:
        Dictionary with keys: addr_tx, tx_addr, tx_tx, addr_addr
    """
    return {
        "addr_tx": [
            # Source wallets → transactions
            AddrTxEdge(input_address=WALLET_A, txid=TX_1001, is_synthetic=True, provenance=PROVENANCE),
            AddrTxEdge(input_address=WALLET_A, txid=TX_1006, is_synthetic=True, provenance=PROVENANCE),
            # Mixer → outbound transactions
            AddrTxEdge(input_address=WALLET_B, txid=TX_1002, is_synthetic=True, provenance=PROVENANCE),
            AddrTxEdge(input_address=WALLET_B, txid=TX_1003, is_synthetic=True, provenance=PROVENANCE),
            AddrTxEdge(input_address=WALLET_B, txid=TX_1007, is_synthetic=True, provenance=PROVENANCE),
            # Layering hop → consolidation
            AddrTxEdge(input_address=WALLET_C, txid=TX_1004, is_synthetic=True, provenance=PROVENANCE),
            # Clean exchange → legit tx
            AddrTxEdge(input_address=WALLET_E, txid=TX_1005, is_synthetic=True, provenance=PROVENANCE),
        ],
        "tx_addr": [
            # Source txs → mixer
            TxAddrEdge(txid=TX_1001, output_address=WALLET_B, is_synthetic=True, provenance=PROVENANCE),
            TxAddrEdge(txid=TX_1006, output_address=WALLET_B, is_synthetic=True, provenance=PROVENANCE),
            # Mixer splits → layering hop
            TxAddrEdge(txid=TX_1002, output_address=WALLET_C, is_synthetic=True, provenance=PROVENANCE),
            TxAddrEdge(txid=TX_1003, output_address=WALLET_C, is_synthetic=True, provenance=PROVENANCE),
            # Consolidation → cash-out
            TxAddrEdge(txid=TX_1004, output_address=WALLET_D, is_synthetic=True, provenance=PROVENANCE),
            # Fast turnaround → cash-out
            TxAddrEdge(txid=TX_1007, output_address=WALLET_D, is_synthetic=True, provenance=PROVENANCE),
            # Legit tx → legit user
            TxAddrEdge(txid=TX_1005, output_address=WALLET_E, is_synthetic=True, provenance=PROVENANCE),
        ],
        "tx_tx": [
            # Source → mixer output splits
            TxTxEdge(source_txid=TX_1001, target_txid=TX_1002, is_synthetic=True, provenance=PROVENANCE),
            TxTxEdge(source_txid=TX_1001, target_txid=TX_1003, is_synthetic=True, provenance=PROVENANCE),
            # Splits → consolidation
            TxTxEdge(source_txid=TX_1002, target_txid=TX_1004, is_synthetic=True, provenance=PROVENANCE),
            TxTxEdge(source_txid=TX_1003, target_txid=TX_1004, is_synthetic=True, provenance=PROVENANCE),
            # Second source → fast turnaround
            TxTxEdge(source_txid=TX_1006, target_txid=TX_1007, is_synthetic=True, provenance=PROVENANCE),
        ],
        "addr_addr": [
            # Source → mixer
            AddrAddrEdge(input_address=WALLET_A, output_address=WALLET_B, is_synthetic=True, provenance=PROVENANCE),
            # Mixer → layering
            AddrAddrEdge(input_address=WALLET_B, output_address=WALLET_C, is_synthetic=True, provenance=PROVENANCE),
            # Mixer → cash-out (direct path)
            AddrAddrEdge(input_address=WALLET_B, output_address=WALLET_D, is_synthetic=True, provenance=PROVENANCE),
            # Layering → cash-out
            AddrAddrEdge(input_address=WALLET_C, output_address=WALLET_D, is_synthetic=True, provenance=PROVENANCE),
        ],
    }


class DemoDataset:
    """Complete demonstration dataset for SIH live demo.

    All data is explicitly synthetic and marked accordingly.
    """

    def __init__(self) -> None:
        self.transactions = _build_transactions()
        self.wallets = _build_wallets()
        self.risk_scores = _build_risk_scores()
        self.correlations = _build_correlations()
        self.graph_edges = _build_graph_edges()
        self.alerts = self._generate_alerts()

    def _generate_alerts(self) -> dict[str, InvestigativeAlert]:
        """Generate alerts from risk scores that meet the medium threshold."""
        alerts: dict[str, InvestigativeAlert] = {}
        for rs in self.risk_scores.values():
            alert = generate_alert(
                rs,
                evidence_ledger=rs.evidence_ledger,
            )
            if alert:
                alerts[alert.alert_id] = alert
        return alerts


def build_demo_dataset() -> DemoDataset:
    """Build and return the complete Operation Shadow Mixer demo dataset."""
    return DemoDataset()
