"""Structured Evidence Ledger generator and investigator explainability engine.

Compiles and categorizes contributing evidence across ML behavioral, anomaly,
graph structural, temporal, and synthetic network domains into natural,
human-understandable narratives.

Forensic Rules:
    1. Human-facing explanations MUST use known interpretable behavioral
       dimensions (volume, fees, size, degree, relay hops).
    2. Anonymized feature names (`Local_feature_*`, `Aggregate_feature_*`)
       must NEVER appear in human-facing explanations.
    3. Anomaly explanations MUST clearly state that deviance indicates
       novelty, not confirmed illicitness.
    4. Graph explanations MUST state that graph proximity or centrality alone
       does not prove guilt.
    5. Synthetic network and correlation records MUST carry `is_synthetic=True`
       and explicit non-attribution disclaimers.
"""

import uuid
from typing import Any

from backend.domain.correlation import (
    CandidateIPRole,
    TemporalCorrelation,
    TransactionIPCorrelation,
)
from backend.domain.ml import MLScore
from backend.domain.risk import EvidenceCategory, EvidenceLedger, EvidenceRecord
from backend.domain.transaction import Transaction
from backend.domain.types import EntityClass, EvidenceType


def _format_btc_amount(amount: float | None) -> str:
    """Format BTC amount gracefully."""
    if amount is None:
        return "unspecified amount"
    return f"{amount:,.4f} BTC"


def build_evidence_ledger(
    entity_id: str,
    entity_type: str = "transaction",
    tx: Transaction | None = None,
    interpretable_features: dict[str, Any] | None = None,
    ml_score: MLScore | None = None,
    anomaly_score: float | None = None,
    graph_metrics: dict[str, float] | None = None,
    temporal_correlations: list[TemporalCorrelation] | None = None,
    tx_ip_correlations: list[TransactionIPCorrelation] | None = None,
    known_indicator_match: str | None = None,
) -> EvidenceLedger:
    """Build a structured Evidence Ledger compiling all forensic evidence for an entity.

    Args:
        entity_id: Transaction ID or wallet address being evaluated.
        entity_type: Type of entity ('transaction' or 'wallet').
        tx: Optional canonical Transaction record with interpretable features.
        interpretable_features: Optional dictionary of domain-level interpretable metrics.
        ml_score: Optional supervised classification result.
        anomaly_score: Optional unsupervised anomaly deviance score.
        graph_metrics: Optional graph-derived metrics (degrees, PageRank, centrality).
        temporal_correlations: Optional temporal window correlations.
        tx_ip_correlations: Optional network transaction-to-IP correlations.
        known_indicator_match: Optional description of watchlist or indicator match.

    Returns:
        Structured EvidenceLedger with categorized, human-understandable records.
    """
    ledger = EvidenceLedger(entity_id=str(entity_id), entity_type=entity_type)

    # 1. ML Behavioral Evidence (Supervised Model)
    if ml_score is not None:
        # Robustly resolve predicted label and probability
        if hasattr(ml_score, "predicted_label") and ml_score.predicted_label is not None:
            pred_label = ml_score.predicted_label
            pred_label_name = (
                pred_label.name.capitalize() if hasattr(pred_label, "name") else str(pred_label)
            )
            pred_value = pred_label.value if hasattr(pred_label, "value") else pred_label
        elif getattr(ml_score, "prediction", None) in (1, EntityClass.ILLICIT):
            pred_label = EntityClass.ILLICIT
            pred_label_name = "Illicit"
            pred_value = EntityClass.ILLICIT.value
        else:
            pred_label = EntityClass.LICIT
            pred_label_name = "Licit"
            pred_value = EntityClass.LICIT.value

        prob = getattr(ml_score, "illicit_probability", None)
        if prob is None:
            prob = getattr(ml_score, "probability", 0.0) or 0.0

        pct = prob * 100.0

        behavioral_narratives: list[str] = [
            f"Supervised model ({ml_score.model_version}) classified this transaction as "
            f"{pred_label_name} with an illicit probability of {pct:.1f}%."
        ]

        metrics: dict[str, Any] = {
            "predicted_label": pred_value,
            "illicit_probability": prob,
            "model_version": ml_score.model_version,
        }

        # Enrich with known interpretable transaction attributes (never anonymized columns)
        if tx is not None:
            if tx.total_btc is not None:
                metrics["total_btc"] = tx.total_btc
                behavioral_narratives.append(
                    f"Total value transferred is {_format_btc_amount(tx.total_btc)}."
                )
            if tx.fees is not None:
                metrics["fees"] = tx.fees
                behavioral_narratives.append(
                    f"Transaction fee is {_format_btc_amount(tx.fees)}."
                )
            if tx.num_input_addresses is not None and tx.num_output_addresses is not None:
                metrics["inputs"] = tx.num_input_addresses
                metrics["outputs"] = tx.num_output_addresses
                behavioral_narratives.append(
                    f"Involves {int(tx.num_input_addresses)} input address(es) and "
                    f"{int(tx.num_output_addresses)} output address(es)."
                )

        if interpretable_features:
            for k, v in interpretable_features.items():
                if not k.startswith("Local_feature_") and not k.startswith("Aggregate_feature_"):
                    metrics[k] = v

        ledger.add_record(
            EvidenceRecord(
                evidence_id=f"ev-ml-{uuid.uuid4().hex[:8]}",
                category=EvidenceCategory.ML_BEHAVIORAL,
                evidence_type=EvidenceType.MODEL_PREDICTION,
                source_entity_id=str(entity_id),
                source_entity_type=entity_type,
                headline=f"Supervised Classification: {pred_label_name} ({pct:.1f}% likelihood)",
                description=" ".join(behavioral_narratives),
                supporting_metrics=metrics,
                confidence=prob,
                is_synthetic=False,
                provenance=f"classifier_{ml_score.model_version}",
            )
        )

    # 2. Anomaly Evidence (Statistical Novelty)
    if anomaly_score is not None:
        pct_anomaly = anomaly_score * 100.0
        if anomaly_score >= 0.75:
            severity = "high statistical deviance"
        elif anomaly_score >= 0.50:
            severity = "moderate statistical deviance"
        else:
            severity = "low statistical deviance"

        ledger.add_record(
            EvidenceRecord(
                evidence_id=f"ev-anom-{uuid.uuid4().hex[:8]}",
                category=EvidenceCategory.ANOMALY,
                evidence_type=EvidenceType.MODEL_PREDICTION,
                source_entity_id=str(entity_id),
                source_entity_type=entity_type,
                headline=f"Statistical Anomaly Detection: {severity} ({pct_anomaly:.1f}%)",
                description=(
                    f"Observation exhibits {severity} relative to the learned baseline "
                    f"distribution (deviance score: {anomaly_score:.2f}). Important: Anomaly "
                    "detection measures statistical novelty, NOT confirmed illicit "
                    "activity."
                ),
                supporting_metrics={"anomaly_score": anomaly_score, "severity": severity},
                confidence=anomaly_score,
                is_synthetic=False,
                provenance="isolation_forest_anomaly_detector",
            )
        )

    # 3. Graph Structural Evidence
    if graph_metrics or (tx and (tx.in_txs_degree is not None or tx.out_txs_degree is not None)):
        graph_details: list[str] = []
        g_metrics: dict[str, Any] = {}

        if tx and tx.in_txs_degree is not None:
            g_metrics["in_txs_degree"] = tx.in_txs_degree
            graph_details.append(
                f"In-degree of {int(tx.in_txs_degree)} transaction input connections"
            )
        if tx and tx.out_txs_degree is not None:
            g_metrics["out_txs_degree"] = tx.out_txs_degree
            graph_details.append(
                f"out-degree of {int(tx.out_txs_degree)} transaction output connections"
            )

        if graph_metrics:
            for k, v in graph_metrics.items():
                g_metrics[k] = v
                if k == "pagerank":
                    graph_details.append(f"PageRank centrality of {v:.6f}")
                elif k == "betweenness":
                    graph_details.append(f"betweenness centrality score of {v:.4f}")

        description_text = (
            f"Relational structure exhibits {', '.join(graph_details)}. "
            "Forensic caveat: Structural graph centrality or connectivity alone is "
            "never proof of illicit conduct."
        )

        ledger.add_record(
            EvidenceRecord(
                evidence_id=f"ev-graph-{uuid.uuid4().hex[:8]}",
                category=EvidenceCategory.GRAPH_STRUCTURAL,
                evidence_type=EvidenceType.OBSERVATION,
                source_entity_id=str(entity_id),
                source_entity_type=entity_type,
                headline="Graph Structural Patterns",
                description=description_text,
                supporting_metrics=g_metrics,
                confidence=None,
                is_synthetic=False,
                provenance="elliptic_plus_plus_graph",
            )
        )

    # 4. Temporal Correlation Evidence
    if temporal_correlations:
        best_corr = max(temporal_correlations, key=lambda c: c.temporal_proximity)
        time_step_info = f"time_step {best_corr.time_step}"
        window_start_str = best_corr.window_start.strftime("%Y-%m-%d")
        window_end_str = best_corr.window_end.strftime("%Y-%m-%d")
        window_desc = f"{window_start_str} to {window_end_str}"

        if best_corr.delta_seconds == 0.0:
            proximity_desc = "perfect alignment inside the synthetic execution window"
        else:
            hours = best_corr.delta_seconds / 3600.0
            proximity_desc = f"{hours:.1f} hours outside the window boundary"

        headline_pct = best_corr.temporal_proximity * 100.0
        ledger.add_record(
            EvidenceRecord(
                evidence_id=f"ev-temp-{uuid.uuid4().hex[:8]}",
                category=EvidenceCategory.TEMPORAL,
                evidence_type=EvidenceType.CORRELATION,
                source_entity_id=str(entity_id),
                source_entity_type=entity_type,
                headline=f"Temporal Window Alignment ({headline_pct:.1f}% proximity)",
                description=(
                    f"Transaction aligns with synthetic {time_step_info} ({window_desc}) "
                    f"with {proximity_desc}. (Synthetic temporal mapping: does not "
                    "represent real-world historical timestamps)."
                ),
                supporting_metrics={
                    "time_step": best_corr.time_step,
                    "temporal_proximity": best_corr.temporal_proximity,
                    "delta_seconds": best_corr.delta_seconds,
                },
                confidence=best_corr.temporal_proximity,
                is_synthetic=True,
                provenance="deterministic_temporal_mapping",
            )
        )

    # 5. Synthetic Network Evidence
    if tx_ip_correlations:
        origin_corrs = [c for c in tx_ip_correlations if c.role == CandidateIPRole.ORIGIN]
        relay_corrs = [c for c in tx_ip_correlations if c.role == CandidateIPRole.RELAY]

        net_summaries: list[str] = []
        if origin_corrs:
            orig = origin_corrs[0]
            geo_desc = f" (ASN {orig.asn}, {orig.country})" if orig.asn or orig.country else ""
            conf_pct = orig.correlation_confidence * 100.0
            net_summaries.append(
                f"Candidate broadcast origin IP {orig.ip}{geo_desc} with {conf_pct:.1f}% "
                "correlation confidence"
            )
        if relay_corrs:
            net_summaries.append(f"{len(relay_corrs)} observed network relay peer(s)")

        best_conf = max(c.correlation_confidence for c in tx_ip_correlations)
        headline_net = f"Synthetic Network Correlation ({len(tx_ip_correlations)} IP observations)"
        ledger.add_record(
            EvidenceRecord(
                evidence_id=f"ev-net-{uuid.uuid4().hex[:8]}",
                category=EvidenceCategory.SYNTHETIC_NETWORK,
                evidence_type=EvidenceType.CORRELATION,
                source_entity_id=str(entity_id),
                source_entity_type=entity_type,
                headline=headline_net,
                description=(
                    "Transaction propagation observed across network: "
                    f"{'; '.join(net_summaries)}. "
                    "Mandatory caveat: Simulated network data. Temporal correlation alone "
                    "must NOT be treated as proof of wallet/IP ownership."
                ),
                supporting_metrics={
                    "total_ip_observations": len(tx_ip_correlations),
                    "candidate_origin_ip": origin_corrs[0].ip if origin_corrs else None,
                    "max_correlation_confidence": best_conf,
                },
                confidence=best_conf,
                is_synthetic=True,
                provenance="synthetic_network_generator",
            )
        )

    # 6. Known Indicator Evidence
    if known_indicator_match:
        ledger.add_record(
            EvidenceRecord(
                evidence_id=f"ev-ind-{uuid.uuid4().hex[:8]}",
                category=EvidenceCategory.KNOWN_INDICATOR,
                evidence_type=EvidenceType.OBSERVATION,
                source_entity_id=str(entity_id),
                source_entity_type=entity_type,
                headline="High-Confidence Indicator Match",
                description=(
                    f"Entity matched known investigative heuristic or watchlist: "
                    f"{known_indicator_match}."
                ),
                supporting_metrics={"indicator": known_indicator_match},
                confidence=1.0,
                is_synthetic=False,
                provenance="known_indicators_list",
            )
        )

    return ledger


def generate_narrative_explanation(ledger: EvidenceLedger) -> str:
    """Generate a coherent, plain-language summary paragraph from an EvidenceLedger."""
    if not ledger.records:
        return "No specific evidence recorded for this entity."

    parts: list[str] = [f"Investigative review for {ledger.entity_type} {ledger.entity_id}:"]

    for r in ledger.records:
        parts.append(f"• {r.headline}: {r.description}")

    if ledger.has_synthetic_evidence():
        parts.append(
            "(Contains synthetic network/temporal evidence. Temporal alignment "
            "does not imply real-world ownership)."
        )

    return "\n".join(parts)
