"""Risk scoring and multi-signal synthesis engine.

Synthesizes independent analytical signals (supervised ML classification,
anomaly deviance, graph structural patterns, temporal/network correlation,
and known indicators) into an explainable investigative prioritization score.

Forensic Rules:
    1. The final risk score represents an INVESTIGATIVE PRIORITIZATION AND
       TRIAGE RANKING, NOT a probability of criminality, proof of guilt,
       or direct model confidence.
    2. Anomaly ≠ Illicitness: Statistical deviance from baseline indicates
       novelty, not confirmed criminality. Uncorroborated anomaly scores are
       capped and cannot unilaterally place an entity in CRITICAL tier.
    3. Correlation ≠ Attribution: Correlation confidence measures connection
       plausibility; it does not prove wallet ownership.
    4. Scoring rules are documented, configurable, and avoid arbitrary unvalidated
       constants.
"""

from dataclasses import dataclass, field
from enum import StrEnum

from backend.domain.risk import RiskScore, SignalInput
from backend.domain.types import PriorityTier


class SynthesisPolicy(StrEnum):
    """Synthesis policy defining how independent signals combine."""

    # Corroborated signals boosted; isolated anomalies capped
    CORROBORATED_TIER = "corroborated_tier"
    # Standard dynamic normalization over active signals
    DYNAMIC_NORMALIZED = "dynamic_normalized"
    # Dominant signal upper bound with corroboration factor
    MAX_CORROBORATED = "max_corroborated"


@dataclass(frozen=True)
class PriorityTierDefinition:
    """Definition of a single investigative priority tier."""

    tier: PriorityTier
    min_score: float
    description: str


@dataclass(frozen=True)
class PriorityTierConfig:
    """Configurable priority tier definitions.

    Defines how continuous risk scores (0–100) map to investigative
    priority tiers. Tiers are evaluated in order from highest to lowest;
    the first tier whose min_score is met is assigned.

    Default tiers:
        CRITICAL ≥ 80: Requires immediate investigator attention
        HIGH     ≥ 60: Should be reviewed promptly
        MEDIUM   ≥ 40: May warrant further analysis
        LOW      < 40: Routine / no immediate concern
    """

    tiers: tuple[PriorityTierDefinition, ...] = (
        PriorityTierDefinition(
            tier=PriorityTier.CRITICAL,
            min_score=80.0,
            description=(
                "Requires immediate investigator attention; "
                "multiple corroborating signals present"
            ),
        ),
        PriorityTierDefinition(
            tier=PriorityTier.HIGH,
            min_score=60.0,
            description="Should be reviewed promptly",
        ),
        PriorityTierDefinition(
            tier=PriorityTier.MEDIUM,
            min_score=40.0,
            description="May warrant further analysis",
        ),
        PriorityTierDefinition(
            tier=PriorityTier.LOW,
            min_score=0.0,
            description="Routine; no immediate concern",
        ),
    )

    def determine_tier(self, score: float) -> PriorityTierDefinition:
        """Map a continuous risk score to its priority tier definition.

        Tiers are evaluated highest-first; the first tier whose
        min_score threshold is met is returned.
        """
        for tier_def in self.tiers:
            if score >= tier_def.min_score:
                return tier_def
        # Fallback to last tier (should always be LOW with min_score=0)
        return self.tiers[-1]


DEFAULT_TIER_CONFIG = PriorityTierConfig()


@dataclass(frozen=True)
class SynthesisConfig:
    """Configurable weights, thresholds, and caps for risk synthesis."""

    policy: SynthesisPolicy = SynthesisPolicy.CORROBORATED_TIER

    # Relative signal weights (dynamically normalized across active signals)
    behavioral_weight: float = 0.35
    graph_weight: float = 0.25
    anomaly_weight: float = 0.20
    correlation_weight: float = 0.20
    known_indicator_weight: float = 0.40

    # Configurable priority tier definitions
    tier_config: PriorityTierConfig = field(
        default_factory=PriorityTierConfig
    )

    # Safety cap for anomaly scores lacking corroboration
    uncorroborated_anomaly_cap: float = 60.0

    # Corroboration multiplier when 2+ independent signals agree
    corroboration_factor: float = 1.15

    # Custom signal weights mapping for open-ended extensions
    custom_weights: dict[str, float] = field(default_factory=dict)


DEFAULT_SYNTHESIS_CONFIG = SynthesisConfig()


def determine_priority_tier(
    score: float, config: SynthesisConfig
) -> PriorityTier:
    """Map continuous risk score (0–100) to an investigative priority tier."""
    return config.tier_config.determine_tier(score).tier


def synthesize_risk_score(
    signals: SignalInput,
    config: SynthesisConfig | None = None,
) -> RiskScore:
    """Synthesize multi-source signals into an investigative priority RiskScore.

    Combines present signals dynamically so that missing signals do not artificially
    suppress the score. Applies corroboration analysis and anomaly safety capping.

    Args:
        signals: SignalInput domain model containing core and custom signals.
        config: Optional SynthesisConfig defining policy, weights, and thresholds.

    Returns:
        RiskScore domain object with priority tier, active signal accounting,
        and forensic narrative.
    """
    cfg = config or DEFAULT_SYNTHESIS_CONFIG

    # Collect active signals and their configured weights
    active_pairs: list[tuple[str, float, float]] = []

    if signals.illicit_probability is not None:
        active_pairs.append(
            ("illicit_probability", signals.illicit_probability, cfg.behavioral_weight)
        )
    if signals.anomaly_score is not None:
        active_pairs.append(("anomaly_score", signals.anomaly_score, cfg.anomaly_weight))
    if signals.graph_signal is not None:
        active_pairs.append(("graph_signal", signals.graph_signal, cfg.graph_weight))
    if signals.correlation_confidence is not None:
        active_pairs.append(
            ("correlation_confidence", signals.correlation_confidence, cfg.correlation_weight)
        )
    if signals.known_indicator_signal is not None:
        active_pairs.append(
            ("known_indicator_signal", signals.known_indicator_signal, cfg.known_indicator_weight)
        )

    # Process open-ended custom signals
    for name, val in signals.custom_signals.items():
        w = cfg.custom_weights.get(name, 0.20)
        active_pairs.append((name, max(0.0, min(1.0, float(val))), w))

    # If no signals are present, return neutral zero score
    if not active_pairs:
        return RiskScore(
            entity_id=signals.entity_id,
            entity_type=signals.entity_type,
            score=0.0,
            priority_tier=PriorityTier.LOW,
            active_signals=[],
            explanation="No investigative signals were supplied for evaluation.",
            contains_synthetic_input=signals.contains_synthetic_input,
        )

    active_names = [name for name, _, _ in active_pairs]
    total_weight = sum(w for _, _, w in active_pairs)

    # Dynamic normalized weighted average over active signals
    if total_weight > 0:
        base_score_0_1 = sum(val * w for _, val, w in active_pairs) / total_weight
    else:
        base_score_0_1 = sum(val for _, val, _ in active_pairs) / len(active_pairs)

    # Scale to standard [0.0, 100.0] investigative priority range
    raw_score = base_score_0_1 * 100.0

    # Apply Corroborated Tier Policy
    if cfg.policy == SynthesisPolicy.CORROBORATED_TIER:
        # Check for multi-signal corroboration (e.g. supervised probability + graph/anomaly)
        has_high_supervised = (
            signals.illicit_probability is not None and signals.illicit_probability >= 0.60
        )
        has_corroborating_evidence = any([
            signals.graph_signal is not None and signals.graph_signal >= 0.50,
            signals.anomaly_score is not None and signals.anomaly_score >= 0.60,
            signals.known_indicator_signal is not None and signals.known_indicator_signal >= 0.50,
        ])

        if has_high_supervised and has_corroborating_evidence:
            # Corroborated boost
            raw_score = min(100.0, raw_score * cfg.corroboration_factor)

        # Anomaly safety cap: if anomaly is the only strong signal, prevent false elevation
        is_isolated_anomaly = (
            signals.anomaly_score is not None
            and signals.anomaly_score >= 0.60
            and (signals.illicit_probability is None or signals.illicit_probability < 0.40)
            and (signals.graph_signal is None or signals.graph_signal < 0.40)
            and (signals.known_indicator_signal is None or signals.known_indicator_signal < 0.40)
        )
        if is_isolated_anomaly:
            raw_score = min(raw_score, cfg.uncorroborated_anomaly_cap)

    elif cfg.policy == SynthesisPolicy.MAX_CORROBORATED:
        max_signal_val = max(val for _, val, _ in active_pairs)
        raw_score = max_signal_val * 100.0
        if len(active_pairs) >= 2:
            raw_score = min(100.0, raw_score * 1.10)

    final_score = round(max(0.0, min(100.0, float(raw_score))), 2)
    tier_def = cfg.tier_config.determine_tier(final_score)
    tier = tier_def.tier

    # Provenance tracking: includes synthetic if inputs rely on synthetic data or correlation
    has_synthetic = (
        signals.contains_synthetic_input or (signals.correlation_confidence is not None)
    )

    # Generate investigative explanation summary
    explanation_parts: list[str] = [
        f"Investigative priority score {final_score:.1f}/100 ({tier.value.upper()} tier)",
        f"synthesized across {len(active_pairs)} active signals ({', '.join(active_names)}).",
    ]
    if signals.anomaly_score is not None and signals.illicit_probability is None:
        explanation_parts.append(
            "Note: Flagged primarily due to statistical anomaly deviance without "
            "confirmed illicit classification."
        )
    if has_synthetic:
        explanation_parts.append(
            "(Includes synthetic network or temporal correlation signals)."
        )

    return RiskScore(
        entity_id=signals.entity_id,
        entity_type=signals.entity_type,
        score=final_score,
        priority_tier=tier,
        tier_description=tier_def.description,
        behavioral_signal=signals.illicit_probability,
        graph_signal=signals.graph_signal,
        anomaly_signal=signals.anomaly_score,
        correlation_signal=signals.correlation_confidence,
        known_indicator_signal=signals.known_indicator_signal,
        active_signals=active_names,
        explanation=" ".join(explanation_parts),
        contains_synthetic_input=has_synthetic,
    )
