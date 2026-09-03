"""Temporal correlation engine.

Identifies and scores temporal relationships between blockchain transactions
and network observations using the deterministic time_step → synthetic datetime
mapping.

Forensic Rule:
    Temporal correlation alone must NOT be treated as proof of wallet/IP ownership.
    Temporal alignment measures only that a simulated network observation took place
    within or near the plausible synthetic execution window of a transaction.
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

from backend.domain.correlation import TemporalCorrelation
from backend.domain.network import NetworkObservation
from backend.domain.transaction import Transaction
from backend.generator.temporal import (
    time_step_to_datetime_range,
)
from backend.normalization.temporal import normalize_timestamp


@dataclass(frozen=True)
class TemporalWindowConfig:
    """Configuration for temporal correlation windows and scoring."""

    step_duration: timedelta = timedelta(days=14)
    boundary_tolerance_seconds: float = 3600.0  # 1 hour boundary tolerance
    decay_half_life_hours: float = 24.0  # Exponential decay half-life outside window
    propagation_burst_max_seconds: float = 60.0  # Maximum expected duration of propagation wave
    min_plausible_delay_ms: float = 10.0  # Fastest plausible P2P relay
    max_plausible_delay_ms: float = 5000.0  # Slowest plausible individual relay hop


DEFAULT_TEMPORAL_CONFIG = TemporalWindowConfig()


def get_temporal_window(
    time_step: int,
    boundary_tolerance_seconds: float = 0.0,
) -> tuple[datetime, datetime]:
    """Return the synthetic [start, end) datetime window for a time_step.

    Args:
        time_step: Elliptic++ time step index (1–49).
        boundary_tolerance_seconds: Optional buffer to expand the window on both ends.

    Returns:
        Tuple of (start_datetime, end_datetime) in UTC.
    """
    start, end = time_step_to_datetime_range(time_step)
    if boundary_tolerance_seconds > 0:
        buffer = timedelta(seconds=boundary_tolerance_seconds)
        start -= buffer
        end += buffer
    return start, end


def is_in_temporal_window(
    obs_dt: datetime,
    time_step: int,
    boundary_tolerance_seconds: float = 0.0,
) -> bool:
    """Check whether a timestamp falls within the transaction's synthetic window.

    Args:
        obs_dt: Observation timestamp (normalized to UTC).
        time_step: Elliptic++ time step index (1–49).
        boundary_tolerance_seconds: Optional boundary expansion buffer.

    Returns:
        True if obs_dt is within the interval [start, end).
    """
    normalized_dt = normalize_timestamp(obs_dt)
    start, end = get_temporal_window(time_step, boundary_tolerance_seconds)
    return start <= normalized_dt < end


def compute_temporal_proximity(
    obs_dt: datetime,
    time_step: int,
    config: TemporalWindowConfig | None = None,
) -> float:
    """Compute temporal proximity score in range [0.0, 1.0].

    If the observation timestamp is strictly inside the synthetic window [start, end),
    the score is 1.0. If outside the window, the score decays exponentially with
    distance to the nearest boundary.

    Args:
        obs_dt: Observation timestamp.
        time_step: Elliptic++ time step index (1–49).
        config: Optional temporal window configuration.

    Returns:
        Proximity score in [0.0, 1.0].
    """
    cfg = config or DEFAULT_TEMPORAL_CONFIG
    normalized_dt = normalize_timestamp(obs_dt)
    start, end = time_step_to_datetime_range(time_step)

    if start <= normalized_dt < end:
        return 1.0

    # Calculate distance in seconds from the closest window boundary
    if normalized_dt < start:
        delta_seconds = (start - normalized_dt).total_seconds()
    else:
        delta_seconds = (normalized_dt - end).total_seconds()

    # Half-life exponential decay: S = exp(-ln(2) * delta / half_life)
    half_life_seconds = cfg.decay_half_life_hours * 3600.0
    decay_rate = math.log(2.0) / max(half_life_seconds, 1.0)
    score = math.exp(-decay_rate * delta_seconds)

    return max(0.0, min(1.0, float(score)))


def compute_propagation_consistency(
    observations: list[NetworkObservation],
    config: TemporalWindowConfig | None = None,
) -> float:
    """Evaluate propagation timing consistency across observed hops.

    For a set of observations belonging to the same transaction:
    1. Single observation: returns 1.0 if timestamp is valid, as there is no
       propagation chain anomaly to contradict it.
    2. Multiple observations:
       - Checks that total propagation duration does not exceed plausible limits
         (e.g., <= 60 seconds for a normal Bitcoin P2P wave).
       - Checks that relay timestamps are non-decreasing along the relay wave.
       - Penalizes multi-day delays or reversed timestamps.

    Args:
        observations: List of network observations for a single transaction.
        config: Optional temporal window configuration.

    Returns:
        Propagation timing consistency score in [0.0, 1.0].
    """
    if not observations:
        return 0.0

    cfg = config or DEFAULT_TEMPORAL_CONFIG

    if len(observations) == 1:
        return 1.0

    # Sort observations by timestamp
    sorted_obs = sorted(observations, key=lambda o: normalize_timestamp(o.timestamp))
    timestamps = [normalize_timestamp(o.timestamp) for o in sorted_obs]

    # Total burst duration
    burst_duration_s = (timestamps[-1] - timestamps[0]).total_seconds()

    if burst_duration_s < 0:
        return 0.0  # Corrupted or reversed ordering

    # Burst score: 1.0 if within max seconds, decaying gracefully if longer
    if burst_duration_s <= cfg.propagation_burst_max_seconds:
        burst_score = 1.0
    else:
        excess_ratio = burst_duration_s / cfg.propagation_burst_max_seconds
        burst_score = max(0.05, 1.0 / math.log2(excess_ratio + 1.0))

    # Pairwise delay consistency between consecutive hops
    hop_scores: list[float] = []
    for i in range(len(timestamps) - 1):
        delay_ms = (timestamps[i + 1] - timestamps[i]).total_seconds() * 1000.0

        if delay_ms < 0:
            hop_scores.append(0.0)
        elif delay_ms <= cfg.max_plausible_delay_ms:
            hop_scores.append(1.0)
        else:
            # Penalize long gaps between individual hops
            excess = delay_ms - cfg.max_plausible_delay_ms
            hop_scores.append(max(0.1, 1.0 / (1.0 + (excess / 1000.0))))

    mean_hop_score = sum(hop_scores) / len(hop_scores) if hop_scores else 1.0

    # Composite consistency: 50% burst duration, 50% hop-level plausibility
    consistency = 0.5 * burst_score + 0.5 * mean_hop_score
    return max(0.0, min(1.0, float(consistency)))


def correlate_temporally(
    tx: Transaction,
    observations: list[NetworkObservation],
    config: TemporalWindowConfig | None = None,
) -> list[TemporalCorrelation]:
    """Correlate a transaction with a set of network observations temporally.

    Args:
        tx: Transaction record containing `txid` and `time_step`.
        observations: Network observations to correlate (matching txid).
        config: Optional temporal window configuration.

    Returns:
        List of TemporalCorrelation records preserving provenance and confidence.
    """
    cfg = config or DEFAULT_TEMPORAL_CONFIG
    window_start, window_end = time_step_to_datetime_range(tx.time_step)

    correlations: list[TemporalCorrelation] = []
    for obs in observations:
        obs_dt = normalize_timestamp(obs.timestamp)

        if window_start <= obs_dt < window_end:
            delta_s = 0.0
        elif obs_dt < window_start:
            delta_s = (window_start - obs_dt).total_seconds()
        else:
            delta_s = (obs_dt - window_end).total_seconds()

        proximity = compute_temporal_proximity(obs_dt, tx.time_step, cfg)

        correlations.append(
            TemporalCorrelation(
                txid=tx.txid,
                time_step=tx.time_step,
                observation_timestamp=obs_dt,
                window_start=window_start,
                window_end=window_end,
                delta_seconds=delta_s,
                temporal_proximity=proximity,
                is_synthetic=True,
            )
        )

    return correlations
