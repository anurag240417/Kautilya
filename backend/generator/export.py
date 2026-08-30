"""Export module for generated synthetic network data.

Exports synthetic network observations and ground truth metadata
with full provenance metadata and strict ``is_synthetic=True`` verification.

See DATASET.md §9–10 and AGENTS.md §3.3 for synthetic data rules.
"""

import logging
from pathlib import Path

import pandas as pd

from backend.domain.network import NetworkObservation
from backend.generator.ground_truth import GroundTruthTracker

logger = logging.getLogger(__name__)


def observations_to_dataframe(
    observations: list[NetworkObservation],
) -> pd.DataFrame:
    """Convert a list of NetworkObservation domain models into a DataFrame.

    Validates that every row has ``is_synthetic=True``.

    Args:
        observations: List of NetworkObservation instances.

    Returns:
        pandas DataFrame containing observation fields.
    """
    if not observations:
        return pd.DataFrame(
            columns=[
                "txid",
                "src_ip",
                "dst_ip",
                "src_port",
                "dst_port",
                "protocol",
                "timestamp",
                "asn",
                "country",
                "script_type",
                "is_synthetic",
                "generation_run_id",
                "scenario_id",
                "generator_version",
            ]
        )

    rows = []
    for obs in observations:
        if not obs.is_synthetic:
            msg = f"Non-synthetic record detected in synthetic network export for txid {obs.txid}"
            raise ValueError(msg)

        rows.append(
            {
                "txid": obs.txid,
                "src_ip": obs.src_ip,
                "dst_ip": obs.dst_ip,
                "src_port": obs.src_port,
                "dst_port": obs.dst_port,
                "protocol": obs.protocol,
                "timestamp": obs.timestamp.isoformat(),
                "asn": obs.asn,
                "country": obs.country,
                "script_type": str(obs.script_type) if obs.script_type else None,
                "is_synthetic": True,
                "generation_run_id": obs.generation_run_id,
                "scenario_id": obs.scenario_id,
                "generator_version": obs.generator_version,
            }
        )

    df = pd.DataFrame(rows)
    logger.info("Converted %d network observations to DataFrame", len(df))
    return df


def export_observations_to_csv(
    observations: list[NetworkObservation] | pd.DataFrame,
    output_path: Path,
) -> Path:
    """Export network observations to a CSV file.

    Args:
        observations: List of NetworkObservation objects or a pre-formatted DataFrame.
        output_path: Target CSV file path.

    Returns:
        The Path to the written file.
    """
    if isinstance(observations, list):
        df = observations_to_dataframe(observations)
    else:
        df = observations

    # Ensure output parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Verify is_synthetic column
    if "is_synthetic" in df.columns:
        if not df["is_synthetic"].all():
            msg = "Export failed: DataFrame contains non-synthetic rows in network layer"
            raise ValueError(msg)
    else:
        df["is_synthetic"] = True

    df.to_csv(output_path, index=False)
    logger.info("Exported %d network observations to %s", len(df), output_path)
    return output_path


def export_ground_truth_to_csv(
    ground_truth: GroundTruthTracker | pd.DataFrame,
    output_path: Path,
) -> Path:
    """Export ground truth records to a CSV file.

    Args:
        ground_truth: GroundTruthTracker instance or DataFrame.
        output_path: Target CSV file path.

    Returns:
        The Path to the written file.
    """
    if isinstance(ground_truth, GroundTruthTracker):
        df = ground_truth.to_dataframe()
    else:
        df = ground_truth

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Exported %d ground truth records to %s", len(df), output_path)
    return output_path
