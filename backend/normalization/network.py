"""Network observation normalization.

Converts generated synthetic network observations into
canonical ``NetworkObservation`` domain objects. Ensures
``is_synthetic=True`` is preserved.
"""

from collections.abc import Iterator

import pandas as pd

from backend.domain.network import NetworkObservation
from backend.normalization.temporal import normalize_timestamp


def normalize_network_observations(df: pd.DataFrame) -> Iterator[NetworkObservation]:
    """Normalize raw network DataFrames into domain objects.

    Args:
        df: DataFrame of synthetic network observations.

    Yields:
        NetworkObservation objects for each row.
    """
    for row in df.itertuples(index=False):
        obs_obj = NetworkObservation(
            txid=row.txid,
            src_ip=row.src_ip,
            dst_ip=row.dst_ip,
            src_port=row.src_port,
            dst_port=row.dst_port,
            protocol=getattr(row, "protocol", None),
            timestamp=normalize_timestamp(row.timestamp),
            asn=getattr(row, "asn", None),
            country=getattr(row, "country", None),
            script_type=getattr(row, "script_type", None),
            is_synthetic=True,  # Force to true as per domain model rules
            generation_run_id=getattr(row, "generation_run_id", None),
            scenario_id=getattr(row, "scenario_id", None),
            generator_version=getattr(row, "generator_version", None),
        )
        yield obs_obj
