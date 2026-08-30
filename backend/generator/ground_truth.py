"""Ground truth recording for synthetic scenarios.

Records the ground truth labels, scenario metadata, and origin information
for injected synthetic scenarios and anomalies. This enables rigorous
evaluation of downstream ML detection capabilities (e.g., comparing M1, M2, and M3).

See CONTEXT.md §3 and DATASET.md §9–10 for provenance rules.
"""

import pandas as pd
from pydantic import BaseModel, Field

from backend.domain.types import EntityClass
from backend.generator.scenarios import ScenarioType


class GroundTruthRecord(BaseModel):
    """Ground truth record for a single transaction's synthetic network generation.

    Attributes:
        txid: Elliptic++ transaction ID.
        time_step: Temporal step index (1–49).
        label: Original dataset label (1=Illicit, 2=Licit, 3=Unknown).
        scenario_type: Network behavior scenario assigned to this transaction.
        is_anomalous: True if assigned a non-normal scenario.
        origin_node_id: ID of the synthetic origin node.
        origin_ip: IP address of the synthetic origin node.
        origin_country: Country code of the origin node.
        origin_asn: ASN of the origin node.
        generation_run_id: Unique generation run identifier.
        is_synthetic: Always True.
    """

    txid: int = Field(description="Elliptic++ transaction ID")
    time_step: int = Field(ge=1, le=49, description="Temporal step index")
    label: EntityClass | None = Field(default=None, description="Dataset label")
    scenario_type: str = Field(description="Scenario identifier (e.g. 'tor_origin')")
    is_anomalous: bool = Field(description="True if non-normal scenario")
    origin_node_id: int = Field(description="Origin node ID in topology")
    origin_ip: str = Field(description="Origin IP address")
    origin_country: str | None = Field(default=None, description="Origin country code")
    origin_asn: int | None = Field(default=None, description="Origin ASN")
    generation_run_id: str | None = Field(default=None, description="Generation run ID")
    is_synthetic: bool = Field(default=True, description="Always True")


class GroundTruthTracker:
    """Collector for synthetic network ground truth records."""

    def __init__(self) -> None:
        self._records: list[GroundTruthRecord] = []

    def record(
        self,
        txid: int,
        time_step: int,
        label: EntityClass | int | None,
        scenario_type: str | ScenarioType,
        origin_node_id: int,
        origin_ip: str,
        origin_country: str | None = None,
        origin_asn: int | None = None,
        generation_run_id: str | None = None,
    ) -> GroundTruthRecord:
        """Record ground truth for a transaction."""
        scenario_str = (
            scenario_type.value
            if isinstance(scenario_type, ScenarioType)
            else str(scenario_type)
        )
        is_anomalous = scenario_str != ScenarioType.NORMAL.value

        entity_label = None
        if label is not None:
            entity_label = EntityClass(int(label))

        rec = GroundTruthRecord(
            txid=txid,
            time_step=time_step,
            label=entity_label,
            scenario_type=scenario_str,
            is_anomalous=is_anomalous,
            origin_node_id=origin_node_id,
            origin_ip=origin_ip,
            origin_country=origin_country,
            origin_asn=origin_asn,
            generation_run_id=generation_run_id,
            is_synthetic=True,
        )
        self._records.append(rec)
        return rec

    def get_records(self) -> list[GroundTruthRecord]:
        """Return all recorded ground truth objects."""
        return list(self._records)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert ground truth records to a pandas DataFrame."""
        if not self._records:
            return pd.DataFrame(
                columns=[
                    "txid",
                    "time_step",
                    "label",
                    "scenario_type",
                    "is_anomalous",
                    "origin_node_id",
                    "origin_ip",
                    "origin_country",
                    "origin_asn",
                    "generation_run_id",
                    "is_synthetic",
                ]
            )

        data = []
        for r in self._records:
            data.append(
                {
                    "txid": r.txid,
                    "time_step": r.time_step,
                    "label": int(r.label) if r.label is not None else None,
                    "scenario_type": r.scenario_type,
                    "is_anomalous": r.is_anomalous,
                    "origin_node_id": r.origin_node_id,
                    "origin_ip": r.origin_ip,
                    "origin_country": r.origin_country,
                    "origin_asn": r.origin_asn,
                    "generation_run_id": r.generation_run_id,
                    "is_synthetic": True,
                }
            )
        return pd.DataFrame(data)

    def __len__(self) -> int:
        return len(self._records)
