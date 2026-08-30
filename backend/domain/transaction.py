"""Transaction domain models.

Defines the canonical schema for Elliptic++ transaction records.

Important:
    ``txid`` values are anonymized numeric dataset IDs assigned by
    Elliptic++. They are NOT real 64-character Bitcoin transaction
    hashes and must not be presented as such.
"""


from pydantic import BaseModel, Field

from backend.domain.types import EntityClass


class Transaction(BaseModel):
    """Canonical transaction record with interpretable features.

    Interpretable features may be used in human-facing explanations.
    All feature fields are optional because they are populated during
    normalization (Phase 2), not at parse time.
    """

    txid: int = Field(description="Elliptic++ anonymized transaction ID (NOT a real Bitcoin TXID)")
    time_step: int = Field(ge=1, le=49, description="Temporal step index (1–49)")
    label: EntityClass | None = Field(
        default=None, description="1=Illicit, 2=Licit, 3=Unknown"
    )

    # --- Interpretable features (columns 168–184 in txs_features.csv) ---
    total_btc: float | None = None
    fees: float | None = None
    size: float | None = None
    num_input_addresses: float | None = None
    num_output_addresses: float | None = None
    in_txs_degree: float | None = None
    out_txs_degree: float | None = None
    in_btc_min: float | None = None
    in_btc_max: float | None = None
    in_btc_mean: float | None = None
    in_btc_median: float | None = None
    in_btc_total: float | None = None
    out_btc_min: float | None = None
    out_btc_max: float | None = None
    out_btc_mean: float | None = None
    out_btc_median: float | None = None
    out_btc_total: float | None = None


class TransactionFeatures(BaseModel):
    """Anonymized feature vectors for a transaction.

    These features may feed ML models internally, but MUST NOT appear
    directly in human-facing explanations per ARCHITECTURE.md and
    DATASET.md §7.2.

    The 93 local features and 72 aggregate features are stored as
    ordered lists matching their column order in txs_features.csv.
    """

    txid: int = Field(description="Elliptic++ anonymized transaction ID")
    local_features: list[float] = Field(
        description="Local_feature_1 through Local_feature_93"
    )
    aggregate_features: list[float] = Field(
        description="Aggregate_feature_1 through Aggregate_feature_72"
    )
