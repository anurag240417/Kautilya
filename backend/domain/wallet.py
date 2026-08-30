"""Wallet domain models.

Defines the canonical schema for Elliptic++ wallet/address records.

Note on actual CSV headers:
    The ``wallets_features.csv`` header uses ``num_txs_as receiver``
    (with a space instead of underscore). Normalization must map this
    to the canonical ``num_txs_as_receiver`` field.
"""


from pydantic import BaseModel, Field

from backend.domain.types import EntityClass


class StatsSummary(BaseModel):
    """Five-number summary reused across wallet feature groups.

    Used for BTC amounts, fees, block intervals, and address
    interaction statistics where the dataset provides
    total/min/max/mean/median variants.
    """

    total: float | None = None
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    median: float | None = None


class Wallet(BaseModel):
    """Canonical wallet/address record with interpretable features.

    All feature fields are optional because they are populated during
    normalization (Phase 2), not at parse time.
    """

    address: str = Field(description="Elliptic++ wallet address identifier")
    time_step: int = Field(ge=1, le=49, description="Temporal step index (1–49)")
    label: EntityClass | None = Field(
        default=None, description="1=Illicit, 2=Licit, 3=Unknown; many addresses lack labels"
    )

    # --- Scalar features ---
    num_txs_as_sender: float | None = None
    num_txs_as_receiver: float | None = None
    first_block_appeared_in: float | None = None
    last_block_appeared_in: float | None = None
    lifetime_in_blocks: float | None = None
    total_txs: float | None = None
    first_sent_block: float | None = None
    first_received_block: float | None = None
    num_timesteps_appeared_in: float | None = None
    num_addr_transacted_multiple: float | None = None

    # --- Grouped stats features ---
    btc_transacted: StatsSummary | None = None
    btc_sent: StatsSummary | None = None
    btc_received: StatsSummary | None = None
    fees: StatsSummary | None = None
    fees_as_share: StatsSummary | None = None
    blocks_btwn_txs: StatsSummary | None = None
    blocks_btwn_input_txs: StatsSummary | None = None
    blocks_btwn_output_txs: StatsSummary | None = None
    transacted_w_address: StatsSummary | None = None
