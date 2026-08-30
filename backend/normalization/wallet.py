"""Wallet normalization.

Converts parsed Elliptic++ wallet records into canonical
``Wallet`` domain objects.
"""

from collections.abc import Iterator
from typing import Any

import pandas as pd

from backend.domain.wallet import StatsSummary, Wallet


def _make_stats(row: Any, prefix: str) -> StatsSummary:  # noqa: ANN401
    """Create a StatsSummary object from a named tuple row using a column prefix."""
    # Handle the fact that some columns might be missing or capitalized differently
    # Standard prefixes in wallets_features:
    # btc_transacted, btc_sent, btc_received, fees, fees_as_share,
    # blocks_btwn_txs, blocks_btwn_input_txs, blocks_btwn_output_txs, transacted_w_address
    return StatsSummary(
        total=getattr(row, f"{prefix}_total", None),
        min=getattr(row, f"{prefix}_min", None),
        max=getattr(row, f"{prefix}_max", None),
        mean=getattr(row, f"{prefix}_mean", None),
        median=getattr(row, f"{prefix}_median", None),
    )


def normalize_wallets(
    df_features: pd.DataFrame, df_classes: pd.DataFrame
) -> Iterator[Wallet]:
    """Normalize raw wallet DataFrames into domain objects.

    Args:
        df_features: DataFrame of wallet features (from wallets_features.csv).
        df_classes: DataFrame of wallet classes (from wallets_classes.csv).

    Yields:
        Wallet objects for each row in the features dataset.
    """
    # Rename class to label to avoid Python reserved keyword issues in itertuples
    df_classes = df_classes.rename(columns={"class": "label"})
    # Merge features and classes on address
    merged = pd.merge(df_features, df_classes, on="address", how="left")

    for row in merged.itertuples(index=False):
        label = None if pd.isna(getattr(row, "label", None)) else int(getattr(row, "label"))

        wallet_obj = Wallet(
            address=row.address,
            time_step=row.time_step,
            label=label,
            num_txs_as_sender=getattr(row, "num_txs_as_sender", None),
            num_txs_as_receiver=getattr(row, "num_txs_as_receiver", None),
            first_block_appeared_in=getattr(row, "first_block_appeared_in", None),
            last_block_appeared_in=getattr(row, "last_block_appeared_in", None),
            lifetime_in_blocks=getattr(row, "lifetime_in_blocks", None),
            total_txs=getattr(row, "total_txs", None),
            first_sent_block=getattr(row, "first_sent_block", None),
            first_received_block=getattr(row, "first_received_block", None),
            num_timesteps_appeared_in=getattr(row, "num_timesteps_appeared_in", None),
            num_addr_transacted_multiple=getattr(row, "num_addr_transacted_multiple", None),
            btc_transacted=_make_stats(row, "btc_transacted"),
            btc_sent=_make_stats(row, "btc_sent"),
            btc_received=_make_stats(row, "btc_received"),
            fees=_make_stats(row, "fees"),
            fees_as_share=_make_stats(row, "fees_as_share"),
            blocks_btwn_txs=_make_stats(row, "blocks_btwn_txs"),
            blocks_btwn_input_txs=_make_stats(row, "blocks_btwn_input_txs"),
            blocks_btwn_output_txs=_make_stats(row, "blocks_btwn_output_txs"),
            transacted_w_address=_make_stats(row, "transacted_w_address"),
        )
        yield wallet_obj
