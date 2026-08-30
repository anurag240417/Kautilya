"""Transaction normalization.

Converts parsed Elliptic++ transaction records into canonical
``Transaction`` and ``TransactionFeatures`` domain objects.
"""

from collections.abc import Iterator

import pandas as pd

from backend.domain.transaction import Transaction, TransactionFeatures


def normalize_transactions(
    df_features: pd.DataFrame, df_classes: pd.DataFrame
) -> Iterator[tuple[Transaction, TransactionFeatures]]:
    """Normalize raw transaction DataFrames into domain objects.

    Args:
        df_features: DataFrame of transaction features (from txs_features.csv).
        df_classes: DataFrame of transaction classes (from txs_classes.csv).

    Yields:
        Tuples of (Transaction, TransactionFeatures) for each row.
    """
    # Rename class to label to avoid Python reserved keyword issues in itertuples
    df_classes = df_classes.rename(columns={"class": "label"})
    # Merge features and classes on txId
    merged = pd.merge(df_features, df_classes, on="txId", how="left")

    for row in merged.itertuples(index=False):
        # Extract local features (Local_feature_1 ... Local_feature_93)
        local_features = [
            getattr(row, f"Local_feature_{i}") for i in range(1, 94)
        ]
        # Extract aggregate features (Aggregate_feature_1 ... Aggregate_feature_72)
        aggregate_features = [
            getattr(row, f"Aggregate_feature_{i}") for i in range(1, 73)
        ]

        features_obj = TransactionFeatures(
            txid=row.txId,
            local_features=local_features,
            aggregate_features=aggregate_features,
        )

        # Handle label mapping. Unknown is typically 3 in the dataset.
        # Use pandas isna to safely handle missing values.
        label_val = getattr(row, "label", None)
        label = None if pd.isna(label_val) else int(row.label)

        transaction_obj = Transaction(
            txid=row.txId,
            time_step=row.time_step,
            label=label,
            total_btc=getattr(row, "total_BTC", None),
            fees=getattr(row, "fees", None),
            size=getattr(row, "size", None),
            num_input_addresses=getattr(row, "num_input_addresses", None),
            num_output_addresses=getattr(row, "num_output_addresses", None),
            in_txs_degree=getattr(row, "in_txs_degree", None),
            out_txs_degree=getattr(row, "out_txs_degree", None),
            in_btc_min=getattr(row, "in_BTC_min", None),
            in_btc_max=getattr(row, "in_BTC_max", None),
            in_btc_mean=getattr(row, "in_BTC_mean", None),
            in_btc_median=getattr(row, "in_BTC_median", None),
            in_btc_total=getattr(row, "in_BTC_total", None),
            out_btc_min=getattr(row, "out_BTC_min", None),
            out_btc_max=getattr(row, "out_BTC_max", None),
            out_btc_mean=getattr(row, "out_BTC_mean", None),
            out_btc_median=getattr(row, "out_BTC_median", None),
            out_btc_total=getattr(row, "out_BTC_total", None),
        )

        yield transaction_obj, features_obj
