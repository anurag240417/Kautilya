"""Graph edgelist normalization.

Converts parsed Elliptic++ edgelists into canonical
Pydantic domain objects: ``TxTxEdge``, ``AddrTxEdge``,
``TxAddrEdge``, and ``AddrAddrEdge``.

Forces ``is_synthetic=False`` on all records because
the native Elliptic++ data is not synthetic. Sets
``provenance='elliptic_pp'`` and ``confidence=1.0``
to explicitly identify the data source.
"""

from collections.abc import Iterator

import pandas as pd

from backend.domain.graph import (
    AddrAddrEdge,
    AddrTxEdge,
    TxAddrEdge,
    TxTxEdge,
)


def normalize_tx_tx_edges(df: pd.DataFrame) -> Iterator[TxTxEdge]:
    """Normalize txs_edgelist DataFrames.

    Args:
        df: DataFrame with columns 'txId1', 'txId2'.

    Yields:
        TxTxEdge objects for each row.
    """
    for row in df.itertuples(index=False):
        yield TxTxEdge(
            source_txid=row.txId1,
            target_txid=row.txId2,
            is_synthetic=False,
            confidence=1.0,
            provenance="elliptic_pp",
        )


def normalize_addr_tx_edges(df: pd.DataFrame) -> Iterator[AddrTxEdge]:
    """Normalize AddrTx_edgelist DataFrames.

    Args:
        df: DataFrame with columns 'input_address', 'txId'.

    Yields:
        AddrTxEdge objects for each row.
    """
    for row in df.itertuples(index=False):
        yield AddrTxEdge(
            input_address=row.input_address,
            txid=row.txId,
            is_synthetic=False,
            confidence=1.0,
            provenance="elliptic_pp",
        )


def normalize_tx_addr_edges(df: pd.DataFrame) -> Iterator[TxAddrEdge]:
    """Normalize TxAddr_edgelist DataFrames.

    Args:
        df: DataFrame with columns 'txId', 'output_address'.

    Yields:
        TxAddrEdge objects for each row.
    """
    for row in df.itertuples(index=False):
        yield TxAddrEdge(
            txid=row.txId,
            output_address=row.output_address,
            is_synthetic=False,
            confidence=1.0,
            provenance="elliptic_pp",
        )


def normalize_addr_addr_edges(df: pd.DataFrame) -> Iterator[AddrAddrEdge]:
    """Normalize AddrAddr_edgelist DataFrames.

    Args:
        df: DataFrame with columns 'input_address', 'output_address'.

    Yields:
        AddrAddrEdge objects for each row.
    """
    for row in df.itertuples(index=False):
        yield AddrAddrEdge(
            input_address=row.input_address,
            output_address=row.output_address,
            is_synthetic=False,
            confidence=1.0,
            provenance="elliptic_pp",
        )

