"""Graph edge domain models.

Defines the four native Elliptic++ edgelist schemas plus the
``is_synthetic`` provenance flag.

Native Elliptic++ edges are real/public data (``is_synthetic=False``).
Synthetic network edges added by the generator will set
``is_synthetic=True``.

Field names match the actual CSV column headers:
    - txs_edgelist.csv:      txId1, txId2
    - AddrTx_edgelist.csv:   input_address, txId
    - TxAddr_edgelist.csv:   txId, output_address
    - AddrAddr_edgelist.csv: input_address, output_address
"""

from pydantic import BaseModel, Field


class TxTxEdge(BaseModel):
    """Transaction-to-transaction edge from ``txs_edgelist.csv``.

    Represents a money-flow relationship between two transactions.
    """

    source_txid: int = Field(description="Source transaction ID (txId1)")
    target_txid: int = Field(description="Target transaction ID (txId2)")
    is_synthetic: bool = Field(default=False)


class AddrTxEdge(BaseModel):
    """Address-to-transaction edge from ``AddrTx_edgelist.csv``.

    Represents an input address contributing to a transaction.
    """

    input_address: str = Field(description="Input address")
    txid: int = Field(description="Transaction ID")
    is_synthetic: bool = Field(default=False)


class TxAddrEdge(BaseModel):
    """Transaction-to-address edge from ``TxAddr_edgelist.csv``.

    Represents a transaction sending to an output address.
    """

    txid: int = Field(description="Transaction ID")
    output_address: str = Field(description="Output address")
    is_synthetic: bool = Field(default=False)


class AddrAddrEdge(BaseModel):
    """Address-to-address edge from ``AddrAddr_edgelist.csv``.

    Represents a direct address-to-address relationship derived
    from shared transaction participation.
    """

    input_address: str = Field(description="Input/source address")
    output_address: str = Field(description="Output/destination address")
    is_synthetic: bool = Field(default=False)
