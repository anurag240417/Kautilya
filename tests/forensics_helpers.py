"""Shared builders for forensics tests."""

from __future__ import annotations

import pandas as pd

from backend.forensics.dataset import dataset_from_frame

T0 = pd.Timestamp("2025-03-01T00:00:00Z")


def make_ds(txs: list[dict]):
    """Build a RawDataset from compact tx dicts.

    Each dict: ``t`` (seconds after T0), ``ins`` [(addr, amt)], ``outs`` [(addr, amt)],
    optional ``ip``, ``txid``.
    """
    rows = []
    for i, tx in enumerate(txs):
        rows.append(
            {
                "timestamp": T0 + pd.Timedelta(seconds=tx["t"]),
                "src_ip": tx.get("ip", "8.8.8.8"),
                "dst_ip": "9.9.9.9",
                "src_port": 40000,
                "dst_port": 8333,
                "txid": tx.get("txid", f"tx{i:04d}"),
                "input_addresses": [a for a, _ in tx["ins"]],
                "output_addresses": [a for a, _ in tx["outs"]],
                "input_amounts": [v for _, v in tx["ins"]],
                "output_amounts": [v for _, v in tx["outs"]],
                "fee": tx.get("fee", 0.0001),
                "script_type": "P2WPKH",
                "geo_country": "US",
                "asn": 15169,
            }
        )
    return dataset_from_frame(pd.DataFrame(rows))
