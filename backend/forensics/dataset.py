"""Columnar dataset for raw Bitcoin transaction + network metadata.

The problem statement schema is one row per observed transaction::

    timestamp, src_ip, dst_ip, src_port, dst_port, txid,
    input_addresses[], output_addresses[], input_amounts[], output_amounts[],
    fee, script_type, geo_country, asn

Analysis at scale cannot carry Python lists per row, so ``RawDataset`` holds
the same information flattened into integer-indexed pandas tables:

* ``tx``       one row per transaction, sorted by first-seen time
* ``inputs``   one row per (tx, spent address)
* ``outputs``  one row per (tx, paid address)
* ``obs``      one row per network observation (a txid can be seen many times)

Addresses are interned to ``int32`` ids; ``addresses[id]`` gives the string.
Amounts are BTC (floats).  Timestamps are int64 nanoseconds since epoch (UTC).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from itertools import chain

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NS_PER_S = 1_000_000_000

CANONICAL_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "txid",
    "input_addresses",
    "output_addresses",
    "input_amounts",
    "output_amounts",
    "fee",
    "script_type",
    "geo_country",
    "asn",
)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "txid",
    "input_addresses",
    "output_addresses",
    "input_amounts",
    "output_amounts",
)


@dataclass
class GroundTruth:
    """Planted labels, present only for synthetic data.

    Attributes:
        addr_entity: true owning entity id for every address id.
        entities: DataFrame[entity_id, is_illicit, scenario, role].
        tx_scenario: scenario label for every tx row (``"normal"`` if none).
    """

    addr_entity: np.ndarray
    entities: pd.DataFrame
    tx_scenario: np.ndarray


@dataclass
class IngestReport:
    """What ingestion accepted, repaired, and dropped (for the audit trail)."""

    source: str = ""
    rows_read: int = 0
    rows_dropped: dict[str, int] = field(default_factory=dict)
    duplicate_txids_merged: int = 0
    amount_unit: str = "btc"
    columns_renamed: dict[str, str] = field(default_factory=dict)
    geo_resolved_rows: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def rows_kept(self) -> int:
        return self.rows_read - sum(self.rows_dropped.values())

    def drop(self, reason: str, n: int) -> None:
        if n:
            self.rows_dropped[reason] = self.rows_dropped.get(reason, 0) + int(n)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "rows_read": self.rows_read,
            "rows_kept": self.rows_kept,
            "rows_dropped": dict(self.rows_dropped),
            "duplicate_txids_merged": self.duplicate_txids_merged,
            "amount_unit": self.amount_unit,
            "columns_renamed": dict(self.columns_renamed),
            "geo_resolved_rows": self.geo_resolved_rows,
            "notes": list(self.notes),
        }


@dataclass
class RawDataset:
    """Flattened, integer-indexed view of a raw transaction dataset."""

    addresses: np.ndarray
    tx: pd.DataFrame
    inputs: pd.DataFrame
    outputs: pd.DataFrame
    obs: pd.DataFrame
    truth: GroundTruth | None = None
    report: IngestReport | None = None

    @property
    def n_tx(self) -> int:
        return len(self.tx)

    @property
    def n_addresses(self) -> int:
        return len(self.addresses)

    def summary(self) -> dict:
        ts = self.tx["ts"].to_numpy()
        return {
            "transactions": int(self.n_tx),
            "addresses": int(self.n_addresses),
            "observations": int(len(self.obs)),
            "first_seen": pd.Timestamp(ts.min(), tz="UTC").isoformat() if len(ts) else None,
            "last_seen": pd.Timestamp(ts.max(), tz="UTC").isoformat() if len(ts) else None,
            "has_ground_truth": self.truth is not None,
        }

    # ------------------------------------------------------------------
    # Export back to the problem-statement schema
    # ------------------------------------------------------------------
    def to_frame(self, limit: int | None = None) -> pd.DataFrame:
        """Rebuild the list-column DataFrame (one row per observation)."""
        n = self.n_tx if limit is None else min(limit, self.n_tx)
        addr = self.addresses
        ins = self.inputs[self.inputs["tx_idx"] < n]
        outs = self.outputs[self.outputs["tx_idx"] < n]

        def lists(frame: pd.DataFrame, col: str, conv) -> dict[int, list]:
            res: dict[int, list] = {}
            for t, v in zip(frame["tx_idx"].to_numpy(), frame[col].to_numpy(), strict=True):
                res.setdefault(int(t), []).append(conv(v))
            return res

        in_a = lists(ins, "addr", lambda a: str(addr[a]))
        in_v = lists(ins, "amount", float)
        out_a = lists(outs, "addr", lambda a: str(addr[a]))
        out_v = lists(outs, "amount", float)

        o = self.obs[self.obs["tx_idx"] < n]
        t = self.tx
        idx = o["tx_idx"].to_numpy()
        return pd.DataFrame(
            {
                "timestamp": pd.to_datetime(o["ts"].to_numpy(), utc=True).strftime(
                    "%Y-%m-%dT%H:%M:%S.%fZ"
                ),
                "src_ip": o["src_ip"].to_numpy(),
                "dst_ip": o["dst_ip"].to_numpy(),
                "src_port": o["src_port"].to_numpy(),
                "dst_port": o["dst_port"].to_numpy(),
                "txid": t["txid"].to_numpy()[idx],
                "input_addresses": [in_a.get(int(i), []) for i in idx],
                "output_addresses": [out_a.get(int(i), []) for i in idx],
                "input_amounts": [in_v.get(int(i), []) for i in idx],
                "output_amounts": [out_v.get(int(i), []) for i in idx],
                "fee": t["fee"].to_numpy()[idx],
                "script_type": t["script_type"].to_numpy()[idx],
                "geo_country": o["country"].to_numpy(),
                "asn": o["asn"].to_numpy(),
            }
        )

    def write_csv(self, path, limit: int | None = None) -> None:
        """CSV export; list columns are JSON strings."""
        df = self.to_frame(limit)
        for c in ("input_addresses", "output_addresses", "input_amounts", "output_amounts"):
            df[c] = df[c].map(json.dumps)
        df.to_csv(path, index=False)

    def write_json(self, path, limit: int | None = None) -> None:
        df = self.to_frame(limit)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(df.to_dict(orient="records"), f, default=_json_default)

    def write_xml(self, path, limit: int | None = None) -> None:
        from xml.sax.saxutils import escape

        df = self.to_frame(limit)
        with open(path, "w", encoding="utf-8") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n<transactions>\n')
            for rec in df.to_dict(orient="records"):
                f.write("  <transaction>\n")
                for k, v in rec.items():
                    if isinstance(v, list):
                        inner = "".join(f"<item>{escape(str(x))}</item>" for x in v)
                        f.write(f"    <{k}>{inner}</{k}>\n")
                    else:
                        f.write(f"    <{k}>{escape(str(v))}</{k}>\n")
                f.write("  </transaction>\n")
            f.write("</transactions>\n")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ======================================================================
# Building from a list-column frame
# ======================================================================


def dataset_from_frame(df: pd.DataFrame, report: IngestReport | None = None) -> RawDataset:
    """Flatten a normalised list-column frame into a ``RawDataset``.

    ``df`` must already have the canonical columns with list-valued address
    and amount cells and ``timestamp`` as tz-aware datetimes (see
    ``ingest.normalize_frame``).  Rows are grouped by ``txid``; each row is
    kept as a network observation, the chain fields come from the earliest.
    """
    report = report or IngestReport()
    df = df.copy()
    df["_ts"] = _to_ns(df["timestamp"])

    df = df.sort_values("_ts", kind="stable").reset_index(drop=True)
    first = ~df["txid"].duplicated(keep="first")
    report.duplicate_txids_merged = int((~first).sum())

    tx_rows = df[first].reset_index(drop=True)
    tx_index = pd.Series(np.arange(len(tx_rows)), index=tx_rows["txid"].to_numpy())
    obs_tx_idx = tx_index.reindex(df["txid"].to_numpy()).to_numpy()

    # --- flatten lists -------------------------------------------------
    n_in = tx_rows["input_addresses"].map(len).to_numpy()
    n_out = tx_rows["output_addresses"].map(len).to_numpy()
    in_addr_str = list(chain.from_iterable(tx_rows["input_addresses"]))
    out_addr_str = list(chain.from_iterable(tx_rows["output_addresses"]))
    in_amt = np.fromiter(chain.from_iterable(tx_rows["input_amounts"]), dtype=np.float64)
    out_amt = np.fromiter(chain.from_iterable(tx_rows["output_amounts"]), dtype=np.float64)

    codes, uniques = pd.factorize(np.array(in_addr_str + out_addr_str, dtype=object))
    n_in_total = len(in_addr_str)
    in_codes = codes[:n_in_total].astype(np.int32)
    out_codes = codes[n_in_total:].astype(np.int32)

    tx_ids = np.arange(len(tx_rows), dtype=np.int64)
    inputs = pd.DataFrame(
        {"tx_idx": np.repeat(tx_ids, n_in).astype(np.int64), "addr": in_codes, "amount": in_amt}
    )
    out_pos = np.concatenate([np.arange(k) for k in n_out]) if n_out.sum() else np.array([], int)
    outputs = pd.DataFrame(
        {
            "tx_idx": np.repeat(tx_ids, n_out).astype(np.int64),
            "addr": out_codes,
            "amount": out_amt,
            "pos": out_pos.astype(np.int32),
        }
    )

    fee = pd.to_numeric(tx_rows.get("fee"), errors="coerce").to_numpy(dtype=np.float64)
    calc_fee = (
        inputs.groupby("tx_idx")["amount"].sum().reindex(tx_ids, fill_value=0.0).to_numpy()
        - outputs.groupby("tx_idx")["amount"].sum().reindex(tx_ids, fill_value=0.0).to_numpy()
    )
    coinbase = n_in == 0
    fee = np.where(np.isnan(fee), np.where(coinbase, 0.0, np.maximum(calc_fee, 0.0)), fee)

    tx = pd.DataFrame(
        {
            "txid": tx_rows["txid"].to_numpy(),
            "ts": tx_rows["_ts"].to_numpy(dtype=np.int64),
            "fee": fee,
            "script_type": tx_rows["script_type"].fillna("UNKNOWN").to_numpy(),
            "n_in": n_in.astype(np.int32),
            "n_out": n_out.astype(np.int32),
        }
    )

    obs = pd.DataFrame(
        {
            "tx_idx": obs_tx_idx.astype(np.int64),
            "ts": df["_ts"].to_numpy(dtype=np.int64),
            "src_ip": df["src_ip"].fillna("").to_numpy(),
            "dst_ip": df["dst_ip"].fillna("").to_numpy(),
            "src_port": pd.to_numeric(df["src_port"], errors="coerce")
            .fillna(0)
            .astype(np.int32)
            .to_numpy(),
            "dst_port": pd.to_numeric(df["dst_port"], errors="coerce")
            .fillna(0)
            .astype(np.int32)
            .to_numpy(),
            "country": df["geo_country"].fillna("").to_numpy(),
            "asn": pd.to_numeric(df["asn"], errors="coerce").fillna(0).astype(np.int64).to_numpy(),
        }
    )

    ds = RawDataset(
        addresses=np.asarray(uniques, dtype=object),
        tx=tx,
        inputs=inputs,
        outputs=outputs,
        obs=obs,
        report=report,
    )
    logger.info("Built RawDataset: %s", ds.summary())
    return ds


def _to_ns(ts: pd.Series) -> np.ndarray:
    """tz-aware datetime Series -> int64 nanoseconds (robust to us/ms units)."""
    idx = pd.DatetimeIndex(pd.to_datetime(ts, utc=True))
    return idx.as_unit("ns").asi8
