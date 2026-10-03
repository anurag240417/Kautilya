"""Ingest raw transaction/network metadata from CSV, JSON, JSONL or XML.

Real-world exports vary, so ingestion is deliberately tolerant about *form*
(column names, list encodings, timestamp formats, satoshi vs BTC) and strict
about *content* (mismatched address/amount lengths, missing txids, bad
timestamps and negative amounts are dropped and counted in ``IngestReport``).
"""

from __future__ import annotations

import ast
import json
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from backend.forensics.dataset import (
    CANONICAL_COLUMNS,
    REQUIRED_COLUMNS,
    IngestReport,
    RawDataset,
    dataset_from_frame,
)

logger = logging.getLogger(__name__)

COLUMN_ALIASES: dict[str, str] = {
    "time": "timestamp",
    "ts": "timestamp",
    "datetime": "timestamp",
    "block_time": "timestamp",
    "first_seen": "timestamp",
    "tx_id": "txid",
    "tx_hash": "txid",
    "txhash": "txid",
    "hash": "txid",
    "source_ip": "src_ip",
    "src_address": "src_ip",
    "destination_ip": "dst_ip",
    "dest_ip": "dst_ip",
    "source_port": "src_port",
    "destination_port": "dst_port",
    "dest_port": "dst_port",
    "inputs": "input_addresses",
    "input_addrs": "input_addresses",
    "input_wallets": "input_addresses",
    "outputs": "output_addresses",
    "output_addrs": "output_addresses",
    "output_wallets": "output_addresses",
    "input_values": "input_amounts",
    "output_values": "output_amounts",
    "fee_btc": "fee",
    "fees": "fee",
    "script": "script_type",
    "country": "geo_country",
    "geo": "geo_country",
    "as_number": "asn",
}

_LIST_SPLIT = re.compile(r"[|;,\s]+")
SATOSHI_THRESHOLD = 1e6  # median output amount above this looks like satoshis


# ======================================================================
# Cell parsing
# ======================================================================


def parse_list_cell(value, numeric: bool = False) -> list:
    """Parse one list-valued cell into a Python list.

    Accepts real lists, JSON strings (``'["a","b"]'``), Python-literal strings,
    and delimiter-separated strings (``a|b|c``, ``a;b``, ``a,b``).  Missing
    values become ``[]``.  Numeric cells that fail to parse raise ``ValueError``.
    """
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        items = list(value)
    elif isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if s[0] == "[":
            try:
                items = json.loads(s)
            except json.JSONDecodeError:
                items = ast.literal_eval(s)
        else:
            items = [p for p in _LIST_SPLIT.split(s) if p]
    else:
        items = [value]
    if numeric:
        return [float(x) for x in items]
    return [str(x).strip() for x in items]


def _parse_timestamp(series: pd.Series) -> pd.Series:
    """Parse ISO strings or epoch seconds/milliseconds into tz-aware UTC."""
    if pd.api.types.is_numeric_dtype(series):
        med = series.dropna().abs().median() if series.notna().any() else 0
        unit = "ms" if med > 1e11 else "s"
        return pd.to_datetime(series, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce", format="mixed")


# ======================================================================
# File readers
# ======================================================================


def read_raw_file(path: Path | str) -> pd.DataFrame:
    """Read CSV / JSON / JSONL / XML into a raw (un-normalised) DataFrame."""
    path = Path(path)
    if not path.exists():
        msg = f"Input file not found: {path}"
        raise FileNotFoundError(msg)
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv", ".txt"):
        return pd.read_csv(
            path, sep="\t" if suffix == ".tsv" else ",", dtype=str, keep_default_na=False
        )
    if suffix == ".jsonl" or suffix == ".ndjson":
        return pd.read_json(path, lines=True, dtype=False)
    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for key in ("transactions", "records", "data", "items"):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
            else:
                data = [data]
        return pd.DataFrame(data)
    if suffix == ".xml":
        return _read_xml(path)
    msg = f"Unsupported input format '{suffix}' (expected .csv, .json, .jsonl, .xml)"
    raise ValueError(msg)


def _read_xml(path: Path) -> pd.DataFrame:
    """Read records whose list fields are nested elements (``<item>``) or delimited text."""
    root = ET.parse(path).getroot()
    # Records are the repeated children of the root element.
    records = list(root)
    rows = []
    for rec in records:
        row: dict = {}
        for child in rec:
            sub = list(child)
            row[child.tag] = (
                [(s.text or "").strip() for s in sub] if sub else (child.text or "").strip()
            )
        rows.append(row)
    return pd.DataFrame(rows)


# ======================================================================
# Normalisation
# ======================================================================


def normalize_frame(
    raw: pd.DataFrame,
    report: IngestReport | None = None,
    amount_unit: str = "auto",
) -> pd.DataFrame:
    """Rename columns, parse lists/timestamps, validate, and report drops.

    ``amount_unit``: ``"btc"``, ``"sat"`` or ``"auto"`` (satoshis if the median
    positive amount exceeds one million).
    """
    report = report or IngestReport()
    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    renames = {
        c: COLUMN_ALIASES[c]
        for c in df.columns
        if c in COLUMN_ALIASES and COLUMN_ALIASES[c] not in df.columns
    }
    if renames:
        df = df.rename(columns=renames)
        report.columns_renamed.update(renames)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        msg = f"Missing required columns: {missing}. Found: {sorted(df.columns)}"
        raise ValueError(msg)
    for c in CANONICAL_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan if c in ("fee", "src_port", "dst_port", "asn") else ""
            report.notes.append(f"column '{c}' absent; filled with empty values")

    report.rows_read += len(df)

    # Timestamps
    df["timestamp"] = _parse_timestamp(df["timestamp"])
    bad_ts = df["timestamp"].isna()
    report.drop("unparseable_timestamp", bad_ts.sum())
    df = df[~bad_ts]

    # Missing txid
    bad_id = df["txid"].isna() | (df["txid"].astype(str).str.strip() == "")
    report.drop("missing_txid", bad_id.sum())
    df = df[~bad_id].copy()
    df["txid"] = df["txid"].astype(str).str.strip()

    # Lists
    ok = np.ones(len(df), dtype=bool)
    parsed: dict[str, list] = {}
    for col, numeric in (
        ("input_addresses", False),
        ("output_addresses", False),
        ("input_amounts", True),
        ("output_amounts", True),
    ):
        out: list = []
        for i, v in enumerate(df[col].to_numpy()):
            try:
                out.append(parse_list_cell(v, numeric=numeric))
            except (ValueError, SyntaxError, json.JSONDecodeError):
                out.append([])
                ok[i] = False
        parsed[col] = out
    report.drop("malformed_list_cell", (~ok).sum())
    for col, vals in parsed.items():
        df[col] = vals

    len_ok = (df["input_addresses"].map(len) == df["input_amounts"].map(len)) & (
        df["output_addresses"].map(len) == df["output_amounts"].map(len)
    )
    report.drop("address_amount_length_mismatch", (ok & ~len_ok.to_numpy()).sum())
    ok &= len_ok.to_numpy()

    no_out = df["output_addresses"].map(len) == 0
    report.drop("no_outputs", (ok & no_out.to_numpy()).sum())
    ok &= ~no_out.to_numpy()

    neg = df["input_amounts"].map(lambda a: any(x < 0 for x in a)) | df["output_amounts"].map(
        lambda a: any(x < 0 for x in a)
    )
    report.drop("negative_amount", (ok & neg.to_numpy()).sum())
    ok &= ~neg.to_numpy()

    df = df[ok].copy()

    # Units
    unit = amount_unit
    if unit == "auto":
        sample = [x for a in df["output_amounts"].head(5000) for x in a if x > 0]
        unit = "sat" if sample and float(np.median(sample)) > SATOSHI_THRESHOLD else "btc"
    if unit == "sat":
        for col in ("input_amounts", "output_amounts"):
            df[col] = df[col].map(lambda a: [x / 1e8 for x in a])
        df["fee"] = pd.to_numeric(df["fee"], errors="coerce") / 1e8
    report.amount_unit = unit

    df["fee"] = pd.to_numeric(df["fee"], errors="coerce")
    df["script_type"] = df["script_type"].replace("", np.nan).str.upper()
    return df.reset_index(drop=True)


def load_raw_transactions(
    path: Path | str,
    amount_unit: str = "auto",
    resolve_geo: bool = True,
    geoip=None,
) -> RawDataset:
    """Full ingestion: file -> validated, flattened ``RawDataset``.

    If ``geo_country``/``asn`` are empty and ``resolve_geo`` is set, source IPs
    are resolved with the offline GeoIP resolver (``geoip`` or the default).
    """
    report = IngestReport(source=str(path))
    raw = read_raw_file(path)
    df = normalize_frame(raw, report, amount_unit)
    if len(df) == 0:
        msg = f"No valid rows after validation: {report.to_dict()}"
        raise ValueError(msg)
    ds = dataset_from_frame(df, report)
    if resolve_geo:
        from backend.forensics.geo import resolve_observation_geo

        report.geo_resolved_rows = resolve_observation_geo(ds, geoip)
    logger.info("Ingested %s: %s", path, report.to_dict())
    return ds
