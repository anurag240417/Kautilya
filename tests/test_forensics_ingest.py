"""Ingestion: list-cell parsing, validation, formats, units."""

import json

import numpy as np
import pandas as pd
import pytest

from backend.forensics.ingest import (
    load_raw_transactions,
    normalize_frame,
    parse_list_cell,
    read_raw_file,
)
from backend.forensics.synth import SynthConfig, generate_dataset


@pytest.mark.parametrize(
    "value,numeric,expected",
    [
        ('["a","b"]', False, ["a", "b"]),
        ("a|b|c", False, ["a", "b", "c"]),
        ("a;b", False, ["a", "b"]),
        ("a, b", False, ["a", "b"]),
        ("[1.5, 2]", True, [1.5, 2.0]),
        ("0.1|0.2", True, [0.1, 0.2]),
        ("", False, []),
        (None, False, []),
        (float("nan"), True, []),
        (["x", "y"], False, ["x", "y"]),
        ("solo", False, ["solo"]),
    ],
)
def test_parse_list_cell(value, numeric, expected):
    assert parse_list_cell(value, numeric) == expected


def test_parse_list_cell_bad_number_raises():
    with pytest.raises(ValueError):
        parse_list_cell("a|b", numeric=True)


def _raw(**over):
    base = {
        "timestamp": ["2025-01-01T00:00:00Z"],
        "txid": ["t1"],
        "input_addresses": ['["a"]'],
        "output_addresses": ['["b","c"]'],
        "input_amounts": ["[1.0]"],
        "output_amounts": ["[0.6,0.39]"],
    }
    base.update(over)
    return pd.DataFrame(base)


def test_normalize_aliases_and_defaults():
    raw = _raw().rename(columns={"txid": "tx_hash", "input_addresses": "inputs"})
    from backend.forensics.dataset import IngestReport

    rep = IngestReport()
    df = normalize_frame(raw, rep)
    assert {"txid", "input_addresses"} <= set(df.columns)
    assert rep.columns_renamed["tx_hash"] == "txid"
    assert len(df) == 1 and df["input_addresses"][0] == ["a"]


def test_missing_required_column_raises():
    with pytest.raises(ValueError, match="Missing required columns"):
        normalize_frame(_raw().drop(columns=["output_amounts"]))


def test_invalid_rows_dropped_and_counted():
    from backend.forensics.dataset import IngestReport

    raw = pd.concat(
        [
            _raw(),
            _raw(txid=["t2"], timestamp=["not a date"]),
            _raw(txid=[""]),
            _raw(txid=["t4"], output_amounts=["[0.6]"]),  # length mismatch
            _raw(txid=["t5"], output_amounts=["[-1, 2]"]),  # negative
            _raw(txid=["t6"], output_addresses=[""], output_amounts=[""]),  # no outputs
            _raw(txid=["t7"], input_amounts=["[x]"]),  # malformed
        ],
        ignore_index=True,
    )
    rep = IngestReport()
    df = normalize_frame(raw, rep)
    assert list(df["txid"]) == ["t1"]
    assert rep.rows_dropped == {
        "unparseable_timestamp": 1,
        "missing_txid": 1,
        "malformed_list_cell": 1,
        "address_amount_length_mismatch": 1,
        "no_outputs": 1,
        "negative_amount": 1,
    }


def test_satoshi_amounts_autodetected():
    raw = _raw(
        input_amounts=["[100000000]"], output_amounts=["[60000000, 39990000]"], fee=["10000"]
    )
    df = normalize_frame(raw)
    assert df["output_amounts"][0] == pytest.approx([0.6, 0.3999])
    assert df["fee"][0] == pytest.approx(0.0001)


def test_duplicate_txid_merged_into_observations(tmp_path):
    p = tmp_path / "d.csv"
    pd.concat([_raw(), _raw(timestamp=["2025-01-01T00:00:01Z"])]).to_csv(p, index=False)
    ds = load_raw_transactions(p, resolve_geo=False)
    assert ds.n_tx == 1 and len(ds.obs) == 2 and ds.report.duplicate_txids_merged == 1


def test_unsupported_and_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_raw_file(tmp_path / "nope.csv")
    bad = tmp_path / "x.parquet"
    bad.write_text("x")
    with pytest.raises(ValueError, match="Unsupported"):
        read_raw_file(bad)


@pytest.mark.parametrize("fmt", ["csv", "json", "xml"])
def test_roundtrip_all_formats(tmp_path, fmt):
    ds = generate_dataset(SynthConfig(n_tx=1500, seed=3))
    path = tmp_path / f"d.{fmt}"
    getattr(ds, f"write_{fmt}")(path, limit=400)
    back = load_raw_transactions(path)
    assert back.n_tx == 400
    # Same transactions (order may differ by sub-second first-seen ties), same structure and amounts.
    assert set(back.tx["txid"]) == set(ds.tx["txid"][:400])
    orig = ds.tx.iloc[:400].set_index("txid")
    got = back.tx.set_index("txid")
    assert (got.loc[orig.index, "n_in"].to_numpy() == orig["n_in"].to_numpy()).all()
    assert (got.loc[orig.index, "n_out"].to_numpy() == orig["n_out"].to_numpy()).all()
    a = np.sort(ds.outputs[ds.outputs["tx_idx"] < 400]["amount"].to_numpy())
    b = np.sort(back.outputs["amount"].to_numpy())
    assert np.allclose(a, b, atol=1e-9)
    assert back.report.rows_dropped == {}


def test_json_wrapped_object(tmp_path):
    p = tmp_path / "w.json"
    rec = {
        "timestamp": "2025-01-01T00:00:00Z",
        "txid": "a",
        "input_addresses": ["x"],
        "output_addresses": ["y"],
        "input_amounts": [1.0],
        "output_amounts": [0.9],
    }
    p.write_text(json.dumps({"transactions": [rec]}))
    assert load_raw_transactions(p, resolve_geo=False).n_tx == 1
