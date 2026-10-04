"""Pipeline, model, evidence, graph view, cases and report."""

import numpy as np
import pytest

from backend.forensics.cases import (
    CaseStore,
    dataset_fingerprint,
    simulate_seed_labels,
    uncertain_entities,
)
from backend.forensics.evidence import build_entity_report
from backend.forensics.geo import (
    GeoIPDatabase,
    flag_hosting_vpn,
    flag_tor,
    ip_to_int,
    tor_exit_ips,
)
from backend.forensics.graphview import entity_subgraph, money_trail
from backend.forensics.pipeline import ForensicsConfig, run_forensics
from backend.forensics.report import render_report_html, verify_report
from backend.forensics.synth import SynthConfig, generate_dataset


@pytest.fixture(scope="module")
def ds():
    return generate_dataset(
        SynthConfig(n_tx=8000, seed=21, n_ransomware=4, n_darknet=2, n_layering=6, n_dust=3)
    )


@pytest.fixture(scope="module")
def res(ds):
    return run_forensics(ds, ForensicsConfig(n_estimators=80, seed=21))


# ---------------------------------------------------------------- data
def test_synthetic_conserves_value_and_has_ground_truth(ds):
    inp = ds.inputs.groupby("tx_idx")["amount"].sum().reindex(range(ds.n_tx), fill_value=0)
    out = ds.outputs.groupby("tx_idx")["amount"].sum().reindex(range(ds.n_tx), fill_value=0)
    spend = ds.tx["n_in"].to_numpy() > 0
    assert np.allclose((inp - out - ds.tx["fee"]).to_numpy()[spend], 0, atol=1e-9)
    assert (ds.tx["fee"] >= 0).all()
    assert ds.truth is not None and ds.truth.entities["is_illicit"].any()
    assert ds.tx["ts"].is_monotonic_increasing


def test_synthetic_is_deterministic():
    a = generate_dataset(SynthConfig(n_tx=1500, seed=5))
    b = generate_dataset(SynthConfig(n_tx=1500, seed=5))
    assert list(a.tx["txid"]) == list(b.tx["txid"])
    assert dataset_fingerprint(a) == dataset_fingerprint(b)
    assert dataset_fingerprint(a) != dataset_fingerprint(
        generate_dataset(SynthConfig(n_tx=1500, seed=6))
    )


# ---------------------------------------------------------------- pipeline
def test_top_alerts_are_mostly_illicit(res):
    top = res.alert_table(25)
    truth = res.labels.loc[top.index, "is_illicit"]
    assert truth.mean() >= 0.8


def test_fused_ranking_beats_chance_by_far(res):
    r = res.metrics["ranking"]["fused_score"]
    assert r["pr_auc"] > 0.7 and r["pr_auc"] > 10 * r["base_rate"]


def test_scores_are_cross_fitted(res):
    # every active entity is scored by a model that did not train on its fold
    assert len(res.models) == res.cfg.n_folds
    assert set(np.unique(res.entity_fold)) <= set(range(res.cfg.n_folds))


def test_heuristics_validated_against_truth(res):
    hv = res.metrics["heuristic_validation"]
    assert hv["coinjoin"]["precision"] == 1.0
    assert hv["change_detection"]["precision"] > 0.9


def test_clustering_is_pure(res):
    assert res.metrics["clustering"]["address_weighted_purity"] > 0.99


def test_rank_is_dense_and_ordered(res):
    t = res.scores[res.scores["rank"] > 0].sort_values("rank")
    assert t["rank"].tolist() == list(range(1, len(t) + 1))
    assert t["score"].is_monotonic_decreasing


# ---------------------------------------------------------------- evidence
def test_entity_report_contents(res):
    eid = int(res.alert_table(1).index[0])
    r = build_entity_report(res, eid)
    lo, hi = r["score"]["interval90"]
    assert 0 <= lo <= r["score"]["illicit_probability"] <= hi <= 1
    assert r["heuristic_evidence"] and all(
        h["type"] in ("observation", "inference") for h in r["heuristic_evidence"]
    )
    assert r["model_contributions"] and "contribution" in r["model_contributions"][0]
    assert "not proof" in r["summary"]
    assert r["provenance"]["synthetic_data"] is True
    assert r["key_transactions"]
    link = r["network_evidence"].get("wallet_ip_link")
    if link:
        assert link["ci95"][0] <= link["share"] <= link["ci95"][1]
        assert "not attribution" in link["interpretation"]


def test_explanations_are_occlusion_deltas(res):
    eid = int(res.alert_table(1).index[0])
    m = res.model_for(eid)
    feats = res.X.loc[[eid]]
    contrib = m.explain(feats, top_k=50)[0]
    assert contrib
    # For a single feature, replacing it with its median must change p by exactly its contribution.
    singles = [c for c in contrib if c["feature"] in m.medians.index]
    if singles:
        top = max(singles, key=lambda c: abs(c["contribution"]))
        alt = feats.copy()
        alt[top["feature"]] = m.medians[top["feature"]]
        assert m.predict_proba(feats)[0] - m.predict_proba(alt)[0] == pytest.approx(
            top["contribution"], abs=1e-9
        )


def test_top_leads_always_have_an_explanation(res):
    """Clear-cut leads have overlapping signals; single-feature occlusion alone returns nothing."""
    ids = [int(i) for i in res.alert_table(10).index]
    for eid in ids:
        contrib = res.model_for(eid).explain(res.X.loc[[eid]])[0]
        assert contrib, f"E-{eid} has no model explanation"
        assert any(c["contribution"] > 0 for c in contrib)
    groups = [c for c in contrib if c["feature"].startswith("group:")]
    assert all(c["value"] is None and c["description"] for c in groups)


# ---------------------------------------------------------------- graph view
def test_subgraph_and_trail(res):
    eid = int(res.alert_table(1).index[0])
    g = entity_subgraph(res, eid, radius=2, max_nodes=40)
    ids = {n["id"] for n in g["nodes"]}
    assert eid in ids and len(ids) <= 40
    assert all(e["src"] in ids and e["dst"] in ids for e in g["edges"])
    ts = [ev["ts"] for ev in g["events"]]
    assert ts == sorted(ts)
    trail = money_trail(res, eid, "forward")
    if trail:
        times = [s["ts"] for s in trail[1:]]
        assert times == sorted(times)  # strictly follows time


# ---------------------------------------------------------------- cases & feedback
def test_feedback_latest_wins_and_unsure_clears():
    s = CaseStore(":memory:")
    fp = "abc"
    s.add_feedback(fp, 1, "confirmed")
    s.add_feedback(fp, 1, "false_positive")
    s.add_feedback(fp, 2, "confirmed")
    s.add_feedback(fp, 2, "unsure")
    s.add_feedback(fp, 3, "confirmed")
    lab = s.feedback_labels(fp)
    assert lab.to_dict() == {1: 0.0, 3: 1.0}
    assert s.feedback_labels("other").empty
    with pytest.raises(ValueError):
        s.add_feedback(fp, 1, "maybe")


def test_cases_roundtrip():
    s = CaseStore(":memory:")
    c = s.create_case("fp", "Ransom wave", [5, 9], "analyst", "n")
    assert c["entity_ids"] == [5, 9] and c["status"] == "open"
    assert s.update_case(c["id"], status="closed")["status"] == "closed"
    assert len(s.list_cases("fp")) == 1 and s.list_cases("zz") == []


def test_active_learning_queue_skips_labelled(res):
    q = uncertain_entities(res, set(), k=5)
    assert len(q) == 5 and (q["uncertainty"].diff().dropna() <= 1e-12).all()
    done = {int(q.index[0])}
    assert int(q.index[0]) not in set(uncertain_entities(res, done, k=5).index)


def test_retrain_with_labels_scores_unseen_entities(ds, res):
    seed = simulate_seed_labels(res.labels, res.scores["active"].to_numpy(), 0.2, 1)
    new = run_forensics(ds, ForensicsConfig(n_estimators=60, seed=21), labels=seed)
    assert new.label_source == "analyst_labels"
    assert "ranking_unlabelled" in new.metrics
    # model never saw the unlabelled entities, yet still ranks illicit ones high
    assert new.metrics["ranking_unlabelled"]["fused_score"]["pr_auc"] > 0.5


# ---------------------------------------------------------------- report
def test_report_chain_of_custody_and_tamper_detection(res):
    ids = [int(i) for i in res.alert_table(2).index]
    html = render_report_html(res, ids, "Test report", "tester", "some notes")
    ok, recorded, recomputed = verify_report(html)
    assert ok and recorded == recomputed and len(recorded) == 64
    assert "Synthetic data" in html and "not proof" in html
    tampered = html.replace('"final":', '"final":99', 1) if '"final":' in html else html + "x"
    assert not verify_report(tampered)[0]


def test_report_escapes_user_text(res):
    eid = int(res.alert_table(1).index[0])
    html = render_report_html(
        res, [eid], "<script>alert(1)</script>", "<b>x</b>", "</script><img src=x>"
    )
    assert "<script>alert(1)</script>" not in html.split('id="evidence"')[0]
    assert "<img src=x>" not in html.split('id="evidence"')[0]
    assert verify_report(html)[0]


# ---------------------------------------------------------------- geo
def test_ip_to_int_and_flags():
    assert ip_to_int(["1.0.0.0", "255.255.255.255", "bad", "1.2.3", "::1", ""]).tolist() == [
        16777216,
        4294967295,
        -1,
        -1,
        -1,
        -1,
    ]
    tor = sorted(tor_exit_ips())[0]
    assert flag_tor([tor, "8.8.8.8"]).tolist() == [True, False]
    assert flag_hosting_vpn(np.array([9009, 15169])).tolist() == [True, False]


def test_geoip_uses_real_database_files_when_present(tmp_path):
    (tmp_path / "dbip-country-lite.csv").write_text(
        '"8.8.8.0","8.8.8.255","ZZ"\n"9.0.0.0","9.0.0.255","YY"\n'
    )
    (tmp_path / "asn.csv").write_text(
        "network,autonomous_system_number,autonomous_system_organization\n8.8.8.0/24,64500,Test\n"
    )
    db = GeoIPDatabase(tmp_path)
    country, asn = db.lookup_many(["8.8.8.8", "9.0.0.7", "3.3.3.3"])
    assert country[0] == "ZZ" and country[1] == "YY"
    assert asn[0] == 64500
    assert country[2] == "US"  # falls back to builtin table outside the DB ranges
    assert "dbip" in db.source


def test_geoip_corrupt_file_falls_back(tmp_path):
    (tmp_path / "dbip-country-lite.csv").write_text("garbage,,\n,,,\n\x00")
    db = GeoIPDatabase(tmp_path)
    assert db.lookup_many(["8.8.8.8"])[0][0] != ""


def test_unlabelled_data_uses_transfer_model():
    d = generate_dataset(SynthConfig(n_tx=3000, seed=2))
    d.truth = None  # pretend real data without labels
    r = run_forensics(d, ForensicsConfig(n_estimators=40, train_synth_n_tx=6000, seed=2))
    assert r.label_source == "transfer_from_synthetic"
    assert len(r.models) == 1 and (r.scores["rank"] > 0).any()
    assert r.labels is None and "ranking" not in r.metrics
