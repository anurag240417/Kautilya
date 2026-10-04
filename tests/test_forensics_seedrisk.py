"""Seed-based risk propagation: algorithm properties and pipeline integration."""

import numpy as np
import pytest
from scipy import sparse

from backend.forensics.evidence import build_entity_report
from backend.forensics.pipeline import ForensicsConfig, run_forensics
from backend.forensics.seedrisk import propagate_seed_risk, seed_boost
from backend.forensics.synth import SynthConfig, generate_dataset


def _graph(edges, n):
    rows, cols, vals = [], [], []
    for a, b, w in edges:
        rows += [a, b]
        cols += [b, a]
        vals += [w, w]
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


def test_risk_decays_with_distance_along_a_chain():
    adj = _graph([(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1)], 6)  # node 5 is isolated
    sr = propagate_seed_risk(adj, np.array([0]), max_hops=4)
    assert sr.hops.tolist() == [0, 1, 2, 3, 4, -1]
    assert sr.risk[0] == 1.0 and sr.risk[5] == 0.0
    assert sr.risk[1] > sr.risk[2] > sr.risk[3] > sr.risk[4] > 0
    assert sr.nearest[:5].tolist() == [0] * 5 and sr.nearest[5] == -1
    assert ((sr.risk >= 0) & (sr.risk <= 1)).all()


def test_max_hops_limits_reach():
    adj = _graph([(0, 1, 1), (1, 2, 1), (2, 3, 1)], 4)
    sr = propagate_seed_risk(adj, np.array([0]), max_hops=2)
    assert sr.hops.tolist() == [0, 1, 2, -1] and sr.risk[3] == 0.0


def test_nearest_of_several_seeds_wins():
    adj = _graph([(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1)], 5)
    sr = propagate_seed_risk(adj, np.array([0, 4]), max_hops=4)
    assert sr.hops.tolist() == [0, 1, 2, 1, 0]
    assert sr.nearest[1] == 0 and sr.nearest[3] == 4
    assert sr.n_seeds == 2 and sr.is_seed.tolist() == [True, False, False, False, True]


def test_hub_absorbs_risk_but_does_not_pass_it_on():
    # seed 0 -> hub 1 -> customer 2
    adj = _graph([(0, 1, 5), (1, 2, 5)], 3)
    plain = propagate_seed_risk(adj, np.array([0]))
    assert plain.risk[2] > 0  # without the hub rule, the customer is tainted
    gated = propagate_seed_risk(adj, np.array([0]), absorbing=np.array([False, True, False]))
    assert gated.risk[2] == 0.0 and gated.hops[2] == -1
    assert gated.risk[1] == 0.0  # the hub itself is not a suspect...
    assert gated.hops[1] == 1  # ...but its distance to the seed is still reported


def test_a_seed_that_is_a_hub_still_spreads_risk():
    adj = _graph([(0, 1, 1)], 2)
    sr = propagate_seed_risk(adj, np.array([0]), absorbing=np.array([True, False]))
    assert sr.risk[0] == 1.0 and sr.risk[1] > 0


def test_no_seeds_or_empty_graph():
    adj = _graph([(0, 1, 1)], 3)
    none = propagate_seed_risk(adj, np.array([], dtype=int))
    assert none.n_seeds == 0 and not none.risk.any() and (none.hops == -1).all()
    empty = propagate_seed_risk(sparse.csr_matrix((3, 3)), np.array([1]))
    assert empty.risk[1] == 1.0 and empty.risk.sum() == 1.0


def test_out_of_range_seeds_ignored_and_deterministic():
    adj = _graph([(0, 1, 1)], 2)
    a = propagate_seed_risk(adj, np.array([0, 99, -3]))
    b = propagate_seed_risk(adj, np.array([0]))
    assert a.n_seeds == 1 and np.array_equal(a.risk, b.risk)


def test_boost_never_lowers_a_score():
    score = np.array([0.0, 30.0, 90.0, 100.0])
    risk = np.array([1.0, 0.0, 0.5, 1.0])
    out = seed_boost(score, risk, 0.5)
    assert (out >= score).all() and (out <= 100).all()
    assert out[1] == 30.0  # no proximity -> unchanged
    assert out[0] == pytest.approx(50.0)
    assert out[2] == pytest.approx(92.5)


# ---------------------------------------------------------------- integration
@pytest.fixture(scope="module")
def res():
    ds = generate_dataset(
        SynthConfig(n_tx=8000, seed=21, n_ransomware=4, n_darknet=2, n_layering=6, n_dust=3)
    )
    return run_forensics(ds, ForensicsConfig(n_estimators=60, seed=21))


def test_pipeline_uses_a_simulated_watch_list(res):
    s = res.scores
    assert res.seed is not None and res.seed.n_seeds == int(s["is_seed"].sum()) > 0
    # seeds are only ever drawn from entities that really are illicit
    assert res.labels.loc[s.index[s["is_seed"]], "is_illicit"].all()


def test_seed_boost_only_raises_scores_and_is_logged(res):
    s = res.scores
    assert (s["score"] >= s["score_base"] - 1e-9).all()
    boosted = s[s["score"] > s["score_base"]]
    assert len(boosted) > 0
    assert boosted["active_signals"].str.contains("seed_proximity").all()


def test_seeds_are_excluded_from_reported_accuracy(res):
    ranking = res.metrics["ranking"]
    assert "seed_propagation_only" in ranking and "fused_without_seed_boost" in ranking
    n_eval = ranking["fused_score"]["n"]
    active_labelled = int((res.scores["active"] & res.labels["is_illicit"].notna()).sum())
    assert n_eval == active_labelled - int((res.scores["is_seed"] & res.scores["active"]).sum())


def test_entities_near_seeds_are_enriched_for_illicit(res):
    seeds = res.metrics["seeds"]
    assert seeds["illicit_share_within_2_hops"] > 3 * seeds["illicit_share_overall"]


def test_hubs_are_never_given_seed_risk(res):
    s = res.scores
    hubs = s[s["is_service"] & ~s["is_seed"]]
    assert (hubs["seed_risk"] == 0).all()


def test_evidence_reports_seed_proximity(res):
    near = res.scores[
        (res.scores["seed_hops"] >= 1) & (res.scores["seed_risk"] > 0.05) & res.scores["active"]
    ]
    eid = int(near["seed_risk"].idxmax())
    r = build_entity_report(res, eid)
    sp = r["seed_proximity"]
    assert sp["hops"] >= 1 and sp["nearest_seed"].startswith("E-") and sp["risk"] > 0
    assert sp["boost_points"] >= 0 and "not guilt" in sp["note"]
    assert r["score"]["seed_proximity"] == pytest.approx(sp["risk"])
    seed_id = int(res.scores.index[res.scores["is_seed"]][0])
    assert build_entity_report(res, seed_id)["seed_proximity"]["is_seed"] is True


def test_analyst_confirmed_leads_become_seeds(res):
    import pandas as pd

    pos = [int(i) for i in res.scores.index[res.labels["is_illicit"] & res.scores["active"]][:30]]
    neg = [
        int(i) for i in res.scores.index[~res.labels["is_illicit"] & res.scores["active"]][:200]
    ]
    labels = pd.Series([1.0] * len(pos) + [0.0] * len(neg), index=pos + neg)
    new = run_forensics(res.ds, ForensicsConfig(n_estimators=40, seed=21), labels=labels)
    assert new.label_source == "analyst_labels"
    assert set(np.flatnonzero(new.seed.is_seed)) == set(pos)


def test_seeds_are_not_boosted_or_ranked_as_new_leads(res):
    s = res.scores
    seeds = s[s["is_seed"]]
    assert len(seeds) > 0
    assert (seeds["rank"] == 0).all()  # known entities are not new leads
    assert (seeds["score"] == seeds["score_base"]).all()  # no boost for merely being a seed
    ranked = s[s["rank"] > 0]
    assert ranked["rank"].tolist() == list(range(1, len(ranked) + 1)) or sorted(
        ranked["rank"]
    ) == list(range(1, len(ranked) + 1))


def test_leads_list_hides_seeds_unless_asked(res):
    from backend.forensics.cases import CaseStore
    from backend.forensics.service import ForensicsService

    svc = ForensicsService(CaseStore(":memory:"))
    svc._install(res)
    default = svc.alerts(limit=500)
    assert default["alerts"] and not any(a["is_seed"] for a in default["alerts"])
    with_seeds = svc.alerts(limit=500, include_seeds=True)
    listed_seeds = [a for a in with_seeds["alerts"] if a["is_seed"]]
    assert listed_seeds and all(a["rank"] == 0 for a in listed_seeds)
    assert with_seeds["total"] > default["total"]
    # with seeds shown they are interleaved by score, not pushed to the end
    scores = [a["score"] for a in with_seeds["alerts"]]
    assert scores == sorted(scores, reverse=True)
    assert any(a["is_seed"] for a in with_seeds["alerts"][:50])
    entity = svc.entity(listed_seeds[0]["entity_id"])
    assert "known-illicit seed" in entity["summary"] and entity["score"]["rank"] == 0
