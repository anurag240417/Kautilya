"""Forensics HTTP API, including input validation and path-traversal defences."""

import os

import pytest

from backend.api import TestClient, create_app
from backend.forensics.cases import CaseStore
from backend.forensics.report import verify_report
from backend.forensics.service import ForensicsService, set_forensics_service
from backend.forensics.synth import SynthConfig, generate_dataset


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    ds = generate_dataset(SynthConfig(n_tx=4000, seed=9))
    ds.write_csv(d / "sample.csv", limit=800)
    (d.parent / "secret.csv").write_text("x")
    os.environ["CHAINTRACE_DATA_DIR"] = str(d)
    svc = ForensicsService(CaseStore(":memory:"))
    set_forensics_service(svc)
    c = TestClient(create_app())
    assert (
        c.request(
            "POST", "/forensics/load", json_body={"source": "synthetic", "n_tx": 6000, "seed": 9}
        ).status_code
        == 200
    )
    yield c
    set_forensics_service(None)
    os.environ.pop("CHAINTRACE_DATA_DIR", None)


def test_status(client):
    j = client.request("GET", "/forensics/status").json()
    assert j["loaded"] and j["n_alerts"] > 0 and j["label_source"] == "ground_truth_synthetic"
    assert j["dataset"]["has_ground_truth"] and "fused_score" in j["metrics"]["ranking"]


def test_alerts_listing_filters_and_paging(client):
    a = client.request("GET", "/forensics/alerts", query_params={"limit": 5}).json()
    assert len(a["alerts"]) == 5 and a["alerts"][0]["rank"] == 1
    assert a["alerts"][0]["score"] >= a["alerts"][-1]["score"]
    crit = client.request(
        "GET", "/forensics/alerts", query_params={"tier": "critical", "limit": 500}
    ).json()
    assert all(x["tier"] == "critical" for x in crit["alerts"])
    page2 = client.request(
        "GET", "/forensics/alerts", query_params={"limit": 5, "offset": 5}
    ).json()
    assert page2["alerts"][0]["rank"] == 6
    one = client.request(
        "GET", "/forensics/alerts", query_params={"search": a["alerts"][0]["label"]}
    ).json()
    assert one["total"] == 1


def test_bad_inputs_rejected_with_400(client):
    assert (
        client.request("GET", "/forensics/alerts", query_params={"tier": "bogus"}).status_code
        == 400
    )
    assert (
        client.request("GET", "/forensics/alerts", query_params={"limit": "abc"}).status_code
        == 400
    )
    assert client.request("GET", "/forensics/entities/notanumber").status_code == 400
    assert client.request("GET", "/forensics/entities/99999999").status_code == 404
    assert (
        client.request(
            "POST", "/forensics/feedback", json_body={"entity_id": 1, "verdict": "maybe"}
        ).status_code
        == 400
    )
    assert client.request("POST", "/forensics/feedback", json_body={"nope": 1}).status_code == 400
    assert (
        client.request("POST", "/forensics/load", json_body={"source": "ftp"}).status_code == 400
    )
    assert (
        client.request(
            "POST", "/forensics/load", json_body={"source": "synthetic", "n_tx": 10}
        ).status_code
        == 400
    )
    assert client.request("GET", "/forensics/report").status_code == 400
    assert (
        client.request("GET", "/forensics/report", query_params={"entities": "1,x"}).status_code
        == 400
    )


def test_load_file_rejects_path_traversal(client):
    for bad in (
        "../secret.csv",
        "..\\secret.csv",
        "/etc/passwd",
        "C:/Windows/win.ini",
        "nope.csv",
    ):
        r = client.request("POST", "/forensics/load", json_body={"source": "file", "path": bad})
        assert r.status_code == 400, bad


def test_entity_graph_feedback_report_flow(client):
    a = client.request("GET", "/forensics/alerts", query_params={"limit": 3}).json()["alerts"]
    eid = a[0]["entity_id"]
    e = client.request("GET", f"/forensics/entities/{eid}").json()
    assert e["entity_id"] == eid and e["model_contributions"]
    g = client.request("GET", f"/forensics/graph/{eid}", query_params={"radius": 2}).json()
    assert g["center"] == eid and any(n["id"] == eid for n in g["nodes"])

    r = client.request(
        "POST",
        "/forensics/feedback",
        json_body={"entity_id": eid, "verdict": "confirmed", "analyst": "t"},
    )
    assert r.status_code == 201
    fb = client.request("GET", "/forensics/feedback").json()["feedback"]
    assert fb[0]["entity_id"] == eid
    assert (
        client.request("GET", "/forensics/alerts", query_params={"limit": 1}).json()["alerts"][0][
            "verdict"
        ]
        == "confirmed"
    )
    uncertain = client.request("GET", "/forensics/uncertain", query_params={"k": 3}).json()[
        "entities"
    ]
    assert eid not in [u["entity_id"] for u in uncertain]

    rep = client.request(
        "GET",
        "/forensics/report",
        query_params={"entities": f"{eid},{a[1]['entity_id']}", "title": "T"},
    )
    assert rep.status_code == 200 and rep.headers["Content-Type"].startswith("text/html")
    assert verify_report(rep.text)[0]

    c = client.request(
        "POST", "/forensics/cases", json_body={"title": "Case A", "entity_ids": [eid]}
    )
    assert c.status_code == 201
    assert client.request("GET", "/forensics/cases").json()["cases"][0]["title"] == "Case A"
    assert client.request("POST", "/forensics/cases", json_body={"title": "x"}).status_code == 400


def test_retrain_then_reset(client):
    r = client.request("POST", "/forensics/retrain", json_body={"seed_fraction": 0.2})
    assert r.status_code == 200
    j = r.json()
    assert j["seed_labels"] > 0 and "fused_score" in j["after_feedback"]
    assert client.request("GET", "/forensics/status").json()["label_source"] == "analyst_labels"
    assert (
        client.request("POST", "/forensics/retrain", json_body={"seed_fraction": 5}).status_code
        == 400
    )
    assert (
        client.request("POST", "/forensics/reset-model").json()["label_source"]
        == "ground_truth_synthetic"
    )


def test_load_csv_file_from_data_dir(client):
    r = client.request(
        "POST", "/forensics/load", json_body={"source": "file", "path": "sample.csv"}
    )
    assert r.status_code == 200
    j = r.json()
    assert j["dataset"]["transactions"] == 800 and j["label_source"] == "transfer_from_synthetic"
    assert j["ingest"]["rows_dropped"] == {}
    assert not j["dataset"]["has_ground_truth"]
