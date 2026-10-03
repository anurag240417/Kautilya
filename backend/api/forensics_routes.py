"""Forensics API: raw-transaction analysis, alerts, evidence, graph, feedback, reports.

GET  /forensics/status
POST /forensics/load                 {"source": "synthetic"|"file", ...}
GET  /forensics/files                data files available to load
GET  /forensics/alerts               ?limit&offset&min_score&tier&search
GET  /forensics/entities/{id}
GET  /forensics/graph/{id}           ?radius&max_nodes
POST /forensics/feedback             {"entity_id", "verdict", "analyst", "note"}
GET  /forensics/feedback
GET  /forensics/uncertain            ?k
POST /forensics/retrain              {"seed_fraction"}
POST /forensics/reset-model
GET  /forensics/report               ?entities=1,2&title&analyst&notes   (HTML)
GET  /forensics/cases   POST /forensics/cases
GET  /forensics/benchmark            latest saved benchmark report
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.api.app import Request, Response
from backend.api.schemas import ErrorDetail, ErrorResponse
from backend.forensics.service import get_forensics_service, list_data_files


def _err(code: str, message: str, status: int) -> Response:
    return Response(
        body=ErrorResponse(error=ErrorDetail(code=code, message=message)), status_code=status
    )


def _int(req: Request, key: str, default: int, lo: int, hi: int) -> int:
    raw = req.get_query_param(key)
    if raw is None:
        return default
    try:
        v = int(raw)
    except ValueError:
        raise ValueError(f"'{key}' must be an integer") from None
    return min(max(v, lo), hi)


def _float(req: Request, key: str, default: float) -> float:
    raw = req.get_query_param(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"'{key}' must be a number") from None


def _body(req: Request) -> dict:
    try:
        b = req.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ValueError("request body must be valid JSON") from None
    return b if isinstance(b, dict) else {}


def _entity_id(req: Request) -> int:
    raw = req.path_params.get("entity_id", "").lower().removeprefix("e-")
    if not raw.isdigit():
        raise ValueError("entity id must be an integer")
    return int(raw)


def handle_status(req: Request) -> Response:
    return Response(body=get_forensics_service().status())


def handle_load(req: Request) -> Response:
    b = _body(req)
    svc = get_forensics_service()
    source = b.get("source", "synthetic")
    if source == "synthetic":
        return Response(body=svc.load_synthetic(int(b.get("n_tx", 20_000)), int(b.get("seed", 7))))
    if source == "file":
        if not b.get("path"):
            raise ValueError("'path' is required for source=file")
        return Response(body=svc.load_file(str(b["path"])))
    raise ValueError("source must be 'synthetic' or 'file'")


def handle_files(req: Request) -> Response:
    return Response(body={"files": list_data_files()})


def handle_alerts(req: Request) -> Response:
    svc = get_forensics_service()
    tier = req.get_query_param("tier")
    if tier and tier not in ("critical", "high", "medium", "low"):
        raise ValueError("tier must be critical, high, medium or low")
    return Response(
        body=svc.alerts(
            limit=_int(req, "limit", 50, 1, 500),
            offset=_int(req, "offset", 0, 0, 10_000_000),
            min_score=_float(req, "min_score", 0.0),
            tier=tier,
            search=req.get_query_param("search"),
        )
    )


def handle_entity(req: Request) -> Response:
    try:
        return Response(body=get_forensics_service().entity(_entity_id(req)))
    except KeyError as e:
        return _err("NOT_FOUND", str(e.args[0]), 404)


def handle_graph(req: Request) -> Response:
    try:
        return Response(
            body=get_forensics_service().graph(
                _entity_id(req), _int(req, "radius", 2, 1, 3), _int(req, "max_nodes", 60, 5, 150)
            )
        )
    except KeyError as e:
        return _err("NOT_FOUND", str(e.args[0]), 404)


def handle_post_feedback(req: Request) -> Response:
    b = _body(req)
    try:
        eid = int(b["entity_id"])
        verdict = str(b["verdict"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("'entity_id' (int) and 'verdict' are required") from None
    try:
        out = get_forensics_service().add_feedback(
            eid, verdict, str(b.get("analyst", ""))[:80], str(b.get("note", ""))[:500]
        )
    except KeyError as e:
        return _err("NOT_FOUND", str(e.args[0]), 404)
    return Response(body=out, status_code=201)


def handle_get_feedback(req: Request) -> Response:
    return Response(body={"feedback": get_forensics_service().feedback()})


def handle_uncertain(req: Request) -> Response:
    return Response(
        body={"entities": get_forensics_service().uncertain(_int(req, "k", 10, 1, 100))}
    )


def handle_retrain(req: Request) -> Response:
    b = _body(req)
    frac = float(b.get("seed_fraction", 0.1))
    if not 0.01 <= frac <= 0.9:
        raise ValueError("seed_fraction must be between 0.01 and 0.9")
    return Response(body=get_forensics_service().retrain(frac))


def handle_reset_model(req: Request) -> Response:
    return Response(body=get_forensics_service().reset_model())


def handle_report(req: Request) -> Response:
    raw = req.get_query_param("entities", "") or ""
    try:
        ids = [int(x.strip().lower().removeprefix("e-")) for x in raw.split(",") if x.strip()]
    except ValueError:
        raise ValueError("'entities' must be a comma-separated list of integers") from None
    if not ids:
        raise ValueError("'entities' is required")
    try:
        html_text = get_forensics_service().report_html(
            ids,
            (req.get_query_param("title") or "ChainTrace investigation report")[:120],
            (req.get_query_param("analyst") or "")[:80],
            (req.get_query_param("notes") or "")[:1000],
        )
    except KeyError as e:
        return _err("NOT_FOUND", str(e.args[0]), 404)
    return Response(
        body=html_text,
        content_type="text/html; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="chaintrace_report.html"'}
        if req.get_query_param("download")
        else {},
    )


def handle_get_cases(req: Request) -> Response:
    return Response(body={"cases": get_forensics_service().cases()})


def handle_post_case(req: Request) -> Response:
    b = _body(req)
    ids = b.get("entity_ids")
    if not b.get("title") or not isinstance(ids, list) or not ids:
        raise ValueError("'title' and a non-empty 'entity_ids' list are required")
    case = get_forensics_service().create_case(
        str(b["title"])[:120],
        [int(i) for i in ids],
        str(b.get("analyst", ""))[:80],
        str(b.get("notes", ""))[:2000],
    )
    return Response(body=case, status_code=201)


def handle_benchmark(req: Request) -> Response:
    p = Path.cwd() / "reports" / "forensics_benchmark.json"
    if not p.exists():
        return _err(
            "NOT_FOUND",
            "no benchmark saved yet; run `python -m scripts.run_forensic_benchmark`",
            404,
        )
    return Response(body=json.loads(p.read_text(encoding="utf-8")))


def register(app) -> None:
    app.add_route("GET", "/forensics/status", handle_status)
    app.add_route("POST", "/forensics/load", handle_load)
    app.add_route("GET", "/forensics/files", handle_files)
    app.add_route("GET", "/forensics/alerts", handle_alerts)
    app.add_route("GET", "/forensics/entities/{entity_id}", handle_entity)
    app.add_route("GET", "/forensics/graph/{entity_id}", handle_graph)
    app.add_route("POST", "/forensics/feedback", handle_post_feedback)
    app.add_route("GET", "/forensics/feedback", handle_get_feedback)
    app.add_route("GET", "/forensics/uncertain", handle_uncertain)
    app.add_route("POST", "/forensics/retrain", handle_retrain)
    app.add_route("POST", "/forensics/reset-model", handle_reset_model)
    app.add_route("GET", "/forensics/report", handle_report)
    app.add_route("GET", "/forensics/cases", handle_get_cases)
    app.add_route("POST", "/forensics/cases", handle_post_case)
    app.add_route("GET", "/forensics/benchmark", handle_benchmark)
