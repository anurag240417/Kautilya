"""API handlers for live simulation and transaction injection."""

from backend.api.app import Request, Response
from backend.api.service import get_investigation_service


def handle_get_simulation_status(req: Request) -> Response:
    """Return the current simulation status and progress."""
    service = get_investigation_service()
    status = service.get_simulation_status()
    return Response(
        body=status,
        status_code=200,
        headers={"Access-Control-Allow-Origin": "*"},
    )


def handle_post_simulation_inject(req: Request) -> Response:
    """Inject the next transaction or batch in the simulation sequence."""
    service = get_investigation_service()

    # Check for batch size in query params or body
    batch_param = req.get_query_param("batch")
    if not batch_param:
        body = req.json()
        if isinstance(body, dict):
            batch_param = body.get("batch")

    if batch_param:
        try:
            batch_size = int(batch_param)
            if batch_size > 1:
                result = service.inject_simulation_batch(batch_size=batch_size)
                return Response(
                    body=result,
                    status_code=200,
                    headers={"Access-Control-Allow-Origin": "*"},
                )
        except (ValueError, TypeError):
            pass

    result = service.inject_simulation_step()
    return Response(
        body=result,
        status_code=200,
        headers={"Access-Control-Allow-Origin": "*"},
    )


def handle_post_simulation_reset(req: Request) -> Response:
    """Reset the simulation to baseline (Step 0) or full (Step 6)."""
    mode = req.get_query_param("mode")
    if not mode:
        body = req.json()
        if isinstance(body, dict):
            mode = body.get("mode")
    if not mode:
        mode = "baseline"

    service = get_investigation_service()
    status = service.reset_simulation(mode=mode)
    return Response(
        body=status,
        status_code=200,
        headers={"Access-Control-Allow-Origin": "*"},
    )
