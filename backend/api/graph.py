"""Graph API endpoints.

GET /graph/{entity_id}
GET /graph/path

Provides graph queries, ego network extraction, and shortest-path forensics.
"""

from backend.api.app import Request, Response
from backend.api.schemas import ErrorDetail, ErrorResponse
from backend.api.service import get_investigation_service


def handle_get_graph(req: Request) -> Response:
    """Handle GET /graph/{entity_id}."""
    entity_id = req.path_params.get("entity_id", "").strip()
    if not entity_id:
        return Response(
            body=ErrorResponse(
                error=ErrorDetail(
                    code="INVALID_ENTITY_ID",
                    message="Entity ID must not be empty.",
                )
            ),
            status_code=400,
        )

    # 1. Parse depth
    raw_depth = req.get_query_param("depth", "1")
    try:
        depth = int(raw_depth)
        if depth < 1 or depth > 3:
            raise ValueError
    except ValueError:
        return Response(
            body=ErrorResponse(
                error=ErrorDetail(
                    code="INVALID_DEPTH",
                    message="depth must be an integer between 1 and 3.",
                )
            ),
            status_code=400,
        )

    # 2. Parse relationship filter
    relationship = req.get_query_param("relationship")
    if relationship:
        valid_rels = ("tx_tx", "addr_tx", "tx_addr", "addr_addr")
        if relationship not in valid_rels:
            return Response(
                body=ErrorResponse(
                    error=ErrorDetail(
                        code="INVALID_RELATIONSHIP",
                        message=f"relationship must be one of: {', '.join(valid_rels)}.",
                    )
                ),
                status_code=400,
            )

    # 3. Parse max_nodes
    raw_max_nodes = req.get_query_param("max_nodes", "100")
    try:
        max_nodes = int(raw_max_nodes)
        if max_nodes <= 0 or max_nodes > 500:
            raise ValueError
    except ValueError:
        return Response(
            body=ErrorResponse(
                error=ErrorDetail(
                    code="INVALID_MAX_NODES",
                    message="max_nodes must be an integer between 1 and 500.",
                )
            ),
            status_code=400,
        )

    service = get_investigation_service()
    subgraph = service.get_subgraph(
        entity_id=entity_id,
        depth=depth,
        relationship=relationship,
        max_nodes=max_nodes,
    )

    if not subgraph:
        return Response(
            body=ErrorResponse(
                error=ErrorDetail(
                    code="NOT_FOUND",
                    message=f"Entity '{entity_id}' not found in investigation graph.",
                )
            ),
            status_code=404,
        )

    return Response(body=subgraph, status_code=200)


def handle_get_graph_path(req: Request) -> Response:
    """Handle GET /graph/path?source={source}&target={target}."""
    source = req.get_query_param("source", "").strip()
    target = req.get_query_param("target", "").strip()

    if not source or not target:
        return Response(
            body=ErrorResponse(
                error=ErrorDetail(
                    code="MISSING_QUERY_PARAMS",
                    message="Both 'source' and 'target' query parameters are required.",
                )
            ),
            status_code=400,
        )

    service = get_investigation_service()
    path_resp = service.find_entity_path(source=source, target=target)

    if path_resp is None:
        return Response(
            body=ErrorResponse(
                error=ErrorDetail(
                    code="NOT_FOUND",
                    message=(
                        f"Either source '{source}' or target '{target}' "
                        "does not exist in graph."
                    ),
                )
            ),
            status_code=404,
        )

    return Response(body=path_resp, status_code=200)
