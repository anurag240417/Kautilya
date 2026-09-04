"""Investigation API module.

Exposes backend capabilities to the frontend per ARCHITECTURE.md §api/.
"""

from backend.api.alerts import (
    handle_get_alert_by_id,
    handle_get_alerts,
    handle_patch_alert,
)
from backend.api.app import ChainTraceAPI, Request, Response, TestClient
from backend.api.graph import handle_get_graph, handle_get_graph_path
from backend.api.schemas import (
    AlertListResponse,
    AlertUpdatePayload,
    ErrorDetail,
    ErrorResponse,
    GraphEdge,
    GraphNode,
    GraphPathResponse,
    GraphResponse,
    HealthResponse,
    TransactionResponse,
    WalletResponse,
)
from backend.api.service import (
    InvestigationService,
    get_investigation_service,
    set_investigation_service,
)
from backend.api.transactions import handle_get_transaction
from backend.api.wallets import handle_get_wallet


def create_app() -> ChainTraceAPI:
    """Factory function to build and configure the ChainTrace API application."""
    app = ChainTraceAPI(title="ChainTrace Investigation API")

    # Health check
    @app.get("/health")
    def health_check(req: Request) -> Response:
        return Response(body=HealthResponse(status="healthy", version="0.1.0"), status_code=200)

    # Transactions
    app.add_route("GET", "/transactions/{txid}", handle_get_transaction)

    # Wallets
    app.add_route("GET", "/wallets/{address}", handle_get_wallet)

    # Alerts
    app.add_route("GET", "/alerts", handle_get_alerts)
    app.add_route("GET", "/alerts/{alert_id}", handle_get_alert_by_id)
    app.add_route("PATCH", "/alerts/{alert_id}", handle_patch_alert)

    # Graph (specific paths before parameterized paths)
    app.add_route("GET", "/graph/path", handle_get_graph_path)
    app.add_route("GET", "/graph/{entity_id}", handle_get_graph)

    return app


__all__ = [
    # Application & Client
    "ChainTraceAPI",
    "create_app",
    "TestClient",
    "Request",
    "Response",
    # Service
    "InvestigationService",
    "get_investigation_service",
    "set_investigation_service",
    # Route Handlers
    "handle_get_transaction",
    "handle_get_wallet",
    "handle_get_alerts",
    "handle_get_alert_by_id",
    "handle_patch_alert",
    "handle_get_graph",
    "handle_get_graph_path",
    # Schemas
    "ErrorDetail",
    "ErrorResponse",
    "TransactionResponse",
    "WalletResponse",
    "AlertListResponse",
    "AlertUpdatePayload",
    "GraphNode",
    "GraphEdge",
    "GraphResponse",
    "GraphPathResponse",
    "HealthResponse",
]
