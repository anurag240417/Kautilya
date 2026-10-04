"""Investigation API module.

Exposes backend capabilities to the frontend per ARCHITECTURE.md §api/.
"""

from backend.api.alerts import (
    handle_get_alert_by_id,
    handle_get_alerts,
    handle_patch_alert,
)
from backend.api.app import KautilyaAPI, Request, Response, TestClient
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
    StatisticsResponse,
    TransactionResponse,
    WalletResponse,
)
from backend.api.service import (
    InvestigationService,
    get_investigation_service,
    set_investigation_service,
)
from backend.api.simulation_routes import (
    handle_get_simulation_status,
    handle_post_simulation_inject,
    handle_post_simulation_reset,
)
from backend.api.transactions import handle_get_transaction
from backend.api.wallets import handle_get_wallet


def create_app() -> KautilyaAPI:
    """Factory function to build and configure the Kautilya API application."""
    app = KautilyaAPI(title="Kautilya Investigation API")

    # Health check
    @app.get("/health")
    def health_check(req: Request) -> Response:
        return Response(body=HealthResponse(status="healthy", version="0.1.0"), status_code=200)

    # Statistics / Dashboard summary
    @app.get("/statistics")
    def statistics_summary(req: Request) -> Response:
        service = get_investigation_service()
        stats = service.get_statistics()
        return Response(body=stats, status_code=200)

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

    # Simulation / Live Stream Ingestion
    app.add_route("GET", "/simulation/status", handle_get_simulation_status)
    app.add_route("POST", "/simulation/inject", handle_post_simulation_inject)
    app.add_route("POST", "/simulation/reset", handle_post_simulation_reset)

    # Raw-transaction forensics (graph ML, heuristics, network correlation, cases)
    from backend.api import forensics_routes

    forensics_routes.register(app)

    return app


__all__ = [
    # Application & Client
    "KautilyaAPI",
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
    "StatisticsResponse",
]

