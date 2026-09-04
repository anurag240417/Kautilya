"""Wallet API endpoint.

GET /wallets/{address}

Retrieves canonical wallet metadata, statistics summary, aggregated entity
risk score with transaction traceability, and mandatory forensic disclaimers.
"""

from backend.api.app import Request, Response
from backend.api.schemas import ErrorDetail, ErrorResponse
from backend.api.service import get_investigation_service
from backend.risk.aggregation import AggregationMethod


def handle_get_wallet(req: Request) -> Response:
    """Handle GET /wallets/{address}."""
    address = req.path_params.get("address", "").strip()

    if not address:
        return Response(
            body=ErrorResponse(
                error=ErrorDetail(
                    code="INVALID_ADDRESS",
                    message="Wallet address parameter must not be empty.",
                )
            ),
            status_code=400,
        )

    # Parse and validate aggregation method
    method_str = req.get_query_param("aggregation_method", "max").lower()
    try:
        method = AggregationMethod(method_str)
    except ValueError:
        valid_methods = [m.value for m in AggregationMethod]
        return Response(
            body=ErrorResponse(
                error=ErrorDetail(
                    code="INVALID_AGGREGATION_METHOD",
                    message=(
                        f"Invalid aggregation_method '{method_str}'. "
                        f"Allowed methods: {', '.join(valid_methods)}."
                    ),
                )
            ),
            status_code=400,
        )

    service = get_investigation_service()
    wallet_resp = service.get_wallet(address, aggregation_method=method)

    if not wallet_resp:
        return Response(
            body=ErrorResponse(
                error=ErrorDetail(
                    code="NOT_FOUND",
                    message=f"Wallet address '{address}' not found.",
                )
            ),
            status_code=404,
        )

    return Response(body=wallet_resp, status_code=200)
