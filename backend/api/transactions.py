"""Transaction API endpoint.

GET /transactions/{txid}

Retrieves canonical transaction details, interpretable behavioral features,
synthesized risk scores, and correlated network observations.
"""

from backend.api.app import Request, Response
from backend.api.schemas import ErrorDetail, ErrorResponse
from backend.api.service import get_investigation_service


def handle_get_transaction(req: Request) -> Response:
    """Handle GET /transactions/{txid}."""
    raw_txid = req.path_params.get("txid", "")

    try:
        txid = int(raw_txid)
        if txid < 0:
            msg = "Transaction ID must be non-negative"
            raise ValueError(msg)
    except (ValueError, TypeError):
        return Response(
            body=ErrorResponse(
                error=ErrorDetail(
                    code="INVALID_TXID",
                    message=f"Invalid transaction ID '{raw_txid}'. Expected positive integer.",
                )
            ),
            status_code=400,
        )

    service = get_investigation_service()
    tx_resp = service.get_transaction(txid)

    if not tx_resp:
        return Response(
            body=ErrorResponse(
                error=ErrorDetail(
                    code="NOT_FOUND",
                    message=f"Transaction {txid} not found.",
                )
            ),
            status_code=404,
        )

    return Response(body=tx_resp, status_code=200)
