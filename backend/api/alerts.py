"""Alert API endpoints.

GET /alerts
GET /alerts/{alert_id}
PATCH /alerts/{alert_id}

Manages investigator triage queues, priority filtering, and alert lifecycle.
"""

from backend.api.app import Request, Response
from backend.api.schemas import (
    AlertListResponse,
    AlertUpdatePayload,
    ErrorDetail,
    ErrorResponse,
)
from backend.api.service import get_investigation_service
from backend.domain.alert import AlertFilter
from backend.domain.types import AlertStatus, PriorityTier


def handle_get_alerts(req: Request) -> Response:
    """Handle GET /alerts with priority filtering and pagination."""
    # 1. Parse min_tier
    raw_tier = req.get_query_param("min_tier")
    min_tier = None
    if raw_tier:
        try:
            min_tier = PriorityTier(raw_tier.lower())
        except ValueError:
            return Response(
                body=ErrorResponse(
                    error=ErrorDetail(
                        code="INVALID_FILTER",
                        message=(
                            f"Invalid min_tier '{raw_tier}'. "
                            "Expected critical, high, medium, or low."
                        ),
                    )
                ),
                status_code=400,
            )

    # 2. Parse status
    raw_status = req.get_query_param("status")
    status = None
    if raw_status:
        try:
            status = AlertStatus(raw_status.lower())
        except ValueError:
            return Response(
                body=ErrorResponse(
                    error=ErrorDetail(
                        code="INVALID_FILTER",
                        message=f"Invalid status '{raw_status}'.",
                    )
                ),
                status_code=400,
            )

    # 3. Parse entity_type
    entity_type = req.get_query_param("entity_type")
    if entity_type and entity_type.lower() not in ("transaction", "wallet"):
        return Response(
            body=ErrorResponse(
                error=ErrorDetail(
                    code="INVALID_FILTER",
                    message="entity_type must be 'transaction' or 'wallet'.",
                )
            ),
            status_code=400,
        )

    # 4. Parse include_synthetic
    raw_synthetic = req.get_query_param("include_synthetic", "true").lower()
    include_synthetic = raw_synthetic not in ("false", "0", "no")

    # 5. Parse min_score
    raw_min_score = req.get_query_param("min_score")
    min_score = None
    if raw_min_score:
        try:
            min_score = float(raw_min_score)
            if min_score < 0.0 or min_score > 100.0:
                raise ValueError
        except ValueError:
            return Response(
                body=ErrorResponse(
                    error=ErrorDetail(
                        code="INVALID_FILTER",
                        message="min_score must be a float between 0.0 and 100.0.",
                    )
                ),
                status_code=400,
            )

    # 6. Parse pagination (limit, offset)
    raw_limit = req.get_query_param("limit", "50")
    try:
        limit = int(raw_limit)
        if limit <= 0 or limit > 500:
            raise ValueError
    except ValueError:
        return Response(
            body=ErrorResponse(
                error=ErrorDetail(
                    code="INVALID_PAGINATION",
                    message="limit must be an integer between 1 and 500.",
                )
            ),
            status_code=400,
        )

    raw_offset = req.get_query_param("offset", "0")
    try:
        offset = int(raw_offset)
        if offset < 0:
            raise ValueError
    except ValueError:
        return Response(
            body=ErrorResponse(
                error=ErrorDetail(
                    code="INVALID_PAGINATION",
                    message="offset must be a non-negative integer.",
                )
            ),
            status_code=400,
        )

    filter_criteria = AlertFilter(
        min_tier=min_tier,
        entity_type=entity_type.lower() if entity_type else None,
        status=status,
        include_synthetic=include_synthetic,
        min_score=min_score,
        limit=limit,
    )

    service = get_investigation_service()
    total, filtered_count, alerts = service.list_alerts(filter_criteria, offset=offset)

    resp = AlertListResponse(
        total=total,
        filtered_count=filtered_count,
        alerts=alerts,
    )
    return Response(body=resp, status_code=200)


def handle_get_alert_by_id(req: Request) -> Response:
    """Handle GET /alerts/{alert_id}."""
    alert_id = req.path_params.get("alert_id", "").strip()
    if not alert_id:
        return Response(
            body=ErrorResponse(
                error=ErrorDetail(
                    code="INVALID_ALERT_ID",
                    message="Alert ID must not be empty.",
                )
            ),
            status_code=400,
        )

    service = get_investigation_service()
    alert = service.get_alert(alert_id)
    if not alert:
        return Response(
            body=ErrorResponse(
                error=ErrorDetail(
                    code="NOT_FOUND",
                    message=f"Alert '{alert_id}' not found.",
                )
            ),
            status_code=404,
        )

    return Response(body=alert, status_code=200)


def handle_patch_alert(req: Request) -> Response:
    """Handle PATCH /alerts/{alert_id}."""
    alert_id = req.path_params.get("alert_id", "").strip()
    if not alert_id:
        return Response(
            body=ErrorResponse(
                error=ErrorDetail(
                    code="INVALID_ALERT_ID",
                    message="Alert ID must not be empty.",
                )
            ),
            status_code=400,
        )

    body = req.json()
    if not body or not isinstance(body, dict):
        return Response(
            body=ErrorResponse(
                error=ErrorDetail(
                    code="INVALID_PAYLOAD",
                    message="Request body must be a valid JSON object.",
                )
            ),
            status_code=400,
        )

    # Validate using Pydantic schema
    try:
        payload = AlertUpdatePayload(**body)
    except Exception as e:
        return Response(
            body=ErrorResponse(
                error=ErrorDetail(
                    code="VALIDATION_ERROR",
                    message=f"Invalid alert update payload: {e}",
                )
            ),
            status_code=400,
        )

    service = get_investigation_service()
    updated = service.update_alert_status(
        alert_id=alert_id,
        new_status=payload.status,
        reviewer_notes=payload.reviewer_notes,
    )

    if not updated:
        return Response(
            body=ErrorResponse(
                error=ErrorDetail(
                    code="NOT_FOUND",
                    message=f"Alert '{alert_id}' not found.",
                )
            ),
            status_code=404,
        )

    return Response(body=updated, status_code=200)
