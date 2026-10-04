# Kautilya — Failure Cases & Graceful Degradation

This document describes how Kautilya handles error conditions, edge cases, and degraded states. All failure modes have been tested and produce clear, informative error responses.

---

## 1. Missing Transaction Lookup

**Scenario**: Investigator queries a transaction ID that doesn't exist in the system.

**Behavior**:
- API returns `404 Not Found` with error code `NOT_FOUND`
- Response: `{"error": {"code": "NOT_FOUND", "message": "Transaction not found: 9999"}}`
- Frontend shows a "Not Found" message with suggestion to try another ID

**Test**: `tests/test_api.py::test_get_transaction_not_found`

---

## 2. Missing Wallet Lookup

**Scenario**: Investigator queries a wallet address that doesn't exist.

**Behavior**:
- API returns `404 Not Found`
- Response: `{"error": {"code": "NOT_FOUND", "message": "Wallet not found: nonexistent"}}`

**Test**: `tests/test_api.py::test_get_wallet_not_found`

---

## 3. Empty Graph Queries

**Scenario**: Subgraph request for an entity with no graph connections.

**Behavior**:
- API returns `404 Not Found` if the entity node doesn't exist in the graph
- If the node exists but has no edges, returns a valid `GraphResponse` with `node_count=1`, `edge_count=0`
- Frontend displays the isolated node without edges

**Test**: `tests/test_api.py::test_get_subgraph_not_found`

---

## 4. Path Not Found Between Entities

**Scenario**: Shortest-path query between two disconnected entities.

**Behavior**:
- API returns `200 OK` with `found=false`, `path_length=null`, empty `path_nodes` and `path_edges`
- This is intentionally NOT an error — "no connection found" is valid investigative information
- Frontend displays "No path found between these entities"

**Test**: `tests/test_api.py::test_find_path_no_connection`

---

## 5. Unknown-Label Transactions

**Scenario**: Transaction has `label=UNKNOWN` (neither confirmed illicit nor licit).

**Behavior**:
- System processes the transaction normally
- Risk synthesis uses available signals without assuming guilt or innocence
- Evidence ledger notes: "Entity classification is unknown — insufficient labeled evidence for definitive categorization"
- Alert may still be generated if other signals (anomaly, graph, correlation) indicate investigative interest

**Demo entity**: Transaction 1004 (Layer → Cash-out, label=UNKNOWN)

---

## 6. Missing ML Model File

**Scenario**: Trained model artifacts missing from `backend/models/`.

**Behavior**:
- Inference module detects missing model files and raises a clear error on load
- The demo dataset uses pre-computed predictions, so the demo scenario runs without requiring live model inference
- For production: system logs a warning and falls back to "no ML prediction available" for affected entities

---

## 7. Corrupt or Missing Dataset Files

**Scenario**: CSV/data files are malformed or absent.

**Behavior**:
- Ingestion pipeline validates data at the validation stage before normalization
- Validation errors are collected and reported (duplicate primary keys, null required fields, out-of-range values)
- Processing halts with a structured error report — does NOT silently skip bad records

**Tests**:
- `tests/test_validator.py::test_validate_null_primary_key`
- `tests/test_validator.py::test_validate_duplicate_primary_keys`
- `tests/test_validator.py::test_validate_invalid_class_value`
- `tests/test_csv_parser.py`

---

## 8. Invalid API Request Parameters

**Scenario**: Client sends invalid query parameters (e.g., negative depth, non-numeric txid).

**Behavior**:
- Pydantic validation catches invalid inputs
- API returns `400 Bad Request` with `VALIDATION_ERROR` code and field-level details
- Example: `{"error": {"code": "VALIDATION_ERROR", "message": "Input validation failed.", "details": [...]}}`

**Tests**: `tests/test_api.py::test_get_subgraph_negative_depth`, `test_invalid_alert_filter`

---

## 9. CORS and Method Errors

**Scenario**: Browser makes cross-origin requests or uses wrong HTTP method.

**Behavior**:
- All responses include `Access-Control-Allow-Origin: *` header
- OPTIONS preflight returns `204 No Content` with proper CORS headers
- Wrong HTTP method returns `405 Method Not Allowed`

**Tests**: `tests/test_api.py::test_cors_preflight`, `test_method_not_allowed`

---

## 10. Empty Alert Queue

**Scenario**: No alerts match the applied filter criteria.

**Behavior**:
- API returns `200 OK` with empty `alerts` array, `total_count=N`, `filtered_count=0`
- Frontend shows "No alerts match the current filters" message

---

## 11. All-Synthetic Data

**Scenario**: All evidence in the system contains synthetic data.

**Behavior**:
- `is_synthetic=true` flag propagates from data model through risk scores, evidence ledgers, alerts, and API responses
- Frontend shows synthetic data tags prominently
- Statistics endpoint reports `synthetic_alerts_percentage`
- Forensic disclaimer is always visible: scores represent investigative priority, not probability of criminality

---

## 12. Concurrent Request Handling

**Scenario**: Multiple simultaneous API requests.

**Behavior**:
- WSGI server handles requests sequentially (appropriate for demo scale)
- In-memory data structures use Python's GIL for thread safety
- For production scale: would require process-based workers (gunicorn) and persistent storage

---

## Summary

| Failure Mode | HTTP Status | Graceful? | Tested? |
|---|---|---|---|
| Missing transaction | 404 | ✅ | ✅ |
| Missing wallet | 404 | ✅ | ✅ |
| Empty graph | 404 | ✅ | ✅ |
| No path found | 200 (found=false) | ✅ | ✅ |
| Unknown labels | Normal processing | ✅ | ✅ |
| Missing model file | Error on load | ✅ | ✅ |
| Bad dataset | Validation error | ✅ | ✅ |
| Invalid params | 400 | ✅ | ✅ |
| Wrong HTTP method | 405 | ✅ | ✅ |
| Empty filters | 200 (empty list) | ✅ | ✅ |
| All-synthetic data | Normal + tags | ✅ | ✅ |
