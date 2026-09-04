"""ChainTrace WSGI API router and application.

Implements a clean, zero-dependency REST API engine using Python standard
library WSGI and Pydantic v2 validation.
"""

import io
import json
import logging
import re
from collections.abc import Callable
from urllib.parse import parse_qs

from pydantic import BaseModel, ValidationError

from backend.api.schemas import ErrorDetail, ErrorResponse

logger = logging.getLogger(__name__)


class Request:
    """Encapsulates an incoming HTTP request within WSGI."""

    def __init__(
        self,
        method: str,
        path: str,
        query_params: dict[str, list[str]],
        headers: dict[str, str],
        body_bytes: bytes,
        path_params: dict[str, str] | None = None,
    ) -> None:
        self.method = method.upper()
        self.path = path
        self.query_params = query_params
        self.headers = headers
        self.body_bytes = body_bytes
        self.path_params = path_params or {}

    def get_query_param(self, key: str, default: str | None = None) -> str | None:
        """Get a single query parameter value."""
        values = self.query_params.get(key)
        if values and len(values) > 0:
            return values[0]
        return default

    def json(self) -> object:
        """Parse request body as JSON."""
        if not self.body_bytes:
            return None
        return json.loads(self.body_bytes.decode("utf-8"))


class Response:
    """HTTP response container."""

    def __init__(
        self,
        body: object,
        status_code: int = 200,
        content_type: str = "application/json",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.content_type = content_type
        self.headers = headers or {}
        self.body = body

    def to_bytes(self) -> bytes:
        """Serialize body to bytes."""
        if isinstance(self.body, bytes):
            return self.body
        if isinstance(self.body, str):
            return self.body.encode("utf-8")
        if isinstance(self.body, BaseModel):
            return self.body.model_dump_json().encode("utf-8")
        return json.dumps(self.body, default=str).encode("utf-8")


class Route:
    """Represents a matched URL route pattern."""

    def __init__(self, method: str, path_pattern: str, handler: Callable) -> None:
        self.method = method.upper()
        self.path_pattern = path_pattern
        self.handler = handler

        # Compile {param} into regex group
        pattern = re.sub(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", r"(?P<\1>[^/]+)", path_pattern)
        self.regex = re.compile(f"^{pattern}$")

    def match(self, method: str, path: str) -> dict[str, str] | None:
        """Return extracted path params if route matches, else None."""
        if self.method != method.upper():
            return None
        m = self.regex.match(path)
        if m:
            return m.groupdict()
        return None


class ChainTraceAPI:
    """WSGI-compliant micro-application for ChainTrace Investigation API."""

    def __init__(self, title: str = "ChainTrace API") -> None:
        self.title = title
        self.routes: list[Route] = []

    def add_route(self, method: str, path: str, handler: Callable) -> None:
        """Register a route handler."""
        self.routes.append(Route(method, path, handler))

    def get(self, path: str) -> Callable:
        """Decorator to register a GET route."""
        def decorator(handler: Callable) -> Callable:
            self.add_route("GET", path, handler)
            return handler
        return decorator

    def post(self, path: str) -> Callable:
        """Decorator to register a POST route."""
        def decorator(handler: Callable) -> Callable:
            self.add_route("POST", path, handler)
            return handler
        return decorator

    def patch(self, path: str) -> Callable:
        """Decorator to register a PATCH route."""
        def decorator(handler: Callable) -> Callable:
            self.add_route("PATCH", path, handler)
            return handler
        return decorator

    def options(self, path: str) -> Callable:
        """Decorator to register an OPTIONS route."""
        def decorator(handler: Callable) -> Callable:
            self.add_route("OPTIONS", path, handler)
            return handler
        return decorator

    def handle_request(self, req: Request) -> Response:
        """Dispatch request to matching route handler."""
        # Handle CORS preflight
        if req.method == "OPTIONS":
            return Response(
                body="",
                status_code=204,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, PATCH, OPTIONS",
                    "Access-Control-Allow-Headers": (
                        "Content-Type, Authorization, X-Requested-With"
                    ),
                },
            )

        # Match routes
        matched_path = False
        for route in self.routes:
            params = route.match(req.method, req.path)
            if params is not None:
                req.path_params = params
                try:
                    res = route.handler(req)
                    if isinstance(res, Response):
                        response = res
                    elif isinstance(res, tuple) and len(res) == 2:
                        code, data = res
                        response = Response(body=data, status_code=code)
                    else:
                        response = Response(body=res, status_code=200)

                    # Ensure CORS header
                    response.headers["Access-Control-Allow-Origin"] = "*"
                    return response

                except ValidationError as ve:
                    err_resp = ErrorResponse(
                        error=ErrorDetail(
                            code="VALIDATION_ERROR",
                            message="Input validation failed.",
                            details=ve.errors(),
                        )
                    )
                    return Response(
                        body=err_resp,
                        status_code=400,
                        headers={"Access-Control-Allow-Origin": "*"},
                    )
                except ValueError as ve:
                    err_resp = ErrorResponse(
                        error=ErrorDetail(
                            code="INVALID_REQUEST",
                            message=str(ve),
                        )
                    )
                    return Response(
                        body=err_resp,
                        status_code=400,
                        headers={"Access-Control-Allow-Origin": "*"},
                    )
                except Exception as exc:
                    logger.exception(
                        "Internal error processing request %s %s: %s",
                        req.method,
                        req.path,
                        exc,
                    )
                    err_resp = ErrorResponse(
                        error=ErrorDetail(
                            code="INTERNAL_SERVER_ERROR",
                            message="An unexpected server error occurred during investigation.",
                        )
                    )
                    return Response(
                        body=err_resp,
                        status_code=500,
                        headers={"Access-Control-Allow-Origin": "*"},
                    )

            # Check if path matched with a different method for 405
            for r in self.routes:
                if r.regex.match(req.path):
                    matched_path = True
                    break

        if matched_path:
            err = ErrorResponse(
                error=ErrorDetail(
                    code="METHOD_NOT_ALLOWED",
                    message=f"Method {req.method} not allowed for {req.path}",
                )
            )
            return Response(
                body=err,
                status_code=405,
                headers={"Access-Control-Allow-Origin": "*"},
            )

        err = ErrorResponse(
            error=ErrorDetail(
                code="NOT_FOUND",
                message=f"Path not found: {req.path}",
            )
        )
        return Response(
            body=err,
            status_code=404,
            headers={"Access-Control-Allow-Origin": "*"},
        )

    def __call__(
        self,
        environ: dict[str, object],
        start_response: Callable,
    ) -> list[bytes]:
        """WSGI entry point callable."""
        method = str(environ.get("REQUEST_METHOD", "GET"))
        path = str(environ.get("PATH_INFO", "/"))
        query_string = str(environ.get("QUERY_STRING", ""))
        query_params = parse_qs(query_string, keep_blank_values=True)

        # Extract headers
        headers: dict[str, str] = {}
        for key, value in environ.items():
            if key.startswith("HTTP_"):
                h_name = key[5:].replace("_", "-").title()
                headers[h_name] = str(value)
            elif key in ("CONTENT_TYPE", "CONTENT_LENGTH"):
                headers[key.replace("_", "-").title()] = str(value)

        # Read body
        body_bytes = b""
        try:
            content_length = int(str(environ.get("CONTENT_LENGTH", 0)))
        except (ValueError, TypeError):
            content_length = 0

        if content_length > 0:
            stream = environ.get("wsgi.input")
            if hasattr(stream, "read"):
                body_bytes = stream.read(content_length)

        req = Request(
            method=method,
            path=path,
            query_params=query_params,
            headers=headers,
            body_bytes=body_bytes,
        )

        response = self.handle_request(req)
        body_data = response.to_bytes()

        status_str = f"{response.status_code} {self._status_text(response.status_code)}"
        resp_headers = [
            ("Content-Type", response.content_type),
            ("Content-Length", str(len(body_data))),
        ]
        for hk, hv in response.headers.items():
            resp_headers.append((hk, hv))

        start_response(status_str, resp_headers)
        return [body_data]

    @staticmethod
    def _status_text(code: int) -> str:
        texts = {
            200: "OK",
            201: "Created",
            204: "No Content",
            400: "Bad Request",
            404: "Not Found",
            405: "Method Not Allowed",
            500: "Internal Server Error",
        }
        return texts.get(code, "Unknown")


class TestResponse:
    """Client test response for fast in-memory assertions."""

    __test__ = False

    def __init__(self, status_code: int, headers: dict[str, str], body_bytes: bytes) -> None:
        self.status_code = status_code
        self.headers = headers
        self.body_bytes = body_bytes

    @property
    def text(self) -> str:
        return self.body_bytes.decode("utf-8")

    def json(self) -> object:
        return json.loads(self.text)


class TestClient:
    """In-memory test client that executes WSGI requests synchronously."""

    __test__ = False

    def __init__(self, app: ChainTraceAPI) -> None:
        self.app = app

    def request(
        self,
        method: str,
        path: str,
        query_params: dict[str, object] | None = None,
        json_body: object | None = None,
        headers: dict[str, str] | None = None,
    ) -> TestResponse:
        """Issue an in-memory HTTP request against the WSGI app."""
        body_bytes = b""
        req_headers = dict(headers or {})
        if json_body is not None:
            body_bytes = json.dumps(json_body).encode("utf-8")
            req_headers["Content-Type"] = "application/json"
            req_headers["Content-Length"] = str(len(body_bytes))

        # Query string construction
        query_parts = []
        if query_params:
            for k, v in query_params.items():
                if v is not None:
                    query_parts.append(f"{k}={v}")
        query_string = "&".join(query_parts)

        # Build WSGI environ
        environ: dict[str, object] = {
            "REQUEST_METHOD": method.upper(),
            "PATH_INFO": path,
            "QUERY_STRING": query_string,
            "CONTENT_LENGTH": str(len(body_bytes)),
            "CONTENT_TYPE": req_headers.get("Content-Type", ""),
            "wsgi.input": io.BytesIO(body_bytes),
            "SERVER_NAME": "localhost",
            "SERVER_PORT": "8000",
            "wsgi.version": (1, 0),
        }
        for hk, hv in req_headers.items():
            environ[f"HTTP_{hk.upper().replace('-', '_')}"] = hv

        status_result: dict[str, int] = {}
        headers_result: dict[str, str] = {}

        def start_response(status: str, resp_headers: list[tuple[str, str]]) -> None:
            code = int(status.split()[0])
            status_result["code"] = code
            for hk, hv in resp_headers:
                headers_result[hk] = hv

        chunks = self.app(environ, start_response)
        body = b"".join(chunks)

        return TestResponse(
            status_code=status_result.get("code", 500),
            headers=headers_result,
            body_bytes=body,
        )

    def get(
        self,
        path: str,
        params: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
    ) -> TestResponse:
        return self.request("GET", path, query_params=params, headers=headers)

    def post(
        self,
        path: str,
        json: object | None = None,
        headers: dict[str, str] | None = None,
    ) -> TestResponse:
        return self.request("POST", path, json_body=json, headers=headers)

    def patch(
        self,
        path: str,
        json: object | None = None,
        headers: dict[str, str] | None = None,
    ) -> TestResponse:
        return self.request("PATCH", path, json_body=json, headers=headers)

    def options(self, path: str, headers: dict[str, str] | None = None) -> TestResponse:
        return self.request("OPTIONS", path, headers=headers)
