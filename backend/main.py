"""ChainTrace application entry point.

Initializes logging, the investigation service, and serves the WSGI API.
"""

import logging
import os
import threading
from socketserver import ThreadingMixIn
from wsgiref.simple_server import WSGIServer, make_server

from backend.api import create_app
from backend.config import configure_logging

logger = logging.getLogger(__name__)


class ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    """Handle each request in its own thread so a slow forensic load never blocks the UI."""

    daemon_threads = True


def _warm_forensics() -> None:
    """Build the default forensic dataset in the background so the first page view is instant."""
    try:
        from backend.forensics.service import get_forensics_service

        get_forensics_service().ensure_loaded()
        logger.info("Forensics Lab ready.")
    except Exception:  # never let warm-up take the server down
        logger.exception("Forensics warm-up failed; it will retry on first request.")


def main(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Initialize and run the ChainTrace application."""
    configure_logging()
    logger.info("ChainTrace starting Investigation API on %s:%d...", host, port)

    app = create_app()
    from backend.api.service import get_investigation_service
    svc = get_investigation_service()
    svc.reset_simulation(mode="baseline")
    logger.info("Simulation engine ready: 110 transactions queued for live injection (starting from baseline).")

    threading.Thread(target=_warm_forensics, name="forensics-warmup", daemon=True).start()
    server = make_server(host, port, app, server_class=ThreadingWSGIServer)
    logger.info("Serving ChainTrace Investigation API on http://%s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("ChainTrace API shutting down.")


if __name__ == "__main__":
    main(
        host=os.environ.get("CHAINTRACE_HOST", "127.0.0.1"),
        port=int(os.environ.get("CHAINTRACE_PORT", "8000")),
    )
