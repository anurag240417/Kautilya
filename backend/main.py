"""ChainTrace application entry point.

Initializes logging, the investigation service, and serves the WSGI API.
"""

import logging
from wsgiref.simple_server import make_server

from backend.api import create_app
from backend.config import configure_logging

logger = logging.getLogger(__name__)


def main(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Initialize and run the ChainTrace application."""
    configure_logging()
    logger.info("ChainTrace starting Investigation API on %s:%d...", host, port)

    app = create_app()
    server = make_server(host, port, app)
    logger.info("Serving ChainTrace Investigation API on http://%s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("ChainTrace API shutting down.")


if __name__ == "__main__":
    main()
