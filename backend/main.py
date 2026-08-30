"""ChainTrace application entry point."""

import logging

from backend.config import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Initialize and run the ChainTrace application."""
    configure_logging()
    logger.info("ChainTrace starting...")
    # Pipeline stages will be wired here as they are implemented.


if __name__ == "__main__":
    main()
