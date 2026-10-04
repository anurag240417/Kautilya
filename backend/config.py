"""Kautilya configuration.

Reads settings from environment variables with sensible defaults.
All paths are resolved relative to the project root unless overridden.
"""

import logging
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset directory names as they exist on disk (note: leading space in EllipticPlusPlus dir)
_ELLIPTIC_PP_SUBDIR = " EllipticPlusPlus"
_ELLIPTIC_PP_DATA_SUBDIR = "data csv"


class Settings:
    """Application settings resolved from environment variables.

    Environment variables use the ``KAUTILYA_`` prefix.
    """

    def __init__(self) -> None:
        self.dataset_dir = Path(
            os.environ.get("KAUTILYA_DATASET_DIR", str(PROJECT_ROOT / "dataset"))
        )
        self.elliptic_pp_dir = self.dataset_dir / _ELLIPTIC_PP_SUBDIR / _ELLIPTIC_PP_DATA_SUBDIR
        self.models_dir = Path(
            os.environ.get("KAUTILYA_MODELS_DIR", str(PROJECT_ROOT / "backend" / "models"))
        )
        self.log_level = os.environ.get("KAUTILYA_LOG_LEVEL", "INFO")

    # --- Elliptic++ file paths ---

    @property
    def txs_features_path(self) -> Path:
        """Path to Elliptic++ transaction features CSV."""
        return self.elliptic_pp_dir / "txs_features.csv"

    @property
    def txs_classes_path(self) -> Path:
        """Path to Elliptic++ transaction labels CSV."""
        return self.elliptic_pp_dir / "txs_classes.csv"

    @property
    def wallets_features_path(self) -> Path:
        """Path to Elliptic++ wallet features CSV."""
        return self.elliptic_pp_dir / "wallets_features.csv"

    @property
    def wallets_classes_path(self) -> Path:
        """Path to Elliptic++ wallet labels CSV."""
        return self.elliptic_pp_dir / "wallets_classes.csv"

    @property
    def txs_edgelist_path(self) -> Path:
        """Path to transaction-to-transaction edgelist CSV."""
        return self.elliptic_pp_dir / "txs_edgelist.csv"

    @property
    def addr_tx_edgelist_path(self) -> Path:
        """Path to address-to-transaction edgelist CSV."""
        return self.elliptic_pp_dir / "AddrTx_edgelist.csv"

    @property
    def tx_addr_edgelist_path(self) -> Path:
        """Path to transaction-to-address edgelist CSV."""
        return self.elliptic_pp_dir / "TxAddr_edgelist.csv"

    @property
    def addr_addr_edgelist_path(self) -> Path:
        """Path to address-to-address edgelist CSV."""
        return self.elliptic_pp_dir / "AddrAddr_edgelist.csv"


def get_settings() -> Settings:
    """Create and return application settings."""
    return Settings()


def configure_logging(level: str | None = None) -> None:
    """Configure application-wide logging.

    Args:
        level: Logging level string. Falls back to settings if not provided.
    """
    resolved_level = level or get_settings().log_level
    logging.basicConfig(
        level=getattr(logging, resolved_level.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
