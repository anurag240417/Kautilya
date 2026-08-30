"""Generic XML parser for offline input support.

Provides parsing for XML files as required by the SIH contract.
Uses pandas.read_xml to load the XML into a DataFrame.
"""

import logging
from pathlib import Path

import pandas as pd

from backend.ingestion.errors import IngestionError

logger = logging.getLogger(__name__)


def parse_xml(path: Path) -> pd.DataFrame:
    """Parse a generic XML file into a DataFrame.

    Args:
        path: Path to the XML file.

    Returns:
        A pandas DataFrame representing the parsed XML.

    Raises:
        IngestionError: If the file is missing or contains invalid XML.
    """
    if not path.exists():
        msg = f"XML file not found: {path}"
        raise IngestionError(msg)

    try:
        df = pd.read_xml(path)
        logger.info("Parsed %s as DataFrame: %d rows", path.name, len(df))
        return df
    except Exception as exc:
        msg = f"Failed to parse XML file {path}: {exc}"
        raise IngestionError(msg) from exc
