"""Generic JSON parser for offline input support.

Provides parsing for JSON files as required by the SIH contract.
Attempts to load the JSON into a pandas DataFrame if it has a
tabular structure, otherwise returns a raw dictionary.
"""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from backend.ingestion.errors import IngestionError

logger = logging.getLogger(__name__)


def parse_json(path: Path) -> pd.DataFrame | dict[str, Any] | list[Any]:
    """Parse a generic JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        A pandas DataFrame if the JSON is an array of objects,
        or a generic dictionary/list otherwise.

    Raises:
        IngestionError: If the file is missing or contains invalid JSON.
    """
    if not path.exists():
        msg = f"JSON file not found: {path}"
        raise IngestionError(msg)

    try:
        # Load with standard json module first
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # If it's a list of dicts, try to convert to DataFrame
        if isinstance(data, list) and all(isinstance(x, dict) for x in data):
            df = pd.DataFrame(data)
            logger.info("Parsed %s as DataFrame: %d rows", path.name, len(df))
            return df

        logger.info("Parsed %s as %s", path.name, type(data).__name__)
        return data
    except Exception as exc:
        msg = f"Failed to parse JSON file {path}: {exc}"
        raise IngestionError(msg) from exc
