"""Custom exceptions for the ingestion module."""


class IngestionError(Exception):
    """Raised when data ingestion fails.

    Covers: missing files, wrong schemas, unreadable data.
    """
