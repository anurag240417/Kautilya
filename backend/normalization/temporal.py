"""Temporal and timestamp normalization.

Provides canonical functions to normalize diverse timestamp representations
(ISO-8601 strings, UNIX epoch seconds/ms, naive/aware datetimes, Pandas Timestamps)
into timezone-aware UTC datetime objects.

Forensic Rule:
    Synthetic network timestamps and temporal mappings must be consistently
    represented in UTC to prevent timezone offsets from distorting temporal
    correlation windows.
"""

from datetime import UTC, datetime

import pandas as pd

TimestampInput = datetime | pd.Timestamp | int | float | str


def normalize_timestamp(ts: TimestampInput) -> datetime:
    """Normalize any timestamp representation to a timezone-aware UTC datetime.

    Supported inputs:
    - datetime objects (naive assumed UTC; aware converted to UTC)
    - pandas Timestamp objects
    - numeric epoch timestamps (int/float, auto-detecting seconds vs. milliseconds)
    - ISO-8601 formatted strings (e.g. '2019-01-07T00:00:00Z', '2019-01-07 00:00:00')

    Args:
        ts: Raw timestamp input.

    Returns:
        datetime with tzinfo=timezone.utc.

    Raises:
        ValueError: If string format cannot be parsed or value is out of range.
        TypeError: If input type is unsupported or None.
    """
    if ts is None:
        raise TypeError("Timestamp cannot be None")

    # Handle pandas Timestamp
    if isinstance(ts, pd.Timestamp):
        if pd.isna(ts):
            raise ValueError("Timestamp is NaT/null")
        dt = ts.to_pydatetime()
        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)

    # Handle Python datetime
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=UTC)
        return ts.astimezone(UTC)

    # Handle numeric epoch timestamp (int or float)
    if isinstance(ts, (int, float)):
        if pd.isna(ts):
            raise ValueError("Timestamp is NaN")
        val = float(ts)
        # Auto-detect milliseconds vs seconds:
        # 1e11 seconds is in year 5138; 1e11 milliseconds is ~1973.
        if abs(val) >= 1e11:
            val = val / 1000.0
        try:
            return datetime.fromtimestamp(val, tz=UTC)
        except (OverflowError, OSError, ValueError) as exc:
            raise ValueError(f"Epoch timestamp out of range: {ts}") from exc

    # Handle string timestamps
    if isinstance(ts, str):
        cleaned = ts.strip()
        if not cleaned:
            raise ValueError("Timestamp string is empty")

        # Normalize 'Z' suffix to '+00:00' for ISO parsing
        if cleaned.endswith("Z"):
            cleaned = cleaned[:-1] + "+00:00"

        # Try ISO format directly
        try:
            dt = datetime.fromisoformat(cleaned)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=UTC)
            return dt.astimezone(UTC)
        except ValueError:
            pass

        # Try common fallback patterns via strptime
        fallback_formats = [
            "%Y-%m-%d %H:%M:%S.%f%z",
            "%Y-%m-%d %H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%d/%m/%Y %H:%M:%S",
            "%d-%m-%Y %H:%M:%S",
        ]
        for fmt in fallback_formats:
            try:
                dt = datetime.strptime(cleaned, fmt)
                if dt.tzinfo is None:
                    return dt.replace(tzinfo=UTC)
                return dt.astimezone(UTC)
            except ValueError:
                continue

        # Try numeric string (epoch timestamp in string form)
        try:
            num_val = float(cleaned)
            return normalize_timestamp(num_val)
        except (ValueError, TypeError):
            pass

        raise ValueError(f"Unable to parse timestamp string: {ts!r}")

    raise TypeError(f"Unsupported timestamp type: {type(ts).__name__}")


def normalize_time_step(step: int | float | str) -> int:
    """Validate and normalize a time_step index into [1, 49].

    Args:
        step: Raw time_step input (int, float, or string).

    Returns:
        Integer time_step in range [1, 49].

    Raises:
        ValueError: If step is not in [1, 49] or cannot be parsed as integer.
        TypeError: If step is None.
    """
    if step is None:
        raise TypeError("time_step cannot be None")

    try:
        val = int(step)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Cannot cast time_step to integer: {step!r}") from exc

    if not (1 <= val <= 49):
        raise ValueError(f"time_step must be between 1 and 49, got {val}")

    return val
