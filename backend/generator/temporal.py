"""Deterministic time_step → synthetic datetime mapping.

Elliptic++ provides a coarse 49-step ``time_step`` index (1–49)
without real-world timestamps. This module maps each step to a
synthetic datetime window so that network observations can be
generated with meaningful timestamps for temporal correlation.

The mapping spans roughly two years (Jan 2019 – Dec 2020),
with each step covering approximately two weeks.

This mapping is SYNTHETIC and does NOT represent the actual
historical dates of the Elliptic++ transactions.
"""

from datetime import UTC, datetime, timedelta

# Anchor date for time_step=1.
_EPOCH = datetime(2019, 1, 7, 0, 0, 0, tzinfo=UTC)

# Duration of each time step (~2 weeks).
_STEP_DURATION = timedelta(days=14)

# Valid step range from the Elliptic++ dataset.
_MIN_STEP = 1
_MAX_STEP = 49


def _validate_step(step: int) -> None:
    """Validate that a time step is within [1, 49]."""
    if not (_MIN_STEP <= step <= _MAX_STEP):
        msg = f"time_step must be between {_MIN_STEP} and {_MAX_STEP}, got {step}"
        raise ValueError(msg)


def time_step_to_datetime(step: int) -> datetime:
    """Map a time_step index to its synthetic start datetime.

    Args:
        step: Elliptic++ time step (1–49).

    Returns:
        The synthetic start datetime for this step (UTC).

    Raises:
        ValueError: If step is outside [1, 49].
    """
    _validate_step(step)
    return _EPOCH + _STEP_DURATION * (step - 1)


def time_step_to_datetime_range(step: int) -> tuple[datetime, datetime]:
    """Map a time_step to its full synthetic datetime window.

    Returns the [start, end) interval for the step. The end of
    one step equals the start of the next.

    Args:
        step: Elliptic++ time step (1–49).

    Returns:
        Tuple of (start_datetime, end_datetime) in UTC.

    Raises:
        ValueError: If step is outside [1, 49].
    """
    _validate_step(step)
    start = _EPOCH + _STEP_DURATION * (step - 1)
    end = start + _STEP_DURATION
    return start, end


def get_all_step_mappings() -> dict[int, datetime]:
    """Return the complete mapping of all 49 time steps to datetimes.

    Returns:
        Dictionary mapping step (1–49) to its synthetic start datetime.
    """
    return {step: time_step_to_datetime(step) for step in range(_MIN_STEP, _MAX_STEP + 1)}


def datetime_to_time_step(dt: datetime) -> int | None:
    """Map a synthetic datetime back to its Elliptic++ time_step index (1–49).

    Args:
        dt: UTC datetime to map. If naive, assumed to be UTC.

    Returns:
        Integer time_step in [1, 49] if within the synthetic epoch range,
        or None if outside [step 1 start, step 49 end).
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)

    offset_seconds = (dt - _EPOCH).total_seconds()
    step_duration_seconds = _STEP_DURATION.total_seconds()

    step = 1 + int(offset_seconds // step_duration_seconds)
    if _MIN_STEP <= step <= _MAX_STEP:
        return step
    return None
