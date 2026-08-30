"""Data validator for ingested records.

Validates parsed DataFrames against expected constraints before they
enter the normalization pipeline. Catches malformed, missing, or
duplicate records.

Validation does NOT modify the data — it reports findings.
"""

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating a parsed DataFrame.

    Attributes:
        is_valid: True if no errors were found.
        errors: Critical issues that must be fixed before processing.
        warnings: Non-critical issues (e.g. duplicates).
        stats: Informational statistics (row count, nulls, distributions).
    """

    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def add_error(self, msg: str) -> None:
        """Record a validation error and mark result as invalid."""
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        """Record a non-critical warning."""
        self.warnings.append(msg)


def _check_required_columns(
    df: pd.DataFrame,
    required: list[str],
    result: ValidationResult,
) -> None:
    """Check that all required columns are present."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        result.add_error(f"Missing required columns: {missing}")


def _check_no_null_primary_key(
    df: pd.DataFrame,
    key_col: str,
    result: ValidationResult,
) -> None:
    """Check that the primary key column has no null values."""
    null_count = df[key_col].isna().sum()
    if null_count > 0:
        result.add_error(f"Found {null_count:,} null values in primary key '{key_col}'")


def _check_duplicate_primary_key(
    df: pd.DataFrame,
    key_col: str,
    result: ValidationResult,
) -> None:
    """Check for duplicate primary keys — warning, not error."""
    dupe_count = df[key_col].duplicated().sum()
    if dupe_count > 0:
        result.add_warning(f"Found {dupe_count:,} duplicate values in '{key_col}'")


def _check_value_range(
    df: pd.DataFrame,
    col: str,
    valid_values: set,
    result: ValidationResult,
) -> None:
    """Check that all values in a column are within the valid set."""
    invalid = df[~df[col].isin(valid_values)]
    if len(invalid) > 0:
        result.add_error(
            f"Found {len(invalid):,} invalid values in '{col}'. "
            f"Expected one of {valid_values}, got: {set(invalid[col].unique())}"
        )


def _check_numeric_range(
    df: pd.DataFrame,
    col: str,
    min_val: int | float,
    max_val: int | float,
    result: ValidationResult,
) -> None:
    """Check that numeric values fall within [min_val, max_val]."""
    below = df[df[col] < min_val]
    above = df[df[col] > max_val]
    if len(below) > 0:
        result.add_error(
            f"Found {len(below):,} values in '{col}' below {min_val} (min found: {df[col].min()})"
        )
    if len(above) > 0:
        result.add_error(
            f"Found {len(above):,} values in '{col}' above {max_val} (max found: {df[col].max()})"
        )


# --- Public validator functions ---


def validate_txs_features(df: pd.DataFrame) -> ValidationResult:
    """Validate a parsed txs_features DataFrame.

    Checks:
        - Required columns present (txId, time_step)
        - No null txIds
        - time_step range [1, 49]
        - Reports row count and feature statistics
    """
    result = ValidationResult()

    _check_required_columns(df, ["txId", "time_step"], result)
    if not result.is_valid:
        return result

    _check_no_null_primary_key(df, "txId", result)
    _check_duplicate_primary_key(df, "txId", result)
    _check_numeric_range(df, "time_step", 1, 49, result)

    # Stats
    result.stats["row_count"] = len(df)
    result.stats["column_count"] = len(df.columns)
    result.stats["null_txId_count"] = int(df["txId"].isna().sum())

    if "time_step" in df.columns:
        ts = df["time_step"]
        result.stats["time_step_min"] = int(ts.min())
        result.stats["time_step_max"] = int(ts.max())
        result.stats["time_step_unique"] = int(ts.nunique())

    logger.info(
        "Validated txs_features: %d rows, valid=%s, errors=%d, warnings=%d",
        len(df),
        result.is_valid,
        len(result.errors),
        len(result.warnings),
    )
    return result


def validate_txs_classes(df: pd.DataFrame) -> ValidationResult:
    """Validate a parsed txs_classes DataFrame.

    Checks:
        - Required columns present (txId, class)
        - No null txIds
        - class values in {1, 2, 3}
        - Reports class distribution
    """
    result = ValidationResult()

    _check_required_columns(df, ["txId", "class"], result)
    if not result.is_valid:
        return result

    _check_no_null_primary_key(df, "txId", result)
    _check_duplicate_primary_key(df, "txId", result)
    _check_value_range(df, "class", {1, 2, 3}, result)

    # Stats
    result.stats["row_count"] = len(df)
    if "class" in df.columns:
        dist = df["class"].value_counts().sort_index().to_dict()
        result.stats["class_distribution"] = dist

    logger.info(
        "Validated txs_classes: %d rows, valid=%s",
        len(df),
        result.is_valid,
    )
    return result


def validate_wallets_features(df: pd.DataFrame) -> ValidationResult:
    """Validate a parsed wallets_features DataFrame.

    Checks:
        - Required columns present (address, time_step)
        - No null addresses
        - Duplicate addresses are expected (same address at different time_steps)
        - Reports row count and unique address count
    """
    result = ValidationResult()

    _check_required_columns(df, ["address", "time_step"], result)
    if not result.is_valid:
        return result

    _check_no_null_primary_key(df, "address", result)
    # Duplicates are expected — same address at different time_steps
    dupe_count = df["address"].duplicated().sum()
    if dupe_count > 0:
        result.stats["duplicate_address_rows"] = int(dupe_count)
        result.stats["note"] = "Duplicate addresses expected — same address at multiple time_steps"

    # Stats
    result.stats["row_count"] = len(df)
    result.stats["column_count"] = len(df.columns)
    result.stats["unique_addresses"] = int(df["address"].nunique())

    logger.info(
        "Validated wallets_features: %d rows, %d unique addresses, valid=%s",
        len(df),
        result.stats["unique_addresses"],
        result.is_valid,
    )
    return result


def validate_wallets_classes(df: pd.DataFrame) -> ValidationResult:
    """Validate a parsed wallets_classes DataFrame.

    Checks:
        - Required columns present (address, class)
        - No null addresses
        - class values in {1, 2, 3}
        - Reports class distribution
    """
    result = ValidationResult()

    _check_required_columns(df, ["address", "class"], result)
    if not result.is_valid:
        return result

    _check_no_null_primary_key(df, "address", result)
    _check_duplicate_primary_key(df, "address", result)
    _check_value_range(df, "class", {1, 2, 3}, result)

    # Stats
    result.stats["row_count"] = len(df)
    if "class" in df.columns:
        dist = df["class"].value_counts().sort_index().to_dict()
        result.stats["class_distribution"] = dist

    logger.info(
        "Validated wallets_classes: %d rows, valid=%s",
        len(df),
        result.is_valid,
    )
    return result


def validate_edgelist(df: pd.DataFrame, name: str) -> ValidationResult:
    """Validate a parsed edgelist DataFrame.

    Checks:
        - Exactly 2 columns
        - No null values in either column
        - Reports row count and duplicate edge count

    Args:
        df: Parsed edgelist DataFrame.
        name: Human-readable name for logging (e.g. "txs_edgelist").
    """
    result = ValidationResult()

    if len(df.columns) != 2:
        result.add_error(f"{name}: Expected 2 columns, got {len(df.columns)}: {list(df.columns)}")
        return result

    col1, col2 = df.columns[0], df.columns[1]

    null_col1 = int(df[col1].isna().sum())
    null_col2 = int(df[col2].isna().sum())
    if null_col1 > 0:
        result.add_error(f"{name}: {null_col1:,} null values in '{col1}'")
    if null_col2 > 0:
        result.add_error(f"{name}: {null_col2:,} null values in '{col2}'")

    # Duplicate edges
    dupe_count = int(df.duplicated().sum())
    if dupe_count > 0:
        result.add_warning(f"{name}: {dupe_count:,} duplicate edges")

    # Stats
    result.stats["row_count"] = len(df)
    result.stats["duplicate_edges"] = dupe_count
    result.stats["null_col1"] = null_col1
    result.stats["null_col2"] = null_col2

    logger.info(
        "Validated %s: %d edges, valid=%s",
        name,
        len(df),
        result.is_valid,
    )
    return result
