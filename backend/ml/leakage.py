"""Data leakage audit for ML experiments.

Checks for three categories of leakage:

1. **Temporal leakage**: Train and test time_steps must not overlap.
2. **Feature leakage**: Target/identity columns must not appear in X.
3. **Target leakage**: Training target must contain only valid binary
   values (0 or 1) — no Unknown labels.

See AGENTS.md §7 and CONTEXT.md — Experimental Progression.
"""

import logging
from dataclasses import dataclass, field

import pandas as pd

from backend.ml.feature_registry import FORBIDDEN_FEATURE_COLUMNS

logger = logging.getLogger(__name__)


@dataclass
class LeakageAuditResult:
    """Result of a leakage audit."""

    passed: bool = True
    checks: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def add_pass(self, check: str) -> None:
        """Record a passed check."""
        self.checks.append(f"PASS: {check}")

    def add_fail(self, check: str) -> None:
        """Record a failed check."""
        self.passed = False
        self.checks.append(f"FAIL: {check}")
        self.failures.append(check)


def audit_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    time_col: str = "time_step",
    target_col: str = "target",
) -> LeakageAuditResult:
    """Run a comprehensive leakage audit on train/test data.

    Args:
        train_df: Training DataFrame.
        test_df: Testing DataFrame.
        feature_columns: List of column names used as features (X).
        time_col: Name of the time step column.
        target_col: Name of the binary target column.

    Returns:
        A ``LeakageAuditResult`` with pass/fail status and explanations.
    """
    result = LeakageAuditResult()

    # ------------------------------------------------------------------
    # 1. Temporal leakage: no time_step overlap between train and test
    # ------------------------------------------------------------------
    if time_col in train_df.columns and time_col in test_df.columns:
        train_steps = set(train_df[time_col].unique())
        test_steps = set(test_df[time_col].unique())
        overlap = train_steps & test_steps

        if overlap:
            result.add_fail(f"Temporal leakage: train/test share time_steps {sorted(overlap)}")
        else:
            result.add_pass("No temporal leakage (zero time_step overlap)")

        # Also check ordering: max(train) must be < min(test)
        train_max = train_df[time_col].max()
        test_min = test_df[time_col].min()
        if train_max >= test_min:
            result.add_fail(
                f"Temporal ordering violated: train max={train_max} >= test min={test_min}"
            )
        else:
            result.add_pass(
                f"Temporal ordering valid: train max={train_max} < test min={test_min}"
            )
    else:
        result.add_fail(f"Column '{time_col}' missing from train or test DataFrame")

    # ------------------------------------------------------------------
    # 2. Feature leakage: forbidden columns must not be in feature set
    # ------------------------------------------------------------------
    feature_set = set(feature_columns)
    leaked_cols = feature_set & FORBIDDEN_FEATURE_COLUMNS

    if leaked_cols:
        result.add_fail(
            f"Feature leakage: forbidden columns in feature set: {sorted(leaked_cols)}"
        )
    else:
        result.add_pass("No forbidden columns in feature set")

    # ------------------------------------------------------------------
    # 3. Target leakage: training target must be binary (0 or 1 only)
    # ------------------------------------------------------------------
    if target_col in train_df.columns:
        unique_targets = set(train_df[target_col].dropna().unique())
        valid_targets = {0, 1}

        if not unique_targets.issubset(valid_targets):
            invalid = unique_targets - valid_targets
            result.add_fail(
                f"Target leakage: invalid values in training target: {sorted(invalid)}. "
                "Only 0 (Licit) and 1 (Illicit) are permitted."
            )
        else:
            result.add_pass("Training target contains only valid binary values (0, 1)")

        # Check for NaN targets
        null_count = train_df[target_col].isna().sum()
        if null_count > 0:
            result.add_fail(f"Target leakage: {null_count} null values in training target")
        else:
            result.add_pass("No null values in training target")
    else:
        result.add_fail(f"Column '{target_col}' missing from training DataFrame")

    if result.passed:
        logger.info("Leakage audit PASSED: all %d checks clean", len(result.checks))
    else:
        logger.warning(
            "Leakage audit FAILED: %d failures out of %d checks",
            len(result.failures),
            len(result.checks),
        )

    return result
