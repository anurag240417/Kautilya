"""Dataset preparation for ML experiments.

Loads Elliptic++ transaction features and labels, joins them,
and produces train/test splits via temporal splitting.

The pipeline enforces leakage auditing before returning data.
See AGENTS.md §7 and CONTEXT.md — Experimental Progression.
"""

import logging

import numpy as np
import pandas as pd

from backend.config import Settings, get_settings
from backend.domain.experiment import ExperimentConfig
from backend.ingestion.csv_parser import parse_txs_classes, parse_txs_features
from backend.ml.feature_registry import FORBIDDEN_FEATURE_COLUMNS
from backend.ml.leakage import audit_leakage
from backend.ml.splitting import temporal_train_test_split

logger = logging.getLogger(__name__)


def load_transaction_dataset(settings: Settings | None = None) -> pd.DataFrame:
    """Load and join Elliptic++ transaction features with labels.

    Reads ``txs_features.csv`` and ``txs_classes.csv``, joins on
    ``txId``, and renames the ``class`` column to ``label`` to avoid
    clashes with Python's ``class`` keyword and with the forbidden-
    feature-column list.

    Args:
        settings: Application settings. Uses defaults if not provided.

    Returns:
        DataFrame with columns: ``txId``, ``time_step``, 182 feature
        columns, and ``label`` (1=Illicit, 2=Licit, 3=Unknown).

    Raises:
        FileNotFoundError: If dataset files are missing.
        ValueError: If the join produces an empty result.
    """
    if settings is None:
        settings = get_settings()

    logger.info("Loading transaction features from %s", settings.txs_features_path)
    features_df = parse_txs_features(settings.txs_features_path)

    logger.info("Loading transaction classes from %s", settings.txs_classes_path)
    classes_df = parse_txs_classes(settings.txs_classes_path)

    # Rename 'class' → 'label' to avoid confusion and forbidden-column issues
    classes_df = classes_df.rename(columns={"class": "label"})

    # Inner join: only keep transactions that have both features and labels
    merged = features_df.merge(classes_df, on="txId", how="inner")

    if len(merged) == 0:
        msg = "No transactions matched between features and classes files"
        raise ValueError(msg)

    logger.info(
        "Loaded %d transactions (%d illicit, %d licit, %d unknown)",
        len(merged),
        (merged["label"] == 1).sum(),
        (merged["label"] == 2).sum(),
        (merged["label"] == 3).sum(),
    )

    return merged


def prepare_ml_splits(
    df: pd.DataFrame,
    feature_cols: list[str],
    train_cutoff: int = 34,
    experiment_id: str = "default",
    dataset_version: str = "elliptic_pp_v1",
    feature_configuration: str = "M1",
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ExperimentConfig]:
    """Prepare train/test splits with leakage auditing.

    1. Validates that all requested feature columns exist in the DataFrame.
    2. Performs temporal train/test split via ``temporal_train_test_split``.
    3. Runs ``audit_leakage`` to catch temporal, feature, and target leakage.
    4. Handles NaN values in features via median imputation.
    5. Extracts X (feature matrix) and y (target vector) arrays.

    Args:
        df: DataFrame from ``load_transaction_dataset``.
        feature_cols: List of column names to use as features.
        train_cutoff: Last time_step in training (default 34).
        experiment_id: Experiment identifier.
        dataset_version: Dataset version string.
        feature_configuration: Feature set name (M1/M2/M3).
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of ``(X_train, y_train, X_test, y_test, ExperimentConfig)``.

    Raises:
        ValueError: If feature columns are missing, leakage is detected,
            or splits are empty.
    """
    # --- 1. Validate feature columns exist ---
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        msg = f"Missing feature columns in DataFrame: {missing[:10]}"
        if len(missing) > 10:
            msg += f" ... and {len(missing) - 10} more"
        raise ValueError(msg)

    # --- 2. Check for forbidden columns in features ---
    forbidden_in_features = set(feature_cols) & FORBIDDEN_FEATURE_COLUMNS
    if forbidden_in_features:
        msg = f"Forbidden columns in feature list: {sorted(forbidden_in_features)}"
        raise ValueError(msg)

    # --- 3. Temporal split ---
    train_df, test_df, config = temporal_train_test_split(
        df=df,
        train_cutoff=train_cutoff,
        experiment_id=experiment_id,
        dataset_version=dataset_version,
        feature_configuration=feature_configuration,
        random_seed=random_seed,
    )

    # --- 4. Leakage audit ---
    audit_result = audit_leakage(
        train_df=train_df,
        test_df=test_df,
        feature_columns=feature_cols,
    )
    if not audit_result.passed:
        msg = "Leakage audit failed:\n" + "\n".join(audit_result.failures)
        raise ValueError(msg)

    logger.info("Leakage audit passed: %d checks clean", len(audit_result.checks))

    # --- 5. Extract feature matrices and handle NaN ---
    X_train = train_df[feature_cols].copy()
    y_train = train_df["target"].values

    X_test = test_df[feature_cols].copy()
    y_test = test_df["target"].values

    # Median imputation for NaN (some Elliptic++ features may have missing values)
    train_nan_count = X_train.isna().sum().sum()
    test_nan_count = X_test.isna().sum().sum()

    if train_nan_count > 0 or test_nan_count > 0:
        logger.warning(
            "NaN values found: %d in train, %d in test. Applying median imputation.",
            train_nan_count,
            test_nan_count,
        )
        # Compute medians from training set only (no test-set information leakage)
        train_medians = X_train.median()
        X_train = X_train.fillna(train_medians)
        X_test = X_test.fillna(train_medians)

    X_train_arr = X_train.values.astype(np.float64)
    X_test_arr = X_test.values.astype(np.float64)

    logger.info(
        "ML splits ready: X_train=%s, X_test=%s, train_pos_rate=%.3f, test_pos_rate=%.3f",
        X_train_arr.shape,
        X_test_arr.shape,
        y_train.mean(),
        y_test.mean(),
    )

    return X_train_arr, y_train, X_test_arr, y_test, config
