"""Temporal train/test splitting for ML experiments.

Implements time_step-based splitting as mandated by CONTEXT.md:
    "ML evaluation must use temporal splits based on time_step;
     earlier steps are used for training and later steps for testing."

Random shuffling across time steps is a data-leakage defect.
"""

import logging

import pandas as pd

from backend.domain.experiment import ExperimentConfig
from backend.ml.feature_registry import (
    BINARY_TARGET_MAP,
    TRAIN_LABELS,
)

logger = logging.getLogger(__name__)

# Default split point: train on steps 1–34, test on steps 35–49.
# This gives roughly a 70/30 temporal split across 49 steps.
DEFAULT_TRAIN_CUTOFF: int = 34


def temporal_train_test_split(
    df: pd.DataFrame,
    train_cutoff: int = DEFAULT_TRAIN_CUTOFF,
    time_col: str = "time_step",
    label_col: str = "label",
    experiment_id: str = "default",
    dataset_version: str = "elliptic_pp_v1",
    feature_configuration: str = "M1",
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, ExperimentConfig]:
    """Split a DataFrame temporally by time_step.

    Rows with time_step <= train_cutoff go to training.
    Rows with time_step > train_cutoff go to testing.

    Only rows with labels in TRAIN_LABELS (Illicit, Licit) are kept.
    Unknown labels are dropped from both sets.

    A binary target column ``target`` is added:
        Illicit (1) → 1, Licit (2) → 0.

    Args:
        df: DataFrame containing at least ``time_col`` and ``label_col``.
        train_cutoff: Last time_step included in training (default 34).
        time_col: Name of the time step column.
        label_col: Name of the label column.
        experiment_id: Unique experiment identifier.
        dataset_version: Dataset version string.
        feature_configuration: Feature set identifier (M1/M2/M3).
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, test_df, ExperimentConfig).

    Raises:
        ValueError: If time_col or label_col is missing, or if
            the split produces empty train/test sets.
    """
    if time_col not in df.columns:
        msg = f"Column '{time_col}' not found in DataFrame"
        raise ValueError(msg)
    if label_col not in df.columns:
        msg = f"Column '{label_col}' not found in DataFrame"
        raise ValueError(msg)

    # Filter to supervised labels only (drop Unknown)
    supervised_df = df[df[label_col].isin(TRAIN_LABELS)].copy()

    if len(supervised_df) == 0:
        msg = "No rows with supervised labels (Illicit/Licit) found"
        raise ValueError(msg)

    # Add binary target column
    supervised_df["target"] = supervised_df[label_col].map(BINARY_TARGET_MAP)

    # Split temporally
    train_steps = list(range(1, train_cutoff + 1))
    test_steps = list(range(train_cutoff + 1, 50))

    train_df = supervised_df[supervised_df[time_col] <= train_cutoff].copy()
    test_df = supervised_df[supervised_df[time_col] > train_cutoff].copy()

    if len(train_df) == 0:
        msg = f"Empty training set with cutoff={train_cutoff}"
        raise ValueError(msg)
    if len(test_df) == 0:
        msg = f"Empty test set with cutoff={train_cutoff}"
        raise ValueError(msg)

    # Validate no temporal overlap
    train_max = train_df[time_col].max()
    test_min = test_df[time_col].min()
    if train_max >= test_min:
        msg = (
            f"Temporal leakage detected: train max time_step={train_max} "
            f">= test min time_step={test_min}"
        )
        raise ValueError(msg)

    config = ExperimentConfig(
        experiment_id=experiment_id,
        dataset_version=dataset_version,
        feature_configuration=feature_configuration,
        split_strategy="temporal",
        random_seed=random_seed,
        train_time_steps=train_steps,
        test_time_steps=test_steps,
    )

    logger.info(
        "Temporal split: train=%d rows (steps 1–%d), test=%d rows (steps %d–49)",
        len(train_df),
        train_cutoff,
        len(test_df),
        train_cutoff + 1,
    )

    return train_df, test_df, config
