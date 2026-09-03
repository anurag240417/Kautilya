"""Anomaly detection.

May use the anonymized Local_feature and Aggregate_feature columns
for internal scoring, but anomaly explanations must not surface
anonymized feature names directly to users.

See ARCHITECTURE.md §ml/ and DATASET.md §7.2.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

# Human-readable anomaly severity thresholds.
# Scores are normalized to [0, 1] where higher = more anomalous.
_SEVERITY_THRESHOLDS = [
    (0.8, "Highly anomalous — behavioral pattern strongly deviates from normal"),
    (0.6, "Moderately anomalous — notable deviation from typical patterns"),
    (0.4, "Mildly anomalous — some unusual characteristics detected"),
    (0.0, "Normal — behavioral pattern within expected range"),
]


class AnomalyDetector:
    """Isolation Forest anomaly detector for transaction analysis.

    Uses anonymized features (``Local_feature_*`` and
    ``Aggregate_feature_*``) internally for scoring. Human-facing
    explanations MUST NOT expose anonymized feature names — use
    ``explain_score()`` for category-level interpretation.

    Attributes:
        model_version: Unique identifier for this model instance.
        is_fitted: Whether the model has been fitted.
    """

    def __init__(
        self,
        model_version: str | None = None,
        contamination: float = 0.1,
        n_estimators: int = 200,
        random_state: int = 42,
        feature_names: list[str] | None = None,
        feature_medians: dict[str, float] | None = None,
    ) -> None:
        """Initialize the anomaly detector.

        Args:
            model_version: Unique model identifier. Auto-generated if None.
            contamination: Expected proportion of anomalies (default 10%).
            n_estimators: Number of trees in the isolation forest.
            random_state: Random seed for reproducibility.
            feature_names: Optional list of feature column names.
            feature_medians: Optional dictionary of feature median values for imputation.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self.model_version = model_version or f"iforest_v1_{timestamp}"
        self.feature_names: list[str] = list(feature_names) if feature_names else []
        self.feature_medians: dict[str, float] = dict(feature_medians) if feature_medians else {}
        self.is_fitted: bool = False

        self._contamination = contamination
        self._n_estimators = n_estimators
        self._random_state = random_state

        self._model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X_train: np.ndarray) -> None:
        """Fit the anomaly detector on training data.

        This is unsupervised — no labels are needed.

        Args:
            X_train: Training feature matrix of shape (n_samples, n_features).

        Raises:
            ValueError: If training data is empty.
        """
        if len(X_train) == 0:
            msg = "Training data is empty"
            raise ValueError(msg)

        logger.info(
            "Fitting anomaly detector: %d samples, %d features",
            X_train.shape[0],
            X_train.shape[1],
        )

        self._model.fit(X_train)
        self.is_fitted = True

        logger.info("Anomaly detector fitted (model_version=%s)", self.model_version)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for each sample.

        Scores are normalized to [0, 1] where:
        - 1.0 = maximally anomalous
        - 0.0 = maximally normal

        Internally uses sklearn's ``decision_function`` (which returns
        negative scores for anomalies) and inverts/normalizes the result.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of anomaly scores in [0, 1], shape (n_samples,).

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._check_fitted()

        # sklearn decision_function: lower (more negative) = more anomalous
        raw_scores = self._model.decision_function(X)

        # Normalize to [0, 1] where higher = more anomalous
        return _normalize_anomaly_scores(raw_scores)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of predictions: 1 = anomaly, 0 = normal.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._check_fitted()

        # sklearn predict: -1 = anomaly, 1 = normal
        raw_preds = self._model.predict(X)
        # Convert to: 1 = anomaly, 0 = normal
        return (raw_preds == -1).astype(int)

    @staticmethod
    def explain_score(score_value: float) -> str:
        """Map an anomaly score to a human-readable explanation.

        This method provides category-level interpretation WITHOUT
        exposing anonymized feature names. It must be used instead
        of surfacing raw feature importances.

        Args:
            score_value: Anomaly score in [0, 1].

        Returns:
            Human-readable explanation string.
        """
        for threshold, explanation in _SEVERITY_THRESHOLDS:
            if score_value >= threshold:
                return explanation
        return _SEVERITY_THRESHOLDS[-1][1]

    def save(self, models_dir: Path) -> Path:
        """Save the anomaly detector and metadata to disk.

        Args:
            models_dir: Directory to save model artifacts.

        Returns:
            Path to the saved model file.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._check_fitted()
        models_dir.mkdir(parents=True, exist_ok=True)

        model_path = models_dir / f"{self.model_version}.joblib"
        metadata_path = models_dir / f"{self.model_version}_metadata.json"

        joblib.dump(self._model, model_path)
        logger.info("Anomaly model saved to %s", model_path)

        metadata = {
            "model_version": self.model_version,
            "model_type": "IsolationForest",
            "contamination": self._contamination,
            "n_estimators": self._n_estimators,
            "random_state": self._random_state,
            "feature_names": self.feature_names,
            "feature_medians": self.feature_medians,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Anomaly metadata saved to %s", metadata_path)

        return model_path

    @classmethod
    def load(cls, model_path: Path) -> "AnomalyDetector":
        """Load a saved anomaly detector.

        Args:
            model_path: Path to the ``.joblib`` model file.

        Returns:
            A fitted ``AnomalyDetector`` instance.

        Raises:
            FileNotFoundError: If model or metadata file is missing.
        """
        if not model_path.exists():
            msg = f"Model file not found: {model_path}"
            raise FileNotFoundError(msg)

        metadata_path = model_path.with_name(
            model_path.stem + "_metadata.json"
        )
        if not metadata_path.exists():
            msg = f"Metadata file not found: {metadata_path}"
            raise FileNotFoundError(msg)

        with open(metadata_path) as f:
            metadata = json.load(f)

        instance = cls(
            model_version=metadata["model_version"],
            contamination=metadata.get("contamination", 0.1),
            n_estimators=metadata.get("n_estimators", 200),
            random_state=metadata.get("random_state", 42),
            feature_names=metadata.get("feature_names", []),
            feature_medians=metadata.get("feature_medians", {}),
        )
        instance._model = joblib.load(model_path)
        instance.is_fitted = True

        logger.info("Loaded anomaly detector %s", instance.model_version)
        return instance

    def _check_fitted(self) -> None:
        """Raise if the model has not been fitted."""
        if not self.is_fitted:
            msg = "Anomaly detector has not been fitted. Call fit() first."
            raise RuntimeError(msg)


def _normalize_anomaly_scores(raw_scores: np.ndarray) -> np.ndarray:
    """Normalize sklearn decision_function scores to [0, 1].

    sklearn's IsolationForest decision_function returns values where
    more negative = more anomalous. We invert and scale to [0, 1].
    """
    # Negate so that higher = more anomalous
    inverted = -raw_scores

    score_min = inverted.min()
    score_max = inverted.max()

    if score_max - score_min < 1e-10:
        # All scores are the same — return 0.5 (indeterminate)
        return np.full_like(inverted, 0.5)

    return (inverted - score_min) / (score_max - score_min)
