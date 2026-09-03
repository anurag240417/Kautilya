"""Model inference pipeline.

Loads saved/versioned model artifacts from the ``models/`` directory
and performs prediction on new data. Must not trigger training.

See ARCHITECTURE.md §ml/ and AGENTS.md §7.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from backend.config import get_settings
from backend.domain.ml import MLScore
from backend.ml.anomaly import AnomalyDetector
from backend.ml.classifier import TransactionClassifier

logger = logging.getLogger(__name__)


def find_latest_model(models_dir: Path, prefix: str) -> Path | None:
    """Find the most recently created model artifact with a given prefix.

    Args:
        models_dir: Directory containing model artifacts (.joblib).
        prefix: Model version prefix (e.g. 'rf_M1', 'iforest').

    Returns:
        Path to the latest .joblib file, or None if not found.
    """
    if not models_dir.exists():
        return None

    candidates = sorted(
        models_dir.glob(f"{prefix}*.joblib"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_classifier(
    models_dir: Path | None = None,
    feature_set_name: str = "M1",
) -> TransactionClassifier:
    """Load the latest trained classifier for a given feature set.

    Args:
        models_dir: Directory containing model files. Defaults to settings.
        feature_set_name: Feature configuration ('M1', 'M2', 'M3').

    Returns:
        Fitted TransactionClassifier.

    Raises:
        FileNotFoundError: If no matching model is found.
    """
    settings = get_settings()
    target_dir = models_dir or settings.models_dir
    model_path = find_latest_model(target_dir, f"rf_{feature_set_name}")

    if model_path is None:
        msg = f"No saved classifier found for feature set '{feature_set_name}' in {target_dir}"
        raise FileNotFoundError(msg)

    logger.info("Loading classifier from %s", model_path)
    return TransactionClassifier.load(model_path)


def load_anomaly_detector(
    models_dir: Path | None = None,
) -> AnomalyDetector:
    """Load the latest trained anomaly detector.

    Args:
        models_dir: Directory containing model files. Defaults to settings.

    Returns:
        Fitted AnomalyDetector.

    Raises:
        FileNotFoundError: If no anomaly detector model is found.
    """
    settings = get_settings()
    target_dir = models_dir or settings.models_dir
    model_path = find_latest_model(target_dir, "iforest")

    if model_path is None:
        msg = f"No saved anomaly detector found in {target_dir}"
        raise FileNotFoundError(msg)

    logger.info("Loading anomaly detector from %s", model_path)
    return AnomalyDetector.load(model_path)


class MLInferencePipeline:
    """End-to-end inference pipeline for scoring transactions.

    Handles alignment of input features against model requirements,
    automatic missing value imputation using training-set medians,
    supervised classification, and unsupervised anomaly scoring.
    """

    def __init__(
        self,
        classifier: TransactionClassifier,
        anomaly_detector: AnomalyDetector | None = None,
    ) -> None:
        """Initialize the inference pipeline.

        Args:
            classifier: Fitted TransactionClassifier.
            anomaly_detector: Optional fitted AnomalyDetector.
        """
        self.classifier = classifier
        self.anomaly_detector = anomaly_detector

    def _prepare_vector(
        self,
        features: dict[str, float] | pd.Series,
        expected_feature_names: list[str],
        feature_medians: dict[str, float],
    ) -> np.ndarray:
        """Convert a feature mapping into a 2D numpy array with imputation.

        Missing features and NaNs are imputed using the recorded training medians.
        """
        if isinstance(features, pd.Series):
            feature_dict = features.to_dict()
        else:
            feature_dict = dict(features)

        row = []
        for name in expected_feature_names:
            val = feature_dict.get(name)
            if val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)):
                val = feature_medians.get(name, 0.0)
            row.append(float(val))

        return np.array([row], dtype=np.float64)

    def predict_transaction(
        self,
        features: dict[str, float] | pd.Series,
        txid: str | int = "unknown",
        contains_synthetic_input: bool = False,
    ) -> MLScore:
        """Score a single transaction.

        Args:
            features: Dictionary or Series of feature names to values.
            txid: Identifier for the transaction.
            contains_synthetic_input: Whether synthetic data (e.g. network) was used.

        Returns:
            Structured MLScore domain object.
        """
        # Prepare feature vector for classifier
        clf_names = self.classifier.feature_names
        clf_medians = self.classifier.feature_medians
        X_clf = self._prepare_vector(features, clf_names, clf_medians)

        # Classification prediction & probability
        pred_label = int(self.classifier.predict(X_clf)[0])
        probabilities = self.classifier.predict_proba(X_clf)[0]
        # Confidence is the probability of the predicted class
        confidence = float(probabilities[pred_label])

        # Anomaly score if anomaly detector is available
        anomaly_score: float | None = None
        if self.anomaly_detector is not None:
            anom_names = self.anomaly_detector.feature_names or [
                c for c in clf_names if c.startswith(("Local_feature_", "Aggregate_feature_"))
            ]
            anom_medians = self.anomaly_detector.feature_medians or clf_medians
            X_anom = self._prepare_vector(features, anom_names, anom_medians)
            scores = self.anomaly_detector.score(X_anom)
            anomaly_score = float(scores[0])

        return MLScore(
            entity_id=str(txid),
            entity_type="transaction",
            prediction=pred_label,
            probability=confidence,
            anomaly_score=anomaly_score,
            model_version=self.classifier.model_version,
            feature_set=self.classifier.feature_set_name,
            contains_synthetic_input=contains_synthetic_input,
        )
