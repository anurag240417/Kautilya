"""Supervised classifier for transaction/wallet classification.

Must train only on interpretable Elliptic++ features for the
primary classifier that feeds human-facing explanations.
Uses temporal train/test splits based on time_step.

See ARCHITECTURE.md §ml/ and AGENTS.md §7.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from backend.domain.experiment import ExperimentConfig

logger = logging.getLogger(__name__)


class TransactionClassifier:
    """Random-forest-based transaction classifier.

    Designed for the M1/M2/M3 experimental progression defined in
    CONTEXT.md. Uses ``class_weight="balanced"`` by default since
    illicit transactions are the minority class.

    Attributes:
        model_version: Unique identifier for this model instance.
        feature_set_name: Feature configuration (M1/M2/M3).
        feature_names: List of feature column names (set after training).
        is_fitted: Whether the model has been trained.
    """

    def __init__(
        self,
        model_version: str | None = None,
        feature_set_name: str = "M1",
        n_estimators: int = 200,
        random_state: int = 42,
        class_weight: str = "balanced",
    ) -> None:
        """Initialize the classifier.

        Args:
            model_version: Unique model identifier. Auto-generated if None.
            feature_set_name: Feature set name (M1/M2/M3).
            n_estimators: Number of trees in the random forest.
            random_state: Random seed for reproducibility.
            class_weight: Class weighting strategy.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self.model_version = model_version or f"rf_{feature_set_name}_v1_{timestamp}"
        self.feature_set_name = feature_set_name
        self.feature_names: list[str] = []
        self.is_fitted: bool = False

        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=-1,
        )
        self._n_estimators = n_estimators
        self._random_state = random_state
        self._class_weight = class_weight

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> dict:
        """Train the classifier.

        Args:
            X_train: Training feature matrix of shape (n_samples, n_features).
            y_train: Training target vector of shape (n_samples,).
            feature_names: Optional list of feature column names.
                If provided, stored for later validation and importance mapping.

        Returns:
            Dictionary of training metrics (accuracy, precision, recall, f1).

        Raises:
            ValueError: If inputs are empty or shapes don't match.
        """
        if len(X_train) == 0:
            msg = "Training data is empty"
            raise ValueError(msg)
        if len(X_train) != len(y_train):
            msg = f"Shape mismatch: X_train has {len(X_train)} rows, y_train has {len(y_train)}"
            raise ValueError(msg)

        if feature_names is not None:
            if len(feature_names) != X_train.shape[1]:
                msg = (
                    f"feature_names length ({len(feature_names)}) does not match "
                    f"X_train columns ({X_train.shape[1]})"
                )
                raise ValueError(msg)
            self.feature_names = list(feature_names)

        logger.info(
            "Training %s classifier: %d samples, %d features",
            self.feature_set_name,
            X_train.shape[0],
            X_train.shape[1],
        )

        self._model.fit(X_train, y_train)
        self.is_fitted = True

        # Compute training metrics
        y_pred = self._model.predict(X_train)
        metrics = _compute_metrics(y_train, y_pred, y_prob=None, prefix="train")

        logger.info(
            "Training complete: accuracy=%.4f, f1=%.4f, precision=%.4f, recall=%.4f",
            metrics["train_accuracy"],
            metrics["train_f1"],
            metrics["train_precision"],
            metrics["train_recall"],
        )

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of predicted class labels (0 or 1).

        Raises:
            RuntimeError: If the model has not been trained.
        """
        self._check_fitted()
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples, 2) with probabilities for
            [class_0, class_1].

        Raises:
            RuntimeError: If the model has not been trained.
        """
        self._check_fitted()
        return self._model.predict_proba(X)

    def feature_importances(self) -> dict[str, float]:
        """Return feature importance mapping.

        Returns:
            Dictionary mapping feature names (or indices if names unavailable)
            to their importance scores, sorted by descending importance.

        Raises:
            RuntimeError: If the model has not been trained.
        """
        self._check_fitted()
        importances = self._model.feature_importances_

        if self.feature_names:
            names = self.feature_names
        else:
            names = [f"feature_{i}" for i in range(len(importances))]

        importance_dict = dict(zip(names, importances))
        # Sort by descending importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate the classifier on test data.

        Computes accuracy, precision, recall, F1, ROC-AUC, and
        confusion matrix using temporal test data.

        Args:
            X_test: Test feature matrix.
            y_test: Test target vector.

        Returns:
            Dictionary of test metrics.

        Raises:
            RuntimeError: If the model has not been trained.
            ValueError: If inputs are empty.
        """
        self._check_fitted()

        if len(X_test) == 0:
            msg = "Test data is empty"
            raise ValueError(msg)

        y_pred = self._model.predict(X_test)
        y_prob = self._model.predict_proba(X_test)[:, 1]

        metrics = _compute_metrics(y_test, y_pred, y_prob, prefix="test")

        logger.info(
            "Evaluation: accuracy=%.4f, f1=%.4f, precision=%.4f, recall=%.4f, roc_auc=%.4f",
            metrics["test_accuracy"],
            metrics["test_f1"],
            metrics["test_precision"],
            metrics["test_recall"],
            metrics["test_roc_auc"],
        )

        return metrics

    def save(
        self,
        models_dir: Path,
        experiment_config: ExperimentConfig | None = None,
        test_metrics: dict | None = None,
    ) -> Path:
        """Save the model and metadata to disk.

        Creates two files under ``models_dir``:
        - ``{model_version}.joblib`` — serialized model
        - ``{model_version}_metadata.json`` — provenance metadata

        Args:
            models_dir: Directory to save model artifacts.
            experiment_config: Experiment configuration for provenance.
            test_metrics: Optional test metrics to store in metadata.

        Returns:
            Path to the saved model file (.joblib).

        Raises:
            RuntimeError: If the model has not been trained.
        """
        self._check_fitted()
        models_dir.mkdir(parents=True, exist_ok=True)

        model_path = models_dir / f"{self.model_version}.joblib"
        metadata_path = models_dir / f"{self.model_version}_metadata.json"

        # Save model
        joblib.dump(self._model, model_path)
        logger.info("Model saved to %s", model_path)

        # Build metadata
        metadata = {
            "model_version": self.model_version,
            "feature_set_name": self.feature_set_name,
            "feature_names": self.feature_names,
            "n_features": len(self.feature_names) if self.feature_names else self._model.n_features_in_,
            "n_estimators": self._n_estimators,
            "random_state": self._random_state,
            "class_weight": self._class_weight,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        if experiment_config is not None:
            metadata["experiment"] = experiment_config.model_dump()

        if test_metrics is not None:
            metadata["test_metrics"] = _serialize_metrics(test_metrics)

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Metadata saved to %s", metadata_path)

        return model_path

    @classmethod
    def load(cls, model_path: Path) -> "TransactionClassifier":
        """Load a saved model and its metadata.

        Args:
            model_path: Path to the ``.joblib`` model file.

        Returns:
            A fitted ``TransactionClassifier`` instance.

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

        # Load metadata
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Create instance
        instance = cls(
            model_version=metadata["model_version"],
            feature_set_name=metadata.get("feature_set_name", "unknown"),
            n_estimators=metadata.get("n_estimators", 200),
            random_state=metadata.get("random_state", 42),
            class_weight=metadata.get("class_weight", "balanced"),
        )

        # Load the fitted model
        instance._model = joblib.load(model_path)
        instance.feature_names = metadata.get("feature_names", [])
        instance.is_fitted = True

        logger.info(
            "Loaded model %s (%d features, %s)",
            instance.model_version,
            len(instance.feature_names),
            instance.feature_set_name,
        )

        return instance

    def _check_fitted(self) -> None:
        """Raise if the model has not been trained."""
        if not self.is_fitted:
            msg = "Model has not been trained. Call train() first."
            raise RuntimeError(msg)


# --- Private helpers ---


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    prefix: str,
) -> dict:
    """Compute classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities for the positive class (optional).
        prefix: Metric name prefix (e.g. 'train', 'test').

    Returns:
        Dictionary of named metrics.
    """
    metrics = {
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_prob is not None:
        try:
            metrics[f"{prefix}_roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            # ROC-AUC is undefined if only one class is present
            metrics[f"{prefix}_roc_auc"] = None

    return metrics


def _serialize_metrics(metrics: dict) -> dict:
    """Ensure all metric values are JSON-serializable."""
    result = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            result[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating)):
            result[k] = v.item()
        else:
            result[k] = v
    return result
