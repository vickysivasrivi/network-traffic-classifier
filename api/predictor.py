"""Model loading, feature scaling, SHAP explainability, and inference logic."""

import logging
import os

import joblib
import numpy as np
import shap

from api.schemas import BatchPredictionRequest, FeatureImpact, PredictionResponse

log: logging.Logger = logging.getLogger(__name__)

THREAT_LEVELS: dict[str, str] = {
    "BENIGN": "LOW",
    "Bot": "HIGH",
    "DDoS": "HIGH",
    "DoS GoldenEye": "HIGH",
    "DoS Hulk": "HIGH",
    "DoS Slowhttptest": "HIGH",
    "DoS slowloris": "HIGH",
    "FTP-Patator": "MEDIUM",
    "Heartbleed": "CRITICAL",
    "Infiltration": "CRITICAL",
    "PortScan": "MEDIUM",
    "SSH-Patator": "MEDIUM",
    "Web Attack \ufffd Brute Force": "MEDIUM",
    "Web Attack \ufffd Sql Injection": "HIGH",
    "Web Attack \ufffd XSS": "MEDIUM",
}


class Predictor:
    """Loads all model artifacts once and exposes predict / predict_batch."""

    def __init__(
        self,
        models_dir: str = "models",
        processed_dir: str = "data/processed",
    ) -> None:
        """Load model, scaler, encoder, feature names, and SHAP explainer.

        Args:
            models_dir: Directory containing ``best_model.joblib``.
            processed_dir: Directory containing scaler and encoder artifacts.
        """
        self.model = self._load_model(models_dir)
        self.scaler = joblib.load(os.path.join(processed_dir, "scaler.joblib"))
        self.label_encoder = joblib.load(os.path.join(processed_dir, "label_encoder.joblib"))

        feature_names_path: str = os.path.join(processed_dir, "feature_names.txt")
        with open(feature_names_path, "r", encoding="utf-8") as fh:
            self.feature_names: list[str] = fh.read().splitlines()

        with open(os.path.join(models_dir, "best_model_name.txt"), "r", encoding="utf-8") as fh:
            self.model_name: str = fh.read().strip()

        log.info("Building SHAP TreeExplainer for '%s' ...", self.model_name)
        self.explainer: shap.TreeExplainer = shap.TreeExplainer(self.model)
        log.info("Predictor ready — %d features, %d classes.", len(self.feature_names), len(self.label_encoder.classes_))

    def _load_model(self, models_dir: str):
        """Try the MLflow Model Registry first; fall back to the local joblib artifact.

        Registry-first loading enables zero-downtime model updates: once
        ``src/promote.py`` registers a better model as *Production*, every new
        prediction request will pick it up without redeploying the container.

        Args:
            models_dir: Directory containing ``best_model.joblib`` as fallback.

        Returns:
            A fitted sklearn-compatible classifier with ``predict`` and
            ``predict_proba`` methods.
        """
        try:
            import mlflow
            model = mlflow.sklearn.load_model("models:/network-threat@production")
            log.info("Loaded model from MLflow Registry (models:/network-threat@production).")
            return model
        except Exception as exc:
            log.info("MLflow Registry not available (%s); loading local artifact.", exc)
            return joblib.load(os.path.join(models_dir, "best_model.joblib"))

    def _top_features(
        self,
        shap_values: list[np.ndarray] | np.ndarray,
        class_idx: int,
        n: int = 3,
    ) -> list[FeatureImpact]:
        """Extract the top-n features by absolute SHAP value for one prediction.

        Args:
            shap_values: SHAP values returned by ``TreeExplainer.shap_values``.
            class_idx: Index of the predicted class.
            n: Number of top features to return.

        Returns:
            Ordered list of :class:`FeatureImpact` from highest to lowest impact.
        """
        if isinstance(shap_values, list):
            # Multi-class: list of (n_samples, n_features), one per class
            class_shap: np.ndarray = np.array(shap_values[class_idx])[0]
        else:
            # Single array: (n_samples, n_features, n_classes) or (n_samples, n_features)
            class_shap = (
                shap_values[0, :, class_idx]
                if shap_values.ndim == 3
                else shap_values[0]
            )

        top_idx: np.ndarray = np.argsort(np.abs(class_shap))[::-1][:n]
        return [
            FeatureImpact(
                feature=self.feature_names[i],
                impact=round(float(class_shap[i]), 4),
            )
            for i in top_idx
        ]

    def predict(self, features: list[float]) -> PredictionResponse:
        """Classify one network flow and return prediction with SHAP explanation.

        Args:
            features: 70 raw feature values in ``feature_names.txt`` order.

        Returns:
            :class:`PredictionResponse` with prediction, confidence, threat level,
            and top-3 SHAP features.
        """
        X: np.ndarray = np.array([features], dtype=np.float64)
        X_scaled: np.ndarray = self.scaler.transform(X)

        class_idx: int = int(self.model.predict(X_scaled)[0])
        confidence: float = float(self.model.predict_proba(X_scaled)[0][class_idx])
        class_name: str = str(self.label_encoder.inverse_transform([class_idx])[0])
        threat_level: str = THREAT_LEVELS.get(class_name, "HIGH")

        shap_values = self.explainer.shap_values(X_scaled)
        top_features: list[FeatureImpact] = self._top_features(shap_values, class_idx)

        return PredictionResponse(
            prediction=class_name,
            confidence=round(confidence, 4),
            threat_level=threat_level,
            top_features=top_features,
        )

    def predict_batch(self, request: BatchPredictionRequest) -> list[PredictionResponse]:
        """Classify a batch of flows using vectorised scaling and SHAP.

        Args:
            request: Batch request containing multiple network flows.

        Returns:
            Ordered list of :class:`PredictionResponse` matching the input flows.
        """
        X: np.ndarray = np.array(
            [flow.features for flow in request.flows], dtype=np.float64
        )
        X_scaled: np.ndarray = self.scaler.transform(X)

        class_indices: np.ndarray = self.model.predict(X_scaled)
        probabilities: np.ndarray = self.model.predict_proba(X_scaled)
        shap_values = self.explainer.shap_values(X_scaled)

        results: list[PredictionResponse] = []
        for i, class_idx in enumerate(class_indices):
            class_idx = int(class_idx)
            class_name: str = str(self.label_encoder.inverse_transform([class_idx])[0])
            confidence: float = round(float(probabilities[i][class_idx]), 4)
            threat_level: str = THREAT_LEVELS.get(class_name, "HIGH")

            if isinstance(shap_values, list):
                row_shap: list[np.ndarray] = [sv[[i]] for sv in shap_values]
            else:
                row_shap = shap_values[[i]]

            top_features: list[FeatureImpact] = self._top_features(row_shap, class_idx)

            results.append(
                PredictionResponse(
                    prediction=class_name,
                    confidence=confidence,
                    threat_level=threat_level,
                    top_features=top_features,
                )
            )
        return results
