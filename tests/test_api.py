"""Integration tests for the threat detection API endpoints."""

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.main import app

_N_FEATURES: int = 70
_N_CLASSES: int = 15
_VALID_FEATURES: list[float] = [0.0] * _N_FEATURES
_VALID_THREAT_LEVELS: frozenset[str] = frozenset({"LOW", "MEDIUM", "HIGH", "CRITICAL"})
_EXPECTED_METRIC_KEYS: frozenset[str] = frozenset({
    "best_model", "accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro",
})


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Shared TestClient with the model loaded once for the whole module."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class TestHealth:
    """Tests for the GET /health endpoint."""

    def test_returns_200(self, client: TestClient) -> None:
        """Health endpoint must return HTTP 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_response_structure(self, client: TestClient) -> None:
        """Response must contain status, model, and n_features keys."""
        data: dict = client.get("/health").json()
        assert "status" in data
        assert "model" in data
        assert "n_features" in data

    def test_status_is_ok(self, client: TestClient) -> None:
        """status field must equal 'ok'."""
        data: dict = client.get("/health").json()
        assert data["status"] == "ok"

    def test_n_features_is_70(self, client: TestClient) -> None:
        """n_features must be 70 (matching CICIDS 2017 feature set)."""
        data: dict = client.get("/health").json()
        assert data["n_features"] == _N_FEATURES


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------


class TestPredict:
    """Tests for the POST /predict endpoint."""

    def test_returns_200_for_valid_input(self, client: TestClient) -> None:
        """Valid 70-feature input must return HTTP 200."""
        response = client.post("/predict", json={"features": _VALID_FEATURES})
        assert response.status_code == 200

    def test_response_has_required_keys(self, client: TestClient) -> None:
        """Response must contain prediction, confidence, threat_level, top_features."""
        data: dict = client.post("/predict", json={"features": _VALID_FEATURES}).json()
        assert "prediction" in data
        assert "confidence" in data
        assert "threat_level" in data
        assert "top_features" in data

    def test_confidence_is_probability(self, client: TestClient) -> None:
        """Confidence must be a float in [0.0, 1.0]."""
        data: dict = client.post("/predict", json={"features": _VALID_FEATURES}).json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_threat_level_is_valid(self, client: TestClient) -> None:
        """threat_level must be one of the four defined levels."""
        data: dict = client.post("/predict", json={"features": _VALID_FEATURES}).json()
        assert data["threat_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

    def test_top_features_returns_three(self, client: TestClient) -> None:
        """top_features must contain exactly 3 entries."""
        data: dict = client.post("/predict", json={"features": _VALID_FEATURES}).json()
        assert len(data["top_features"]) == 3

    def test_top_features_structure(self, client: TestClient) -> None:
        """Each top_features entry must have 'feature' (str) and 'impact' (float)."""
        data: dict = client.post("/predict", json={"features": _VALID_FEATURES}).json()
        for item in data["top_features"]:
            assert "feature" in item
            assert "impact" in item
            assert isinstance(item["feature"], str)
            assert isinstance(item["impact"], float)

    def test_too_few_features_returns_422(self, client: TestClient) -> None:
        """Input with fewer than 70 features must return HTTP 422."""
        response = client.post("/predict", json={"features": [0.0] * 5})
        assert response.status_code == 422

    def test_too_many_features_returns_422(self, client: TestClient) -> None:
        """Input with more than 70 features must return HTTP 422."""
        response = client.post("/predict", json={"features": [0.0] * 80})
        assert response.status_code == 422

    def test_empty_features_returns_422(self, client: TestClient) -> None:
        """Empty features list must return HTTP 422."""
        response = client.post("/predict", json={"features": []})
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# /predict/batch
# ---------------------------------------------------------------------------


class TestPredictBatch:
    """Tests for the POST /predict/batch endpoint."""

    def test_returns_200_for_valid_batch(self, client: TestClient) -> None:
        """Valid batch of 3 flows must return HTTP 200."""
        payload = {"flows": [{"features": _VALID_FEATURES}] * 3}
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200

    def test_predictions_count_matches_input(self, client: TestClient) -> None:
        """Number of predictions must equal number of input flows."""
        n: int = 4
        payload = {"flows": [{"features": _VALID_FEATURES}] * n}
        data: dict = client.post("/predict/batch", json=payload).json()
        assert len(data["predictions"]) == n

    def test_each_prediction_has_required_keys(self, client: TestClient) -> None:
        """Each item in predictions must have all required response keys."""
        payload = {"flows": [{"features": _VALID_FEATURES}] * 2}
        data: dict = client.post("/predict/batch", json=payload).json()
        for pred in data["predictions"]:
            assert "prediction" in pred
            assert "confidence" in pred
            assert "threat_level" in pred
            assert "top_features" in pred

    def test_single_flow_batch(self, client: TestClient) -> None:
        """Batch with a single flow must return exactly one prediction."""
        payload = {"flows": [{"features": _VALID_FEATURES}]}
        data: dict = client.post("/predict/batch", json=payload).json()
        assert len(data["predictions"]) == 1


# ---------------------------------------------------------------------------
# /features
# ---------------------------------------------------------------------------


class TestFeatures:
    """Tests for the GET /features endpoint."""

    def test_returns_200(self, client: TestClient) -> None:
        """Features endpoint must return HTTP 200."""
        response = client.get("/features")
        assert response.status_code == 200

    def test_response_has_features_and_count_keys(self, client: TestClient) -> None:
        """Response must contain 'features' and 'count' keys."""
        data: dict = client.get("/features").json()
        assert "features" in data
        assert "count" in data

    def test_features_is_list_of_strings(self, client: TestClient) -> None:
        """'features' must be a list of strings."""
        data: dict = client.get("/features").json()
        assert isinstance(data["features"], list)
        assert all(isinstance(f, str) for f in data["features"])

    def test_returns_exactly_70_features(self, client: TestClient) -> None:
        """Endpoint must return exactly 70 feature names."""
        data: dict = client.get("/features").json()
        assert len(data["features"]) == _N_FEATURES

    def test_count_matches_list_length(self, client: TestClient) -> None:
        """'count' must equal len(features)."""
        data: dict = client.get("/features").json()
        assert data["count"] == len(data["features"])

    def test_features_are_unique(self, client: TestClient) -> None:
        """All feature names must be unique."""
        data: dict = client.get("/features").json()
        assert len(data["features"]) == len(set(data["features"]))

    def test_known_feature_present(self, client: TestClient) -> None:
        """'Flow Duration' must appear in the feature list."""
        data: dict = client.get("/features").json()
        assert "Flow Duration" in data["features"]


# ---------------------------------------------------------------------------
# /classes
# ---------------------------------------------------------------------------


class TestClasses:
    """Tests for the GET /classes endpoint."""

    def test_returns_200(self, client: TestClient) -> None:
        """Classes endpoint must return HTTP 200."""
        response = client.get("/classes")
        assert response.status_code == 200

    def test_response_has_classes_and_count_keys(self, client: TestClient) -> None:
        """Response must contain 'classes' and 'count' keys."""
        data: dict = client.get("/classes").json()
        assert "classes" in data
        assert "count" in data

    def test_returns_exactly_15_classes(self, client: TestClient) -> None:
        """Endpoint must return exactly 15 attack classes."""
        data: dict = client.get("/classes").json()
        assert len(data["classes"]) == _N_CLASSES

    def test_count_matches_list_length(self, client: TestClient) -> None:
        """'count' must equal len(classes)."""
        data: dict = client.get("/classes").json()
        assert data["count"] == len(data["classes"])

    def test_each_class_has_name_and_threat_level(self, client: TestClient) -> None:
        """Each class entry must have 'name' and 'threat_level' keys."""
        data: dict = client.get("/classes").json()
        for item in data["classes"]:
            assert "name" in item
            assert "threat_level" in item

    def test_all_threat_levels_are_valid(self, client: TestClient) -> None:
        """Every threat_level must be one of the four defined levels."""
        data: dict = client.get("/classes").json()
        for item in data["classes"]:
            assert item["threat_level"] in _VALID_THREAT_LEVELS

    def test_benign_class_is_low(self, client: TestClient) -> None:
        """BENIGN class must map to threat level LOW."""
        data: dict = client.get("/classes").json()
        benign = next(c for c in data["classes"] if c["name"] == "BENIGN")
        assert benign["threat_level"] == "LOW"

    def test_heartbleed_is_critical(self, client: TestClient) -> None:
        """Heartbleed class must map to threat level CRITICAL."""
        data: dict = client.get("/classes").json()
        heartbleed = next(c for c in data["classes"] if c["name"] == "Heartbleed")
        assert heartbleed["threat_level"] == "CRITICAL"

    def test_class_names_are_unique(self, client: TestClient) -> None:
        """All class names must be unique."""
        data: dict = client.get("/classes").json()
        names: list[str] = [c["name"] for c in data["classes"]]
        assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    """Tests for the GET /metrics endpoint."""

    def test_returns_200(self, client: TestClient) -> None:
        """Metrics endpoint must return HTTP 200."""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_response_has_all_required_keys(self, client: TestClient) -> None:
        """Response must contain exactly the expected metric keys."""
        data: dict = client.get("/metrics").json()
        assert set(data.keys()) == _EXPECTED_METRIC_KEYS

    def test_best_model_is_non_empty_string(self, client: TestClient) -> None:
        """'best_model' must be a non-empty string."""
        data: dict = client.get("/metrics").json()
        assert isinstance(data["best_model"], str)
        assert len(data["best_model"]) > 0

    def test_accuracy_is_valid_probability(self, client: TestClient) -> None:
        """'accuracy' must be a float in [0.0, 1.0]."""
        data: dict = client.get("/metrics").json()
        assert 0.0 <= data["accuracy"] <= 1.0

    def test_f1_macro_is_valid_probability(self, client: TestClient) -> None:
        """'f1_macro' must be a float in [0.0, 1.0]."""
        data: dict = client.get("/metrics").json()
        assert 0.0 <= data["f1_macro"] <= 1.0

    def test_f1_weighted_is_valid_probability(self, client: TestClient) -> None:
        """'f1_weighted' must be a float in [0.0, 1.0]."""
        data: dict = client.get("/metrics").json()
        assert 0.0 <= data["f1_weighted"] <= 1.0

    def test_precision_macro_is_valid_probability(self, client: TestClient) -> None:
        """'precision_macro' must be a float in [0.0, 1.0]."""
        data: dict = client.get("/metrics").json()
        assert 0.0 <= data["precision_macro"] <= 1.0

    def test_recall_macro_is_valid_probability(self, client: TestClient) -> None:
        """'recall_macro' must be a float in [0.0, 1.0]."""
        data: dict = client.get("/metrics").json()
        assert 0.0 <= data["recall_macro"] <= 1.0

    def test_accuracy_exceeds_reasonable_threshold(self, client: TestClient) -> None:
        """Accuracy regression guard — must not drop below 0.95."""
        data: dict = client.get("/metrics").json()
        assert data["accuracy"] >= 0.95
