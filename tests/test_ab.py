"""Tests for Week 7: A/B routing endpoints and model quality gate.

Covers:
    - GET  /ab-test/config       (TestABConfigEndpoint)
    - POST /ab-test/predict      (TestABPredictEndpoint)
    - src/validate_model.py      (TestValidateModelGate)
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import api.main as main_module
from api.main import _prediction_buffer, app
from src.validate_model import compare_against_champion, evaluate_candidate

_N_FEATURES: int = 70
_VALID_FEATURES: list[float] = [0.0] * _N_FEATURES


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Shared TestClient — loads the model once for the entire module."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def clear_buffer() -> None:
    """Clear the prediction buffer before and after every test."""
    _prediction_buffer.clear()
    yield
    _prediction_buffer.clear()


# ---------------------------------------------------------------------------
# GET /ab-test/config
# ---------------------------------------------------------------------------


class TestABConfigEndpoint:
    """Tests for the A/B router configuration endpoint."""

    def test_returns_200(self, client: TestClient) -> None:
        """Config endpoint must return HTTP 200."""
        response = client.get("/ab-test/config")
        assert response.status_code == 200

    def test_response_has_required_keys(self, client: TestClient) -> None:
        """Response must contain shadow_mode, traffic_split, challenger_loaded."""
        data: dict = client.get("/ab-test/config").json()
        assert "shadow_mode" in data
        assert "traffic_split" in data
        assert "challenger_loaded" in data

    def test_challenger_not_loaded_by_default(self, client: TestClient) -> None:
        """Without CHALLENGER_MODEL_PATH env var the challenger must not be loaded."""
        data: dict = client.get("/ab-test/config").json()
        assert data["challenger_loaded"] is False

    def test_shadow_mode_is_bool(self, client: TestClient) -> None:
        """shadow_mode must be a boolean."""
        data: dict = client.get("/ab-test/config").json()
        assert isinstance(data["shadow_mode"], bool)

    def test_traffic_split_is_float(self, client: TestClient) -> None:
        """traffic_split must be numeric."""
        data: dict = client.get("/ab-test/config").json()
        assert isinstance(data["traffic_split"], float)


# ---------------------------------------------------------------------------
# POST /ab-test/predict
# ---------------------------------------------------------------------------


class TestABPredictEndpoint:
    """Tests for the A/B prediction endpoint."""

    def test_returns_200(self, client: TestClient) -> None:
        """A/B predict endpoint must return HTTP 200 for valid input."""
        response = client.post("/ab-test/predict", json={"features": _VALID_FEATURES})
        assert response.status_code == 200

    def test_response_has_served_by_field(self, client: TestClient) -> None:
        """Response must include the served_by field."""
        data: dict = client.post("/ab-test/predict", json={"features": _VALID_FEATURES}).json()
        assert "served_by" in data

    def test_served_by_is_champion_when_no_challenger(self, client: TestClient) -> None:
        """served_by must be 'champion' when no challenger is configured."""
        data: dict = client.post("/ab-test/predict", json={"features": _VALID_FEATURES}).json()
        assert data["served_by"] == "champion"

    def test_response_has_prediction_field(self, client: TestClient) -> None:
        """Response must include the standard prediction field."""
        data: dict = client.post("/ab-test/predict", json={"features": _VALID_FEATURES}).json()
        assert "prediction" in data

    def test_response_has_confidence_field(self, client: TestClient) -> None:
        """Response must include confidence score."""
        data: dict = client.post("/ab-test/predict", json={"features": _VALID_FEATURES}).json()
        assert "confidence" in data

    def test_ab_predict_adds_to_buffer(self, client: TestClient) -> None:
        """Calling /ab-test/predict must add one entry to the prediction buffer."""
        assert len(_prediction_buffer) == 0
        client.post("/ab-test/predict", json={"features": _VALID_FEATURES})
        assert len(_prediction_buffer) == 1

    def test_rejects_wrong_feature_count(self, client: TestClient) -> None:
        """Request with wrong number of features must return HTTP 422."""
        response = client.post("/ab-test/predict", json={"features": [0.0] * 10})
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# src/validate_model — unit tests (no disk I/O)
# ---------------------------------------------------------------------------


class TestValidateModelGate:
    """Unit tests for the quality-gate helper functions."""

    def _make_model(self, pred_value: int = 0):
        """Return a mock sklearn estimator that always predicts pred_value."""
        model = MagicMock()
        model.predict.return_value = np.array([pred_value] * 10)
        return model

    def test_evaluate_candidate_returns_accuracy(self) -> None:
        """evaluate_candidate must return a dict with 'accuracy' key."""
        model = self._make_model(pred_value=1)
        X = np.zeros((10, _N_FEATURES))
        y = np.ones(10, dtype=int)  # same as pred_value
        result = evaluate_candidate(model, X, y)
        assert "accuracy" in result
        assert result["accuracy"] == pytest.approx(1.0)

    def test_evaluate_candidate_returns_f1_macro(self) -> None:
        """evaluate_candidate must return a dict with 'f1_macro' key."""
        model = self._make_model(pred_value=0)
        X = np.zeros((10, _N_FEATURES))
        y = np.zeros(10, dtype=int)
        result = evaluate_candidate(model, X, y)
        assert "f1_macro" in result

    def test_compare_champion_better_returns_true(self) -> None:
        """compare_against_champion must return True when candidate >= champion."""
        candidate = {"accuracy": 0.997}
        champion = {"accuracy": 0.995}
        assert compare_against_champion(candidate, champion) is True

    def test_compare_champion_worse_returns_false(self) -> None:
        """compare_against_champion must return False when candidate < champion."""
        candidate = {"accuracy": 0.90}
        champion = {"accuracy": 0.995}
        assert compare_against_champion(candidate, champion) is False

    def test_validate_model_gate_bad_path_returns_false(self) -> None:
        """validate_model_gate must return False when model file is missing."""
        from src.validate_model import validate_model_gate
        result = validate_model_gate("/nonexistent/path/model.joblib")
        assert result is False

    def test_validate_model_gate_passes_when_candidate_meets_threshold(
        self, tmp_path
    ) -> None:
        """validate_model_gate must return True when candidate accuracy >= champion."""
        from src.validate_model import validate_model_gate

        # Write a champion metrics file.
        metrics = {"accuracy": 0.90, "f1_macro": 0.89}
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text(json.dumps(metrics))

        # Mock joblib.load to return a perfect mock model and test arrays.
        mock_model = self._make_model(pred_value=0)

        def joblib_side_effect(path):
            path_str = str(path)
            if path_str.endswith("best_model.joblib"):
                return mock_model
            if "X_test" in path_str:
                return np.zeros((10, _N_FEATURES))
            if "y_test" in path_str:
                return np.zeros(10, dtype=int)
            raise FileNotFoundError(path)

        with patch("src.validate_model.joblib.load", side_effect=joblib_side_effect):
            result = validate_model_gate(
                model_path="models/best_model.joblib",
                processed_dir=str(tmp_path),
                champion_metrics_path=str(metrics_file),
            )

        assert result is True
