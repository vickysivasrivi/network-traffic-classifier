"""Tests for the monitoring endpoints: GET /prometheus (Prometheus) and GET /drift."""

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import api.main as main_module
from api.main import _prediction_buffer, app

_N_FEATURES: int = 70
_VALID_FEATURES: list[float] = [0.0] * _N_FEATURES


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Shared TestClient — loads the model once for the entire module."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def clear_buffer() -> None:
    """Clear the prediction buffer before (and after) every test."""
    _prediction_buffer.clear()
    yield
    _prediction_buffer.clear()


# ---------------------------------------------------------------------------
# GET /prometheus  (Prometheus metrics endpoint)
# ---------------------------------------------------------------------------


class TestPrometheusEndpoint:
    """Tests for the Prometheus metrics scrape endpoint at GET /prometheus."""

    def test_returns_200(self, client: TestClient) -> None:
        """Prometheus endpoint must return HTTP 200."""
        response = client.get("/prometheus")
        assert response.status_code == 200

    def test_content_type_is_text_plain(self, client: TestClient) -> None:
        """Content-type must be text/plain (Prometheus text format)."""
        response = client.get("/prometheus")
        assert "text/plain" in response.headers["content-type"]

    def test_contains_http_request_metric(self, client: TestClient) -> None:
        """Response must include the standard Prometheus HTTP request counter."""
        client.get("/health")  # generate at least one request
        response = client.get("/prometheus")
        # prometheus-fastapi-instrumentator exposes this metric
        assert "http_requests_total" in response.text or "http_request" in response.text

    def test_contains_custom_prediction_counter(self, client: TestClient) -> None:
        """Custom network_threat_predictions_total counter must appear after a /predict call."""
        client.post("/predict", json={"features": _VALID_FEATURES})
        response = client.get("/prometheus")
        assert "network_threat_predictions_total" in response.text

    def test_contains_custom_confidence_histogram(self, client: TestClient) -> None:
        """Custom confidence histogram must appear after a /predict call."""
        client.post("/predict", json={"features": _VALID_FEATURES})
        response = client.get("/prometheus")
        assert "network_threat_confidence" in response.text

    def test_contains_drift_gauge(self, client: TestClient) -> None:
        """Drift Gauge must be present in the metrics output."""
        response = client.get("/prometheus")
        assert "network_threat_drift_detected" in response.text


# ---------------------------------------------------------------------------
# GET /drift  (Evidently drift report)
# ---------------------------------------------------------------------------


class TestDriftEndpoint:
    """Tests for the GET /drift endpoint."""

    def test_returns_200(self, client: TestClient) -> None:
        """Drift endpoint must return HTTP 200."""
        response = client.get("/drift")
        assert response.status_code == 200

    def test_response_has_required_keys(self, client: TestClient) -> None:
        """DriftResponse must have drift_detected, n_drifted_features, n_features_total, share_drifted."""
        data: dict = client.get("/drift").json()
        assert "drift_detected" in data
        assert "n_drifted_features" in data
        assert "n_features_total" in data
        assert "share_drifted" in data

    def test_empty_buffer_returns_no_drift(self, client: TestClient) -> None:
        """With an empty prediction buffer, drift_detected must be False."""
        # buffer is cleared by autouse fixture
        data: dict = client.get("/drift").json()
        assert data["drift_detected"] is False

    def test_empty_buffer_returns_zero_drifted_features(self, client: TestClient) -> None:
        """With an empty buffer, n_drifted_features must be 0."""
        data: dict = client.get("/drift").json()
        assert data["n_drifted_features"] == 0

    def test_n_features_total_equals_feature_count(self, client: TestClient) -> None:
        """n_features_total must equal the number of model features (70)."""
        data: dict = client.get("/drift").json()
        assert data["n_features_total"] == _N_FEATURES

    def test_share_drifted_is_zero_when_no_buffer(self, client: TestClient) -> None:
        """share_drifted must be 0.0 when buffer is empty."""
        data: dict = client.get("/drift").json()
        assert data["share_drifted"] == 0.0

    def test_drift_detected_is_bool(self, client: TestClient) -> None:
        """drift_detected must be a boolean."""
        data: dict = client.get("/drift").json()
        assert isinstance(data["drift_detected"], bool)

    def test_report_saved_at_is_none_or_string(self, client: TestClient) -> None:
        """report_saved_at must be None or a string path."""
        data: dict = client.get("/drift").json()
        assert data["report_saved_at"] is None or isinstance(data["report_saved_at"], str)


# ---------------------------------------------------------------------------
# Prediction buffer integration
# ---------------------------------------------------------------------------


class TestPredictionBuffer:
    """Tests that verify the prediction buffer is filled correctly by /predict."""

    def test_predict_adds_to_buffer(self, client: TestClient) -> None:
        """Calling /predict once must add exactly one entry to the buffer."""
        assert len(_prediction_buffer) == 0
        client.post("/predict", json={"features": _VALID_FEATURES})
        assert len(_prediction_buffer) == 1

    def test_batch_predict_adds_all_flows_to_buffer(self, client: TestClient) -> None:
        """Calling /predict/batch with N flows must add N entries to the buffer."""
        n: int = 5
        payload = {"flows": [{"features": _VALID_FEATURES}] * n}
        client.post("/predict/batch", json=payload)
        assert len(_prediction_buffer) == n

    def test_buffer_respects_maxlen(self, client: TestClient) -> None:
        """Buffer must not exceed 500 entries (configured maxlen)."""
        assert _prediction_buffer.maxlen == 500
