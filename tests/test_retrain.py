"""Unit tests for the Prefect retraining tasks in src/retrain_flow.py.

All external I/O (HTTP calls, subprocess, MLflow, file I/O) is mocked so
these tests run fully offline with no running API, DVC, or MLflow server.

Prefect tasks expose their underlying function via ``task.fn(*args)``,
which bypasses the Prefect execution context and runs the function
synchronously — ideal for unit testing.
"""

import json
import os
import sys
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.retrain_flow import (
    check_drift,
    evaluate_new_model,
    promote_if_better,
    retrain_flow,
    run_dvc_pipeline,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MOCK_DRIFT_NO = {
    "drift_detected": False,
    "n_drifted_features": 0,
    "n_features_total": 70,
    "share_drifted": 0.0,
    "report_saved_at": None,
}

_MOCK_DRIFT_YES = {
    "drift_detected": True,
    "n_drifted_features": 18,
    "n_features_total": 70,
    "share_drifted": 0.257,
    "report_saved_at": None,
}

_GOOD_METRICS = {
    "best_model": "xgboost",
    "accuracy": 0.9969,
    "f1_macro": 0.9965,
    "f1_weighted": 0.9969,
    "precision_macro": 0.9963,
    "recall_macro": 0.9967,
}

_POOR_METRICS = {
    "best_model": "random_forest",
    "accuracy": 0.91,
    "f1_macro": 0.89,
    "f1_weighted": 0.90,
    "precision_macro": 0.88,
    "recall_macro": 0.89,
}


# ---------------------------------------------------------------------------
# check_drift task
# ---------------------------------------------------------------------------


class TestCheckDriftTask:
    """Tests for the check_drift Prefect task."""

    def test_returns_dict_with_drift_detected_key(self) -> None:
        """Task must return a dict containing 'drift_detected'."""
        mock_response = MagicMock()
        mock_response.json.return_value = _MOCK_DRIFT_NO

        with patch("src.retrain_flow.requests.get", return_value=mock_response):
            result = check_drift.fn()

        assert isinstance(result, dict)
        assert "drift_detected" in result

    def test_returns_drift_detected_true(self) -> None:
        """Task must propagate drift_detected=True from the API response."""
        mock_response = MagicMock()
        mock_response.json.return_value = _MOCK_DRIFT_YES

        with patch("src.retrain_flow.requests.get", return_value=mock_response):
            result = check_drift.fn()

        assert result["drift_detected"] is True

    def test_calls_correct_endpoint(self) -> None:
        """Task must call GET /drift on the configured API URL."""
        mock_response = MagicMock()
        mock_response.json.return_value = _MOCK_DRIFT_NO

        with patch("src.retrain_flow.requests.get", return_value=mock_response) as mock_get:
            check_drift.fn(api_url="http://test-api:8000")

        mock_get.assert_called_once_with("http://test-api:8000/drift", timeout=30)


# ---------------------------------------------------------------------------
# run_dvc_pipeline task
# ---------------------------------------------------------------------------


class TestRunDvcPipelineTask:
    """Tests for the run_dvc_pipeline Prefect task."""

    def test_calls_dvc_repro(self) -> None:
        """Task must invoke 'dvc repro' via subprocess.run."""
        mock_result = MagicMock()
        mock_result.stdout = "Stage 'train' didn't change, skipping\n"
        mock_result.stderr = ""

        with patch("src.retrain_flow.subprocess.run", return_value=mock_result) as mock_run:
            run_dvc_pipeline.fn()

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args == ["dvc", "repro"]

    def test_returns_output_string(self) -> None:
        """Task must return the combined stdout+stderr string."""
        mock_result = MagicMock()
        mock_result.stdout = "Running stage train\n"
        mock_result.stderr = ""

        with patch("src.retrain_flow.subprocess.run", return_value=mock_result):
            output = run_dvc_pipeline.fn()

        assert "Running stage train" in output


# ---------------------------------------------------------------------------
# evaluate_new_model task
# ---------------------------------------------------------------------------


class TestEvaluateNewModelTask:
    """Tests for the evaluate_new_model Prefect task."""

    def test_reads_metrics_json(self, tmp_path) -> None:
        """Task must read and return the contents of metrics.json."""
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text(json.dumps(_GOOD_METRICS))

        result = evaluate_new_model.fn(metrics_path=str(metrics_file))

        assert result["best_model"] == "xgboost"
        assert result["accuracy"] == pytest.approx(0.9969)

    def test_returns_dict_with_accuracy_key(self, tmp_path) -> None:
        """Returned dict must include 'accuracy' key."""
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text(json.dumps(_GOOD_METRICS))

        result = evaluate_new_model.fn(metrics_path=str(metrics_file))

        assert "accuracy" in result


# ---------------------------------------------------------------------------
# promote_if_better task
# ---------------------------------------------------------------------------


class TestPromoteIfBetterTask:
    """Tests for the promote_if_better Prefect task."""

    def _make_mock_mlflow(self, run_id: str = "abc123"):
        """Return a mock MlflowClient with a single experiment and run."""
        mock_client = MagicMock()
        mock_exp = MagicMock()
        mock_exp.experiment_id = "1"
        mock_client.search_experiments.return_value = [mock_exp]
        mock_run = MagicMock()
        mock_run.info.run_id = run_id
        mock_client.search_runs.return_value = [mock_run]
        return mock_client

    def test_promotes_when_accuracy_exceeds_threshold(self) -> None:
        """Task must call register_and_promote and return True when accuracy ≥ threshold."""
        mock_client = self._make_mock_mlflow()

        with (
            patch("src.retrain_flow.mlflow.MlflowClient", return_value=mock_client),
            patch("src.retrain_flow.register_and_promote", return_value="3") as mock_promote,
        ):
            result = promote_if_better.fn(_GOOD_METRICS, threshold=0.995)

        assert result is True
        mock_promote.assert_called_once_with("runs:/abc123/model")

    def test_skips_promotion_when_accuracy_below_threshold(self) -> None:
        """Task must return False and skip promotion when accuracy < threshold."""
        with patch("src.retrain_flow.register_and_promote") as mock_promote:
            result = promote_if_better.fn(_POOR_METRICS, threshold=0.995)

        assert result is False
        mock_promote.assert_not_called()

    def test_skips_promotion_when_no_experiments(self) -> None:
        """Task must return False when no MLflow experiments are found."""
        mock_client = MagicMock()
        mock_client.search_experiments.return_value = []

        with (
            patch("src.retrain_flow.mlflow.MlflowClient", return_value=mock_client),
            patch("src.retrain_flow.register_and_promote") as mock_promote,
        ):
            result = promote_if_better.fn(_GOOD_METRICS, threshold=0.995)

        assert result is False
        mock_promote.assert_not_called()

    def test_skips_promotion_when_no_runs(self) -> None:
        """Task must return False when the experiment exists but has no runs."""
        mock_client = MagicMock()
        mock_exp = MagicMock()
        mock_exp.experiment_id = "1"
        mock_client.search_experiments.return_value = [mock_exp]
        mock_client.search_runs.return_value = []

        with (
            patch("src.retrain_flow.mlflow.MlflowClient", return_value=mock_client),
            patch("src.retrain_flow.register_and_promote") as mock_promote,
        ):
            result = promote_if_better.fn(_GOOD_METRICS, threshold=0.995)

        assert result is False
        mock_promote.assert_not_called()


# ---------------------------------------------------------------------------
# retrain_flow integration
# ---------------------------------------------------------------------------


class TestRetrainFlow:
    """Tests for the retrain_flow Prefect flow."""

    def test_skips_pipeline_when_no_drift_and_not_forced(self) -> None:
        """Flow must skip dvc repro when drift_detected=False and force=False."""
        with (
            patch("src.retrain_flow.check_drift", return_value=_MOCK_DRIFT_NO),
            patch("src.retrain_flow.run_dvc_pipeline") as mock_dvc,
        ):
            retrain_flow.fn(force=False)

        mock_dvc.assert_not_called()

    def test_runs_pipeline_when_forced(self) -> None:
        """Flow must run the DVC pipeline when force=True, regardless of drift."""
        mock_metrics = _GOOD_METRICS

        with (
            patch("src.retrain_flow.check_drift", return_value=_MOCK_DRIFT_NO),
            patch("src.retrain_flow.run_dvc_pipeline") as mock_dvc,
            patch("src.retrain_flow.evaluate_new_model", return_value=mock_metrics),
            patch("src.retrain_flow.promote_if_better", return_value=False),
        ):
            retrain_flow.fn(force=True)

        mock_dvc.assert_called_once()

    def test_runs_pipeline_when_drift_detected(self) -> None:
        """Flow must run the DVC pipeline when drift is detected."""
        mock_metrics = _GOOD_METRICS

        with (
            patch("src.retrain_flow.check_drift", return_value=_MOCK_DRIFT_YES),
            patch("src.retrain_flow.run_dvc_pipeline") as mock_dvc,
            patch("src.retrain_flow.evaluate_new_model", return_value=mock_metrics),
            patch("src.retrain_flow.promote_if_better", return_value=True),
        ):
            retrain_flow.fn(force=False)

        mock_dvc.assert_called_once()
