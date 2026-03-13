"""Prefect orchestration flow for automated model retraining.

The flow runs three tasks in sequence:

    1. check_drift  — calls GET /drift on the running API
    2. run_dvc_pipeline — shells out ``dvc repro`` to retrain all changed stages
    3. evaluate_new_model — reads ``metrics.json`` produced by the pipeline
    4. promote_if_better — registers the new model in MLflow Registry if it
                           beats the accuracy threshold

Triggering:
    # Manually (force retrain even without drift):
    python -c "from src.retrain_flow import retrain_flow; retrain_flow(force=True)"

    # Via GitHub Actions workflow_dispatch (see .github/workflows/retrain.yml)

    # Via Prefect deployment (scheduled weekly):
    prefect deployment build src/retrain_flow.py:retrain_flow -n weekly
    prefect deployment apply retrain_flow-deployment.yaml
"""

import json
import logging
import os
import subprocess

import mlflow
import requests
from prefect import flow, task

from src.promote import register_and_promote
from src.validate_model import validate_model_gate

log = logging.getLogger(__name__)

API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")
MLFLOW_EXPERIMENT: str = os.getenv("MLFLOW_EXPERIMENT", "network-traffic-classifier")
ACCURACY_THRESHOLD: float = float(os.getenv("ACCURACY_THRESHOLD", "0.995"))


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@task(name="check-drift", retries=2, retry_delay_seconds=10)
def check_drift(api_url: str = API_BASE_URL) -> dict:
    """Call GET /drift on the running API and return the drift summary dict.

    Args:
        api_url: Base URL of the inference API. Defaults to ``API_BASE_URL``.

    Returns:
        Dict with at least the key ``"drift_detected"`` (bool).
    """
    response = requests.get(f"{api_url}/drift", timeout=30)
    response.raise_for_status()
    result: dict = response.json()
    log.info(
        "Drift check: detected=%s, drifted_features=%d/%d",
        result.get("drift_detected"),
        result.get("n_drifted_features", 0),
        result.get("n_features_total", 0),
    )
    return result


@task(name="run-dvc-pipeline")
def run_dvc_pipeline() -> str:
    """Shell out to ``dvc repro`` to re-run all changed pipeline stages.

    DVC will skip stages whose inputs are unchanged, so this is fast when
    only params or code changed (not the raw data).

    Returns:
        Combined stdout + stderr from the dvc command.
    """
    log.info("Running dvc repro ...")
    result = subprocess.run(
        ["dvc", "repro"],
        capture_output=True,
        text=True,
        check=True,
    )
    output = result.stdout + result.stderr
    log.info("dvc repro complete:\n%s", output[-2000:])  # last 2000 chars
    return output


@task(name="evaluate-new-model")
def evaluate_new_model(metrics_path: str = "metrics.json") -> dict:
    """Read ``metrics.json`` produced by the DVC pipeline after retraining.

    Args:
        metrics_path: Path to the metrics JSON file written by ``src/train.py``.

    Returns:
        Dict with keys: ``best_model``, ``accuracy``, ``f1_macro``, etc.
    """
    with open(metrics_path, "r", encoding="utf-8") as fh:
        metrics: dict = json.load(fh)
    log.info(
        "New model: %s  accuracy=%.4f  f1_macro=%.4f",
        metrics.get("best_model"),
        metrics.get("accuracy", 0),
        metrics.get("f1_macro", 0),
    )
    return metrics


@task(name="promote-if-better")
def promote_if_better(
    new_metrics: dict,
    threshold: float = ACCURACY_THRESHOLD,
) -> bool:
    """Register the best model in MLflow Registry and promote to Production if accuracy ≥ threshold.

    Finds the most recent MLflow run in the ``network-traffic-classifier``
    experiment (written by ``src/train.py``) and registers its ``model``
    artifact.

    Args:
        new_metrics: Metrics dict returned by :func:`evaluate_new_model`.
        threshold:   Minimum accuracy required to promote. Defaults to 0.995.

    Returns:
        ``True`` if the model was promoted, ``False`` if accuracy was too low
        or no MLflow run was found.
    """
    accuracy: float = float(new_metrics.get("accuracy", 0.0))
    if accuracy < threshold:
        log.info(
            "Accuracy %.4f < threshold %.4f — skipping promotion.", accuracy, threshold
        )
        return False

    # Find the latest run in the training experiment.
    client = mlflow.MlflowClient()
    experiments = client.search_experiments(filter_string=f"name = '{MLFLOW_EXPERIMENT}'")
    if not experiments:
        log.warning("MLflow experiment '%s' not found — skipping promotion.", MLFLOW_EXPERIMENT)
        return False

    runs = client.search_runs(
        experiment_ids=[experiments[0].experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        log.warning("No MLflow runs found — skipping promotion.")
        return False

    # Run quality gate before touching the registry.
    local_model = os.path.join("models", "best_model.joblib")
    if not validate_model_gate(local_model):
        log.warning("Quality gate failed — skipping promotion.")
        return False

    model_uri = f"runs:/{runs[0].info.run_id}/model"
    version = register_and_promote(model_uri)
    log.info("Promoted model version %s to Production.", version)
    return True


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------


@flow(name="network-threat-retrain", log_prints=True)
def retrain_flow(force: bool = False) -> None:
    """Orchestrate the full MLOps retraining loop.

    Steps:
      1. Check for data drift via the running API.
      2. If drift detected (or ``force=True``), re-run the DVC pipeline.
      3. Read new metrics and promote if accuracy exceeds the threshold.

    Args:
        force: If ``True``, retrain even when no drift is detected.
               Useful for scheduled maintenance runs.
    """
    drift = check_drift()

    if not force and not drift.get("drift_detected", False):
        log.info("No drift detected and force=False — nothing to do.")
        return

    reason = "forced" if force else "drift detected"
    log.info("Retraining triggered (%s).", reason)

    run_dvc_pipeline()
    metrics = evaluate_new_model()
    promoted = promote_if_better(metrics)

    if promoted:
        log.info(
            "New model promoted to Production. "
            "The API will load it from the MLflow Registry on the next /predict request."
        )
    else:
        log.info("Retrain complete but model was not promoted (accuracy below threshold).")


if __name__ == "__main__":
    retrain_flow(force=True)
