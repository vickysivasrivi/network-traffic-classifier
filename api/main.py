"""FastAPI application for real-time network threat detection."""

import collections
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

from api.ab_router import ABRouter
from api.monitoring import DRIFT_DETECTED, DRIFT_FEATURES_COUNT, record_prediction
from api.predictor import THREAT_LEVELS, Predictor
from api.schemas import (
    ABPredictionResponse,
    ABTestConfig,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ClassesResponse,
    ClassInfo,
    DriftResponse,
    FeaturesResponse,
    HealthResponse,
    MetricsResponse,
    NetworkFlowRequest,
    PredictionResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log: logging.Logger = logging.getLogger(__name__)

_predictor: Predictor | None = None
_metrics: dict | None = None
_baseline_df: pd.DataFrame | None = None
_ab_router: ABRouter | None = None

# Rolling buffer of the last 500 prediction feature vectors — fed to Evidently drift checks.
_prediction_buffer: collections.deque = collections.deque(maxlen=500)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the model once on startup and release resources on shutdown."""
    global _predictor, _metrics, _baseline_df, _ab_router
    log.info("Loading model artifacts ...")
    _predictor = Predictor()
    with open("metrics.json", "r", encoding="utf-8") as fh:
        _metrics = json.load(fh)
    log.info("Metrics loaded.")

    # Clear any stale predictions carried over from a previous session.
    _prediction_buffer.clear()

    # Load a sample of training data as the drift reference baseline.
    # X_train.joblib is excluded from git (too large) but present locally after DVC pull.
    try:
        X_train = joblib.load("data/processed/X_train.joblib")
        _baseline_df = pd.DataFrame(X_train[:1000], columns=_predictor.feature_names)
        log.info("Drift baseline loaded — %d reference rows.", len(_baseline_df))
    except Exception as exc:
        log.warning("Could not load drift baseline (drift detection disabled): %s", exc)

    # Initialise A/B router — champion-only by default unless env vars are set.
    _ab_router = ABRouter(
        champion=_predictor,
        challenger_model_path=os.getenv("CHALLENGER_MODEL_PATH"),
        traffic_split=float(os.getenv("AB_TRAFFIC_SPLIT", "0.0")),
        shadow_mode=os.getenv("AB_SHADOW_MODE", "true").lower() == "true",
    )
    log.info(
        "A/B router ready — shadow_mode=%s  traffic_split=%.2f  challenger_loaded=%s",
        _ab_router.shadow_mode,
        _ab_router.traffic_split,
        _ab_router.challenger_loaded,
    )

    log.info("API ready.")
    yield
    _predictor = None
    _metrics = None
    _baseline_df = None
    _ab_router = None


app: FastAPI = FastAPI(
    title="Network Threat Detection API",
    description="Classifies network flows as BENIGN or a specific attack type with SHAP explainability.",
    version="1.0.0",
    lifespan=lifespan,
)

# Expose Prometheus RED metrics at GET /prometheus.
# We use /prometheus (not /metrics) to avoid conflicting with the existing
# model-evaluation metrics endpoint at GET /metrics.
Instrumentator().instrument(app).expose(app, endpoint="/prometheus")


def _get_predictor() -> Predictor:
    """Return the loaded predictor or raise 503 if not initialised.

    Returns:
        Loaded :class:`~api.predictor.Predictor` instance.

    Raises:
        HTTPException: 503 if the model has not been loaded yet.
    """
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return _predictor


def _get_metrics() -> dict:
    """Return loaded metrics dict or raise 503 if not initialised.

    Returns:
        Metrics dictionary loaded from ``metrics.json``.

    Raises:
        HTTPException: 503 if metrics have not been loaded yet.
    """
    if _metrics is None:
        raise HTTPException(status_code=503, detail="Metrics not loaded.")
    return _metrics


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health() -> HealthResponse:
    """Return API liveness status and loaded model metadata."""
    predictor: Predictor = _get_predictor()
    return HealthResponse(
        status="ok",
        model=predictor.model_name,
        n_features=len(predictor.feature_names),
    )


@app.get("/drift", response_model=DriftResponse, tags=["Monitoring"])
def check_drift() -> DriftResponse:
    """Run an Evidently data-drift report comparing recent predictions to the training baseline.

    Uses the last ≤500 prediction feature vectors buffered in memory as the
    *current* dataset and a 1 000-row sample of ``X_train.joblib`` as the
    *reference* dataset.

    Returns:
        :class:`DriftResponse` with drift flag, drifted-feature count, and share.
    """
    predictor = _get_predictor()
    feature_names = predictor.feature_names
    n_features = len(feature_names)

    # Not enough data or no baseline → report no drift without running Evidently.
    if len(_prediction_buffer) < 10 or _baseline_df is None:
        DRIFT_DETECTED.set(0)
        DRIFT_FEATURES_COUNT.set(0)
        return DriftResponse(
            drift_detected=False,
            n_drifted_features=0,
            n_features_total=n_features,
            share_drifted=0.0,
            report_saved_at=None,
        )

    try:
        from evidently.metric_preset import DataDriftPreset  # lazy import — avoids startup cost
        from evidently.report import Report

        current_df = pd.DataFrame(list(_prediction_buffer), columns=feature_names)
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=_baseline_df, current_data=current_df)
        result_dict = report.as_dict()
        drift_result = result_dict["metrics"][0]["result"]

        n_drifted: int = int(drift_result.get("number_of_drifted_columns", 0))
        n_total: int = int(drift_result.get("number_of_columns", n_features))
        drift_detected: bool = bool(drift_result.get("dataset_drift", False))

    except Exception as exc:
        log.warning("Drift check failed: %s", exc)
        return DriftResponse(
            drift_detected=False,
            n_drifted_features=0,
            n_features_total=n_features,
            share_drifted=0.0,
            report_saved_at=None,
        )

    DRIFT_DETECTED.set(1 if drift_detected else 0)
    DRIFT_FEATURES_COUNT.set(n_drifted)

    return DriftResponse(
        drift_detected=drift_detected,
        n_drifted_features=n_drifted,
        n_features_total=n_total,
        share_drifted=round(n_drifted / n_total, 4) if n_total > 0 else 0.0,
        report_saved_at=None,
    )


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


@app.get("/features", response_model=FeaturesResponse, tags=["Metadata"])
def get_features() -> FeaturesResponse:
    """Return the ordered list of 70 feature names clients must supply."""
    predictor: Predictor = _get_predictor()
    return FeaturesResponse(
        features=predictor.feature_names,
        count=len(predictor.feature_names),
    )


@app.get("/classes", response_model=ClassesResponse, tags=["Metadata"])
def get_classes() -> ClassesResponse:
    """Return all 15 attack class names each paired with its threat level."""
    predictor: Predictor = _get_predictor()
    class_list: list[ClassInfo] = [
        ClassInfo(name=cls, threat_level=THREAT_LEVELS.get(cls, "HIGH"))
        for cls in predictor.label_encoder.classes_
    ]
    return ClassesResponse(classes=class_list, count=len(class_list))


@app.get("/metrics", response_model=MetricsResponse, tags=["Metadata"])
def get_metrics() -> MetricsResponse:
    """Return model evaluation metrics from the last training run."""
    return MetricsResponse(**_get_metrics())


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(request: NetworkFlowRequest) -> PredictionResponse:
    """Classify a single network flow and return threat prediction with SHAP explanation.

    Args:
        request: 70 raw feature values in ``feature_names.txt`` order.

    Returns:
        Prediction, confidence score, threat level, and top-3 SHAP features.
    """
    predictor: Predictor = _get_predictor()
    result: PredictionResponse = predictor.predict(request.features)

    # Buffer feature vector for drift detection and update Prometheus counters.
    _prediction_buffer.append(request.features)
    record_prediction(result.prediction, result.threat_level, result.confidence)

    return result


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Classify a batch of network flows in a single vectorised call.

    Args:
        request: List of flows, each with 70 raw feature values.

    Returns:
        Ordered list of predictions matching the input batch.
    """
    predictor: Predictor = _get_predictor()
    predictions: list[PredictionResponse] = predictor.predict_batch(request)

    # Buffer every input flow and update Prometheus counters.
    for flow, pred in zip(request.flows, predictions):
        _prediction_buffer.append(flow.features)
        record_prediction(pred.prediction, pred.threat_level, pred.confidence)

    return BatchPredictionResponse(predictions=predictions)


# ---------------------------------------------------------------------------
# A/B Testing
# ---------------------------------------------------------------------------


@app.get("/ab-test/config", response_model=ABTestConfig, tags=["A/B Testing"])
def get_ab_config() -> ABTestConfig:
    """Return the current A/B router configuration.

    Returns:
        :class:`ABTestConfig` with shadow_mode, traffic_split, and whether a
        challenger model is loaded.
    """
    if _ab_router is None:
        raise HTTPException(status_code=503, detail="A/B router not initialised.")
    return ABTestConfig(
        shadow_mode=_ab_router.shadow_mode,
        traffic_split=_ab_router.traffic_split,
        challenger_loaded=_ab_router.challenger_loaded,
    )


@app.post("/ab-test/predict", response_model=ABPredictionResponse, tags=["A/B Testing"])
def ab_predict(request: NetworkFlowRequest) -> ABPredictionResponse:
    """Classify a network flow via the A/B router.

    The response includes a ``served_by`` field indicating whether the champion
    or challenger model produced the result.  In shadow mode the champion result
    is always returned; the challenger runs silently in the background.

    Args:
        request: 70 raw feature values in ``feature_names.txt`` order.

    Returns:
        :class:`ABPredictionResponse` — standard prediction fields plus
        ``served_by`` (``"champion"`` or ``"challenger"``).
    """
    if _ab_router is None:
        raise HTTPException(status_code=503, detail="A/B router not initialised.")

    result, served_by = _ab_router.route(request.features)

    # Buffer for drift detection and update Prometheus counters.
    _prediction_buffer.append(request.features)
    record_prediction(result.prediction, result.threat_level, result.confidence)

    return ABPredictionResponse(**result.model_dump(), served_by=served_by)
