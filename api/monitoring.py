"""Custom Prometheus metrics for the Network Threat Detection API."""

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# Prediction metrics
# ---------------------------------------------------------------------------

PREDICTION_COUNTER = Counter(
    "network_threat_predictions_total",
    "Total predictions broken down by predicted class and threat level.",
    ["prediction", "threat_level"],
)

CONFIDENCE_HISTOGRAM = Histogram(
    "network_threat_confidence",
    "Distribution of prediction confidence scores.",
    buckets=[0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
)

# ---------------------------------------------------------------------------
# Drift metrics (updated by GET /drift)
# ---------------------------------------------------------------------------

DRIFT_DETECTED = Gauge(
    "network_threat_drift_detected",
    "Set to 1 if data drift was detected in the last /drift check, 0 otherwise.",
)

DRIFT_FEATURES_COUNT = Gauge(
    "network_threat_drift_features",
    "Number of features found to be drifted in the last /drift check.",
)


# ---------------------------------------------------------------------------
# A/B testing metrics
# ---------------------------------------------------------------------------

AB_CHAMPION_COUNTER = Counter(
    "network_threat_ab_champion_total",
    "Total requests served by the champion model in A/B routing.",
)

AB_CHALLENGER_COUNTER = Counter(
    "network_threat_ab_challenger_total",
    "Total requests served (or shadowed) by the challenger model in A/B routing.",
)


def record_prediction(class_name: str, threat_level: str, confidence: float) -> None:
    """Update Prometheus metrics after every prediction.

    Args:
        class_name:   Predicted class label (e.g. ``"BENIGN"``, ``"DDoS"``).
        threat_level: One of ``"LOW"``, ``"MEDIUM"``, ``"HIGH"``, ``"CRITICAL"``.
        confidence:   Model confidence score in [0.0, 1.0].
    """
    PREDICTION_COUNTER.labels(prediction=class_name, threat_level=threat_level).inc()
    CONFIDENCE_HISTOGRAM.observe(confidence)
