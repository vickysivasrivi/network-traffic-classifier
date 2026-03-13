"""Pydantic v2 request and response schemas for the threat detection API."""

from pydantic import BaseModel, Field


class NetworkFlowRequest(BaseModel):
    """Single network flow input — 70 raw feature values in ``feature_names.txt`` order."""

    features: list[float] = Field(..., min_length=70, max_length=70)


class FeatureImpact(BaseModel):
    """SHAP contribution of one feature to a prediction."""

    feature: str
    impact: float


class PredictionResponse(BaseModel):
    """Single-flow prediction result with confidence and SHAP explainability."""

    prediction: str
    confidence: float
    threat_level: str
    top_features: list[FeatureImpact]


class BatchPredictionRequest(BaseModel):
    """Batch of network flows for bulk classification."""

    flows: list[NetworkFlowRequest]


class BatchPredictionResponse(BaseModel):
    """Ordered list of prediction results matching the input batch."""

    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    """API liveness and model metadata."""

    status: str
    model: str
    n_features: int


class FeaturesResponse(BaseModel):
    """Ordered list of the 70 feature names expected in each request."""

    features: list[str]
    count: int


class ClassInfo(BaseModel):
    """A single attack class name paired with its threat level."""

    name: str
    threat_level: str


class ClassesResponse(BaseModel):
    """All attack class names the model can predict, each with its threat level."""

    classes: list[ClassInfo]
    count: int


class MetricsResponse(BaseModel):
    """Model evaluation metrics from the last training run."""

    best_model: str
    accuracy: float
    f1_macro: float
    f1_weighted: float
    precision_macro: float
    recall_macro: float


class DriftResponse(BaseModel):
    """Evidently data-drift report summary for the /drift endpoint."""

    drift_detected: bool
    n_drifted_features: int
    n_features_total: int
    share_drifted: float          # n_drifted / n_features_total
    report_saved_at: str | None = None


class ABTestConfig(BaseModel):
    """Current configuration of the A/B router."""

    shadow_mode: bool
    traffic_split: float
    challenger_loaded: bool


class ABPredictionResponse(PredictionResponse):
    """Prediction result annotated with which model served the request."""

    served_by: str  # "champion" or "challenger"
