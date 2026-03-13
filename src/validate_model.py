"""Model quality gate — prevents regressions before MLflow promotion.

Called by :func:`src.retrain_flow.promote_if_better` before registering a
new model version.  The candidate model must match or beat the current
champion on the held-out test set, otherwise promotion is blocked.

Standalone usage::

    from src.validate_model import validate_model_gate
    passed = validate_model_gate("models/best_model.joblib")
    print("promote" if passed else "skip")
"""

import json
import logging
import os

import joblib
from sklearn.metrics import accuracy_score, f1_score

log = logging.getLogger(__name__)


def load_test_set(processed_dir: str = "data/processed") -> tuple:
    """Load the held-out test arrays produced by the DVC preprocess stage.

    Args:
        processed_dir: Directory containing ``X_test.joblib`` and
                       ``y_test.joblib``.

    Returns:
        Tuple ``(X_test, y_test)`` as numpy arrays.
    """
    X_test = joblib.load(os.path.join(processed_dir, "X_test.joblib"))
    y_test = joblib.load(os.path.join(processed_dir, "y_test.joblib"))
    return X_test, y_test


def evaluate_candidate(model, X_test, y_test) -> dict:
    """Score *model* on the test set and return a metrics dict.

    Args:
        model:  Scikit-learn compatible estimator with a ``predict`` method.
        X_test: Feature matrix (numpy array or compatible).
        y_test: True labels (numpy array or compatible).

    Returns:
        Dict with keys ``"accuracy"`` and ``"f1_macro"`` (both ``float``).
    """
    y_pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }


def compare_against_champion(
    candidate: dict,
    champion: dict,
    min_improvement: float = 0.0,
) -> bool:
    """Return ``True`` if the candidate accuracy is at least as good as the champion.

    Args:
        candidate:       Metrics dict for the newly trained model.
        champion:        Metrics dict for the currently deployed champion.
        min_improvement: Extra accuracy margin the candidate must clear above the
                         champion.  ``0.0`` (default) means equal-or-better.

    Returns:
        ``True`` if ``candidate["accuracy"] >= champion["accuracy"] - min_improvement``.
    """
    return float(candidate.get("accuracy", 0.0)) >= (
        float(champion.get("accuracy", 0.0)) - min_improvement
    )


def validate_model_gate(
    model_path: str,
    processed_dir: str = "data/processed",
    champion_metrics_path: str = "metrics.json",
) -> bool:
    """Load the candidate model, evaluate it, and compare against the champion.

    Args:
        model_path:             Path to the candidate ``best_model.joblib``.
        processed_dir:          Directory containing the held-out test arrays.
        champion_metrics_path:  Path to the current champion ``metrics.json``.

    Returns:
        ``True`` if the candidate passes the quality gate (accuracy ≥ champion).
        ``False`` if any step fails or the candidate is worse.
    """
    try:
        model = joblib.load(model_path)
    except Exception as exc:
        log.warning("Quality gate: could not load candidate model at %s: %s", model_path, exc)
        return False

    try:
        X_test, y_test = load_test_set(processed_dir)
    except Exception as exc:
        log.warning("Quality gate: could not load test set from %s: %s", processed_dir, exc)
        return False

    try:
        with open(champion_metrics_path, "r", encoding="utf-8") as fh:
            champion_metrics = json.load(fh)
    except Exception as exc:
        log.warning(
            "Quality gate: could not load champion metrics from %s: %s",
            champion_metrics_path,
            exc,
        )
        return False

    candidate_metrics = evaluate_candidate(model, X_test, y_test)
    passed = compare_against_champion(candidate_metrics, champion_metrics)

    log.info(
        "Quality gate: candidate accuracy=%.4f  champion accuracy=%.4f  passed=%s",
        candidate_metrics.get("accuracy", 0.0),
        champion_metrics.get("accuracy", 0.0),
        passed,
    )
    return passed
