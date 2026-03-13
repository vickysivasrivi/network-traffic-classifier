"""Model training with MLflow experiment tracking for CICIDS 2017."""

import json
import logging
import os
from typing import Any

import joblib
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

matplotlib.use("Agg")  # Non-interactive backend — safe for scripts without a display


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log: logging.Logger = logging.getLogger(__name__)


def load_params(params_path: str = "params.yaml") -> dict[str, Any]:
    """Load pipeline parameters from a YAML file.

    Args:
        params_path: Path to the YAML config file.

    Returns:
        Nested dictionary of pipeline parameters.
    """
    with open(params_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_processed_data(
    processed_dir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series, list[str]]:
    """Load all processed arrays from disk.

    Args:
        processed_dir: Directory containing joblib artifacts from preprocess.py.

    Returns:
        Tuple ``(X_train, X_test, y_train, y_test, feature_names)``.
    """
    X_train: np.ndarray = joblib.load(os.path.join(processed_dir, "X_train.joblib"))
    X_test: np.ndarray = joblib.load(os.path.join(processed_dir, "X_test.joblib"))
    y_train: np.ndarray = joblib.load(os.path.join(processed_dir, "y_train.joblib"))
    y_test: pd.Series = joblib.load(os.path.join(processed_dir, "y_test.joblib"))

    feature_names_path: str = os.path.join(processed_dir, "feature_names.txt")
    with open(feature_names_path, "r", encoding="utf-8") as fh:
        feature_names: list[str] = fh.read().splitlines()

    log.info(
        "Loaded — X_train: %s  X_test: %s  Classes: %d",
        X_train.shape,
        X_test.shape,
        len(np.unique(y_train)),
    )
    return X_train, X_test, y_train, y_test, feature_names


def build_models(
    params: dict[str, Any],
) -> dict[str, RandomForestClassifier | XGBClassifier | LGBMClassifier]:
    """Instantiate all three classifiers from params.yaml hyperparameters.

    Args:
        params: Full params dict (reads from ``train`` section).

    Returns:
        Dict mapping model name to unfitted classifier instance.
    """
    tp: dict[str, Any] = params["train"]
    random_state: int = tp["random_state"]
    n_jobs: int = tp["n_jobs"]
    rf: dict[str, Any] = tp["random_forest"]
    xgb: dict[str, Any] = tp["xgboost"]
    lgbm: dict[str, Any] = tp["lightgbm"]

    return {
        "random_forest": RandomForestClassifier(
            n_estimators=rf["n_estimators"],
            max_depth=rf["max_depth"],
            min_samples_split=rf["min_samples_split"],
            random_state=random_state,
            n_jobs=n_jobs,
        ),
        "xgboost": XGBClassifier(
            n_estimators=xgb["n_estimators"],
            max_depth=xgb["max_depth"],
            learning_rate=xgb["learning_rate"],
            subsample=xgb["subsample"],
            random_state=random_state,
            n_jobs=n_jobs,
            eval_metric="mlogloss",
            verbosity=0,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=lgbm["n_estimators"],
            max_depth=lgbm["max_depth"],
            learning_rate=lgbm["learning_rate"],
            num_leaves=lgbm["num_leaves"],
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=-1,
        ),
    }


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute classification metrics for a single model run.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        Dict with keys: ``accuracy``, ``f1_macro``, ``f1_weighted``,
        ``precision_macro``, ``recall_macro``.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def save_confusion_matrix_plot(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: str,
) -> None:
    """Save a confusion matrix heatmap to disk.

    Args:
        y_true: Ground-truth labels (integer-encoded).
        y_pred: Predicted labels (integer-encoded).
        class_names: Ordered list of human-readable class names.
        output_path: File path for the saved PNG.
    """
    cm: np.ndarray = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def train_and_log(
    model_name: str,
    model: RandomForestClassifier | XGBClassifier | LGBMClassifier,
    model_params: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray | pd.Series,
    class_names: list[str],
    tmp_dir: str,
) -> dict[str, float]:
    """Train one model inside an MLflow run, log everything, return metrics.

    Args:
        model_name: Human-readable name used as MLflow run name.
        model: Unfitted sklearn-compatible classifier.
        model_params: Hyperparameter dict to log to MLflow.
        X_train: Training feature matrix.
        y_train: Training labels.
        X_test: Test feature matrix.
        y_test: Test labels.
        class_names: Ordered list of class name strings.
        tmp_dir: Temporary directory for artifact files before logging.

    Returns:
        Metrics dict from :func:`compute_metrics`.
    """
    log.info("Training %s ...", model_name)

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(model_params)

        model.fit(X_train, y_train)
        y_pred: np.ndarray = model.predict(X_test)

        metrics: dict[str, float] = compute_metrics(y_test, y_pred)
        mlflow.log_metrics(metrics)

        cm_path: str = os.path.join(tmp_dir, f"{model_name}_confusion_matrix.png")
        save_confusion_matrix_plot(y_test, y_pred, class_names, cm_path)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        mlflow.sklearn.log_model(model, artifact_path="model")

    log.info(
        "%s — f1_macro: %.4f  accuracy: %.4f",
        model_name,
        metrics["f1_macro"],
        metrics["accuracy"],
    )
    return metrics


def save_best_model(
    best_name: str,
    best_model: RandomForestClassifier | XGBClassifier | LGBMClassifier,
    best_metrics: dict[str, float],
    models_dir: str,
) -> None:
    """Persist the best model and write metrics.json for DVC.

    Args:
        best_name: Model name (e.g. ``"random_forest"``).
        best_model: Fitted classifier instance.
        best_metrics: Metrics dict for the best run.
        models_dir: Output directory for model artifacts.
    """
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(best_model, os.path.join(models_dir, "best_model.joblib"))

    with open(os.path.join(models_dir, "best_model_name.txt"), "w", encoding="utf-8") as fh:
        fh.write(best_name)

    metrics_payload: dict[str, Any] = {"best_model": best_name, **best_metrics}
    with open("metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics_payload, fh, indent=2)

    log.info("Best model: %s  (f1_macro=%.4f)", best_name, best_metrics["f1_macro"])
    log.info("Saved to '%s/' and metrics.json.", models_dir)


def run_training(params_path: str = "params.yaml") -> None:
    """Orchestrate training for all models and save the best one.

    Trains Random Forest, XGBoost, and LightGBM. Each run is logged to
    MLflow under the ``"network-traffic-classifier"`` experiment.
    The best model by ``f1_macro`` is saved to ``models/``.

    Args:
        params_path: Path to the YAML parameter file.
    """
    params: dict[str, Any] = load_params(params_path)
    processed_dir: str = params["data"]["processed_dir"]
    models_dir: str = "models"
    tmp_dir: str = "tmp_artifacts"
    os.makedirs(tmp_dir, exist_ok=True)

    X_train, X_test, y_train, y_test, _ = load_processed_data(processed_dir)

    label_encoder = joblib.load(os.path.join(processed_dir, "label_encoder.joblib"))
    class_names: list[str] = list(label_encoder.classes_)

    mlflow.set_experiment("network-traffic-classifier")

    models: dict[str, Any] = build_models(params)
    tp: dict[str, Any] = params["train"]

    model_hyperparams: dict[str, dict[str, Any]] = {
        "random_forest": tp["random_forest"],
        "xgboost": tp["xgboost"],
        "lightgbm": tp["lightgbm"],
    }

    all_metrics: dict[str, dict[str, float]] = {}
    trained_models: dict[str, Any] = {}

    for model_name, model in models.items():
        metrics: dict[str, float] = train_and_log(
            model_name=model_name,
            model=model,
            model_params=model_hyperparams[model_name],
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            class_names=class_names,
            tmp_dir=tmp_dir,
        )
        all_metrics[model_name] = metrics
        trained_models[model_name] = model

    best_name: str = max(all_metrics, key=lambda k: all_metrics[k]["f1_macro"])
    save_best_model(
        best_name=best_name,
        best_model=trained_models[best_name],
        best_metrics=all_metrics[best_name],
        models_dir=models_dir,
    )

    log.info("Training complete. Run `mlflow ui` to compare all runs.")


if __name__ == "__main__":
    run_training()
