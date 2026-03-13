"""Generate evaluation reports for the best trained model."""

import logging
import os
from typing import Any

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

matplotlib.use("Agg")


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


def load_artifacts(
    processed_dir: str,
    models_dir: str,
) -> tuple[np.ndarray, pd.Series, Any, list[str]]:
    """Load test data, best model, and class names from disk.

    Args:
        processed_dir: Directory containing joblib artifacts from preprocess.py.
        models_dir: Directory containing the best model from train.py.

    Returns:
        Tuple ``(X_test, y_test, model, class_names)``.
    """
    X_test: np.ndarray = joblib.load(os.path.join(processed_dir, "X_test.joblib"))
    y_test: pd.Series = joblib.load(os.path.join(processed_dir, "y_test.joblib"))
    model: Any = joblib.load(os.path.join(models_dir, "best_model.joblib"))
    label_encoder = joblib.load(os.path.join(processed_dir, "label_encoder.joblib"))
    class_names: list[str] = list(label_encoder.classes_)

    with open(os.path.join(models_dir, "best_model_name.txt"), "r", encoding="utf-8") as fh:
        model_name: str = fh.read().strip()

    log.info("Evaluating model: %s  |  Test set: %s", model_name, X_test.shape)
    return X_test, y_test, model, class_names


def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: str,
) -> None:
    """Save a normalised confusion matrix heatmap to disk.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        class_names: Ordered list of human-readable class names.
        output_path: File path for the saved PNG.
    """
    cm: np.ndarray = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Normalised count")

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Normalised Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j, i, f"{cm[i, j]:.2f}",
                ha="center", va="center",
                color="white" if cm[i, j] > 0.5 else "black",
                fontsize=6,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info("Saved confusion matrix → %s", output_path)


def save_roc_curves(
    y_true: pd.Series,
    y_prob: np.ndarray,
    class_names: list[str],
    output_path: str,
) -> None:
    """Save One-vs-Rest ROC curves for all classes to disk.

    Args:
        y_true: Ground-truth integer labels.
        y_prob: Predicted probabilities, shape ``(n_samples, n_classes)``.
        class_names: Ordered list of human-readable class names.
        output_path: File path for the saved PNG.
    """
    n_classes: int = len(class_names)
    y_bin: np.ndarray = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = plt.colormaps["tab20"]

    for i, name in enumerate(class_names):
        fpr: np.ndarray
        tpr: np.ndarray
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc: float = float(roc_auc_score(y_bin[:, i], y_prob[:, i]))
        ax.plot(fpr, tpr, color=cmap(i / n_classes), lw=1.5, label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="One-vs-Rest ROC Curves",
        xlim=[0.0, 1.0],
        ylim=[0.0, 1.05],
    )
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info("Saved ROC curves → %s", output_path)


def save_classification_report(
    y_true: pd.Series,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: str,
) -> None:
    """Save a per-class classification report as plain text.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        class_names: Ordered list of human-readable class names.
        output_path: File path for the saved ``.txt`` file.
    """
    report: str = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(report)
    log.info("Saved classification report → %s", output_path)
    log.info("\n%s", report)


def run_evaluation(params_path: str = "params.yaml") -> None:
    """Generate all evaluation reports for the best trained model.

    Saves confusion matrix, ROC curves, and classification report to
    ``reports/``. Invoked by DVC (``dvc repro evaluate``) or directly via
    ``python src/evaluate.py``.

    Args:
        params_path: Path to the YAML parameter file.
    """
    params: dict[str, Any] = load_params(params_path)
    processed_dir: str = params["data"]["processed_dir"]
    models_dir: str = "models"
    reports_dir: str = "reports"
    os.makedirs(reports_dir, exist_ok=True)

    X_test, y_test, model, class_names = load_artifacts(processed_dir, models_dir)

    y_pred: np.ndarray = model.predict(X_test)
    y_prob: np.ndarray = model.predict_proba(X_test)

    save_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        class_names=class_names,
        output_path=os.path.join(reports_dir, "confusion_matrix.png"),
    )
    save_roc_curves(
        y_true=y_test,
        y_prob=y_prob,
        class_names=class_names,
        output_path=os.path.join(reports_dir, "roc_curves.png"),
    )
    save_classification_report(
        y_true=y_test,
        y_pred=y_pred,
        class_names=class_names,
        output_path=os.path.join(reports_dir, "classification_report.txt"),
    )
    log.info("Evaluation complete. Reports saved to '%s/'.", reports_dir)


if __name__ == "__main__":
    run_evaluation()
