"""Register a trained model in the MLflow Model Registry and promote it to Production.

Usage (called automatically by src/retrain_flow.py after a successful retrain):

    from src.promote import register_and_promote
    register_and_promote("runs:/<run_id>/model")

You can also run it manually:

    python -m src.promote runs:/<run_id>/model
"""

import logging
import sys

import mlflow
from mlflow import MlflowClient

log: logging.Logger = logging.getLogger(__name__)

MODEL_NAME: str = "network-threat"


def register_and_promote(model_uri: str, model_name: str = MODEL_NAME) -> str:
    """Register a model artifact and transition it to the *Production* stage.

    Any previously *Production* version is automatically moved to *Archived*,
    so there is always exactly one Production version.

    Args:
        model_uri:  MLflow model URI, e.g. ``"runs:/<run_id>/model"`` or a
                    local path prefixed with ``"file://"``.
        model_name: Registered model name in the MLflow Model Registry.
                    Defaults to ``"network-threat"``.

    Returns:
        The new model version string (e.g. ``"3"``).
    """
    log.info("Registering model '%s' from %s ...", model_name, model_uri)
    result = mlflow.register_model(model_uri, model_name)
    version: str = result.version

    client = MlflowClient()

    # Set the "production" alias on the new version.
    # Aliases replaced the deprecated stage system in MLflow 2.9+.
    # The API now loads models via "models:/network-threat@production".
    client.set_registered_model_alias(
        name=model_name,
        alias="production",
        version=version,
    )
    log.info(
        "Model '%s' version %s tagged as alias 'production'.",
        model_name,
        version,
    )
    return version


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.promote <model_uri> [model_name]")
        sys.exit(1)
    uri = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else MODEL_NAME
    register_and_promote(uri, name)
