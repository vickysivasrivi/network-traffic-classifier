"""Locust load test for the Network Threat Detection API.

Run with a live API server (e.g. ``uvicorn api.main:app``) or against the
Docker stack (``docker-compose up``):

    locust -f tests/load_test.py --host http://localhost:8000 \\
           --headless -u 10 -r 2 --run-time 30s

Or open the Locust web UI for interactive control:

    locust -f tests/load_test.py --host http://localhost:8000

Tasks are weighted so that single /predict calls dominate (realistic traffic):
    - predict_single  → weight 3
    - predict_batch   → weight 1
    - check_health    → weight 1
    - ab_predict      → weight 2
    - get_ab_config   → weight 1
"""

from locust import HttpUser, between, task

_FEATURES: list[float] = [0.0] * 70
_BATCH_SIZE: int = 5


class NetworkThreatUser(HttpUser):
    """Simulates a client that classifies network flows in real time."""

    wait_time = between(0.1, 1.0)

    @task(3)
    def predict_single(self) -> None:
        """POST /predict — single network flow classification."""
        self.client.post(
            "/predict",
            json={"features": _FEATURES},
            name="/predict",
        )

    @task(1)
    def predict_batch(self) -> None:
        """POST /predict/batch — batch of 5 flows."""
        self.client.post(
            "/predict/batch",
            json={"flows": [{"features": _FEATURES}] * _BATCH_SIZE},
            name="/predict/batch",
        )

    @task(1)
    def check_health(self) -> None:
        """GET /health — liveness check."""
        self.client.get("/health", name="/health")

    @task(2)
    def ab_predict(self) -> None:
        """POST /ab-test/predict — A/B router endpoint."""
        self.client.post(
            "/ab-test/predict",
            json={"features": _FEATURES},
            name="/ab-test/predict",
        )

    @task(1)
    def get_ab_config(self) -> None:
        """GET /ab-test/config — A/B configuration inspection."""
        self.client.get("/ab-test/config", name="/ab-test/config")
