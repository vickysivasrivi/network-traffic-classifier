# =============================================================================
# Stage 1 — Builder
#
# Installs Python dependencies into /root/.local (user-space install).
# This stage is discarded after its compiled packages are copied to runtime.
#
# WHY multi-stage?
#   SHAP requires GCC to compile its C extensions. Including GCC in the final
#   image adds ~200 MB and widens the attack surface unnecessarily.
#   Multi-stage lets us build with the compiler and ship without it.
# =============================================================================
FROM python:3.13-slim AS builder

WORKDIR /build

# Install build tools needed by SHAP's C extensions.
# --no-install-recommends keeps this layer as lean as possible.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .

# --user installs packages into /root/.local instead of the system site-packages.
# --no-cache-dir avoids storing the pip download cache in the image layer.
RUN pip install --user --no-cache-dir -r requirements-api.txt


# =============================================================================
# Stage 2 — Runtime
#
# Lean production image. Only compiled packages + inference artifacts.
# No compiler, no pip cache, no training code, no test suite.
# =============================================================================
FROM python:3.13-slim AS runtime

WORKDIR /app

# Transfer the compiled packages from the builder stage.
# Nothing from the builder (compiler, pip cache, build headers) comes with it.
COPY --from=builder /root/.local /root/.local

# --- Targeted artifact copy --------------------------------------------------
# Copying only what the API needs at inference time.
# .dockerignore already excludes the 2 GB of training data from the build
# context; these targeted COPYs add a second line of defence.

# FastAPI application code
COPY api/            ./api/

# Trained XGBoost model (~3.1 MB)
COPY models/         ./models/

# Preprocessing artifacts used at inference time
COPY data/processed/scaler.joblib        ./data/processed/scaler.joblib
COPY data/processed/label_encoder.joblib ./data/processed/label_encoder.joblib
COPY data/processed/feature_names.txt   ./data/processed/feature_names.txt

# Metrics file — loaded by api/main.py lifespan with open("metrics.json", "r")
COPY metrics.json    ./metrics.json

# --- Environment -------------------------------------------------------------
# The uvicorn binary is installed in /root/.local/bin via --user install.
ENV PATH=/root/.local/bin:$PATH

# WORKDIR=/app + PYTHONPATH=/app ensures both:
#   - `from api.predictor import Predictor` resolves (api/ is a package under /app)
#   - open("models/best_model.joblib") resolves (relative to /app)
ENV PYTHONPATH=/app

# --- Network -----------------------------------------------------------------
EXPOSE 8000

# --- Healthcheck -------------------------------------------------------------
# python:3.13-slim does NOT include curl.
# We use Python's stdlib urllib instead — zero extra dependencies.
# start-period=60s: SHAP TreeExplainer initialisation takes ~20 s on first load;
# this prevents ECS/docker-compose from marking the container unhealthy prematurely.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
        || exit 1

# --- Entrypoint --------------------------------------------------------------
# Exec form (JSON array): uvicorn is PID 1 and receives SIGTERM directly.
# Shell form would wrap uvicorn in /bin/sh, which buffers signals and delays
# graceful shutdown on ECS task stops.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
