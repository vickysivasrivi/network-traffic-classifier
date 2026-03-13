# Network Threat Detection — MLOps Pipeline

An end-to-end, production-grade MLOps system that classifies network traffic flows as **BENIGN or one of 14 attack types** in real time. Covers the complete ML lifecycle: data engineering → model training → REST API → monitoring → automated retraining → cloud deployment.

> **Stack:** Python 3.13 · XGBoost · FastAPI · SHAP · Docker · DVC · MLflow · Prefect · Evidently · Prometheus · Grafana · Terraform · AWS ECS Fargate · GitHub Actions

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Dataset](#dataset)
4. [Model Performance](#model-performance)
5. [API Reference](#api-reference)
6. [Monitoring Stack](#monitoring-stack)
7. [Automated Retraining](#automated-retraining)
8. [A/B Testing](#ab-testing)
9. [Infrastructure as Code](#infrastructure-as-code)
10. [Project Structure](#project-structure)
11. [Getting Started](#getting-started)
12. [CI/CD Pipeline](#cicd-pipeline)
13. [AWS Deployment](#aws-deployment)
14. [Week-by-Week Breakdown](#week-by-week-breakdown)

---

## Project Overview

| Attribute | Value |
|-----------|-------|
| **Task** | Multi-class network intrusion detection |
| **Dataset** | CIC-IDS-2017 (Canadian Institute for Cybersecurity) |
| **Best Model** | XGBoost |
| **Accuracy** | **99.69%** |
| **F1 (weighted)** | **99.74%** |
| **Classes** | 15 (BENIGN + 14 attack types) |
| **Features** | 70 network flow features |
| **API Endpoints** | 9 (inference, monitoring, A/B testing, metadata) |
| **Tests** | 110 (unit + integration + load) |
| **Deployment** | AWS ECS Fargate (eu-west-1) |
| **Infrastructure** | Terraform (ECR, ECS, IAM, CloudWatch) |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            Data Pipeline (DVC)                           │
│                                                                          │
│  CIC-IDS-2017 CSVs  ──►  preprocess.py  ──►  train.py  ──►  evaluate.py │
│  (8 files, 2.8M rows)    SMOTE · Scale       XGBoost        Reports      │
│                           Encode · Split      MLflow log     SHAP plots  │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │  best_model.joblib
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          FastAPI Service (9 endpoints)                   │
│                                                                          │
│  POST /predict ──► Predictor ──► XGBoost ──► SHAP ──► PredictionResponse│
│  POST /predict/batch            (classify)   (explain)  confidence +     │
│  GET  /drift ──► Evidently ──► DriftReport              threat_level +   │
│  POST /ab-test/predict ──► ABRouter ──► champion|challenger  top_features│
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │  /prometheus metrics
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         Monitoring Stack (local)                         │
│                                                                          │
│  Prometheus ──► scrapes /prometheus every 15s ──► Alert rules           │
│  Grafana    ──► dashboards: request rate, latency, drift, predictions    │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      Automated Retraining (Prefect)                      │
│                                                                          │
│  Drift detected ──► dvc repro ──► evaluate ──► quality gate             │
│                                               ──► MLflow Registry       │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        CI/CD & Cloud Deployment                          │
│                                                                          │
│  git push ──► GitHub Actions CI (tests + docker build)                  │
│  push main ──► GitHub Actions CD ──► ECR push ──► ECS Fargate deploy    │
│  infra change ──► Terraform workflow ──► plan + apply                   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Dataset

**CIC-IDS-2017** — a benchmark network intrusion dataset containing real-world attack traffic generated in a controlled lab environment by the Canadian Institute for Cybersecurity.

- **Size:** ~2.8 million flow records across 8 CSV files
- **Features:** 78 raw → 70 after NaN/Inf cleaning
- **Class imbalance:** Handled with **SMOTE** oversampling in preprocessing

### Traffic Classes & Threat Levels

| Class | Threat Level |
|-------|-------------|
| BENIGN | LOW |
| PortScan | MEDIUM |
| FTP-Patator | MEDIUM |
| SSH-Patator | MEDIUM |
| Web Attack – Brute Force | MEDIUM |
| Web Attack – XSS | MEDIUM |
| Bot | HIGH |
| DDoS | HIGH |
| DoS GoldenEye | HIGH |
| DoS Hulk | HIGH |
| DoS Slowhttptest | HIGH |
| DoS slowloris | HIGH |
| Web Attack – SQL Injection | HIGH |
| Heartbleed | CRITICAL |
| Infiltration | CRITICAL |

---

## Model Performance

Three classifiers were trained and compared via MLflow tracking. XGBoost was selected as the production champion.

| Metric | XGBoost (Champion) |
|--------|-------------------|
| **Accuracy** | **99.69%** |
| **F1 (weighted)** | **99.74%** |
| **F1 (macro)** | **84.37%** |
| **Precision (macro)** | **80.24%** |
| **Recall (macro)** | **95.43%** |

### Training Configuration (`params.yaml`)

```yaml
train:
  xgboost:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    subsample: 0.8
```

All hyperparameters are centralised in `params.yaml` and tracked by DVC. Change a value → `dvc repro` → full pipeline re-runs automatically.

---

## API Reference

The FastAPI service exposes **9 endpoints** on port **8000**. Interactive docs: `http://localhost:8000/docs`

### Monitoring

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check — confirms model is loaded |
| `GET` | `/drift` | Runs Evidently drift report on last 500 predictions vs training baseline |
| `GET` | `/prometheus` | Prometheus metrics scrape endpoint |

### Metadata

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/features` | Returns ordered list of 70 feature names |
| `GET` | `/classes` | Returns all 15 classes with threat levels |
| `GET` | `/metrics` | Returns model evaluation metrics from last training run |

### Inference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Classify a single network flow with SHAP explanation |
| `POST` | `/predict/batch` | Classify multiple flows in one vectorised call |

### A/B Testing

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/ab-test/config` | Returns current A/B router configuration |
| `POST` | `/ab-test/predict` | Routes request through champion-challenger router |

### Example: `POST /predict`

**Request:**
```json
{
  "features": [80, 123456, 1500, ...]
}
```

**Response:**
```json
{
  "prediction": "BENIGN",
  "confidence": 0.991,
  "threat_level": "LOW",
  "top_features": [
    {"feature": "Init_Win_bytes_forward", "impact": 2.6708},
    {"feature": "min_seg_size_forward",   "impact": 0.819},
    {"feature": "Fwd Header Length",      "impact": 0.6702}
  ]
}
```

---

## Monitoring Stack

The monitoring stack runs locally via `docker-compose up` and includes three services.

```
docker-compose up --build

http://localhost:8000/docs   → FastAPI Swagger UI
http://localhost:9090        → Prometheus UI + alert rules
http://localhost:3000        → Grafana dashboards (admin / admin)
```

### Grafana Dashboards

| Panel | Metric |
|-------|--------|
| HTTP Request Rate | Requests per second per endpoint |
| Request Latency (p50 / p95) | Latency percentiles in milliseconds |
| Prediction Class Rate | Distribution of predicted attack classes |
| Data Drift Status | OK / DRIFT (Evidently report) |
| Drifted Features Count | Number of features with statistical drift |

### Prometheus Alert Rules

Four alerting rules defined in `infra/alerting/rules.yml`:

| Alert | Condition |
|-------|-----------|
| `APIDown` | API unreachable for > 1 minute |
| `APIHighErrorRate` | 5xx rate > 5% over 5 minutes |
| `DataDriftDetected` | `network_threat_drift_detected == 1` |
| `HighPredictionLatency` | p95 latency > 2 seconds for 5 minutes |

---

## Automated Retraining

The retraining pipeline is orchestrated with **Prefect** and integrates DVC, MLflow, and a model quality gate.

```
check_drift()
    └── drift detected OR force=True
            └── dvc repro  (preprocess → train → evaluate)
                    └── validate_model_gate()
                            └── candidate accuracy >= champion accuracy
                                    └── register_and_promote() → MLflow Registry
```

### Quality Gate

Before any new model is promoted to production, `validate_model_gate()` evaluates it on the held-out test set and compares it against the current champion. **Promotion is blocked if the new model performs worse** — preventing silent regressions.

### Trigger Retraining

```bash
# Manual trigger via CLI
python -c "from src.retrain_flow import retrain_flow; retrain_flow(force=True)"

# Or via GitHub Actions (Actions → Manual Retrain → Run workflow)
```

---

## A/B Testing

The `ABRouter` implements the **champion-challenger pattern** for safe model rollouts.

### Modes

| Mode | Behaviour |
|------|-----------|
| **Shadow mode** (default) | Both models run on every request; only champion result is returned; challenger predictions logged for offline analysis. Zero production risk. |
| **Traffic split** | Set `AB_TRAFFIC_SPLIT=0.1` to route 10% of live traffic to the challenger. |

### Configuration

```bash
# Enable traffic split (10% to challenger)
export AB_TRAFFIC_SPLIT=0.1

# Load a challenger model
export CHALLENGER_MODEL_PATH=models/challenger/

# Disable shadow mode (use traffic split instead)
export AB_SHADOW_MODE=false
```

**GET /ab-test/config** response:
```json
{
  "shadow_mode": true,
  "traffic_split": 0.0,
  "challenger_loaded": false
}
```

---

## Infrastructure as Code

All AWS infrastructure is defined in Terraform under `infra/terraform/`.

```
infra/terraform/
├── main.tf         # AWS provider, default tags
├── variables.tf    # All configurable inputs
├── outputs.tf      # ECR URL, cluster ARN, log group name
├── ecr.tf          # ECR repository + lifecycle policy (keep last 10 images)
├── ecs.tf          # ECS cluster, task definition, CloudWatch log group
└── iam.tf          # ECS task execution IAM role + policy
```

### Resources Managed

| Resource | Name |
|----------|------|
| ECR Repository | `network-threat-classifier` |
| ECS Cluster | `network-threat-cluster` |
| ECS Task Definition | `network-threat-api` (512 CPU / 1024 MB) |
| CloudWatch Log Group | `/ecs/network-threat` (30-day retention) |
| IAM Role | `network-threat-ecs-task-execution` |

### Commands

```bash
cd infra/terraform
terraform init
terraform plan
terraform apply

# Import existing resources into state if needed
terraform import aws_ecr_repository.api network-threat-classifier
```

---

## Project Structure

```
network-traffic-classifier/
│
├── api/                            # FastAPI inference service
│   ├── main.py                     # 9 endpoints, lifespan model loading
│   ├── predictor.py                # Predictor: scale → infer → SHAP explain
│   ├── ab_router.py                # Champion-challenger A/B router
│   ├── monitoring.py               # Prometheus metric counters
│   └── schemas.py                  # Pydantic request/response models
│
├── src/                            # ML pipeline scripts
│   ├── preprocess.py               # NaN/Inf cleaning, SMOTE, scale, encode
│   ├── train.py                    # Train XGBoost/RF/LightGBM, MLflow tracking
│   ├── evaluate.py                 # Confusion matrix, ROC curves, SHAP plots
│   ├── retrain_flow.py             # Prefect retraining orchestration flow
│   ├── validate_model.py           # Model quality gate
│   └── promote.py                  # MLflow Registry promotion
│
├── tests/                          # 110 pytest tests
│   ├── test_preprocess.py          # Preprocessing unit tests
│   ├── test_api.py                 # API endpoint integration tests
│   ├── test_ab.py                  # A/B router tests
│   ├── test_monitoring.py          # Drift detection + Prometheus tests
│   ├── test_retrain.py             # Retraining flow tests
│   └── load_test.py                # Locust load testing scenarios
│
├── infra/
│   ├── terraform/                  # Terraform IaC (ECR, ECS, IAM, CloudWatch)
│   ├── alerting/
│   │   ├── rules.yml               # 4 Prometheus alerting rules
│   │   └── alertmanager.yml        # Alert routing configuration
│   ├── grafana/
│   │   └── provisioning/           # Auto-provisioned datasource + dashboard
│   ├── prometheus.yml              # Prometheus scrape config
│   └── ecs_task_definition.json    # ECS task template for CD workflow
│
├── .github/workflows/
│   ├── ci.yml                      # Tests + Docker build on every push
│   ├── cd.yml                      # ECR push + ECS deploy on main
│   ├── terraform.yml               # Plan on PR, apply on main
│   └── retrain.yml                 # Manual retraining trigger
│
├── data/
│   ├── raw/                        # CIC-IDS-2017 CSVs (DVC-tracked)
│   └── processed/                  # Scaled features, encoders (DVC-tracked)
│
├── models/
│   ├── best_model.joblib           # Trained XGBoost (3.1 MB)
│   └── best_model_name.txt
│
├── Dockerfile                      # Multi-stage: builder + slim runtime
├── docker-compose.yml              # API + Prometheus + Grafana stack
├── dvc.yaml                        # Pipeline: preprocess → train → evaluate
├── params.yaml                     # All hyperparameters (DVC-tracked)
├── metrics.json                    # Latest model metrics snapshot
└── requirements.txt                # Full dev + training dependencies
```

---

## Getting Started

### Prerequisites

| Tool | Version | Check |
|------|---------|-------|
| Python (Conda) | 3.13 | `python --version` |
| Docker Desktop | Latest | `docker info` |
| Git | Any | `git --version` |
| AWS CLI | v2 | `aws --version` |
| Terraform | 1.7+ | `terraform --version` |

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/network-traffic-classifier.git
cd network-traffic-classifier
pip install -r requirements.txt
```

### 2. Run the ML Pipeline

```bash
# Requires CIC-IDS-2017 CSVs in data/raw/
dvc repro
# Runs: preprocess → train → evaluate
# Produces: models/best_model.joblib, metrics.json, reports/
```

### 3. Start the API

```bash
uvicorn api.main:app --reload
# Open: http://localhost:8000/docs
```

### 4. Start the Full Monitoring Stack

```bash
docker-compose up --build
# API:        http://localhost:8000/docs
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000  (admin / admin)
```

### 5. Run All Tests

```bash
pytest tests/ -v
# Expected: 110 passed
```

### 6. Load Test

```bash
locust -f tests/load_test.py --host http://localhost:8000 \
       --headless -u 10 -r 2 --run-time 30s
```

---

## CI/CD Pipeline

Four GitHub Actions workflows automate the full delivery lifecycle.

| Workflow | Trigger | Action |
|----------|---------|--------|
| `ci.yml` | Every push / PR | Run tests + validate Docker build |
| `cd.yml` | Push to `main` | Build image → push to ECR → deploy to ECS Fargate |
| `terraform.yml` | `infra/terraform/**` changes | `plan` on PR · `apply` on merge to main |
| `retrain.yml` | Manual dispatch | Run Prefect retraining flow |

### Required GitHub Secrets

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |

### Image Tagging Strategy

Every CD run pushes two tags to ECR:

| Tag | Purpose |
|-----|---------|
| `:latest` | Always points to the newest build |
| `:<7-char-git-sha>` | Immutable — enables rollback to any specific commit |

---

## AWS Deployment

**Region:** `eu-west-1` (Ireland)

### Infrastructure (Terraform-managed)

| Component | Resource |
|-----------|---------|
| Container Registry | Amazon ECR — `network-threat-classifier` |
| Compute | ECS Fargate — `network-threat-cluster` |
| Task | `network-threat-api` — 512 CPU / 1024 MB RAM |
| Logs | CloudWatch — `/ecs/network-threat` (30-day retention) |
| IAM | `network-threat-ecs-task-execution` role |

### Push Docker Image to ECR

```powershell
# Authenticate (PowerShell)
$TOKEN = python -m awscli ecr get-login-password --region eu-west-1
$TOKEN | docker login --username AWS --password-stdin <account-id>.dkr.ecr.eu-west-1.amazonaws.com

# Build, tag, push
docker build -t network-threat-classifier .
docker tag network-threat-classifier:latest <account-id>.dkr.ecr.eu-west-1.amazonaws.com/network-threat-classifier:latest
docker push <account-id>.dkr.ecr.eu-west-1.amazonaws.com/network-threat-classifier:latest
```

### Create ECS Service

```bash
aws ecs create-service \
  --cluster network-threat-cluster \
  --service-name network-threat-service \
  --task-definition network-threat-api \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-XXXX],securityGroups=[sg-XXXX],assignPublicIp=ENABLED}" \
  --region eu-west-1
```

---

## Week-by-Week Breakdown

### Week 1 — Data Engineering (`src/preprocess.py`)
- Loaded and merged 8 CIC-IDS-2017 CSV files (~2.8M rows)
- Removed columns with >50% NaN; replaced Inf with column max
- Applied **SMOTE** to oversample minority attack classes
- Fitted **StandardScaler** and **LabelEncoder** — serialised with joblib
- DVC pipeline stage: parameterised via `params.yaml`, reproducible with `dvc repro`

### Week 2 — Model Training & Experiment Tracking (`src/train.py`)
- Trained 3 classifiers: XGBoost, Random Forest, LightGBM
- All experiments tracked in **MLflow** (parameters, metrics, model artifacts)
- Best model selected by weighted F1: **XGBoost (99.74%)**
- Evaluation stage generates confusion matrix, ROC curves, SHAP plots

### Week 3 — FastAPI Inference Service (`api/`)
- 6 REST endpoints with SHAP explainability on every prediction
- Threat level mapping: each class tagged LOW / MEDIUM / HIGH / CRITICAL
- Pydantic v2 request/response validation

### Week 4 — Docker + CI/CD + AWS ECS Fargate
- Multi-stage Dockerfile: builder (GCC for SHAP) + lean runtime
- GitHub Actions CI (test + docker build) and CD (ECR push + ECS deploy)
- AWS ECS Fargate with CloudWatch logging

### Week 5 — Prometheus + Grafana Monitoring
- `prometheus-fastapi-instrumentator` for automatic RED metrics
- Custom counters: prediction class distribution, threat levels, confidence
- Grafana dashboards with auto-provisioned Prometheus datasource

### Week 6 — Evidently Drift Detection + Prefect Retraining
- `GET /drift` runs Evidently data drift report (last 500 predictions vs 1000-row baseline)
- Prefect flow: `check_drift → dvc repro → evaluate → promote_if_better`
- MLflow Model Registry integration for champion model management

### Week 7 — A/B Testing + Model Quality Gate
- `ABRouter` with shadow mode and configurable traffic split
- `validate_model_gate()` blocks promotion if candidate underperforms champion
- New endpoints: `GET /ab-test/config`, `POST /ab-test/predict`

### Week 8 — Terraform IaC + Alerting + Load Testing
- Full Terraform stack: ECR, ECS, IAM, CloudWatch — version-controlled infrastructure
- 4 Prometheus alerting rules with Alertmanager routing
- Locust load testing: sustained throughput validation
- Terraform GitHub Actions workflow: plan on PR, apply on merge

---

## Licence

This project is for educational and portfolio purposes. The CIC-IDS-2017 dataset is provided by the Canadian Institute for Cybersecurity for research use.
