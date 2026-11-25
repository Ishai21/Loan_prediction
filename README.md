# Loan Approval Prediction

A two-stage machine learning service that decides whether a loan application is approved and, if it is, predicts the sanctioned loan amount. The same model is exposed two ways — an interactive **Streamlit** UI for humans and a **FastAPI** JSON endpoint for machines — and both ship in a single container deployed on AWS.

![CI/CD](https://github.com/Ishai21/Loan_prediction/actions/workflows/ci-cd.yml/badge.svg)

---

## Why two stages

Predicting "how much will be sanctioned?" only makes sense for applications that would be approved in the first place. Training a single regressor over all applicants forces it to fit rejected rows that have no meaningful sanctioned amount. So the pipeline splits the problem:

| Stage | Model | Task | Output |
|-------|-------|------|--------|
| 1 | Random Forest **Classifier** | Approve or reject | `approve: 0 \| 1` |
| 2 | Random Forest **Regressor** | Sanctioned amount | `regression_prediction: float` |

Stage 2 runs **only** when stage 1 returns approved; otherwise the amount is returned as `null`. Both stages are saved as complete scikit-learn `Pipeline` objects, so preprocessing (encoding, scaling) travels with the model and inference needs no feature engineering at call time.

Implementation: [`loan_model.py`](loan_model.py) → `LoanApprovalApp.two_stage_predict()`

---

## Features

The model consumes 11 applicant attributes:

| Feature | Type | Notes |
|---|---|---|
| `no_of_dependents` | int | |
| `education` | str | `Graduate` / `Not Graduate` |
| `self_employed` | str | `Yes` / `No` |
| `income_annum` | float | Annual income |
| `loan_amount` | float | Amount requested |
| `loan_term` | int | Term in years |
| `cibil_score` | int | Credit score — the dominant predictor |
| `residential_assets_value` | float | |
| `commercial_assets_value` | float | |
| `luxury_assets_value` | float | |
| `bank_asset_value` | float | |

Trained on `data/loan_approval_dataset.csv` (4,269 records). Exploratory analysis and model selection are in [`analyze.ipynb`](analyze.ipynb).

---

## The two outputs

One container, two ports, one shared model layer.

```
                    ┌──────────────────────────────┐
                    │        Docker container       │
                    │                              │
  Browser  ────────▶│  Streamlit UI      :8501     │
                    │        │                     │
                    │        ▼                     │
  curl / client ───▶│  FastAPI /predict  :8000     │
                    │        │                     │
                    │        ▼                     │
                    │  LoanApprovalApp             │
                    │   ├─ stage_1_rf_classifier   │
                    │   └─ stage_2_rf_regression   │
                    └──────────────────────────────┘
```

Both processes are launched by [`start.sh`](start.sh): uvicorn is backgrounded and Streamlit runs in the foreground as PID 1.

### 1. Streamlit UI — port 8501

A form-driven interface in [`app.py`](app.py). Enter applicant details, press **Predict**, and get an approve/reject verdict plus the predicted amount. The model is loaded once per session via `@st.cache_resource` rather than on every rerun.

### 2. FastAPI service — port 8000

A JSON API in [`api.py`](api.py) with request validation via Pydantic. Interactive docs at `/docs`.

**`POST /predict`**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "no_of_dependents": 2,
    "education": "Graduate",
    "self_employed": "No",
    "income_annum": 9600000,
    "loan_amount": 29900000,
    "loan_term": 12,
    "cibil_score": 778,
    "residential_assets_value": 2400000,
    "commercial_assets_value": 17600000,
    "luxury_assets_value": 22700000,
    "bank_asset_value": 8000000
  }'
```

```json
{
  "status": "success",
  "data": {
    "approve": 1,
    "regression_prediction": 29412847.63
  }
}
```

A rejected application returns `"approve": 0` and `"regression_prediction": null`.

---

## Running locally

### With Python

```bash
git clone https://github.com/Ishai21/Loan_prediction.git
cd Loan_prediction

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

./start.sh
```

### With Docker

```bash
docker build -t loan-prediction .
docker run -d --name loan-prediction -p 8501:8501 -p 8000:8000 loan-prediction
```

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI docs | http://localhost:8000/docs |

---

## AWS deployment

This application is deployed in an **AWS cloud environment built entirely within Free Tier limits**.

```
   git push main
        │
        ▼
  ┌──────────────┐   build & push    ┌─────────┐
  │GitHub Actions│──────────────────▶│   ECR   │  Docker image registry
  └──────────────┘                   └─────────┘
        │                                 │
        │ SSH deploy                      │ docker pull
        ▼                                 ▼
  ┌────────────────────────────────────────────┐
  │  EC2  t2.micro (Free Tier)                 │
  │  ┌──────────────────────────────────┐      │
  │  │ container  :8501 UI  :8000 API   │      │
  │  └──────────────────────────────────┘      │
  └────────────────────────────────────────────┘
        │
        ▼
  ┌─────────┐
  │   S3    │  model artifacts & dataset storage
  └─────────┘
```

### Services used

| Service | Role | Free Tier allowance |
|---|---|---|
| **EC2** | `t2.micro` instance hosting the running container | 750 hrs/month for 12 months |
| **ECR** | Private registry for the Docker image | 500 MB/month storage |
| **S3** | Storage for model artifacts (`.pkl`) and the training dataset | 5 GB standard storage |

### Security group

Inbound rules on the EC2 instance:

| Port | Purpose |
|---|---|
| 22 | SSH (restrict to your own IP) |
| 8501 | Streamlit UI |
| 8000 | FastAPI |

### Why the image is built in CI and not on the instance

A `t2.micro` has **1 GB of RAM**. Installing `scikit-learn`, `pandas` and `matplotlib` wheels in a Docker build on that instance reliably exhausts memory and gets the build killed. Building on GitHub's runners and having EC2 only `docker pull` a finished image keeps the instance inside its means — it needs enough memory to *run* the container, not to compile one.

---

## CI/CD pipeline

[`.github/workflows/ci-cd.yml`](.github/workflows/ci-cd.yml) runs on every push to `main`, every pull request, and on manual dispatch.

```
┌──────────┐     ┌────────────────┐     ┌──────────┐
│   test   │────▶│ build-and-push │────▶│  deploy  │
└──────────┘     └────────────────┘     └──────────┘
 pytest +         Docker image →         SSH to EC2,
 docker build     ECR (sha + latest)     pull & restart
   PRs stop here ─┘
```

**`test`** — installs dependencies on Python 3.11, runs the suite in [`tests/`](tests/), verifies the Docker image builds, and asserts the `starlette` pin is still valid (see *Dependency notes* below).

**`build-and-push`** — authenticates to ECR and pushes the image tagged with both the commit SHA and `latest`. The SHA tag means every deploy is traceable to an exact commit and rollback is a tag change. Runs only on `main`.

**`deploy`** — connects to EC2 over SSH, pulls the new image, and replaces the running container. Then polls both service health endpoints and fails the job if either does not come up, so a broken deploy surfaces in the Actions log instead of silently serving errors.

### Required repository secrets

Set these under **Settings → Secrets and variables → Actions**:

| Secret | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | IAM user key with ECR push rights |
| `AWS_SECRET_ACCESS_KEY` | Corresponding secret |
| `AWS_REGION` | e.g. `ap-south-1` |
| `ECR_REPOSITORY` | ECR repository name, e.g. `loan-prediction` |
| `EC2_HOST` | Instance public IP or DNS |
| `EC2_USER` | `ec2-user` (Amazon Linux) or `ubuntu` |
| `EC2_SSH_KEY` | Full contents of the `.pem` private key |

> **Hardening note:** long-lived IAM access keys are the simplest thing that works, not the best thing. For production, replace `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` with GitHub's OIDC provider and an assumable IAM role, which issues short-lived credentials and removes standing secrets from the repository entirely.

---

## Project structure

```
Loan_prediction/
├── .github/workflows/ci-cd.yml   CI/CD pipeline
├── app.py                        Streamlit UI          (:8501)
├── api.py                        FastAPI service       (:8000)
├── loan_model.py                 Two-stage inference layer
├── analyze.ipynb                 EDA & model development
├── start.sh                      Launches both services
├── Dockerfile                    Container definition
├── requirements.txt              Pinned dependencies
├── tests/                        Test suite
├── models/
│   ├── stage_1_rf_classifier_pipeline.pkl
│   └── stage_2_rf_regression_pipeline.pkl
└── data/
    └── loan_approval_dataset.csv
```

---

## Dependency notes

`requirements.txt` pins **`starlette==1.7.0`**, and that pin is deliberate — do not relax it.

Streamlit 1.60.0 subclasses Starlette's gzip middleware and overrides `send_with_compression`. Starlette **1.8.0** changed the base class to forward the `http.response.start` message immediately for responses that skip compression; Streamlit's override replaces that logic and never forwards it. The result is that uvicorn receives a response body with no preceding start message and raises:

```
RuntimeError: Expected ASGI message 'http.response.start', but got 'http.response.body'.
```

This affects any non-static response with an excluded content type — the `/media/` route behind `st.image`, `st.audio`, `st.video` and `st.download_button`. Starlette 1.7.0 is the last release compatible with Streamlit 1.60.0's middleware. `tests/test_dependencies.py` guards the pin so a careless upgrade fails CI instead of production.

Both Streamlit and FastAPI depend on Starlette, so pip resolves a single shared version for both — which is why an unpinned transitive dependency was able to break only one of the two services.

---

## Next steps

- Load model artifacts from S3 at container startup instead of baking them into the image, so retraining does not require a rebuild
- Add an automated retraining workflow triggered on new data landing in S3
- Put the two services behind nginx or an ALB so both are reachable on port 443 with TLS
- Replace static IAM keys with GitHub OIDC role assumption
- Add model monitoring for prediction drift
