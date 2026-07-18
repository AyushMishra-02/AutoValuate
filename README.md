# AutoValuate Pro — Intelligent Used Car Pricing Engine

### A production-grade SaaS pricing system with uncertainty-aware predictions, MLOps telemetry, and API authentication.

---

## 1. Important Pro Features Added
We recently upgraded AutoValuate to a full "Pro" production state. Here are the key highlights:

- **MLOps & Data Drift Dashboard**: We added an asynchronous SQLite logging system (`prediction_logs.db`). Every time a prediction is made, the inputs and outputs are saved. The Streamlit Dashboard now has a dedicated "MLOps & Drift Monitor" tab to visualize API usage, pricing trends, and query distributions in real-time.
- **API Key Authentication**: The FastAPI backend is secured with an `X-API-Key` header (defaulting to `AUTOVAL-DEMO-KEY`). Unauthorized requests are safely rejected with `401 Unauthorized`.
- **Prometheus Telemetry**: The FastAPI app exposes a `/metrics` endpoint, allowing you to scrape latency, throughput, and error rates using Grafana or Prometheus.
- **PDF Valuation Certificates**: The frontend now features an `fpdf2` integration, allowing users to instantly download an official PDF Certificate containing their car's details and the 80% confidence valuation bounds.
- **AI Negotiation Intelligence (Groq LLM)**: By providing a `GROQ_API_KEY`, the backend connects to the Llama-3 model to provide dynamic, ultra-fast negotiation tactics customized to the exact vehicle specs and SHAP feature importances.
- **Premium UI Upgrades**: The Streamlit interface was entirely redesigned using custom CSS (Glassmorphism, Google Fonts, Animated Gradients, and custom pill-shaped tabs).

---

## 2. Architecture

```text
                    ┌─────────────────┐
                    │   Raw Listings  │
                    │   (CSV / API)   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Data Validation │  ← pydantic schema checks,
                    │ & Cleaning      │    range checks, dedup
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │Feature Eng.     │  ← Depreciation curve pos,
                    │Pipeline (sklearn)  brand reliability, interactions
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
       ┌──────▼─────┐ ┌──────▼─────┐ ┌──────▼─────┐
       │  XGBoost   │ │  LightGBM  │ │  Quantile  │
       │  (point)   │ │  (point)   │ │  Regressor │
       │            │ │            │ │  (bounds)  │
       └──────┬─────┘ └──────┬─────┘ └──────┬─────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────▼────────┐
                    │ Stacked Ensemble│  ← weighted blend
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ SHAP Explainer  │  ← cached at startup
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐  ← /predict (Requires X-API-Key)
                    │ FastAPI Service │  ← /metrics (Prometheus)
                    └────────┬────────┘  ← Asynchronously writes to SQLite
                             │
                    ┌────────▼────────┐  
                    │ Streamlit Dash  │  ← Valuation UI & MLOps Drift Monitor
                    └─────────────────┘  ← PDF Certificate Generator
```

---

## 3. How to Deploy (Production)

We have provided a `docker-compose.yml` to make deployment seamless on any VPS (like AWS EC2, DigitalOcean) or local environment.

**Step 1: Install Docker & Docker Compose**
Ensure your server has Docker installed.

**Step 2: Run the Application**
```bash
docker-compose up --build -d
```
This single command will:
1. Build the shared Python environment.
2. Spin up the **FastAPI Service** on port `8000`.
3. Spin up the **Streamlit Dashboard** on port `8501`.
4. Persist the MLOps prediction logs to your local `./data` folder using a Docker volume.

**Step 3 (Alternative): Deploying to Render / Railway**
Since Render and Railway deploy individual services, you can deploy them separately from this repository:
1. **Backend Service**: Set the start command to `uvicorn api.main:app --host 0.0.0.0 --port 8000`
2. **Frontend Service**: Set the start command to `streamlit run dashboard/app.py`
*(Be sure to set the `AUTOVAL_API_KEY` environment variable in both services so they match!)*
*(Optional: Set the `GROQ_API_KEY` in the Backend Service to enable dynamic AI negotiation insights.)*

---

## 4. Local Development

If you want to run the services manually without Docker:

**Terminal 1 (Backend):**
```bash
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Terminal 2 (Frontend):**
```bash
streamlit run dashboard/app.py
```
