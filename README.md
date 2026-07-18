# AutoValuate — Intelligent Used Car Pricing Engine

### A production-grade dynamic pricing system with uncertainty-aware predictions and explainability

---

## 1. Business Framing

Every pricing system has two failure modes, and they're asymmetric:
- **Overestimate the price** → the platform loses money when reselling the car
- **Underestimate the price** → the seller walks to a competitor

A model optimized purely for RMSE treats both errors equally. A *pricing* system shouldn't. 

> "AutoValuate doesn't just predict a price — it predicts a price *range* with confidence bounds, and explicitly optimizes for asymmetric business cost rather than symmetric error."

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
                    ┌────────▼────────┐
                    │ FastAPI Service │  ← /predict, /health,
                    └────────┬────────┘    /model-info
                             │
                    ┌────────▼────────┐
                    │ Streamlit Dash  │  ← UI with bounds + SHAP
                    └─────────────────┘
```

---

## 3. Advanced Feature Engineering
- **Depreciation Curve Position**: Cars depreciate non-linearly. The model maps the age of the car to a historical depreciation curve learned from the training set.
- **Brand Reliability Score**: Aggregates the average resale-value retention by brand.
- **Interaction Terms**: e.g., `car_age × km_driven`.
- **Robust Outliers**: Handled explicitly in the pipeline instead of silently dropping data.

---

## 4. Modeling Strategy & Results

We evaluated a naïve baseline, tuned XGBoost and LightGBM models via **Optuna (Bayesian Optimization)**, and combined them into a Stacked Ensemble. More importantly, we introduced **Quantile Regression (10th and 90th percentiles)** to provide 80% confidence bounds on pricing.

We also designed a custom **Business Cost Metric** that penalizes overestimation (which causes the platform to lose money) 2x heavier than underestimation.

| Model | RMSE | MAE | R² | Business Cost |
|-------|------|-----|----|---------------|
| Linear Regression | 383,779 | 194,606 | 0.517 | 442,266 |
| XGBoost (Optuna) | 265,785 | 118,189 | 0.768 | 295,145 |
| LightGBM (Optuna) | 282,294 | 126,898 | 0.738 | 313,282 |
| **Stacked Ensemble** | **271,375** | **121,180** | **0.758** | **300,185** |

*(Note: The ensemble provides a highly robust balance of point estimation, bounds, and optimized business cost).*

---

## 5. API & UI Usage

### Streamlit Dashboard (Recommended for Demos)
Run the interactive dashboard:
```bash
streamlit run dashboard/app.py
```
This UI shows the price, the 80% confidence range, and a visual SHAP waterfall chart explaining the top factors driving the price.

### FastAPI Endpoints
Run the production server with Uvicorn:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
- **POST `/predict`**: Returns point estimate, bounds, and SHAP top-3.
- **GET `/health`**: Liveness check.
- **GET `/model-info`**: Versioning and feature schema.

---

## 6. MLOps & Testing
- **Dockerized**: A complete `Dockerfile` is included for zero-config deployments to Render/Railway.
- **CI/CD**: GitHub Actions workflow runs `pytest` on every push.
- **Drift Monitoring**: Includes a lightweight Kolmogorov-Smirnov test script (`src/monitor.py`) to detect input distribution shifts over time.

---
