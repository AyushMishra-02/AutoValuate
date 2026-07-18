# AutoValuate — Used Car Price Prediction Engine

AutoValuate is a machine learning-based REST API designed to predict used car prices and provide explainability for its pricing decisions using SHAP (SHapley Additive exPlanations).

## Project Overview
This project uses the popular CarDekho vehicle dataset to predict the `selling_price` of a car based on various features such as `year`, `km_driven`, `fuel` type, `transmission`, and `seller_type`. 

### Key Features
- **Dynamic Pricing Engine**: Uses an XGBoost/LightGBM model target-encoded for high cardinality features (car brands/models).
- **Explainability**: Leverages SHAP to produce Waterfall and Summary plots showing exactly *why* a particular price was predicted, making the model interpretable.
- **REST API & Web UI**: Served via a Flask application providing a sleek, modern web frontend and a `/predict` JSON endpoint.

## Setup and Installation

1. **Clone and setup virtual environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Data**:
   ```bash
   python src/data_fetch.py
   ```

4. **Train Model**:
   ```bash
   python src/train.py
   ```
   *This will evaluate Linear Regression, XGBoost, and LightGBM, select the best performing model, save it to `model.joblib`, and generate SHAP plots in the `images/` directory.*

5. **Run the Server**:
   ```bash
   python app.py
   ```

## Usage

### 1. Web Interface (Recommended)
Simply open your web browser and navigate to:
**http://localhost:5000/**

You will see a modern, glassmorphism UI where you can input car details and see the AI's price prediction along with a breakdown of exactly which features influenced the price (powered by SHAP).

### 2. API Endpoint
The Flask application also accepts JSON payloads at `http://localhost:5000/predict`.

**Example Request:**
```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{
           \"name\": \"Maruti 800 AC\",
           \"year\": 2007,
           \"km_driven\": 10000,
           \"fuel\": \"Petrol\",
           \"seller_type\": \"Individual\",
           \"transmission\": \"Manual\",
           \"owner\": \"First Owner\"
         }"
```

**Example Response:**
```json
{
  "predicted_price": 65000.5,
  "top_3_shap_features": [
    {"contribution": -25000.34, "feature": "car_age"},
    {"contribution": 15000.21, "feature": "name"},
    {"contribution": -10000.12, "feature": "km_driven"}
  ]
}
```

## Model Evaluation: XGBoost vs LightGBM
During training, multiple models are evaluated. 
- **LightGBM** typically excels in speed and performance with larger datasets due to its leaf-wise tree growth.
- **XGBoost** often provides a robust and highly accurate baseline with level-wise growth.
- Both models vastly outperformed the **Linear Regression** baseline by capturing non-linear relationships like the steep initial depreciation of car prices.
Refer to the console output of `src/train.py` for exact RMSE/MAE metrics to see which model performed best on this dataset.

## Resume Bullet Target
> "Built a used-car pricing engine (XGBoost/LightGBM) with SHAP-based explainability, achieving significant RMSE reduction over baseline linear regression; served via Flask REST API."
