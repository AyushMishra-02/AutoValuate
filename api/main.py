import os
import joblib
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks
from contextlib import asynccontextmanager
from typing import Dict, Any, List
import functools

from .schemas import PredictionRequest, PredictionResponse, SHAPFeature
from .database import SessionLocal, PredictionLog
import sys
import json
from groq import Groq

# Removed global Groq client for security; now passed via HTTP header

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

# Globals for models
pipeline_components = {}
explainer = None
all_feat_names = []

# Dummy API Key for demo purposes
API_KEY = os.getenv("AUTOVAL_API_KEY", "AUTOVAL-DEMO-KEY")

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline_components, explainer, all_feat_names
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, 'models', 'ensemble_pipeline.joblib')
    
    if os.path.exists(model_path):
        print("Loading models and precomputing SHAP explainer...")
        pipeline_components = joblib.load(model_path)
        
        # We will explain the LGBM point model
        lgb_model = pipeline_components['model_lgb_point']
        explainer = shap.TreeExplainer(lgb_model)
        
        preprocessor = pipeline_components['preprocessor']
        
        ohe_feat_names = preprocessor.named_transformers_['ohe'].get_feature_names_out()
        num_feat_names = ['year', 'km_driven', 'car_age', 'km_per_year', 
                          'depreciation_curve_position', 'brand_reliability_score', 
                          'age_km_interaction', 'is_luxury_brand', 'is_outlier_heuristic']
        te_feat_names = ['brand']
        all_feat_names = list(num_feat_names) + list(ohe_feat_names) + list(te_feat_names)
        
        print("Models loaded successfully.")
    else:
        print(f"Warning: Model not found at {model_path}")
        
    yield
    # Cleanup if needed
    pipeline_components.clear()

app = FastAPI(
    title="AutoValuate API Pro",
    description="Intelligent Used Car Pricing Engine with Telemetry, Auth, and Caching.",
    version="3.0",
    lifespan=lifespan
)

# Prometheus Telemetry
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app)
except ImportError:
    print("Prometheus instrumentator not found. Skipping telemetry.")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5173",
        "http://localhost:8501"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import RedirectResponse

@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/model-info")
def model_info(api_key: str = Depends(verify_api_key)):
    if not pipeline_components:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {
        "version": "3.0",
        "models": ["XGBoost", "LightGBM Point", "MAPIE Conformal"],
        "features": all_feat_names
    }

def generate_negotiation_insights(feat_contribs, request, groq_key=None):
    if not groq_key or not request:
        # Fallback heuristic if Groq API key is missing or request is None
        insights = []
        for feat, contrib in feat_contribs:
            if len(insights) >= 2:
                break
            if feat == 'km_driven' and contrib < -10000:
                insights.append(f"Buyer Tactic: High mileage severely depreciates value (by ₹{abs(contrib):,.0f}). Negotiate a discount.")
            elif feat == 'year' and contrib > 20000:
                insights.append(f"Seller Leverage: Recent model year adds ₹{contrib:,.0f}. Hold firm on price.")
            elif feat.startswith('brand_') and contrib > 25000:
                insights.append(f"Market Reality: Premium brand commands ₹{contrib:,.0f} extra. Little wiggle room.")
        if not insights:
            insights.append("Market Reality: Vehicle is priced at market expectations.")
        return insights
        
    try:
        # Construct dynamic prompt for Groq
        top_features = ", ".join([f"{feat} (Impact: ₹{contrib:,.0f})" for feat, contrib in feat_contribs[:3]])
        vehicle_desc = f"{request.year} {request.name} ({request.fuel}, {request.km_driven} km)"
        
        prompt = f"""
You are an expert car dealer and negotiator. Analyze this vehicle: {vehicle_desc}.
The top 3 factors affecting its price are: {top_features}.

Provide exactly two concise negotiation tactics:
1. One tactic for the BUYER to negotiate the price down. Start with 'Buyer Tactic:'.
2. One tactic for the SELLER to justify a higher price. Start with 'Seller Leverage:'.

Format exactly as a JSON array of strings, like this:
["Buyer Tactic: ...", "Seller Leverage: ..."]
Do not output any markdown formatting or extra text.
"""
        client = Groq(api_key=groq_key)
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        content = response.choices[0].message.content.strip()
        insights = json.loads(content)
        return insights
    except Exception as e:
        print(f"Groq API Error: {e}")
        return ["Market Reality: Vehicle is priced exactly at market expectations with no significant anomalies."]

def log_prediction_to_db(req_dict, price, lower, upper):
    """Background task to log prediction to SQLite for Data Drift Monitoring."""
    db = SessionLocal()
    try:
        log = PredictionLog(
            name=req_dict['name'],
            year=req_dict['year'],
            km_driven=req_dict['km_driven'],
            fuel=req_dict['fuel'],
            seller_type=req_dict['seller_type'],
            transmission=req_dict['transmission'],
            owner=req_dict['owner'],
            predicted_price=price,
            lower_bound_80=lower,
            upper_bound_80=upper
        )
        db.add(log)
        db.commit()
    except Exception as e:
        print(f"Error logging prediction: {e}")
    finally:
        db.close()

# LRU Cache for identical requests (simulates Redis edge cache)
@functools.lru_cache(maxsize=1000)
def compute_prediction(req_tuple, groq_key=None):
    # Convert tuple back to dict and reconstruct PredictionRequest object
    req_dict = dict(req_tuple)
    request_obj = PredictionRequest(**req_dict)
    
    df = pd.DataFrame([req_dict])
    
    # 1. Feature Engineering
    fe = pipeline_components['feature_engineer']
    df_fe = fe.transform(df)
    
    # 2. Preprocessing
    preprocessor = pipeline_components['preprocessor']
    X_prep = preprocessor.transform(df_fe)
    
    # 3. Predictions
    xgb_model = pipeline_components['model_xgb']
    lgb_point = pipeline_components['model_lgb_point']
    mapie_model = pipeline_components['mapie_model']
    
    xgb_pred = xgb_model.predict(X_prep)[0]
    lgb_pred = lgb_point.predict(X_prep)[0]
    
    # Ensemble point estimate
    point_pred = 0.4 * xgb_pred + 0.6 * lgb_pred
    
    # Conformal Prediction Bounds (80% confidence interval)
    _, y_pis = mapie_model.predict_interval(X_prep)
    lower_bound = y_pis[0, 0, 0]
    upper_bound = y_pis[0, 1, 0]
    
    # 4. SHAP Explanation
    explanation = []
    insights = []
    if explainer:
        shap_values = explainer(X_prep)
        contributions = shap_values.values[0]
        
        feat_contribs = list(zip(all_feat_names, contributions))
        feat_contribs.sort(key=lambda x: abs(x[1]), reverse=True)
        
        top_3 = feat_contribs[:3]
        explanation = [{'feature': f, 'contribution': round(float(c), 2)} for f, c in top_3]
        
        # Generate insights using LLM
        insights = generate_negotiation_insights(feat_contribs, request_obj, groq_key)
        
    return {
        "predicted_price": round(float(point_pred), 2),
        "confidence_lower_80": round(float(lower_bound), 2),
        "confidence_upper_80": round(float(upper_bound), 2),
        "top_3_shap_features": explanation,
        "negotiation_insights": insights
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_price(request: PredictionRequest, background_tasks: BackgroundTasks, api_key: str = Depends(verify_api_key), x_groq_key: str = Header(None)):
    if not pipeline_components:
        raise HTTPException(status_code=503, detail="Models not loaded")
        
    try:
        req_dict = request.dict()
        # Create a hashable tuple for LRU cache
        req_tuple = tuple(sorted(req_dict.items()))
        
        result = compute_prediction(req_tuple, groq_key=x_groq_key)
        
        # Background task for asynchronous MLOps logging
        background_tasks.add_task(
            log_prediction_to_db, 
            req_dict, 
            result['predicted_price'], 
            result['confidence_lower_80'], 
            result['confidence_upper_80']
        )
        
        return PredictionResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
