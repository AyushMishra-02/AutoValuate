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

def generate_negotiation_insights(feat_contribs, request):
    insights = []
    for feat, contrib in feat_contribs:
        # We only generate insights for the top 2 absolute contributors to keep it punchy
        if len(insights) >= 2:
            break
            
        if feat == 'km_driven' and contrib < -10000:
            insights.append(f"Buyer Tactic: This vehicle has high mileage which heavily depreciates its value (by ₹{abs(contrib):,.0f}). Use impending maintenance milestones to negotiate a further discount.")
        elif feat == 'km_driven' and contrib > 10000:
            insights.append(f"Seller Leverage: The exceptionally low mileage is adding ₹{contrib:,.0f} to the value. Hold firm on your price, as low-mileage variants are rare.")
            
        elif feat == 'year' and contrib < -15000:
            insights.append(f"Buyer Tactic: The age of this model is severely impacting its market value. Highlight potential outdated tech and wear-and-tear to push for a lower price.")
        elif feat == 'year' and contrib > 20000:
            insights.append(f"Seller Leverage: This is a very recent model year, driving up the price by ₹{contrib:,.0f}. Buyers have few alternatives for near-new cars without buying retail.")
            
        elif feat.startswith('brand_') and contrib > 25000:
            brand = feat.replace('brand_', '')
            insights.append(f"Market Reality: The '{brand}' badge commands a massive premium (₹{contrib:,.0f}) due to high brand reliability and liquidity. Don't expect much wiggle room.")
            
        elif feat == 'is_luxury_brand' and contrib < -30000:
             insights.append(f"Buyer Tactic: Luxury brands depreciate violently outside of warranty. Use the fear of expensive out-of-warranty repairs to aggressively negotiate down.")
             
    # Add a fallback if no specific heuristics were triggered
    if not insights:
        insights.append("Market Reality: This vehicle is priced exactly at market expectations with no significant anomalies.")
        
    return insights

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
def compute_prediction(req_tuple):
    # Convert tuple back to dict
    req_dict = dict(req_tuple)
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
        
        # Generate insights
        # Fake object for compatibility with old function signature
        insights = generate_negotiation_insights(feat_contribs, None)
        
    return {
        "predicted_price": round(float(point_pred), 2),
        "confidence_lower_80": round(float(lower_bound), 2),
        "confidence_upper_80": round(float(upper_bound), 2),
        "top_3_shap_features": explanation,
        "negotiation_insights": insights
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_price(request: PredictionRequest, background_tasks: BackgroundTasks, api_key: str = Depends(verify_api_key)):
    if not pipeline_components:
        raise HTTPException(status_code=503, detail="Models not loaded")
        
    try:
        req_dict = request.dict()
        # Create a hashable tuple for LRU cache
        req_tuple = tuple(sorted(req_dict.items()))
        
        result = compute_prediction(req_tuple)
        
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
