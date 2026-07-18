import os
import joblib
import pandas as pd
import numpy as np
import optuna
import shap
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from category_encoders import TargetEncoder
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LinearRegression

# MAPIE for Conformal Prediction
from mapie.regression import SplitConformalRegressor

from features import AdvancedFeatureEngineer
from evaluate import evaluate_model, business_cost

# Constants
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'car_data_clean.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

def build_preprocessor():
    ohe_features = ['fuel', 'seller_type', 'transmission', 'owner']
    # Now including the extracted features from Phase 3
    te_features = ['brand', 'car_model', 'trim']
    num_features = ['year', 'km_driven', 'car_age', 'km_per_year', 
                    'depreciation_curve_position', 'brand_reliability_score', 
                    'age_km_interaction', 'is_luxury_brand', 'is_outlier_heuristic']
    
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ohe_features),
            ('te', TargetEncoder(), te_features)
        ]
    )

def objective_lgb(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'random_state': 42,
        'verbose': -1
    }
    
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        bc = business_cost(y_val, preds)
        mlflow.log_metric("business_cost", bc)
        
    return bc

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Initialize MLflow tracking
    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(BASE_DIR, 'mlruns.db')}")
    mlflow.set_experiment("AutoValuate_Phase3")
    
    with mlflow.start_run(run_name="Full_Pipeline_Run"):
        # 1. Load Data
        df = pd.read_csv(DATA_PATH)
        X = df.drop(columns=['selling_price'])
        y = df['selling_price']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        # 2. Feature Engineering Pipeline
        fe = AdvancedFeatureEngineer()
        fe.fit(X_train, y_train)
        
        X_train_fe = fe.transform(X_train)
        X_train_sub_fe = fe.transform(X_train_sub)
        X_val_fe = fe.transform(X_val)
        X_test_fe = fe.transform(X_test)
        
        preprocessor = build_preprocessor()
        X_train_prep = preprocessor.fit_transform(X_train_fe, y_train)
        X_train_sub_prep = preprocessor.transform(X_train_sub_fe)
        X_val_prep = preprocessor.transform(X_val_fe)
        X_test_prep = preprocessor.transform(X_test_fe)
        
        # 3. Baselines
        print("\n--- Training Baseline ---")
        lr = LinearRegression()
        lr.fit(X_train_prep, y_train)
        evaluate_model(y_test, lr.predict(X_test_prep), "Linear Regression")
        
        # 4. Tune LightGBM with Optuna
        print("\n--- Tuning LightGBM with Optuna (Logged to MLflow) ---")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda t: objective_lgb(t, X_train_sub_prep, y_train_sub, X_val_prep, y_val), n_trials=30)
        best_lgb_params = study.best_params
        best_lgb_params.update({'random_state': 42, 'verbose': -1})
        print(f"Best LGBM Params: {best_lgb_params}")
        mlflow.log_params({f"best_{k}": v for k, v in best_lgb_params.items()})
        
        # 5. Train Core Models
        xgb_model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42)
        xgb_model.fit(X_train_prep, y_train)
        
        lgb_point = lgb.LGBMRegressor(**best_lgb_params)
        lgb_point.fit(X_train_prep, y_train)
        
        # We'll use the LGBM point model as the base estimator for Conformal Prediction
        # MAPIE generates mathematically guaranteed bounds
        mapie_model = SplitConformalRegressor(estimator=lgb_point, prefit=True, confidence_level=0.8)
        mapie_model.conformalize(X_val_prep, y_val) # calibrate on validation set
        
        # 6. Evaluate
        xgb_preds = xgb_model.predict(X_test_prep)
        lgb_preds = lgb_point.predict(X_test_prep)
        
        metrics = evaluate_model(y_test, lgb_preds, "LightGBM")
        mlflow.log_metrics({f"lgbm_{k}": v for k, v in metrics.items()})
        
        # Conformal Prediction Evaluation
        _, y_pis = mapie_model.predict_interval(X_test_prep)
        lower_bounds = y_pis[:, 0, 0]
        upper_bounds = y_pis[:, 1, 0]
        
        coverage = np.mean((y_test >= lower_bounds) & (y_test <= upper_bounds))
        print(f"\n--- Conformal Prediction ---")
        print(f"Target Coverage: 80.00%")
        print(f"Actual Coverage: {coverage*100:.2f}%")
        mlflow.log_metric("conformal_actual_coverage", coverage)
        
        # 7. Save Artifacts
        full_pipeline = {
            'feature_engineer': fe,
            'preprocessor': preprocessor,
            'model_xgb': xgb_model,
            'model_lgb_point': lgb_point,
            'mapie_model': mapie_model
        }
        
        model_filepath = os.path.join(MODEL_DIR, 'ensemble_pipeline.joblib')
        joblib.dump(full_pipeline, model_filepath)
        mlflow.log_artifact(model_filepath)
        
        print(f"\nSaved full Phase 3 pipeline to {model_filepath}")

if __name__ == '__main__':
    main()
