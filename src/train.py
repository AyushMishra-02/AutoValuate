import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from category_encoders import TargetEncoder
import xgboost as xgb
import lightgbm as lgb

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Original shape: {df.shape}")
    
    # Basic cleaning
    # Remove outliers or bad data
    df = df[df['km_driven'] > 0]
    
    # Feature engineering
    current_year = datetime.now().year
    df['car_age'] = current_year - df['year']
    df['car_age'] = df['car_age'].apply(lambda x: x if x > 0 else 1) # Prevent division by zero
    df['km_per_year'] = df['km_driven'] / df['car_age']
    
    # In this dataset, target is 'selling_price'
    
    print(f"Cleaned shape: {df.shape}")
    return df

def build_pipeline(model):
    # Categorical features for one-hot encoding
    ohe_features = ['fuel', 'seller_type', 'transmission', 'owner']
    # Categorical features for target encoding
    te_features = ['name']
    # Numeric features
    num_features = ['year', 'km_driven', 'car_age', 'km_per_year']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ohe_features),
            ('te', TargetEncoder(), te_features)
        ]
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    return pipeline

def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"--- {name} ---")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"R²:   {r2:.4f}")
    return rmse, mae, r2

def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, 'data', 'car_data.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}. Run data_fetch.py first.")
        
    df = load_and_preprocess_data(data_path)
    
    X = df.drop(columns=['selling_price'])
    y = df['selling_price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Linear Regression': LinearRegression(),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, verbose=-1)
    }
    
    results = {}
    best_model = None
    best_rmse = float('inf')
    best_name = ""
    
    for name, m in models.items():
        pipeline = build_pipeline(m)
        pipeline.fit(X_train, y_train)
        
        rmse, mae, r2 = evaluate_model(name, pipeline, X_test, y_test)
        results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'pipeline': pipeline}
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = pipeline
            best_name = name
            
    print(f"\nBest Model: {best_name} with RMSE: {best_rmse:.2f}")
    
    # Save best model
    model_path = os.path.join(base_dir, 'model.joblib')
    joblib.dump(best_model, model_path)
    print(f"Saved best model to {model_path}")
    
    # SHAP Explainability for the best model
    # We need to extract the transformed data and the actual model from the pipeline
    print("Generating SHAP explainability plots...")
    preprocessor = best_model.named_steps['preprocessor']
    actual_model = best_model.named_steps['model']
    
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Get feature names from preprocessor
    ohe_feat_names = preprocessor.named_transformers_['ohe'].get_feature_names_out()
    num_feat_names = ['year', 'km_driven', 'car_age', 'km_per_year']
    te_feat_names = ['name']
    all_feat_names = list(num_feat_names) + list(ohe_feat_names) + list(te_feat_names)
    
    # Initialize JS visualization for SHAP (only needed in notebooks)
    # shap.initjs()
    
    if best_name == 'Linear Regression':
        explainer = shap.LinearExplainer(actual_model, X_train_transformed)
    else:
        explainer = shap.TreeExplainer(actual_model)
        
    shap_values = explainer(X_test_transformed)
    shap_values.feature_names = all_feat_names
    
    images_dir = os.path.join(base_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Summary Plot
    plt.figure()
    shap.summary_plot(shap_values, X_test_transformed, feature_names=all_feat_names, show=False)
    plt.savefig(os.path.join(images_dir, 'shap_summary.png'), bbox_inches='tight')
    plt.close()
    
    # Waterfall Plot for a single prediction (e.g., first test instance)
    plt.figure()
    shap.waterfall_plot(shap_values[0], show=False)
    plt.savefig(os.path.join(images_dir, 'shap_waterfall.png'), bbox_inches='tight')
    plt.close()
    
    print("SHAP plots saved in 'images/' directory.")

if __name__ == '__main__':
    main()
