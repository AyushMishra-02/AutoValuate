import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, render_template
import shap

app = Flask(__name__)

# Load model
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'model.joblib')

pipeline = None
preprocessor = None
actual_model = None
explainer = None
all_feat_names = None

def load_resources():
    global pipeline, preprocessor, actual_model, explainer, all_feat_names
    if os.path.exists(MODEL_PATH):
        pipeline = joblib.load(MODEL_PATH)
        preprocessor = pipeline.named_steps['preprocessor']
        actual_model = pipeline.named_steps['model']
        
        # Check model type to use the correct explainer
        from sklearn.linear_model import LinearRegression
        if isinstance(actual_model, LinearRegression):
            # For linear models, SHAP needs background data, which is heavy for API.
            # We'll just instantiate it here, though it might error if we don't pass data.
            # Usually XGBoost or LightGBM wins, so we handle TreeExplainer mostly.
            pass
        else:
            explainer = shap.TreeExplainer(actual_model)
            
        ohe_feat_names = preprocessor.named_transformers_['ohe'].get_feature_names_out()
        num_feat_names = ['year', 'km_driven', 'car_age', 'km_per_year']
        te_feat_names = ['name']
        all_feat_names = list(num_feat_names) + list(ohe_feat_names) + list(te_feat_names)
        print("Model and explainer loaded successfully.")

load_resources()

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not pipeline:
        return jsonify({'error': 'Model not found on server.'}), 500
        
    try:
        data = request.json
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Feature Engineering
        current_year = datetime.now().year
        df['car_age'] = current_year - df['year']
        df['car_age'] = df['car_age'].apply(lambda x: x if x > 0 else 1)
        df['km_per_year'] = df['km_driven'] / df['car_age']
        
        # Ensure correct column order expected by pipeline (should match training)
        # The pipeline handles it automatically if passed as a DataFrame
        
        # Predict
        prediction = pipeline.predict(df)[0]
        
        # SHAP Explainability
        explanation = {}
        if explainer:
            X_transformed = preprocessor.transform(df)
            shap_values = explainer(X_transformed)
            contributions = shap_values.values[0]
            
            # Combine feature names and contributions
            feat_contribs = list(zip(all_feat_names, contributions))
            # Sort by absolute contribution to find top 3
            feat_contribs.sort(key=lambda x: abs(x[1]), reverse=True)
            
            top_3 = feat_contribs[:3]
            explanation = [{'feature': f, 'contribution': round(float(c), 2)} for f, c in top_3]
            
        return jsonify({
            'predicted_price': round(float(prediction), 2),
            'top_3_shap_features': explanation
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
