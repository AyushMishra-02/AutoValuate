import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import datetime

LUXURY_BRANDS = ['BMW', 'Mercedes-Benz', 'Audi', 'Lexus', 'Jaguar', 'Land Rover', 'Volvo', 'Porsche']

class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.brand_means_ = {}
        self.age_means_ = {}
        self.overall_mean_ = 0
        
    def _extract_name_features(self, df):
        # Name format usually: "Brand Model Trim..." (e.g., "Maruti Swift Dzire VDI")
        
        # 1. Brand (First word)
        df['brand'] = df['name'].apply(lambda x: str(x).split(' ')[0] if pd.notnull(x) else 'Unknown')
        
        # 2. Model (Second and sometimes third word, simplistic heuristic)
        def extract_model(name_str):
            parts = str(name_str).split(' ')
            if len(parts) > 2:
                return f"{parts[0]} {parts[1]}"
            elif len(parts) > 1:
                return parts[1]
            return "Unknown"
            
        df['car_model'] = df['name'].apply(extract_model)
        
        # 3. Trim (Everything after Model)
        def extract_trim(name_str):
            parts = str(name_str).split(' ')
            if len(parts) > 2:
                return " ".join(parts[2:])
            return "Base"
            
        df['trim'] = df['name'].apply(extract_trim)
        return df

    def fit(self, X, y=None):
        df = X.copy()
        if y is not None:
            df['target'] = y
            df = self._extract_name_features(df)
            
            self.overall_mean_ = df['target'].mean()
            self.brand_means_ = df.groupby('brand')['target'].mean().to_dict()
            
            current_year = datetime.datetime.now().year
            df['car_age'] = current_year - df['year']
            df['car_age'] = df['car_age'].apply(lambda x: x if x > 0 else 1)
            
            self.age_means_ = df.groupby('car_age')['target'].mean().to_dict()
            
        return self
        
    def transform(self, X, y=None):
        df = X.copy()
        
        # Extract Name Features
        df = self._extract_name_features(df)
        
        # 1. Base Time Features
        current_year = datetime.datetime.now().year
        df['car_age'] = current_year - df['year']
        df['car_age'] = df['car_age'].apply(lambda x: x if x > 0 else 1)
        df['km_per_year'] = df['km_driven'] / df['car_age']
        
        # 3. Luxury Flag
        df['is_luxury_brand'] = df['brand'].apply(lambda x: 1 if x in LUXURY_BRANDS else 0)
        
        # 4. Depreciation Curve Position (age_means_)
        df['depreciation_curve_position'] = df['car_age'].map(self.age_means_).fillna(self.overall_mean_)
        
        # 5. Brand Reliability / Value Score
        df['brand_reliability_score'] = df['brand'].map(self.brand_means_).fillna(self.overall_mean_)
        
        # 6. Interaction Terms
        df['age_km_interaction'] = df['car_age'] * df['km_driven']
        
        # 7. Outlier Flag (simple heuristic: > 25 years old or > 250k km)
        df['is_outlier_heuristic'] = ((df['car_age'] > 25) | (df['km_driven'] > 250000)).astype(int)
        
        return df
