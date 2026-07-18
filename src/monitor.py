import pandas as pd
from scipy.stats import ks_2samp
import os

def detect_drift(train_data_path, new_data_path, numerical_features):
    """
    Lightweight data drift monitor using the Kolmogorov-Smirnov test.
    Compares the distribution of numerical features between training data and new incoming data.
    """
    if not os.path.exists(new_data_path):
        print("No new data to monitor against.")
        return

    train_df = pd.read_csv(train_data_path)
    new_df = pd.read_csv(new_data_path)

    drift_detected = False
    print("--- Drift Monitoring Report ---")
    
    for feat in numerical_features:
        if feat in train_df.columns and feat in new_df.columns:
            stat, p_value = ks_2samp(train_df[feat].dropna(), new_df[feat].dropna())
            
            # If p-value < 0.05, we reject the null hypothesis (distributions are different)
            if p_value < 0.05:
                print(f"⚠️ Drift detected in '{feat}' (p-value: {p_value:.4f})")
                drift_detected = True
            else:
                print(f"✅ No drift in '{feat}' (p-value: {p_value:.4f})")
                
    if not drift_detected:
        print("Status: All monitored features are stable.")
        
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    train_path = os.path.join(base_dir, 'data', 'processed', 'car_data_clean.csv')
    
    # In a real scenario, this would be a log of recent API inputs
    # For demonstration, we'll just compare train against train to show no drift
    detect_drift(
        train_data_path=train_path,
        new_data_path=train_path,
        numerical_features=['km_driven', 'year']
    )
