import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def business_cost(y_true, y_pred):
    """
    Custom business cost metric.
    Overestimating the price (predicting higher than true value) means the platform 
    loses money when reselling the car. This error is weighted 2x heavier than underestimating.
    """
    error = y_pred - y_true
    # overestimation costs more -> penalty is 2.0 * error^2
    # underestimation -> 1.0 * error^2
    penalized_error = np.where(error > 0, 2.0 * error**2, error**2)
    return np.sqrt(np.mean(penalized_error))

def evaluate_model(y_true, y_pred, model_name="Model"):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    bc = business_cost(y_true, y_pred)
    
    print(f"\n--- {model_name} ---")
    print(f"RMSE:          {rmse:.2f}")
    print(f"MAE:           {mae:.2f}")
    print(f"R²:            {r2:.4f}")
    print(f"Business Cost: {bc:.2f}")
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'BusinessCost': bc
    }
