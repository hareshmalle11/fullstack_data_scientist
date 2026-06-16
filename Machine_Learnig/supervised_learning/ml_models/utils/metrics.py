from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import numpy as np

def regression_metrics(y_true, y_pred):
    """
    Calculate comprehensive regression metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

def explain_metrics():
    """
    Return explanation of metrics
    """
    return """
    **Mean Absolute Error (MAE):** Average absolute difference between predicted and actual values.  Lower is better.
    
    **Mean Squared Error (MSE):** Average of squared differences.  Penalizes larger errors more.  Lower is better.
    
    **Root Mean Squared Error (RMSE):** Square root of MSE. Same unit as target variable.  Lower is better.
    
    **R² Score:** Proportion of variance explained by the model.  Ranges from 0 to 1.  Higher is better (1 = perfect prediction).
    """