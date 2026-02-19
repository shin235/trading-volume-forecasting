import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


def eval_predictions(y_true_model, y_pred_model, use_log1p=True):
    """
    Compute metrics in model space (log1p(y) if enabled, otherwise raw y),
    and also report metrics on the original scale for interpretability.
    """
    y_true_model = np.asarray(y_true_model).reshape(-1)
    y_pred_model = np.asarray(y_pred_model).reshape(-1)

    if use_log1p:
        y_true_orig = np.expm1(y_true_model)
        y_pred_orig = np.expm1(y_pred_model)
    else:
        y_true_orig = y_true_model
        y_pred_orig = y_pred_model

    return {
        "RMSE_model": float(root_mean_squared_error(y_true_model, y_pred_model)),
        "MAE_model":  float(mean_absolute_error(y_true_model, y_pred_model)),
        "R2_model":   float(r2_score(y_true_model, y_pred_model)),
        "RMSE_orig":  float(root_mean_squared_error(y_true_orig, y_pred_orig)),
        "MAE_orig":   float(mean_absolute_error(y_true_orig, y_pred_orig)),
    }
