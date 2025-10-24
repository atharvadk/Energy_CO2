# src/modeling/utils.py
import os
import math
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def save_model(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)

def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def regression_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse(y_true, y_pred)),
        "mape": float(mape(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred))
    }

def save_metrics_df(metrics_list, out_csv):
    df = pd.DataFrame(metrics_list)
    df.to_csv(out_csv, index=False)
    return df
