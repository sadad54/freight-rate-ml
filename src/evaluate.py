"""Regression metrics in business-interpretable units (dollars, %)."""
from __future__ import annotations
import numpy as np

def mae(y_true, y_pred)-> float:
    return float(np.mean(np.abs(y_true-y_pred)))

def rmse(y_true, y_pred)->float:
    return float(np.sqrt(np.mean((y_true-y_pred)**2)))

def mape(y_true, y_pred)->float:
    return float(np.mean(np.abs((y_true-y_pred)/y_true))*100)

def summarize(y_true, y_pred)-> dict:
    return {"MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred), "MAPE_%": mape(y_true, y_pred)}
