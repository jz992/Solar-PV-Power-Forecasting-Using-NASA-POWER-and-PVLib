"""Evaluation utilities for PV forecasting models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import DAYLIGHT_THRESHOLD_KW


def evaluate_regression(y_true, y_pred, model_name: str) -> dict:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n{model_name}")
    print(f"MAE  : {mae:.6f} kW")
    print(f"RMSE : {rmse:.6f} kW")
    print(f"R2   : {r2:.6f}")
    return {"Model": model_name, "MAE_kW": mae, "RMSE_kW": rmse, "R2": r2}


def persistence_baseline(y_test: pd.Series):
    """One-step persistence baseline using previous PV output."""
    y_pred = y_test.shift(1).dropna()
    y_true = y_test.iloc[1:]
    return y_true, y_pred


def daylight_evaluation(y_true, y_pred, model_name: str) -> dict:
    """Evaluate only daylight samples where actual simulated PV output is positive."""
    y_true_series = pd.Series(np.asarray(y_true).reshape(-1))
    y_pred_series = pd.Series(np.asarray(y_pred).reshape(-1))
    daylight_mask = y_true_series > DAYLIGHT_THRESHOLD_KW
    return evaluate_regression(
        y_true_series[daylight_mask],
        y_pred_series[daylight_mask],
        f"{model_name} - Daylight Only",
    )
