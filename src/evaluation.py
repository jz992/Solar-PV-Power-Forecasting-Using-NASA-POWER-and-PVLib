from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_predictions(model_name: str, y_true, y_pred) -> dict[str, float | str]:
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    return {
        "model": model_name,
        "MAE": mean_absolute_error(y_true_array, y_pred_array),
        "RMSE": np.sqrt(mean_squared_error(y_true_array, y_pred_array)),
        "R2": r2_score(y_true_array, y_pred_array),
    }


def save_model_comparison(results: list[dict[str, float | str]], path: Path | str) -> pd.DataFrame:
    comparison = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    comparison.to_csv(path, index=False)
    return comparison
