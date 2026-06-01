from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_current_figure(path: Path | str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_pv_power_timeseries(df: pd.DataFrame, path: Path | str) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df["pv_power_kw"], linewidth=0.8, color="#1f77b4")
    plt.title("Simulated PV Power Output (1 kW system)")
    plt.xlabel("Datetime (UTC)")
    plt.ylabel("PV power (kW)")
    _save_current_figure(path)


def plot_model_comparison(comparison: pd.DataFrame, path: Path | str) -> None:
    plt.figure(figsize=(9, 5))
    ordered = comparison.sort_values("RMSE", ascending=True)
    plt.barh(ordered["model"], ordered["RMSE"], color="#2a9d8f")
    plt.title("Model Comparison by RMSE")
    plt.xlabel("RMSE (kW)")
    _save_current_figure(path)


def plot_actual_vs_predicted(index, y_true, predictions: dict[str, object], path: Path | str) -> None:
    plt.figure(figsize=(13, 6))
    window = min(240, len(y_true))
    plot_index = index[:window]
    plt.plot(plot_index, y_true[:window], label="Actual simulated PV", color="black", linewidth=1.8)
    for name, values in predictions.items():
        series = np.asarray(values)[:window]
        plt.plot(plot_index, series, label=name, linewidth=0.9, alpha=0.85)
    plt.title("Actual vs Predicted Simulated PV Power")
    plt.xlabel("Datetime (UTC)")
    plt.ylabel("PV power (kW)")
    plt.legend(ncol=2)
    _save_current_figure(path)


def plot_residuals(index, y_true, predictions: dict[str, object], path: Path | str) -> None:
    plt.figure(figsize=(13, 6))
    window = min(240, len(y_true))
    plot_index = index[:window]
    true_values = np.asarray(y_true)[:window]
    for name, values in predictions.items():
        residuals = true_values - np.asarray(values)[:window]
        plt.plot(plot_index, residuals, label=name, linewidth=0.9, alpha=0.85)
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Forecast Residuals")
    plt.xlabel("Datetime (UTC)")
    plt.ylabel("Residual (kW)")
    plt.legend(ncol=2)
    _save_current_figure(path)
