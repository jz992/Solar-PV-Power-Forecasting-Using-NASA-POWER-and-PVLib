"""Plotting functions for results and diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import DAYLIGHT_THRESHOLD_KW, FIGURES_DIR


plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "axes.labelsize": 12,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "font.size": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "grid.color": "#d9d9d9",
    "grid.linestyle": "-",
    "grid.linewidth": 0.7,
    "grid.alpha": 0.7,
})

ACTUAL_COLOR = "#1f4e79"
PREDICTED_COLOR = "#c44e52"
BAR_COLOR = "#4c78a8"
SECONDARY_BAR_COLOR = "#6f6f6f"
SCATTER_COLOR = "#2f6f6f"


def ensure_figures_dir(output_dir: str | Path = FIGURES_DIR) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def _finish_plot(output_path: Path) -> None:
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _metric_title(base_title: str, rmse: float | None = None, r2: float | None = None) -> str:
    parts = []
    if rmse is not None:
        parts.append(f"RMSE={rmse:.4f} kW")
    if r2 is not None:
        parts.append(f"R2={r2:.4f}")
    if not parts:
        return base_title
    return f"{base_title} ({', '.join(parts)})"


def plot_pv_timeseries(df: pd.DataFrame, output_dir: str | Path = FIGURES_DIR) -> None:
    out = ensure_figures_dir(output_dir)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index[: 24 * 14], df["pv_power_kw"].iloc[: 24 * 14], color=ACTUAL_COLOR, linewidth=2.0)
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Simulated PV Power (kW)")
    ax.set_title("Simulated PV Power Output - First Two Weeks")
    ax.grid(True)
    _finish_plot(out / "pv_power_timeseries.png")


def plot_irradiance_vs_pv(df: pd.DataFrame, output_dir: str | Path = FIGURES_DIR) -> None:
    out = ensure_figures_dir(output_dir)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["ALLSKY_SFC_SW_DWN"], df["pv_power_kw"], alpha=0.35, s=16, color=SCATTER_COLOR, edgecolors="none")
    ax.set_xlabel("NASA POWER All-Sky Irradiance")
    ax.set_ylabel("Simulated PV Power (kW)")
    ax.set_title("Irradiance vs Simulated PV Power")
    ax.grid(True)
    _finish_plot(out / "irradiance_vs_pv_power.png")


def plot_actual_vs_predicted(
    y_true,
    y_pred,
    model_name: str,
    output_dir: str | Path = FIGURES_DIR,
    filename_suffix: str | None = None,
    rmse: float | None = None,
    r2: float | None = None,
) -> None:
    out = ensure_figures_dir(output_dir)
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    n = min(300, len(y_true))
    x_axis = np.arange(n)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x_axis, y_true[:n], label="Actual", color=ACTUAL_COLOR, linewidth=2.0, linestyle="-")
    ax.plot(
        x_axis,
        y_pred[:n],
        label="Predicted",
        color=PREDICTED_COLOR,
        linewidth=1.8,
        linestyle="--",
        marker="o",
        markersize=2.6,
        markevery=max(1, n // 45),
        alpha=0.95,
    )
    ax.set_xlabel("Time Step")
    ax.set_ylabel("PV Power (kW)")
    ax.set_title(_metric_title(f"Actual vs Predicted PV Power - {model_name}", rmse=rmse, r2=r2))
    ax.grid(True)
    ax.legend(frameon=False)

    if filename_suffix is None:
        filename_suffix = _slug(model_name)
    _finish_plot(out / f"actual_vs_predicted_{filename_suffix}.png")


def plot_daylight_only(y_true, y_pred, model_name: str, output_dir: str | Path = FIGURES_DIR) -> None:
    y_true_series = pd.Series(np.asarray(y_true).reshape(-1))
    y_pred_series = pd.Series(np.asarray(y_pred).reshape(-1))
    daylight_mask = y_true_series > DAYLIGHT_THRESHOLD_KW
    plot_actual_vs_predicted(
        y_true_series[daylight_mask],
        y_pred_series[daylight_mask],
        f"{model_name} - Daylight Only",
        output_dir=output_dir,
        filename_suffix=f"{_slug(model_name)}_daylight_only",
    )


def plot_feature_importance(
    feature_names,
    importances,
    model_name: str = "Random Forest",
    output_dir: str | Path = FIGURES_DIR,
    top_n: int = 15,
    filename: str | None = None,
) -> None:
    out = ensure_figures_dir(output_dir)
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    top_importance = importance_df.sort_values("importance", ascending=False).head(top_n)

    fig_height = max(5.5, 0.36 * len(top_importance) + 1.8)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(top_importance["feature"][::-1], top_importance["importance"][::-1], color=BAR_COLOR)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top {len(top_importance)} {model_name} Feature Importances")
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")

    if filename is None:
        filename = f"feature_importance_{_slug(model_name)}.png"
    _finish_plot(out / filename)


def plot_model_comparison(
    results_df: pd.DataFrame,
    title: str,
    filename: str,
    output_dir: str | Path = FIGURES_DIR,
    metric: str = "RMSE_kW",
) -> None:
    out = ensure_figures_dir(output_dir)
    ascending = metric != "R2"
    sorted_df = results_df.sort_values(metric, ascending=ascending)

    fig_height = max(4.8, 0.46 * len(sorted_df) + 1.6)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    colors = [BAR_COLOR if index == 0 else SECONDARY_BAR_COLOR for index in range(len(sorted_df))]
    ax.barh(sorted_df["Model"], sorted_df[metric], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("R2 Score" if metric == "R2" else "RMSE (kW)")
    ax.set_ylabel("Model")
    ax.set_title(title)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")

    for patch, value in zip(ax.patches, sorted_df[metric]):
        ax.text(
            patch.get_width(),
            patch.get_y() + patch.get_height() / 2,
            f" {value:.4f}",
            va="center",
            ha="left",
            fontsize=9,
            color="#222222",
        )

    if metric == "R2":
        ax.set_xlim(left=max(0, min(sorted_df[metric]) - 0.05), right=min(1.02, max(sorted_df[metric]) + 0.05))
    _finish_plot(out / filename)


def plot_learning_curve(history, model_name: str, output_dir: str | Path = FIGURES_DIR) -> None:
    out = ensure_figures_dir(output_dir)
    epochs = np.arange(1, len(history.history["loss"]) + 1)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(epochs, history.history["loss"], label="Training Loss", color=ACTUAL_COLOR, linewidth=2.0, linestyle="-")
    ax.plot(epochs, history.history["val_loss"], label="Validation Loss", color=PREDICTED_COLOR, linewidth=2.0, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Learning Curve - {model_name}")
    ax.grid(True)
    ax.legend(frameon=False)
    _finish_plot(out / f"learning_curve_{_slug(model_name)}.png")
