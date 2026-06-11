"""Explainability functions for tree-based models."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR, SEED


def save_shap_summary(
    model,
    X_test: pd.DataFrame,
    output_path: str | Path = FIGURES_DIR / "shap_summary_xgboost.png",
    sample_size: int = 500,
    model_name: str = "XGBoost",
) -> bool:
    """Save a SHAP summary plot for a tree-based model."""
    try:
        import shap
    except Exception as exc:
        print(f"{model_name} SHAP plot skipped. Reason: {exc}")
        return False

    try:
        sample = X_test.sample(min(sample_size, len(X_test)), random_state=SEED)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        shap.summary_plot(shap_values, sample, show=False, max_display=15)
        plt.title(f"SHAP Summary - {model_name}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        return True
    except Exception as exc:
        plt.close()
        print(f"{model_name} SHAP plot skipped. Reason: {exc}")
        return False


def save_random_forest_shap_summary(
    model,
    X_test: pd.DataFrame,
    output_path: str | Path = FIGURES_DIR / "shap_summary_random_forest.png",
    sample_size: int = 150,
) -> bool:
    """Optionally save a SHAP summary plot for Random Forest on a small sample."""
    print("Attempting Random Forest SHAP summary on a small sample...")
    return save_shap_summary(
        model,
        X_test,
        output_path=output_path,
        sample_size=sample_size,
        model_name="Random Forest",
    )
