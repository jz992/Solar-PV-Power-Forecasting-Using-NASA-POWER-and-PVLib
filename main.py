"""Full pipeline runner for solar PV power simulation and forecasting."""

from __future__ import annotations

import os
import random
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from src.config import (
    BATCH_SIZE,
    DATA_PATH,
    DAYLIGHT_RESULTS_PATH,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    FIGURES_DIR,
    FULL_RESULTS_PATH,
    RESULTS_DIR,
    SEED,
    SEQUENCE_LENGTH,
    SIMULATED_DATA_PATH,
    TEST_SIZE,
    VALIDATION_SPLIT,
)
from src.data_preprocessing import load_and_preprocess
from src.evaluation import daylight_evaluation, evaluate_regression, persistence_baseline
from src.explainability import save_random_forest_shap_summary, save_shap_summary
from src.feature_engineering import (
    build_feature_matrix,
    chronological_sequence_split,
    chronological_split,
    create_sequences,
)
from src.models_dl import build_cnn, build_cnn_lstm, build_lstm
from src.models_ml import build_random_forest, build_xgboost
from src.plotting import (
    plot_actual_vs_predicted,
    plot_daylight_only,
    plot_feature_importance,
    plot_irradiance_vs_pv,
    plot_learning_curve,
    plot_model_comparison,
    plot_pv_timeseries,
)
from src.pv_simulation import keep_valid_2025_data, simulate_pv_power

warnings.filterwarnings("ignore")


def set_reproducibility(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def train_tabular_models(df: pd.DataFrame):
    X, y, _ = build_feature_matrix(df)
    X_train, X_test, y_train, y_test = chronological_split(X, y, test_size=TEST_SIZE)
    results_full, results_daylight = [], []

    y_true_base, y_pred_base = persistence_baseline(y_test)
    results_full.append(evaluate_regression(y_true_base, y_pred_base, "Persistence Baseline"))
    plot_actual_vs_predicted(y_true_base, y_pred_base, "Persistence Baseline")
    results_daylight.append(daylight_evaluation(y_true_base, y_pred_base, "Persistence Baseline"))
    plot_daylight_only(y_true_base, y_pred_base, "Persistence Baseline")

    rf_model = build_random_forest()
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    results_full.append(evaluate_regression(y_test, rf_pred, "Random Forest"))
    plot_actual_vs_predicted(y_test, rf_pred, "Random Forest")
    results_daylight.append(daylight_evaluation(y_test, rf_pred, "Random Forest"))
    plot_daylight_only(y_test, rf_pred, "Random Forest")
    plot_feature_importance(
        X_train.columns,
        rf_model.feature_importances_,
        model_name="Random Forest",
        filename="feature_importance_random_forest.png",
    )
    save_random_forest_shap_summary(rf_model, X_test)

    xgb_model = build_xgboost()
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    results_full.append(evaluate_regression(y_test, xgb_pred, "XGBoost"))
    plot_actual_vs_predicted(y_test, xgb_pred, "XGBoost")
    results_daylight.append(daylight_evaluation(y_test, xgb_pred, "XGBoost"))
    plot_daylight_only(y_test, xgb_pred, "XGBoost")
    plot_feature_importance(
        X_train.columns,
        xgb_model.feature_importances_,
        model_name="XGBoost",
        filename="feature_importance_xgboost.png",
    )
    save_shap_summary(xgb_model, X_test, model_name="XGBoost")
    return results_full, results_daylight


def train_deep_learning_models(df: pd.DataFrame):
    X_seq, y_seq, timestamps, target_scaler, _ = create_sequences(df, sequence_length=SEQUENCE_LENGTH)
    X_train, X_test, y_train, y_test, _, _ = chronological_sequence_split(
        X_seq,
        y_seq,
        timestamps,
        test_size=TEST_SIZE,
    )
    y_test_inv = target_scaler.inverse_transform(y_test).reshape(-1)
    n_features = X_seq.shape[2]
    results_full, results_daylight = [], []
    dl_models = {
        "LSTM": build_lstm(SEQUENCE_LENGTH, n_features),
        "CNN": build_cnn(SEQUENCE_LENGTH, n_features),
        "CNN-LSTM": build_cnn_lstm(SEQUENCE_LENGTH, n_features),
    }
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )

    for model_name, model in dl_models.items():
        random.seed(SEED)
        np.random.seed(SEED)
        tf.random.set_seed(SEED)
        print(f"\nTraining {model_name}...")
        history = model.fit(
            X_train,
            y_train,
            validation_split=VALIDATION_SPLIT,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=1,
        )
        plot_learning_curve(history, model_name)
        pred_scaled = model.predict(X_test, verbose=0)
        pred = target_scaler.inverse_transform(pred_scaled).reshape(-1)
        results_full.append(evaluate_regression(y_test_inv, pred, model_name))
        plot_actual_vs_predicted(y_test_inv, pred, model_name)
        results_daylight.append(daylight_evaluation(y_test_inv, pred, model_name))
        plot_daylight_only(y_test_inv, pred, model_name)
    return results_full, results_daylight


def main() -> None:
    set_reproducibility(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"NASA POWER CSV not found at {DATA_PATH}. Place the file there and rerun python main.py.")

    print("Loading NASA POWER data...")
    nasa_df = load_and_preprocess(DATA_PATH)
    print("Simulating PV output with PVLib...")
    pv_df = keep_valid_2025_data(simulate_pv_power(nasa_df))
    pv_df.to_csv(SIMULATED_DATA_PATH)
    print("Cleaned dataset start:", pv_df.index.min())
    print("Cleaned dataset end:", pv_df.index.max())
    print("Rows after cleaning:", len(pv_df))
    print(pv_df["pv_power_kw"].describe())

    plot_pv_timeseries(pv_df)
    plot_irradiance_vs_pv(pv_df)
    print("Training Persistence, Random Forest, and XGBoost...")
    tabular_full, tabular_daylight = train_tabular_models(pv_df)
    print("Training LSTM, CNN, and CNN-LSTM...")
    dl_full, dl_daylight = train_deep_learning_models(pv_df)

    results_full_df = pd.DataFrame(tabular_full + dl_full).sort_values("RMSE_kW")
    results_daylight_df = pd.DataFrame(tabular_daylight + dl_daylight).sort_values("RMSE_kW")
    results_full_df.to_csv(FULL_RESULTS_PATH, index=False)
    results_daylight_df.to_csv(DAYLIGHT_RESULTS_PATH, index=False)
    print("\nFull 24-hour evaluation:")
    print(results_full_df)
    print("\nDaylight-only evaluation:")
    print(results_daylight_df)
    plot_model_comparison(
        results_full_df,
        "Model Comparison - Full 24-Hour RMSE",
        "model_comparison_rmse_full_24h.png",
        metric="RMSE_kW",
    )
    plot_model_comparison(
        results_daylight_df,
        "Model Comparison - Daylight-Only RMSE",
        "model_comparison_rmse_daylight_only.png",
        metric="RMSE_kW",
    )
    plot_model_comparison(
        results_full_df,
        "Model Comparison - Full 24-Hour R2",
        "model_comparison_r2_full_24h.png",
        metric="R2",
    )
    plot_model_comparison(
        results_daylight_df,
        "Model Comparison - Daylight-Only R2",
        "model_comparison_r2_daylight_only.png",
        metric="R2",
    )
    print("\nOutputs saved in results/ and results/figures/.")


if __name__ == "__main__":
    main()
