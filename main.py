from pathlib import Path

from src.data_preprocessing import load_nasa_power_data
from src.evaluation import evaluate_predictions, save_model_comparison
from src.feature_engineering import (
    build_feature_matrix,
    chronological_split,
    create_sequence_data,
)
from src.models import (
    train_cnn,
    train_cnn_lstm,
    train_lstm,
    train_persistence,
    train_random_forest,
    train_xgboost,
)
from src.plotting import (
    plot_actual_vs_predicted,
    plot_model_comparison,
    plot_pv_power_timeseries,
    plot_residuals,
)
from src.pv_simulation import simulate_pv_power


DATA_PATH = Path("data/nasa_power_hourly.csv")
RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
TARGET = "pv_power_kw"


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    weather = load_nasa_power_data(DATA_PATH)
    simulated = simulate_pv_power(weather)
    features = build_feature_matrix(simulated, target_col=TARGET)

    train_df, test_df = chronological_split(features, train_size=0.8)
    feature_cols = [col for col in features.columns if col != TARGET]

    results = []
    predictions = {}

    persistence_pred = train_persistence(test_df)
    predictions["Persistence"] = persistence_pred
    results.append(evaluate_predictions("Persistence", test_df[TARGET], persistence_pred))

    rf_pred = train_random_forest(train_df, test_df, feature_cols, TARGET)
    predictions["Random Forest"] = rf_pred
    results.append(evaluate_predictions("Random Forest", test_df[TARGET], rf_pred))

    xgb_pred = train_xgboost(train_df, test_df, feature_cols, TARGET)
    predictions["XGBoost"] = xgb_pred
    results.append(evaluate_predictions("XGBoost", test_df[TARGET], xgb_pred))

    sequence_length = 24
    X_train_seq, y_train_seq, X_test_seq, y_test_seq, sequence_index = create_sequence_data(
        train_df, test_df, feature_cols, TARGET, sequence_length=sequence_length
    )

    lstm_pred = train_lstm(X_train_seq, y_train_seq, X_test_seq)
    predictions["LSTM"] = lstm_pred
    results.append(evaluate_predictions("LSTM", y_test_seq, lstm_pred))

    cnn_pred = train_cnn(X_train_seq, y_train_seq, X_test_seq)
    predictions["CNN"] = cnn_pred
    results.append(evaluate_predictions("CNN", y_test_seq, cnn_pred))

    cnn_lstm_pred = train_cnn_lstm(X_train_seq, y_train_seq, X_test_seq)
    predictions["CNN-LSTM"] = cnn_lstm_pred
    results.append(evaluate_predictions("CNN-LSTM", y_test_seq, cnn_lstm_pred))

    comparison = save_model_comparison(results, RESULTS_DIR / "model_comparison.csv")

    aligned_predictions = {
        "Persistence": predictions["Persistence"].loc[sequence_index],
        "Random Forest": predictions["Random Forest"].loc[sequence_index],
        "XGBoost": predictions["XGBoost"].loc[sequence_index],
        "LSTM": lstm_pred,
        "CNN": cnn_pred,
        "CNN-LSTM": cnn_lstm_pred,
    }

    plot_pv_power_timeseries(simulated, FIGURES_DIR / "pv_power_timeseries.png")
    plot_model_comparison(comparison, FIGURES_DIR / "model_comparison.png")
    plot_actual_vs_predicted(
        sequence_index,
        y_test_seq,
        aligned_predictions,
        FIGURES_DIR / "actual_vs_predicted.png",
    )
    plot_residuals(
        sequence_index,
        y_test_seq,
        aligned_predictions,
        FIGURES_DIR / "residuals.png",
    )

    print("Model comparison saved to results/model_comparison.csv")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
