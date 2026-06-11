"""Feature engineering for ML and deep-learning forecasting models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import TARGET


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    hour = data.index.hour
    dayofyear = data.index.dayofyear
    month = data.index.month
    data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    data["doy_sin"] = np.sin(2 * np.pi * dayofyear / 365.25)
    data["doy_cos"] = np.cos(2 * np.pi * dayofyear / 365.25)
    data["month"] = month
    data["is_daylight"] = (data["ALLSKY_SFC_SW_DWN"] > 0).astype(int)
    return data


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    for lag in [1, 2, 3, 24, 48, 168]:
        data[f"pv_lag_{lag}"] = data[TARGET].shift(lag)
    return data


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    for window in [3, 24, 48, 168]:
        data[f"pv_roll_mean_{window}"] = data[TARGET].shift(1).rolling(window).mean()
        data[f"pv_roll_std_{window}"] = data[TARGET].shift(1).rolling(window).std()
    return data


def build_feature_matrix(df: pd.DataFrame):
    """Create tabular features for Random Forest and XGBoost."""
    data = add_time_features(df)
    data = add_lag_features(data)
    data = add_rolling_features(data)

    feature_cols = [
        "ALLSKY_SFC_SW_DWN", "T2M", "WS10M", "QV2M", "PRECTOTCORR",
        "hour_sin", "hour_cos", "doy_sin", "doy_cos", "month", "is_daylight",
    ]
    feature_cols += [column for column in data.columns if column.startswith("pv_lag_")]
    feature_cols += [column for column in data.columns if column.startswith("pv_roll_")]
    available_features = [column for column in feature_cols if column in data.columns]
    data = data.dropna(subset=available_features + [TARGET])
    return data[available_features], data[TARGET], data


def chronological_split(X, y, test_size: float = 0.2):
    split_index = int(len(X) * (1 - test_size))
    return X.iloc[:split_index], X.iloc[split_index:], y.iloc[:split_index], y.iloc[split_index:]


def create_sequences(df: pd.DataFrame, sequence_length: int = 24):
    """Create multivariate sequences for LSTM, CNN, and CNN-LSTM."""
    data = add_time_features(df).copy()
    input_features = [
        "ALLSKY_SFC_SW_DWN", "T2M", "WS10M", "QV2M", "PRECTOTCORR",
        "hour_sin", "hour_cos", "doy_sin", "doy_cos", "month", TARGET,
    ]
    input_features = [column for column in input_features if column in data.columns]
    data = data.dropna(subset=input_features + [TARGET])

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    scaled_X = feature_scaler.fit_transform(data[input_features])
    scaled_y = target_scaler.fit_transform(data[[TARGET]])

    X_seq = []
    y_seq = []
    timestamps = []
    for index in range(sequence_length, len(data)):
        X_seq.append(scaled_X[index - sequence_length : index])
        y_seq.append(scaled_y[index])
        timestamps.append(data.index[index])

    return np.array(X_seq), np.array(y_seq), np.array(timestamps), target_scaler, data


def chronological_sequence_split(X, y, timestamps, test_size: float = 0.2):
    split_index = int(len(X) * (1 - test_size))
    return X[:split_index], X[split_index:], y[:split_index], y[split_index:], timestamps[:split_index], timestamps[split_index:]
