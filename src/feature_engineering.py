import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def build_feature_matrix(df: pd.DataFrame, target_col: str = "pv_power_kw") -> pd.DataFrame:
    features = df.copy()
    hour = features.index.hour
    day_of_year = features.index.dayofyear

    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features["day_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    features["day_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    for lag in (1, 2, 3, 24):
        features[f"{target_col}_lag_{lag}"] = features[target_col].shift(lag)

    for window in (3, 6, 24):
        features[f"{target_col}_rolling_mean_{window}"] = (
            features[target_col].shift(1).rolling(window=window).mean()
        )

    return features.dropna()


def chronological_split(df: pd.DataFrame, train_size: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = int(len(df) * train_size)
    return df.iloc[:split_index].copy(), df.iloc[split_index:].copy()


def create_sequence_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    sequence_length: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_df[feature_cols])
    test_features = scaler.transform(test_df[feature_cols])

    def make_sequences(values: np.ndarray, target: pd.Series) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        X, y, index = [], [], []
        for end in range(sequence_length, len(values)):
            X.append(values[end - sequence_length : end])
            y.append(target.iloc[end])
            index.append(target.index[end])
        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), pd.DatetimeIndex(index)

    X_train, y_train, _ = make_sequences(train_features, train_df[target_col])
    X_test, y_test, test_index = make_sequences(test_features, test_df[target_col])

    return X_train, y_train, X_test, y_test, test_index
