import random

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


SEED = 42


def _set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)


def train_persistence(test_df: pd.DataFrame, target_col: str = "pv_power_kw") -> pd.Series:
    return test_df[f"{target_col}_lag_1"].clip(lower=0)


def train_random_forest(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> pd.Series:
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=18,
        min_samples_leaf=2,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(train_df[feature_cols], train_df[target_col])
    predictions = model.predict(test_df[feature_cols])
    return pd.Series(np.clip(predictions, 0, 1), index=test_df.index)


def train_xgboost(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> pd.Series:
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=250,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=SEED,
        n_jobs=1,
    )
    model.fit(train_df[feature_cols], train_df[target_col])
    predictions = model.predict(test_df[feature_cols])
    return pd.Series(np.clip(predictions, 0, 1), index=test_df.index)


class LSTMRegressor(torch.nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.lstm = torch.nn.LSTM(n_features, 32, batch_first=True)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :]).squeeze(-1)


class CNNRegressor(torch.nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2)).squeeze(-1)


class CNNLSTMRegressor(torch.nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )
        self.lstm = torch.nn.LSTM(32, 32, batch_first=True)
        self.head = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :]).squeeze(-1)


def _train_torch_model(model, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    _set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for _ in range(12):
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        predictions = model(test_tensor).cpu().numpy()

    return np.clip(predictions, 0, 1)


def train_lstm(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    return _train_torch_model(LSTMRegressor(X_train.shape[-1]), X_train, y_train, X_test)


def train_cnn(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    return _train_torch_model(CNNRegressor(X_train.shape[-1]), X_train, y_train, X_test)


def train_cnn_lstm(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    return _train_torch_model(CNNLSTMRegressor(X_train.shape[-1]), X_train, y_train, X_test)
