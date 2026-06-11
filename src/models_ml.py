"""Machine-learning model definitions."""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.config import SEED


def build_random_forest() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=250,
        max_depth=18,
        min_samples_leaf=2,
        random_state=SEED,
        n_jobs=-1,
    )


def build_xgboost() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=-1,
    )
