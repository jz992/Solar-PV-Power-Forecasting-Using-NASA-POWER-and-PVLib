"""Data loading and preprocessing utilities for NASA POWER hourly data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "YEAR", "MO", "DY", "HR", "ALLSKY_SFC_SW_DWN", "T2M", "WS10M", "QV2M", "PRECTOTCORR",
]

NUMERIC_COLUMNS = [
    "ALLSKY_SFC_SW_DWN", "T2M", "WS10M", "QV2M", "PRECTOTCORR", "PS", "PSC", "WSC",
]


def find_nasa_header_end(file_path: str | Path) -> int:
    """Return the number of rows to skip before the NASA POWER CSV table."""
    with open(file_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file):
            if "-END HEADER-" in line:
                return line_number + 1
    return 0


def load_nasa_power_csv(file_path: str | Path) -> pd.DataFrame:
    """Load a NASA POWER CSV, including files with metadata header lines."""
    skiprows = find_nasa_header_end(file_path)
    df = pd.read_csv(file_path, skiprows=skiprows)
    df.columns = df.columns.str.strip()
    return df


def preprocess_nasa_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean NASA POWER data and create a UTC datetime index."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = df.copy()
    data["datetime"] = pd.to_datetime(
        dict(
            year=data["YEAR"].astype(int),
            month=data["MO"].astype(int),
            day=data["DY"].astype(int),
            hour=data["HR"].astype(int),
        ),
        utc=True,
        errors="coerce",
    )

    for column in NUMERIC_COLUMNS:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["datetime", "ALLSKY_SFC_SW_DWN", "T2M", "WS10M"])
    data = data.sort_values("datetime").set_index("datetime")

    for column in data.select_dtypes(include="number").columns:
        data.loc[data[column] <= -900, column] = np.nan

    return data.ffill().bfill()


def load_and_preprocess(file_path: str | Path) -> pd.DataFrame:
    """Load and preprocess NASA POWER data in one call."""
    return preprocess_nasa_data(load_nasa_power_csv(file_path))
