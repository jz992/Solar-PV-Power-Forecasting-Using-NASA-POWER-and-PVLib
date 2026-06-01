from pathlib import Path

import pandas as pd


NASA_MISSING_VALUE = -999


def _find_data_start(path: Path) -> int:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file):
            if line.startswith("YEAR,MO,DY,HR"):
                return line_number
    raise ValueError("Could not find NASA POWER data header row.")


def load_nasa_power_data(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    header_row = _find_data_start(path)
    df = pd.read_csv(path, skiprows=header_row)

    required_columns = {"YEAR", "MO", "DY", "HR", "ALLSKY_SFC_SW_DWN", "T2M"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required NASA POWER columns: {sorted(missing)}")

    df["datetime"] = pd.to_datetime(
        {
            "year": df["YEAR"],
            "month": df["MO"],
            "day": df["DY"],
            "hour": df["HR"],
        },
        utc=True,
    )
    df = df.drop(columns=["YEAR", "MO", "DY", "HR"]).set_index("datetime").sort_index()
    df = df.replace(NASA_MISSING_VALUE, pd.NA)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.interpolate(method="time", limit_direction="both")
    df = df.dropna()
    df = remove_trailing_zero_irradiance_days(df)

    return df


def remove_trailing_zero_irradiance_days(
    df: pd.DataFrame,
    irradiance_col: str = "ALLSKY_SFC_SW_DWN",
    min_daily_max_wh_m2: float = 1.0,
) -> pd.DataFrame:
    daily_max = df[irradiance_col].resample("D").max()
    valid_days = daily_max[daily_max > min_daily_max_wh_m2]
    if valid_days.empty:
        raise ValueError("No valid irradiance days found in NASA POWER data.")

    last_valid_day = valid_days.index.max()
    cleaned = df.loc[: last_valid_day + pd.Timedelta(days=1) - pd.Timedelta(hours=1)].copy()

    removed_rows = len(df) - len(cleaned)
    if removed_rows > 0:
        print(
            f"Removed {removed_rows} trailing rows with all-zero daily irradiance "
            f"after {last_valid_day.date()}."
        )

    return cleaned
