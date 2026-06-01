# Solar PV Power Forecasting Using NASA POWER and PVLib

This project builds an end-to-end solar PV power forecasting workflow from NASA POWER hourly weather and irradiance data. It uses PVLib to simulate the hourly AC power output of a fixed-tilt 1 kW photovoltaic system, then trains and compares classical machine learning, deep learning, and persistence forecasting models.

Important: `pv_power_kw` is simulated with PVLib from NASA POWER meteorological inputs. It is not measured solar plant output, and this repository does not claim to use real plant SCADA or metered PV generation data.

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
├── data/
│   ├── README.md
│   └── nasa_power_hourly.csv
├── src/
│   ├── data_preprocessing.py
│   ├── pv_simulation.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── evaluation.py
│   └── plotting.py
└── results/
    └── figures/
```

## Dataset

The included file `data/nasa_power_hourly.csv` is a NASA POWER hourly export for latitude 35.9375 and longitude 14.3754 from 2025-01-01 through 2026-05-31 UTC.

NASA POWER variables used include:

- `ALLSKY_SFC_SW_DWN`: all-sky surface shortwave downward irradiance
- `T2M`: 2-meter air temperature
- `WS10M` / `WSC`: wind speed
- `PS` / `PSC`: surface pressure
- `QV2M`: specific humidity
- `PRECTOTCORR`: corrected precipitation

## Methodology

1. Load the NASA POWER CSV while skipping its metadata header.
2. Convert date columns into a UTC datetime index.
3. Replace NASA missing value flags (`-999`), interpolate short gaps, and remove trailing all-zero irradiance days when present.
4. Use PVLib to estimate solar position, decompose GHI into DNI/DHI, calculate plane-of-array irradiance, estimate cell temperature, and simulate AC output for a 1 kW fixed-tilt PV system.
5. Create `pv_power_kw` as the supervised learning target.
6. Add chronological and lagged features.
7. Split data chronologically into train and test sets.
8. Train and compare:
   - Persistence baseline
   - Random Forest
   - XGBoost
   - LSTM
   - CNN
   - CNN-LSTM
9. Evaluate all models with MAE, RMSE, and R2.
10. Save plots and metrics in `results/`.

## Installation

Use Python 3.10 to 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Outputs are written to:

- `results/model_comparison.csv`
- `results/figures/pv_power_timeseries.png`
- `results/figures/model_comparison.png`
- `results/figures/actual_vs_predicted.png`
- `results/figures/residuals.png`

## Reproducibility Notes

- The split is chronological, preserving time order and avoiding leakage from future data.
- Neural models use deterministic seeds where supported.
- The simulated PV system is a simplified fixed-tilt 1 kW PVWatts-style system. Results should be interpreted as a modeling exercise, not as site-validated plant performance.
- The supplied CSV contains all-zero irradiance rows after 2025-12-31; the preprocessing step removes that trailing invalid segment before modeling.

## License

This project is prepared as a GitHub-ready research and portfolio project. Add a license file before public release if redistribution terms are needed.
