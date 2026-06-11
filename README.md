# Solar PV Power Forecasting using NASA POWER and PVLib

Solar PV power forecasting with NASA POWER meteorological data, PVLib simulation, Random Forest, XGBoost, LSTM, CNN, and CNN-LSTM, with full 24-hour and daylight-only evaluation.

## Project Overview

This repository contains a professional, runnable Python pipeline converted from the original notebook into modular `.py` files for solar PV power forecasting using NASA POWER hourly meteorological data and PVLib-based PV system simulation.

NASA POWER provides the meteorological and irradiance inputs. PVLib converts those inputs into simulated photovoltaic power output. The simulated PV output is then used as the forecasting target for machine-learning and deep-learning models.

The PV output in this project is simulated using PVLib. It is not measured output from a physical solar plant.

The original notebook logic has been converted into `main.py` and reusable modules under `src/`. The repository is intended to run as a Python project with `python main.py`.

## Motivation

Solar PV generation depends strongly on irradiance, weather, and time-of-day patterns. This project demonstrates how publicly available meteorological data can be combined with a physics-based PV simulation library and forecasting models to build a complete renewable-energy forecasting workflow.

## Data Source

The input file is a NASA POWER hourly CSV placed at:

```text
data/nasa_power_hourly.csv
```

Required columns are `YEAR`, `MO`, `DY`, `HR`, `ALLSKY_SFC_SW_DWN`, `T2M`, `WS10M`, `QV2M`, and `PRECTOTCORR`.

NASA POWER provides gridded meteorological and irradiance data, not plant-level PV measurements.

## PVLib Simulation

PVLib generates the simulated PV output. The simulation uses NASA POWER irradiance and weather variables to estimate solar position, decompose irradiance, calculate plane-of-array irradiance, estimate cell temperature, and produce AC PV power using PVWatts.

Default PV system assumptions are stored in `src/config.py`, including site coordinates, tilt, azimuth, system capacity, and inverter efficiency.

## Methodology

The final methodology from the original notebook is preserved:

1. Load and preprocess the NASA POWER hourly CSV.
2. Simulate PV output using PVLib.
3. Keep only valid 2025 data and remove invalid zero-only trailing periods.
4. Save `results/simulated_pv_dataset.csv`.
5. Train all models on full 24-hour data.
6. Use chronological train/test splitting to respect time order.
7. Report both full 24-hour metrics and daylight-only metrics.
8. Use `SEED = 42` for reproducibility.

## Models

The pipeline trains and compares:

- Persistence Baseline
- Random Forest
- XGBoost
- LSTM
- CNN
- CNN-LSTM

Random Forest and XGBoost use time, lag, and rolling features. LSTM, CNN, and CNN-LSTM use sequence inputs with a 24-hour sequence length.

## Evaluation Strategy

Models are evaluated with:

- MAE in kW
- RMSE in kW
- R2 score

Two evaluation tables are generated:

```text
results/model_comparison_full_24h.csv
results/model_comparison_daylight_only.csv
```

Full 24-hour metrics can be inflated by night-time zeros because PV output is naturally zero at night. Daylight-only evaluation is included to better assess actual PV generation forecasting during periods when the system can produce power.

## Results

The original workflow found that XGBoost and Random Forest achieved the best results. LSTM also performed well, while CNN and CNN-LSTM were less effective than the tree-based models for this structured time-series forecasting setup.

Generated outputs include model comparison CSV files and figures under `results/figures/`.

## Feature Importance and SHAP Explainability

The repository includes Random Forest feature importance, XGBoost feature importance, and an XGBoost SHAP summary plot. These outputs help explain which meteorological, irradiance, lag, rolling, and time features contribute most to the simulated PV forecasting task.

Feature importance and SHAP explainability are reported for tree-based models because they are efficient and interpretable for structured tabular forecasting data. Deep-learning explainability was not included in this version to keep the workflow lightweight and reproducible.

Key figures:

- [Random Forest feature importance](results/figures/feature_importance_random_forest.png)
- [XGBoost feature importance](results/figures/feature_importance_xgboost.png)
- [XGBoost SHAP summary](results/figures/shap_summary_xgboost.png)
- [Daylight-only RMSE comparison](results/figures/model_comparison_rmse_daylight_only.png)
- [Daylight-only R2 comparison](results/figures/model_comparison_r2_daylight_only.png)
- [LSTM learning curve](results/figures/learning_curve_lstm.png)

## Limitations

- PV output is simulated using PVLib, not measured from a physical solar plant.
- NASA POWER data is gridded and may not match exact on-site sensor conditions.
- The workflow uses one-step-ahead forecasting rather than a full operational day-ahead forecasting system.
- Results are influenced by the deterministic PVLib simulation used to generate the target.
- Deep-learning models may require more data and tuning to outperform tree-based methods on this dataset.

## How to Run

Create and activate a virtual environment, install dependencies, place the NASA POWER CSV in `data/`, and run the pipeline:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

On macOS/Linux, activate the environment with:

```bash
source venv/bin/activate
```

## Repository Structure

```text
Solar-PV-Power-Forecasting-Using-NASA-POWER-and-PVLib/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
├── data/
│   ├── README.md
│   └── nasa_power_hourly.csv
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── pv_simulation.py
│   ├── feature_engineering.py
│   ├── evaluation.py
│   ├── plotting.py
│   ├── models_ml.py
│   ├── models_dl.py
│   └── explainability.py
└── results/
    └── figures/
```

## Future Improvements

- Add real measured PV plant data if available.
- Add multi-step 24-hour ahead forecasting.
- Add hyperparameter tuning with Optuna or scikit-learn search tools.
- Add additional weather forecast inputs.
- Add model persistence and experiment tracking.
- Add a dashboard for forecast visualization.

