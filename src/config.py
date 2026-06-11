"""Project configuration constants."""

from __future__ import annotations

from pathlib import Path


SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
DATA_PATH = DATA_DIR / "nasa_power_hourly.csv"
SIMULATED_DATA_PATH = RESULTS_DIR / "simulated_pv_dataset.csv"
FULL_RESULTS_PATH = RESULTS_DIR / "model_comparison_full_24h.csv"
DAYLIGHT_RESULTS_PATH = RESULTS_DIR / "model_comparison_daylight_only.csv"

SEQUENCE_LENGTH = 24
EPOCHS = 60
BATCH_SIZE = 32
TEST_SIZE = 0.2
VALIDATION_SPLIT = 0.15
EARLY_STOPPING_PATIENCE = 25

TARGET = "pv_power_kw"
DAYLIGHT_THRESHOLD_KW = 0.01

PV_LATITUDE = 35.9375
PV_LONGITUDE = 14.3754
PV_ALTITUDE = 100.0
PV_TILT = 30.0
PV_AZIMUTH = 180.0
PV_SYSTEM_CAPACITY_KW = 1.0
PV_TEMPERATURE_MODEL = "open_rack_glass_glass"
PV_GAMMA_PDC = -0.004
PV_INVERTER_EFFICIENCY = 0.96
