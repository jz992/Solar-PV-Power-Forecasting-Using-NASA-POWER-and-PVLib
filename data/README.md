# Data

`nasa_power_hourly.csv` is a NASA POWER hourly point export for latitude 35.9375 and longitude 14.3754 from 2025-01-01 through 2026-05-31 UTC.

The target variable used by this project, `pv_power_kw`, is not included in the raw data. It is simulated in the pipeline using PVLib and the NASA POWER weather and irradiance variables.

Do not describe `pv_power_kw` as measured solar plant generation. It is modeled output for a simplified fixed-tilt 1 kW PV system.
