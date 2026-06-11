# Data

Download the NASA POWER hourly CSV manually and place it in this directory as:

```text
data/nasa_power_hourly.csv
```

Required columns:

- `YEAR`
- `MO`
- `DY`
- `HR`
- `ALLSKY_SFC_SW_DWN`
- `T2M`
- `WS10M`
- `QV2M`
- `PRECTOTCORR`

NASA POWER provides meteorological and irradiance data. The PV output used by this project is simulated using PVLib and is not measured output from a physical solar plant.
