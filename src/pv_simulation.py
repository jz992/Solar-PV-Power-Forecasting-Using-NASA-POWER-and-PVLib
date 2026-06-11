"""PVLib-based solar PV power simulation."""

from __future__ import annotations

import pandas as pd
import pvlib

from src.config import (
    PV_ALTITUDE, PV_AZIMUTH, PV_GAMMA_PDC, PV_INVERTER_EFFICIENCY, PV_LATITUDE,
    PV_LONGITUDE, PV_SYSTEM_CAPACITY_KW, PV_TEMPERATURE_MODEL, PV_TILT,
)


def simulate_pv_power(
    df: pd.DataFrame,
    latitude: float = PV_LATITUDE,
    longitude: float = PV_LONGITUDE,
    altitude: float = PV_ALTITUDE,
    tilt: float = PV_TILT,
    azimuth: float = PV_AZIMUTH,
    system_capacity_kw: float = PV_SYSTEM_CAPACITY_KW,
    temperature_model: str = PV_TEMPERATURE_MODEL,
) -> pd.DataFrame:
    """Generate simulated PV output from NASA POWER irradiance and weather data."""
    data = df.copy()
    ghi = data["ALLSKY_SFC_SW_DWN"].clip(lower=0)
    temp_air = data["T2M"]
    wind_speed = data["WS10M"].clip(lower=0)

    location = pvlib.location.Location(latitude=latitude, longitude=longitude, tz="UTC", altitude=altitude, name="Selected site")
    solar_position = location.get_solarposition(data.index)
    dni_extra = pvlib.irradiance.get_extra_radiation(data.index)

    erbs = pvlib.irradiance.erbs(ghi=ghi, zenith=solar_position["zenith"], datetime_or_doy=data.index)
    dni = erbs["dni"].fillna(0).clip(lower=0)
    dhi = erbs["dhi"].fillna(0).clip(lower=0)

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        solar_zenith=solar_position["apparent_zenith"],
        solar_azimuth=solar_position["azimuth"],
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        dni_extra=dni_extra,
        model="haydavies",
    )
    poa_global = poa["poa_global"].fillna(0).clip(lower=0)

    temp_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][temperature_model]
    cell_temperature = pvlib.temperature.sapm_cell(poa_global=poa_global, temp_air=temp_air, wind_speed=wind_speed, **temp_params)

    pdc0_w = system_capacity_kw * 1000
    dc_power = pvlib.pvsystem.pvwatts_dc(g_poa_effective=poa_global, temp_cell=cell_temperature, pdc0=pdc0_w, gamma_pdc=PV_GAMMA_PDC)
    ac_power = pvlib.inverter.pvwatts(pdc=dc_power, pdc0=pdc0_w, eta_inv_nom=PV_INVERTER_EFFICIENCY)

    data["dni"] = dni
    data["dhi"] = dhi
    data["poa_global"] = poa_global
    data["cell_temperature"] = cell_temperature
    data["pv_power_kw"] = (ac_power / 1000).clip(lower=0, upper=system_capacity_kw)
    return data.dropna(subset=["pv_power_kw"])


def keep_valid_2025_data(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the valid 2025 window and remove invalid zero-only trailing days."""
    data = df.loc["2025-01-01":"2025-12-31 23:00:00"].copy()
    daily_max = data["ALLSKY_SFC_SW_DWN"].resample("D").max()
    valid_days = daily_max[daily_max > 10]
    if valid_days.empty:
        return data
    return data.loc[valid_days.index.min() : valid_days.index.max() + pd.Timedelta(hours=23)].copy()
