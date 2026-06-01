import numpy as np
import pandas as pd
import pvlib


LATITUDE = 35.9375
LONGITUDE = 14.3754
ALTITUDE_M = 4.38
SYSTEM_CAPACITY_KW = 1.0


def simulate_pv_power(weather: pd.DataFrame) -> pd.DataFrame:
    df = weather.copy()

    ghi = df["ALLSKY_SFC_SW_DWN"].clip(lower=0)
    temp_air = df["T2M"]
    wind_speed = df["WSC"] if "WSC" in df.columns else df.get("WS10M", 1.0)

    location = pvlib.location.Location(
        latitude=LATITUDE,
        longitude=LONGITUDE,
        tz="UTC",
        altitude=ALTITUDE_M,
        name="NASA POWER point",
    )

    solar_position = location.get_solarposition(df.index)
    dni_extra = pvlib.irradiance.get_extra_radiation(df.index)
    decomposition = pvlib.irradiance.erbs(ghi, solar_position["zenith"], df.index)
    dni = decomposition["dni"].fillna(0).clip(lower=0)
    dhi = decomposition["dhi"].fillna(0).clip(lower=0)

    surface_tilt = LATITUDE
    surface_azimuth = 180
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=solar_position["apparent_zenith"],
        solar_azimuth=solar_position["azimuth"],
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        dni_extra=dni_extra,
        model="haydavies",
    )

    cell_temperature = pvlib.temperature.sapm_cell(
        poa_global=poa["poa_global"].clip(lower=0),
        temp_air=temp_air,
        wind_speed=wind_speed,
        **pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"],
    )

    dc_power_w = pvlib.pvsystem.pvwatts_dc(
        effective_irradiance=poa["poa_global"].clip(lower=0),
        temp_cell=cell_temperature,
        pdc0=SYSTEM_CAPACITY_KW * 1000,
        gamma_pdc=-0.0035,
    )
    ac_power_w = pvlib.inverter.pvwatts(
        pdc=dc_power_w,
        pdc0=SYSTEM_CAPACITY_KW * 1000,
        eta_inv_nom=0.96,
        eta_inv_ref=0.9637,
    )

    df["solar_zenith"] = solar_position["apparent_zenith"]
    df["dni"] = dni
    df["dhi"] = dhi
    df["poa_global"] = poa["poa_global"].clip(lower=0)
    df["cell_temperature"] = cell_temperature
    df["pv_power_kw"] = np.clip(ac_power_w / 1000, 0, SYSTEM_CAPACITY_KW)

    return df.dropna()
