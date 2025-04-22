import numpy as np
import pandas as pd
import xarray as xr
from statsmodels.tsa.seasonal import seasonal_decompose


def get_periods(data, fs=1):
    """Gets the dominant period. Set fs=1 if data sampled hourly."""
    T = data.z.values.shape[0]  # Currently works only on z500

    fft_vals = np.fft.rfft(data.z.values, axis=0)
    freqs = np.fft.rfftfreq(T, d=1 / fs)
    power = np.abs(fft_vals) ** 2

    power_nozero = power[1:]
    freqs_nozero = freqs[1:]

    sorted_idx = np.argsort(power_nozero, axis=0)
    dom_idx = sorted_idx[-1, :, :]
    dom_freq = freqs_nozero[dom_idx]
    dom_period = 1.0 / dom_freq
    dom_period_da = xr.DataArray(
        dom_period, coords=[data.lat, data.lon], dims=["lat", "lon"]
    )

    return dom_period_da


def seasonal_decompose_grid(data, dom_period, model="additive"):
    """Perform seasonal_decompose across whole grid"""
    if hasattr(data, "values"):
        data_np = data.z.values
    else:
        data_np = data

    T, nlat, nlon = data_np.shape
    seasonal_full = np.empty_like(data_np, dtype=np.float64)
    residual_full = np.empty_like(data_np, dtype=np.float64)

    for i in range(nlat):
        for j in range(nlon):
            ts = pd.Series(data_np[:, i, j])
            period = dom_period.values[i, j]
            period_int = int(np.round(period))

            result = seasonal_decompose(
                ts, period=period_int, model=model, extrapolate_trend="freq"
            )
            # Define our "seasonal" component as seasonal + mean(trend)
            seasonal_component = result.seasonal.values + np.mean(result.trend.values)
            residual = ts.values - seasonal_component

            seasonal_full[:, i, j] = seasonal_component
            residual_full[:, i, j] = residual

    detrend = xr.Dataset(
        {
            "seasonal": (("time", "lat", "lon"), seasonal_full),
            "resid": (("time", "lat", "lon"), residual_full),
        },
        coords={
            "time": data.time.values,
            "lat": data.lat.values,
            "lon": data.lon.values,
        },
    )

    return detrend


def extrapolate_seasonal(detrend, forecast_steps):
    """Extrapolate a seasonal component by repeating its cycle."""

    T = detrend.time.shape[0]
    n_cycles = int(np.ceil(forecast_steps / T))
    forecast_full = np.tile(detrend.seasonal.values, (n_cycles, 1, 1))
    forecast = forecast_full[:forecast_steps, :, :]

    last_time = pd.to_datetime(detrend.time.values[-1])
    new_times = pd.date_range(
        start=last_time + pd.Timedelta(hours=1), periods=forecast_steps, freq="h"
    )

    seasonal_forecast = xr.DataArray(
        forecast,
        dims=["time", "lat", "lon"],
        coords={
            "time": new_times,
            "lat": detrend.lat.values,
            "lon": detrend.lon.values,
        },
    )

    return seasonal_forecast
