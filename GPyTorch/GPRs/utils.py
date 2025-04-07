"""Utility Functions"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import periodogram, find_peaks
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

def latlon_to_cartesian(lat, lon):
    """ Converting (lat., lon.) to cartesian coordinates on a unit sphere
        Note: WeatherBench data is defined on a constant altitude """

    lat, lon = np.deg2rad(lat), np.deg2rad(lon)

    x = np.cos(lat)[:, None] * np.cos(lon)[None, :]
    y = np.cos(lat)[:, None] * np.sin(lon)[None, :]
    z = np.sin(lat)[:, None] * np.ones_like(lon)[None, :]

    return np.stack([x, y, z], axis=-1) # Out: (nlat, nlon, 3)

def cartesian_to_latlon(xyz):

    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    lat = np.arcsin(z) 
    lon = np.arctan2(y, x)  
    lat_deg = np.degrees(lat)
    lon_deg = np.degrees(lon)
    lon_deg = (lon_deg + 360) % 360
    latlon = np.stack([lat_deg, lon_deg], axis=-1)
    return latlon

def freq_spectrum(val, plot=True, no_peaks=5, fs=1.0):
    """fs is the frequency of the sample in hours"""
    freqs, power = periodogram(val, fs=fs)
    peaks, _ = find_peaks(power, height=1, distance=1)

    # Sort the peaks by their power (descending order), then pick x peaks
    # There might be a peak with period as half the data time?
    sorted_peak_indices = np.argsort(power[peaks])[::-1]
    top_peaks = peaks[sorted_peak_indices[:no_peaks]]

    peak_freqs = freqs[top_peaks]
    peak_powers = power[top_peaks]

    for f, p in zip(peak_freqs, peak_powers):
        print(f"Peak at frequency = {f:.5f} cycles/hour (corresponding period = {1/f:.2f} hrs = {1/f/24:.2f} days = {1/f/(24*30):.2f} months), power = {p/1e5 :.5f} * 1e5")
    
    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(freqs, power)
        plt.xlim(0, 0.005)
        plt.xlabel('Frequency (cycles per hour)')
        plt.ylabel('Power')
        plt.title('Periodogram')
        plt.show()
    
    periods = np.rint(1 / freqs[top_peaks])
    valid_periods = periods[periods < len(val)/2]
    if valid_periods.size > 0:
        return int(valid_periods[0])
    else:
        print("No peaks found")
        return None
    
def splitting_data(data, var, plot=False):
    """Splitting up the weather data into smaller grids based on mean trends.
        Input data with (Time, Lat, Lon) shape then return indices to slice."""
    
    if not isinstance(var, str):
        var = str(var)

    if var not in ['z', 't2m']:
        raise ValueError("Variable must be either 't2m' or 'z'.")
    
    avg_time = data[var].mean(dim=["lat", "lon"])
    avg_lat = data[var].mean(dim=["time", "lon"])
    avg_lon = data[var].mean(dim=["time", "lat"])

    # Find peaks and troughs
    peaks_lat, _ = find_peaks(avg_lat.values, distance=5)
    troughs_lat, _ = find_peaks(-avg_lat.values, distance=5)
    peaks_lon, _ = find_peaks(avg_lon.values, distance=5)
    troughs_lon, _ = find_peaks(-avg_lon.values, distance=5)
    print("Peaks (lat indices):", peaks_lat, "Troughs (lat indices):", troughs_lat)
    print("Peaks (lon indices):", peaks_lon, "Troughs (lon indices):", troughs_lon)

    if plot:
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

        avg_time.plot(ax=axs[0])
        axs[0].set_title("Average Geopotential vs Time")
        axs[0].set_xlabel("Time")
        axs[0].set_ylabel("Geopotential")
        avg_lat.plot(ax=axs[1])
        axs[1].set_title("Average Geopotential vs Latitude")
        axs[1].set_xlabel("Latitude")
        axs[1].set_ylabel("Geopotential")
        avg_lon.plot(ax=axs[2])
        axs[2].set_title("Average Geopotential vs Longitude")
        axs[2].set_xlabel("Longitude")
        axs[2].set_ylabel("Geopotential")

        # Overlay crosses for latitude peaks and troughs
        lat_coords = avg_lat.coords['lat'].values  # coordinate values for latitude
        axs[1].plot(lat_coords[peaks_lat], avg_lat.values[peaks_lat], 'rx', markersize=10, label="Peaks")
        axs[1].plot(lat_coords[troughs_lat], avg_lat.values[troughs_lat], 'kx', markersize=10, label="Troughs")
        axs[1].legend()
        # Overlay crosses for longitude peaks and troughs
        lon_coords = avg_lon.coords['lon'].values  # coordinate values for longitude
        axs[2].plot(lon_coords[peaks_lon], avg_lon.values[peaks_lon], 'rx', markersize=10, label="Peaks")
        axs[2].plot(lon_coords[troughs_lon], avg_lon.values[troughs_lon], 'kx', markersize=10, label="Troughs")
        axs[2].legend()

        plt.tight_layout()
        plt.show()

    lat_boundaries = np.unique(np.r_[0, peaks_lat, troughs_lat, data.lat.size])
    lon_boundaries = np.unique(np.r_[0, peaks_lon, troughs_lon, data.lon.size])

    return lat_boundaries, lon_boundaries

def extreme_points_rel_err(relative_error_mean):
    """Retriving the points with max/min/best(closest to zero) on relative error"""

    max_idx = np.argmax(relative_error_mean)
    min_idx = np.argmin(relative_error_mean)
    best_idx = np.argmin(np.abs(relative_error_mean))

    flat_max_idx = np.unravel_index(max_idx, relative_error_mean.shape)
    flat_min_idx = np.unravel_index(min_idx, relative_error_mean.shape)
    flat_best_idx = np.unravel_index(best_idx, relative_error_mean.shape)

    max_err = relative_error_mean[flat_max_idx[0],flat_max_idx[1]]
    min_err = relative_error_mean[flat_min_idx[0],flat_min_idx[1]]
    best_err = relative_error_mean[flat_best_idx[0],flat_best_idx[1]]

    print(f"Highest relative error: ({int(flat_max_idx[0])},{int(flat_max_idx[1])}), with relative error in percentage: {max_err:.3f}")
    print(f"Lowest relative error: ({int(flat_min_idx[0])},{int(flat_min_idx[1])}), with relative error in percentage: {min_err:.3f}")
    print(f"Smallest relative error: ({int(flat_best_idx[0])},{int(flat_best_idx[1])}), with relative error in percentage: {best_err:.3f}")

    idx = [max_idx, min_idx, best_idx]
    val = [max_err, min_err, best_err]

    return idx, val

def extreme_points_rmse(rmse_error_mean):
    # Best and worst location based on rmse
    rmse_max_idx = np.argmax(rmse_error_mean)
    rmse_min_idx = np.argmin(rmse_error_mean)

    flat_rmse_max_idx = np.unravel_index(rmse_max_idx, rmse_error_mean.shape)
    flat_rmse_min_idx = np.unravel_index(rmse_min_idx, rmse_error_mean.shape)

    rmse_max_err = rmse_error_mean[flat_rmse_max_idx[0],flat_rmse_max_idx[1]]
    rmse_min_err = rmse_error_mean[flat_rmse_min_idx[0],flat_rmse_min_idx[1]]

    print(f"Highest RMSE: ({int(flat_rmse_max_idx[0])},{int(flat_rmse_max_idx[1])}), with value: {rmse_max_err:.3f}")
    print(f"Lowest RMSE: ({int(flat_rmse_min_idx[0])},{int(flat_rmse_min_idx[1])}), with value: {rmse_min_err:.3f}")

    idx = [rmse_max_idx, rmse_min_idx]
    val = [rmse_max_err, rmse_min_err]

    return idx, val

def get_periods(data, fs=1):
    """Gets the dominant period. Set fs=1 if data sampled hourly."""
    T = data.z.values.shape[0] # Currently works only on z500

    fft_vals = np.fft.rfft(data.z.values, axis=0)  
    freqs = np.fft.rfftfreq(T, d=1/fs)  
    power = np.abs(fft_vals)**2  

    power_nozero = power[1:]     
    freqs_nozero = freqs[1:]       

    sorted_idx = np.argsort(power_nozero, axis=0)
    dom_idx = sorted_idx[-1, :, :]
    dom_freq = freqs_nozero[dom_idx] 
    dom_period = 1.0 / dom_freq   
    dom_period_da = xr.DataArray(dom_period, coords=[data.lat, data.lon], dims=["lat", "lon"])

    return dom_period_da

def seasonal_decompose_grid(data, dom_period, model='additive'):
    """Perform seasonal_decompose across whole grid"""
    if hasattr(data, 'values'):
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

            result = seasonal_decompose(ts, period=period_int, model=model, extrapolate_trend='freq')
            # Define our "seasonal" component as seasonal + mean(trend)
            seasonal_component = result.seasonal.values + np.mean(result.trend.values)
            residual = ts.values - seasonal_component
            
            seasonal_full[:, i, j] = seasonal_component
            residual_full[:, i, j] = residual

    detrend = xr.Dataset({
        "seasonal": (("time", "lat", "lon"), seasonal_full),
        "resid": (("time", "lat", "lon"), residual_full)
        },
        coords={
            "time": data.time.values,
            "lat": data.lat.values,
            "lon": data.lon.values
        })
    
    return detrend
