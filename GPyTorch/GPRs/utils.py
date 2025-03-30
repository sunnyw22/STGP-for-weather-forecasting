"""Utility Functions"""

import numpy as np
from scipy.signal import periodogram, find_peaks
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

    # Sort the peaks by their power (descending order), then pick three peaks
    # There might be a peak with period as half the data time
    sorted_peak_indices = np.argsort(power[peaks])[::-1]
    top_peaks = peaks[sorted_peak_indices[:no_peaks]]

    peak_freqs = freqs[top_peaks]
    peak_powers = power[top_peaks]

    for f, p in zip(peak_freqs, peak_powers):
        print(f"Peak at frequency = {f:.5f} cycles/hour (corresponding period = {1/f:.2f} hrs = {1/f/24:.2f} days = {1/f/(24*30):.2f} months), power = {p/1e5 :.5f} * 1e5")
    
    if plot==True:
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