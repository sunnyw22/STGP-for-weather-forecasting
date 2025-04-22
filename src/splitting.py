from typing import List
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def latitude_longitude_splitter(
    locations: np.ndarray,
    latitude_bins: List[float] = None,
    longitude_bins: List[float] = None,
) -> List[np.ndarray[bool]]:
    """
    Split the locations into regions based on the provided latitude and longitude bins.

    Arguments:
        locations: A numpy array of shape (n_locations, 2) where each row is a
        (latitude, longitude) pair.
    Returns:
        region_masks: A list of boolean masks each of shape (n_locations,) indicating
        which locations belong to each region.

    If latitude_bins and latitude_bins are not provided, the respective dimension is not split.
    """
    latitudes = locations[:, 0]
    longitudes = locations[:, 1]

    if latitude_bins is None:
        latitude_bins = [latitudes.min(), latitudes.max() + 1]
    if longitude_bins is None:
        longitude_bins = [longitudes.min(), longitudes.max() + 1]

    # Ensure bins are sorted
    latitude_bins = np.sort(latitude_bins)
    longitude_bins = np.sort(longitude_bins)

    # Create masks for each region based on the bins
    region_masks = []
    for i in range(len(latitude_bins) - 1):
        for j in range(len(longitude_bins) - 1):
            latitude_mask = np.logical_and(
                latitudes >= latitude_bins[i],
                latitudes < latitude_bins[i + 1],
            )
            longitude_mask = np.logical_and(
                longitudes >= longitude_bins[j],
                longitudes < longitude_bins[j + 1],
            )
            mask = np.logical_and(latitude_mask, longitude_mask)
            region_masks.append(mask)

    return region_masks


def splitting_data(data_xarray):
    """
    Splitting up the weather data into smaller grids based on peaks and troughs
    in the average latitude and longitude values.

    Args:
        data: xarray DataArray containing the weather data.

    Returns:
        lat_boundaries: List of indices for latitude boundaries.
        lon_boundaries: List of indices for longitude boundaries.
    """

    avg_lat = data_xarray.mean(dim=["time", "lon"])
    avg_lon = data_xarray.mean(dim=["time", "lat"])

    # Find peaks and troughs
    peaks_lat, _ = find_peaks(avg_lat.values, distance=5)
    troughs_lat, _ = find_peaks(-avg_lat.values, distance=5)
    peaks_lon, _ = find_peaks(avg_lon.values, distance=5)
    troughs_lon, _ = find_peaks(-avg_lon.values, distance=5)
    print("Peaks (lat indices):", peaks_lat, "Troughs (lat indices):", troughs_lat)
    print("Peaks (lon indices):", peaks_lon, "Troughs (lon indices):", troughs_lon)

    lat_boundaries = np.unique(np.r_[0, peaks_lat, troughs_lat, data_xarray.lat.size])
    lon_boundaries = np.unique(np.r_[0, peaks_lon, troughs_lon, data_xarray.lon.size])

    return lat_boundaries, lon_boundaries
