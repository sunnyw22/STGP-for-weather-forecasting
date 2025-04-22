"""
Utilities for holding and processing weather data.
"""

import os
from typing import NamedTuple
import numpy as np
import xarray as xr


class DataContainer(NamedTuple):
    """Container for the raw weather data"""

    x: np.ndarray
    t_train: np.ndarray
    t_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def load_xarray(root: str, var: str) -> xr.DataArray:
    """
    Args:
        root: Root directory, should contain the .nc weather data files.
        var: Weather variable to load. "z" for geopotential, "t2m" for 2m temperature.
    """
    data_path = os.path.join(root, "*.nc")
    data_xarray = xr.open_mfdataset(data_path, combine="by_coords")[var]
    return data_xarray


def process_xarray(
    data_xarray: xr.DataArray,
    time_slice: slice = None,
    latitude_slice: slice = None,
    longitude_slice: slice = None,
):
    """
    Process the xarray DataArray to extract time, locations, and weather variable values.
    Syntax: slice(start, stop, step), where if any of the arguments are None,
    it will take all values.

    Args:
        data: xarray DataArray containing the weather data.
        time_slice: Time slice to subset data. Runs from 1979-01-01 to 2018-12-31.
        lat_slice: Latitude slice to subset the data. Runs from -90 to 90.
        lon_slice: Longitude slice to subset the data. Runs from 0 to 360.
    Returns:
        t: (n_time,) array of time points.
        x: (n_space, 2) array of locations, where each row is a (latitude, longitude) pair.
        y: (n_time, n_space,) array of weather variable values.
    """
    if time_slice is None:
        time_slice = slice("1979-01-01", "2018-12-31")
    if latitude_slice is None:
        latitude_slice = slice(-90, 90)
    if longitude_slice is None:
        longitude_slice = slice(0, 360)
    xarray = data_xarray.sel(time=time_slice, lat=latitude_slice, lon=longitude_slice)

    # Convert time to hours since the first time point
    t = (xarray.time - xarray.time[0]).values / np.timedelta64(1, "h")

    # Convert latitude and longitude to a (n_space, 2) array
    latitudes = xarray.lat.values
    longitudes = xarray.lon.values
    n_space = len(latitudes) * len(longitudes)
    lat_grid, lon_grid = np.meshgrid(latitudes, longitudes, indexing="ij")
    x = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=-1)

    # Convert the spatiotemporal weather variables to a (n_time, n_space) array
    y = xarray.values
    y = y.reshape(-1, n_space)

    assert len(x) == n_space, "Mismatch in number of locations"
    assert y.shape == (len(t), n_space), "Mismatch in data shape"

    return t, x, y
