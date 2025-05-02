import numpy as np


def latlon_to_cartesian(lat, lon):
    """Converting (lat., lon.) to cartesian coordinates on a unit sphere
    Note: WeatherBench data is defined on a constant altitude"""

    lat, lon = np.deg2rad(lat), np.deg2rad(lon)

    x = np.cos(lat)[:, None] * np.cos(lon)[None, :]
    y = np.cos(lat)[:, None] * np.sin(lon)[None, :]
    z = np.sin(lat)[:, None] * np.ones_like(lon)[None, :]

    return np.stack([x, y, z], axis=-1)  # Out: (nlat, nlon, 3)


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
