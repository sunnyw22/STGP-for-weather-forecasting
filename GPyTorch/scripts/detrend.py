import xarray as xr

from GPyTorch.GPRs.utils import get_periods, seasonal_decompose_grid

# Data
z500 = xr.open_mfdataset('data/5.625deg/geopotential_500/*.nc', combine='by_coords')
data = z500.sel(time=slice('2000', '2017'))

# Detrend
print("Obtaining dominant periods...")
dom_period = get_periods(data)
print("Decomposing...")
detrended_dataset = seasonal_decompose_grid(data, dom_period)
print("Done!")

detrended_dataset.to_netcdf('data/5.625deg/detrended/z500/detrended_dataset.nc')
print("Saved!")