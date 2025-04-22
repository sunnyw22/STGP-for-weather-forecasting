"""Functions that prepares the WeatherBench data for training and testing"""

import torch
import numpy as np
from .utils import latlon_to_cartesian

def create_data(data_train, data_test, lead_time, space_subsample=1, time_subsample=1, train=True):
    """Preparing input and output data. X should be the coordinate input into the kernel, 
    while Y should be the forecast target (shifted lead time)."""

    if lead_time == 0:
        X = data_train.isel(time=slice(0, None, time_subsample),
                        lat=slice(0, None, space_subsample),
                        lon=slice(0, None, space_subsample))
    else:
        X = data_train.isel(time=slice(0, -lead_time, time_subsample),
                            lat=slice(0, None, space_subsample),
                            lon=slice(0, None, space_subsample))
        Y = data_train.isel(time=slice(lead_time, None, time_subsample),
                    lat=slice(0, None, space_subsample),
                    lon=slice(0, None, space_subsample)) 

    X_time = (X.time - data_train.time[0]).values / np.timedelta64(1, 'h')
    X_time = torch.tensor(X_time)
    t_steps = X_time.shape[0]

    latitudes = X.lat.values 
    longitudes = X.lon.values 
    cartesian_coords = torch.tensor(latlon_to_cartesian(latitudes, longitudes))
    nlat, nlon, _ = cartesian_coords.shape

    if train:
        print("Processing Training Dataset")
        # Combine spatial and temporal data
        training_coords = torch.empty((t_steps, nlat, nlon, 4))
        # Fill in the spatial part first
        training_coords[..., :3] = cartesian_coords.unsqueeze(0).expand(t_steps, -1, -1, -1)
        # The final element is time
        training_coords[..., 3] = X_time.view(t_steps, 1, 1).expand(t_steps, nlat, nlon)
        X_size = training_coords.element_size() * training_coords.nelement() / (1024**2)
        print(f"Combined coords info --> Shape: {training_coords.shape}, Mem: {X_size:.2f} MB")

        if lead_time == 0:
            Y = torch.tensor(X.values).view(-1)
        else:
            # Flatten to shape (N,)
            Y = torch.tensor(Y.values).view(-1)
            Y_size = Y.element_size() * Y.nelement() / (1024**2)
            print(f"Target info --> Shape: {Y.shape}, Mem: {Y_size:.2f} MB")

        return training_coords, Y
    
    else:
        # Testing Data
        if lead_time == 0:
            X_test = data_test.isel(time=slice(0, None, time_subsample),
                            lat=slice(0, None, space_subsample),
                            lon=slice(0, None, space_subsample))
        else:
            X_test = data_test.isel(time=slice(0, -lead_time, time_subsample),
                                lat=slice(0, None, space_subsample),
                                lon=slice(0, None, space_subsample))
            Y_test = data_test.isel(time=slice(lead_time, None, time_subsample),
                        lat=slice(0, None, space_subsample),
                        lon=slice(0, None, space_subsample))
        X_test_time = (X_test.time - data_train.time[0]).values / np.timedelta64(1, 'h')
        X_test_time = torch.tensor(X_test_time)
        test_t_steps = X_test_time.shape[0]

        print("Processing Test Dataset")
        # Combine spatial and temporal data
        testing_coords = torch.empty((test_t_steps, nlat, nlon, 4))
        # Fill in the spatial part first
        testing_coords[..., :3] = cartesian_coords.unsqueeze(0).expand(test_t_steps, -1, -1, -1)
        # The final element is time
        testing_coords[..., 3] = X_test_time.view(test_t_steps, 1, 1).expand(test_t_steps, nlat, nlon)
        X_size = testing_coords.element_size() * testing_coords.nelement() / (1024**2)
        print(f"Combined coords info --> Shape: {testing_coords.shape}, Mem: {X_size:.2f} MB")

        if lead_time == 0:
            Y_test = torch.tensor(X_test.values).view(-1)
        else:
            # Flatten to shape (N,)
            Y_test = torch.tensor(Y_test.values).view(-1)
            Y_size = Y_test.element_size() * Y_test.nelement() / (1024**2)
            print(f"Target info --> Shape: {Y_test.shape}, Mem: {Y_size:.2f} MB")

        return testing_coords, Y_test, latitudes, longitudes

def create_latlon_data(data_train, data_test, lead_time, space_subsample=1, time_subsample=1, train=True):
    """Preparing input and output data. X should be the coordinate input into the kernel, 
    while Y should be the forecast target (shifted lead time)."""

    if lead_time == 0:
        X = data_train.isel(time=slice(0, None, time_subsample),
                        lat=slice(0, None, space_subsample),
                        lon=slice(0, None, space_subsample))
    else:
        X = data_train.isel(time=slice(0, -lead_time, time_subsample),
                            lat=slice(0, None, space_subsample),
                            lon=slice(0, None, space_subsample))
        Y = data_train.isel(time=slice(lead_time, None, time_subsample),
                    lat=slice(0, None, space_subsample),
                    lon=slice(0, None, space_subsample)) 

    X_time = (X.time - data_train.time[0]).values / np.timedelta64(1, 'h')
    X_time = torch.tensor(X_time)
    ntime, nlat, nlon = X.shape
    lat = X.lat.values 
    lon = X.lon.values 

    lat_grid, lon_grid = torch.meshgrid(torch.tensor(lat), torch.tensor(lon), indexing='ij')  
    latlon_grid = torch.stack([lat_grid, lon_grid], dim=-1)

    if train:
        print("Processing Training Dataset")
        # Combine spatial and temporal data
        training_coords = torch.empty((ntime, nlat, nlon, 3))
        # Fill in the spatial part first
        training_coords[..., :2] = latlon_grid.unsqueeze(0).expand(ntime, -1, -1, -1)
        # The final element is time
        training_coords[..., 2] = X_time.view(ntime, 1, 1).expand(ntime, nlat, nlon)
        X_size = training_coords.element_size() * training_coords.nelement() / (1024**2)
        print(f"Combined coords info --> Shape: {training_coords.shape}, Mem: {X_size:.2f} MB")

        if lead_time == 0:
            Y = torch.tensor(X.values).view(-1)
        else:
            # Flatten to shape (N,)
            Y = torch.tensor(Y.values).view(-1)
            Y_size = Y.element_size() * Y.nelement() / (1024**2)
            print(f"Target info --> Shape: {Y.shape}, Mem: {Y_size:.2f} MB")

        return training_coords, Y
    
    else:
        # Testing Data
        if lead_time == 0:
            X_test = data_test.isel(time=slice(0, None, time_subsample),
                            lat=slice(0, None, space_subsample),
                            lon=slice(0, None, space_subsample))
        else:
            X_test = data_test.isel(time=slice(0, -lead_time, time_subsample),
                                lat=slice(0, None, space_subsample),
                                lon=slice(0, None, space_subsample))
            Y_test = data_test.isel(time=slice(lead_time, None, time_subsample),
                        lat=slice(0, None, space_subsample),
                        lon=slice(0, None, space_subsample))
        X_test_time = (X_test.time - data_train.time[0]).values / np.timedelta64(1, 'h')
        X_test_time = torch.tensor(X_test_time)
        test_t_steps = X_test_time.shape[0]

        print("Processing Test Dataset")
        # Combine spatial and temporal data
        testing_coords = torch.empty((test_t_steps, nlat, nlon, 3))
        # Fill in the spatial part first
        testing_coords[..., :2] = latlon_grid.unsqueeze(0).expand(test_t_steps, -1, -1, -1)
        # The final element is time
        testing_coords[..., 2] = X_test_time.view(test_t_steps, 1, 1).expand(test_t_steps, nlat, nlon)
        X_size = testing_coords.element_size() * testing_coords.nelement() / (1024**2)
        print(f"Combined coords info --> Shape: {testing_coords.shape}, Mem: {X_size:.2f} MB")

        if lead_time == 0:
            Y_test = torch.tensor(X_test.values).view(-1)
        else:
            # Flatten to shape (N,)
            Y_test = torch.tensor(Y_test.values).view(-1)
            Y_size = Y_test.element_size() * Y_test.nelement() / (1024**2)
            print(f"Target info --> Shape: {Y_test.shape}, Mem: {Y_size:.2f} MB")

        return testing_coords, Y_test, lat, lon
