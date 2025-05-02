#!/usr/bin/env python
"""
Script to run in a batch job
"""

import os
import shutil
import gc
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from tqdm import tqdm

import torch
import gpytorch

# Import modules from your package
from GPyTorch.GPRs.mean import latlonmean
from GPyTorch.GPRs.kernel import SphericalMaternKernel
from GPyTorch.GPRs.utils import splitting_data, extreme_points_rel_err, extreme_points_rmse
from GPyTorch.GPRs.process_data import create_latlon_data
from GPyTorch.GPRs.plot import *

os.environ["CUDA_VISIBLE_DEVICES"]  = "4"


def plot_modelparams(params, wrmse, spatial, time, show=False):
    """Plots showing evolution of model params"""

    df = pd.DataFrame(params)
    epochs_range = np.arange(len(df))

    # Create a figure with 3 rows and 2 columns of subplots
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(14, 12))

    # --- Row 1 ---
    # Left: Loss
    axs[0, 0].plot(epochs_range, df["loss"], label=f'Weighted RMSE = {wrmse:.2f}', color='tab:blue')
    axs[0, 0].set_title("Training Loss Over Epochs")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Negative Marginal Log Likelihood", color='tab:blue')
    axs[0, 0].tick_params(axis='y', labelcolor='tab:blue')
    axs[0, 0].legend()
    axs[0, 0].text(
        0.6215, 0.65, 
        f"Train on 2017_12\nTest on 1-7/1/2018\nSubsampling: Rand",
        transform=axs[0, 0].transAxes,  
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round')
    )
    ax_rmse = axs[0, 0].twinx()
    ax_rmse.plot(epochs_range, df["rmse"], color='tab:orange', label="RMSE")
    ax_rmse.set_ylabel("RMSE", color='tab:orange')
    ax_rmse.tick_params(axis='y', labelcolor='tab:orange')

    # Right: Mean and Noise
    axs[0, 1].plot(epochs_range, df["mean"], color='tab:blue', label="Mean")
    axs[0, 1].set_title("GP Mean and Noise")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Mean", color='tab:blue')
    axs[0, 1].tick_params(axis='y', labelcolor='tab:blue')
    ax_noise = axs[0, 1].twinx()
    ax_noise.plot(epochs_range, df["noise"], color='tab:orange', label="Noise")
    ax_noise.set_ylabel("Noise", color='tab:orange')
    ax_noise.tick_params(axis='y', labelcolor='tab:orange')

    # --- Row 2 ---
    # Left: Spatial lengthscales
    axs[1, 0].plot(epochs_range, df["spatial_1_lengthscale"], label=f'Spatial 1 Length Scale ({spatial[0]})', color='tab:blue')
    axs[1, 0].plot(epochs_range, df["spatial_2_lengthscale"], label=f'Spatial 2 Length Scale ({spatial[1]})', color='tab:orange')
    axs[1, 0].plot(epochs_range, df["spatial_3_lengthscale"], label=f'Spatial 3 Length Scale ({spatial[2]})', color='tab:red')
    axs[1, 0].set_title("Spatial Lengthscales")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("Lengthscale")
    axs[1, 0].legend()

    # Right: Spatial outputscale
    # axs[1, 1].plot(epochs_range, df["spatial_outputscale"], label='Output scale from ScaleKernel', color='tab:red')
    axs[1, 1].plot(epochs_range, df["outputscale"], label='Output scale from ScaleKernel', color='tab:red')
    axs[1, 1].set_title("Spatial Outputscale")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("Outputscale")

    # --- Row 3 ---
    # Left: Temporal lengthscales
    axs[2, 0].plot(epochs_range, df["time_period_1"], label=f'Period Length ({time[0]/24:.2f} days)', color='tab:blue')
    #axs[2, 0].plot(epochs_range, df["time_period_2"], label=f'Period Length ({time[1]/24:.2f} days)', color='tab:red')
    #axs[2, 0].plot(epochs_range, df["time_period_3"], label=f'Period Length ({time[2]/24:.2f} days)', color='tab:orange')
    axs[2, 0].set_title("Temporal Period Lengths")
    axs[2, 0].set_xlabel("Epoch")
    axs[2, 0].set_ylabel("Lengthscale")
    axs[2, 0].legend()

    # Right: Temporal periods
    axs[2, 1].plot(epochs_range, df["time_lengthscale_1"], label=f'Period Length ({time[0]/24:.2f} days)', color='tab:blue')
    #axs[2, 1].plot(epochs_range, df["time_lengthscale_2"], label=f'Period Length ({time[1]/24:.2f} days)', color='tab:red')
    #axs[2, 1].plot(epochs_range, df["time_lengthscale_3"], label=f'Period Length ({time[2]/24:.2f} days)', color='tab:orange')
    axs[2, 1].plot(epochs_range, df["time_matern_lengthscale"], label='Matern Length Scale', color='tab:green')
    axs[2, 1].set_title("Temporal Kernel Lengthscale")
    axs[2, 1].set_xlabel("Epoch")
    axs[2, 1].set_ylabel("Period")
    axs[2, 1].legend()

    plt.tight_layout()
    plt.savefig("tmp/output.png")
    if show:
        plt.show()

    return 

class ExactGP(gpytorch.models.ExactGP):
    """Exact GP Model from GPyTorch stores the training data.

    Parameters
    ----------
    train_x : torch.Tensor
        The training input data of shape (N, d), where N is the number of observations
        and d is the dimensionality of the input (d=2 for (lat,lon), d=3 for (x,y,z)). 
        If we have a grid of (lat, lon) points with ntime timesteps, we expect 
        n=nlat*nlon*ntime. 
    train_y : torch.Tensor
        The training target values.
    likelihood : gpytorch.likelihoods.Likelihood
        The likelihood function to be used in the model (e.g., GaussianLikelihood).
    kernel : gpytorch.kernels.Kernel
        The covariance function (kernel).

    TODO: Implement mean_module parametrised by lat, lon.
    """
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module =  latlonmean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Set default tensor type and device
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train_gp(kernel, lead_time_h, data_train, data_test, space_subsample=1, time_subsample=1, frac=0.1, epochs=2500, save_model=False, device=device):
    """Train a Gaussian Process model using GPyTorch. Can add subsampling for efficiency."""
    X_train, Y_train = create_latlon_data(data_train, data_test, lead_time_h, space_subsample=space_subsample, time_subsample=time_subsample)
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    
    # Random sampling
    # N = X_train.shape[0] * X_train.shape[1] * X_train.shape[2] 
    # n_frac = int(frac * N) # Fraction of points to be chosen
    n_frac = 15000
    X_train = X_train.view(-1, 3)
    Y_train = Y_train.view(-1) 
    indices = torch.randperm(X_train.size(0))[:n_frac]
    X_train = X_train[indices]
    Y_train = Y_train[indices]

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = ExactGP(X_train, Y_train, likelihood, kernel).to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.965)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    print("Starting Exact GP regression training:")
    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.time()

    hyperparams_track = []
    for epoch in tqdm(range(epochs), desc="Training Progress", unit="epoch"):
        optimizer.zero_grad()
        output = model(X_train)
        loss_val = -mll(output, Y_train)
        loss_val.backward()
        optimizer.step()
        scheduler.step()

        # Record model parameters
        spatial_kernel = model.covar_module.base_kernel.kernels[0]
        temporal_kernel = model.covar_module.base_kernel.kernels[1]
        hyperparams_track.append({
            "loss": loss_val.item(),
            "rmse": torch.sqrt(torch.mean((output.mean - Y_train)**2)).item(),
            "mean": model.mean_module.constant.item(),
            "noise": likelihood.noise.item(),
            "outputscale": model.covar_module.outputscale.item(),
            "spatial_1_lengthscale": spatial_kernel.kernels[0].lengthscale.item(),
            "spatial_2_lengthscale": spatial_kernel.kernels[1].lengthscale.item(),
            "spatial_3_lengthscale": spatial_kernel.kernels[2].lengthscale.item(),
            "time_matern_lengthscale": temporal_kernel.kernels[0].lengthscale.item(),
            "time_lengthscale_1": temporal_kernel.kernels[1].lengthscale.item(),
            "time_period_1": temporal_kernel.kernels[1].period_length.item(),
            #"time_lengthscale_2": temporal_kernel.kernels[2].lengthscale.item(),
            #"time_period_2": temporal_kernel.kernels[2].period_length.item(),
            #"time_lengthscale_3": temporal_kernel.kernels[3].lengthscale.item(),
            #"time_period_3": temporal_kernel.kernels[3].period_length.item(),
        })
    current_allocated = torch.cuda.memory_allocated(device)
    max_allocated = torch.cuda.max_memory_allocated(device)
    print(f"Current GPU memory allocated: {current_allocated / (1024**2):.2f} MB")
    print(f"Peak GPU memory allocated: {max_allocated / (1024**2):.2f} MB")

    end_time = time.time()
    print("Training completed")
    print(f"Training time: {end_time - start_time:.2f}s")

    if save_model:
        os.makedirs("saved_models", exist_ok=True)
        torch.save(model.state_dict(), "saved_models/gp_model.pth")
        torch.save(likelihood.state_dict(), "saved_models/gp_likelihood.pth")
        print("Saved model")

    return model, likelihood, hyperparams_track, X_train, Y_train


def predictions(model, likelihood, data_train, data_test, data_std, data_mean, lead_time, space_subsample, time_subsample):
    """Produce predictions after training."""
    X_test, Y_test, lat, lon = create_latlon_data(data_train, data_test, lead_time, space_subsample=space_subsample, time_subsample=time_subsample, train=False)
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)
    X_test = X_test.view(-1, 3)
    nlat = len(lat)
    nlon = len(lon)

    model.eval()
    likelihood.eval()
    with torch.no_grad():
        pred_dist = model(X_test)
        Y_pred = pred_dist.mean
        Y_std = pred_dist.stddev
        mse = torch.mean((Y_pred - Y_test)**2)
        rmse = torch.sqrt(mse)
        print("MSE: ", mse.item())
        print("RMSE: ", rmse.item())
        Y_pred_actual = (Y_pred.detach().cpu().numpy() * data_std.values + data_mean.values).reshape(-1, nlat, nlon)
        Y_test_actual = (Y_test.detach().cpu().numpy() * data_std.values + data_mean.values).reshape(-1, nlat, nlon)
        err = Y_pred_actual - Y_test_actual
        weights_lat = np.cos(np.deg2rad(lat))
        weights_lat /= weights_lat.mean()
        wrmse = np.sqrt((err**2 * weights_lat[None, :, None]).mean())
        print("Weighted RMSE: ", wrmse)

    Y_pred = Y_pred.detach().cpu().numpy()  
    Y_test = Y_test.detach().cpu().numpy()  
    Y_std = Y_std.detach().cpu().numpy()

    return Y_pred_actual, Y_test_actual, Y_pred, Y_test, Y_std, lat, lon, wrmse.item()


def main():
    # Open dataset and select time range
    z500 = xr.open_mfdataset('data/5.625deg/geopotential_500/*.nc', combine='by_coords').sel(time=slice('2000', '2018'))
    
    # Use splitting_data to get boundaries (or manually set them)
    lat_bound, lon_bound = splitting_data(z500, 'z')
    print("lat. boundaries:", lat_bound)
    print("lon. boundaries:", lon_bound)
    
    # Manually set boundaries for this run (if desired)
    lat_bound = [0, 13, 18, 32]
    lon_bound = [0, 5, 8, 22, 43, 50, 60, 64]
    
    nlat = z500.lat.size
    nlon = z500.lon.size
    num_lat_bounds = len(lat_bound) - 1
    num_lon_bounds = len(lon_bound) - 1
    
    # Initialize full error grids for later reconstruction (full grid of shape nlat x nlon)
    full_relative_error_grid = np.full((nlat, nlon), np.nan)
    full_rmse_error_grid = np.full((nlat, nlon), np.nan)
    wrmse_full = []
    # Loop over subgrids
    for i in range(num_lat_bounds):
        for j in range(num_lon_bounds):
            lat_slice = slice(lat_bound[i], lat_bound[i+1])
            lon_slice = slice(lon_bound[j], lon_bound[j+1])
            
            z500_train = z500.sel(time=slice('2017.12', '2017.12')).isel(lat=lat_slice, lon=lon_slice)['z']
            z500_test = z500.sel(time=slice('2018', '2018.1.7')).isel(lat=lat_slice, lon=lon_slice)['z']
            
            data_mean = z500_train.mean().load()
            data_std = z500_train.std().load()
            
            data_train = (z500_train - data_mean) / data_std
            data_test = (z500_test - data_mean) / data_std

            # Spatial kernel components
            spatial_lengths = [0.08, 0.25, 0.5]
            matern_spatial_1 = SphericalMaternKernel(nu=1.5, active_dims=[0, 1]).to(device)
            matern_spatial_1.lengthscale = torch.tensor(spatial_lengths[0]).to(device)
            matern_spatial_2 = SphericalMaternKernel(nu=1.5, active_dims=[0, 1]).to(device)
            matern_spatial_2.lengthscale = torch.tensor(spatial_lengths[1]).to(device)
            matern_spatial_3 = SphericalMaternKernel(nu=1.5, active_dims=[0, 1]).to(device)
            matern_spatial_3.lengthscale = torch.tensor(spatial_lengths[2]).to(device)

            spatial_kernel = matern_spatial_1 + matern_spatial_2 + matern_spatial_3

            # Temporal kernel components
            #period_lengths = [24, 24*30*3, 24*30*12]
            period_lengths = [24]
            time_kernel1 = gpytorch.kernels.PeriodicKernel(active_dims=[2]).to(device)  # Daily
            time_kernel1.period_length = torch.tensor(period_lengths[0]).to(device)
            matern_time_1 = gpytorch.kernels.MaternKernel(nu=1.5, active_dims=[2]).to(device)

            # time_kernel = matern_time_1 + time_kernel1 + time_kernel2 + time_kernel3
            time_kernel = matern_time_1 + time_kernel1 
            # Assemble the full kernel, wrapping scaling to the whole kernel
            kernel = gpytorch.kernels.ScaleKernel(spatial_kernel * time_kernel).to(device)
            
            lead_time = 0
            space_subsample = 1
            time_subsample = 6
            frac = 0.8
            
            # Train GP model on this subgrid
            model, likelihood, hyperparams_track, X_train, Y_train = train_gp(kernel, lead_time, data_train, data_test, space_subsample, time_subsample, frac=frac)
            
            # Get predictions
            Y_pred_actual, Y_test_actual, Y_pred, Y_test, Y_std, lat_sub, lon_sub, wrmse = predictions(model, likelihood, data_train, data_test, data_std, data_mean, lead_time, space_subsample, time_subsample)
            wrmse_full.append(wrmse)

            # Fit again on training data to get posterior for plotting
            model.eval()
            likelihood.eval()
            with torch.no_grad():
                pred_train = model(X_train)
                train_mean = pred_train.mean
                train_std = pred_train.stddev
            
            sorted_idx = torch.argsort(X_train[:, 2])
            train_mean = train_mean[sorted_idx].detach().cpu().numpy()
            train_std = train_std[sorted_idx].detach().cpu().numpy()
            Y_train_arr = Y_train[sorted_idx].detach().cpu().numpy()
            
            # Generate plots (these functions save plots in the "tmp" folder)
            rand_coords = X_train.detach().cpu().numpy()
            plot_coords_dist(rand_coords)
            plot_modelparams(hyperparams_track, wrmse, spatial_lengths, period_lengths)
            relative_error = (Y_pred_actual - Y_test_actual) / Y_test_actual * 100
            relative_error_mean = plot_relative_err(lat_sub, lon_sub, relative_error)
            rmse_error = np.sqrt((Y_pred_actual - Y_test_actual)**2)
            rmse_error_mean = plot_rmse(lat_sub, lon_sub, rmse_error)
            rel_err_idx, rel_err_val = extreme_points_rel_err(relative_error_mean)
            rmse_idx, rmse_val = extreme_points_rmse(rmse_error_mean)
            plot_relative_err_posterior(rel_err_idx, rel_err_val, time_subsample, relative_error_mean, Y_pred, Y_test, Y_std, train_mean, train_std, Y_train_arr)
            plot_rmse_posterior(rmse_idx, rmse_val, time_subsample, rmse_error_mean, Y_pred, Y_test, Y_std, train_mean, train_std, Y_train_arr)
            
            # Move files from tmp to target folder for this subgrid
            target_folder = (
                f"plots/5.625deg/z500/Train (2017_12), test (2018_1_7)/ExactGP_full(2500epochs,latlon)/"
                f"lat_{lat_bound[i]}-{lat_bound[i+1]}_lon_{lon_bound[j]}-{lon_bound[j+1]}"
            )
            os.makedirs(target_folder, exist_ok=True)
            tmp_folder = "tmp"
            for filename in os.listdir(tmp_folder):
                src_path = os.path.join(tmp_folder, filename)
                dst_path = os.path.join(target_folder, filename)
                shutil.move(src_path, dst_path)
            
            # Fill in the full error grids with time-averaged errors (these error arrays should match the subgrid shape)
            full_relative_error_grid[lat_bound[i]:lat_bound[i+1], lon_bound[j]:lon_bound[j+1]] = relative_error_mean
            full_rmse_error_grid[lat_bound[i]:lat_bound[i+1], lon_bound[j]:lon_bound[j+1]] = rmse_error_mean
            
            # Free GPU and CPU memory for this iteration
            del model, likelihood, X_train, Y_train, pred_train, train_mean, train_std, Y_pred, Y_test, Y_std
            torch.cuda.empty_cache()
            gc.collect()
    
    # After processing all subgrids, plot the full grid errors.
    lat_full = z500.lat.values
    lon_full = z500.lon.values
    plot_relative_err(lat_full, lon_full, full_relative_error_grid)
    plot_rmse(lat_full, lon_full, full_rmse_error_grid)
    
    target_folder_final = "plots/5.625deg/z500/Train (2017_12), test (2018_1_7)/ExactGP_full(2500epochs,latlon)/final"
    os.makedirs(target_folder_final, exist_ok=True)
    tmp_folder = "tmp"
    for filename in os.listdir(tmp_folder):
        src_path = os.path.join(tmp_folder, filename)
        dst_path = os.path.join(target_folder_final, filename)
        shutil.move(src_path, dst_path)

    print(f"Avg WRMSE: {sum(wrmse_full)/len(wrmse_full)}")
    w_lat = np.cos(np.deg2rad(lat_full))
    w_lat /= w_lat.mean() 
    mse_weighted_mean = np.mean(full_rmse_error_grid**2 * w_lat[:, None])
    wrmse_global = np.sqrt(mse_weighted_mean)
    print("Global wrmse:", wrmse_global)

if __name__ == "__main__":
    main()
