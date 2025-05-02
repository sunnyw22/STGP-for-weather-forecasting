"""Plotting Functions"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import cartesian_to_latlon

def plot_coords_dist(X_train, show=False):
    """Plots showing the distribution of coordinates from random sampling
       X_train should be numpy array."""
    
    # rand_coords = X_train[:,:3]
    # rand_time = X_train[:,3]

    rand_time = X_train[:,2]

    # latlon = cartesian_to_latlon(rand_coords)
    # lat = latlon[:, 0]
    # lon = latlon[:, 1]
    lat = X_train[:, 0]
    lon = X_train[:, 1]

    nlat = len(np.unique(np.round(lat, 6)))
    nlon = len(np.unique(np.round(lon, 6)))

    fig, axs = plt.subplots(ncols=2, figsize=(16, 6))
    # Left plot: Histogram of time coordinates
    axs[0].hist(rand_time, bins=500, color='tab:blue')
    axs[0].set_xlabel('Time (hours)')
    axs[0].set_ylabel('Counts')
    axs[0].set_title('Distribution of Time Coordinates')
    # Right plot: 2D histogram for spatial coordinates
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_under('white')
    h = axs[1].hist2d(lon, lat, bins=[nlon, nlat], cmap=cmap, vmin=1)
    axs[1].set_xlabel('Longitude')
    axs[1].set_ylabel('Latitude')
    axs[1].set_title('2D Histogram of (lat, lon) Points')
    fig.colorbar(h[3], ax=axs[1], label='Count')

    plt.tight_layout()
    plt.savefig("tmp/rand coords dist.png")
    if show:
        plt.show()

    return 


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
        f"Train on 2017-12\nTest on 1-7/1/2018\nSubsampling: Rand",
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
    axs[2, 0].plot(epochs_range, df["time_period_2"], label=f'Period Length ({time[1]/24:.2f} days)', color='tab:red')
    axs[2, 0].plot(epochs_range, df["time_period_3"], label=f'Period Length ({time[2]/24:.2f} days)', color='tab:orange')
    axs[2, 0].set_title("Temporal Period Lengths")
    axs[2, 0].set_xlabel("Epoch")
    axs[2, 0].set_ylabel("Lengthscale")
    axs[2, 0].legend()

    # Right: Temporal periods
    axs[2, 1].plot(epochs_range, df["time_lengthscale_1"], label=f'Period Length ({time[0]/24:.2f} days)', color='tab:blue')
    axs[2, 1].plot(epochs_range, df["time_lengthscale_2"], label=f'Period Length ({time[1]/24:.2f} days)', color='tab:red')
    axs[2, 1].plot(epochs_range, df["time_lengthscale_3"], label=f'Period Length ({time[2]/24:.2f} days)', color='tab:orange')
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


def plot_relative_err(lat, lon, relative_error, show=False):
    """Plots showing relative error across grid."""
    full_grid = False
    if relative_error.ndim > 2:
        # Averaging across the time dimension
        relative_error_mean = relative_error.mean(axis=0)
    else:
        # Assuming we have already averaged across time
        relative_error_mean = relative_error
        full_grid = True

    low_q = np.percentile(relative_error_mean, 5)
    high_q = np.percentile(relative_error_mean, 95)
    masked_error = np.ma.masked_outside(relative_error_mean, low_q, high_q)

    # Create a figure with 2 columns
    fig, axs = plt.subplots(ncols=2, figsize=(16, 6))

    # Left plot: Full relative error mean grid
    im0 = axs[0].imshow(relative_error_mean, cmap='RdBu', aspect='auto',
                        extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                        origin='lower')
    axs[0].set_title('Mean Relative Error (%) Across the Grid')
    axs[0].set_xlabel('Longitude index')
    axs[0].set_ylabel('Latitude index')
    cbar0 = fig.colorbar(im0, ax=axs[0], label='Relative Error (%)')

    # Right plot: Clipped (masked) relative error grid
    im1 = axs[1].imshow(masked_error, cmap='RdBu', aspect='auto', 
                        extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                        origin='lower')
    axs[1].set_title('Relative Error Clipped to 5th and 95th percentile')
    axs[1].set_xlabel('Longitude index')
    axs[1].set_ylabel('Latitude index')
    cbar1 = fig.colorbar(im1, ax=axs[1], label='Relative Error (%)')

    plt.tight_layout()
    if full_grid:
        plt.savefig("tmp/relative err full grid.png")
        plt.show()
        return
    else:
        plt.savefig("tmp/relative err grid.png")
        if show:
            plt.show()
        return relative_error_mean
    

def plot_rmse(lat, lon, rmse_error, show=False):
    """Plots showing rmse across grid"""
    full_grid = False
    if rmse_error.ndim > 2:
        # Averaging across the time dimension
        rmse_error_mean = rmse_error.mean(axis=0)
    else:
        # Assuming we have already averaged across time
        rmse_error_mean = rmse_error
        full_grid = True

    low_q_rmse = np.percentile(rmse_error_mean, 0)
    high_q_rmse = np.percentile(rmse_error_mean, 95)
    masked_error_rmse = np.ma.masked_outside(rmse_error_mean, low_q_rmse, high_q_rmse)

    # Create a 2-column plot
    fig, axs = plt.subplots(ncols=2, figsize=(16, 6))

    # Left subplot: Full RMSE grid
    im0 = axs[0].imshow(rmse_error_mean, cmap='RdBu', aspect='auto', 
                        extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                        origin='lower')
    axs[0].set_title('Root Mean Squared Error Across the Grid')
    axs[0].set_xlabel('Longitude')
    axs[0].set_ylabel('Latitude')
    cbar0 = fig.colorbar(im0, ax=axs[0], label='RMSE')

    # Right subplot: Masked RMSE grid
    im1 = axs[1].imshow(masked_error_rmse, cmap='RdBu', aspect='auto', 
                        extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                        origin='lower')
    axs[1].set_title('RMSE Masked to 95th Percentile')
    axs[1].set_xlabel('Longitude')
    axs[1].set_ylabel('Latitude')
    cbar1 = fig.colorbar(im1, ax=axs[1], label='RMSE')

    plt.tight_layout()
    if full_grid:
        plt.savefig("tmp/rmse full grid.png")
        plt.show()
        return
    else:
        plt.savefig("tmp/rmse grid.png")
        if show:
            plt.show()
        return rmse_error_mean 


def plot_relative_err_posterior(idx, val, time_subsample, relative_error_mean, Y_pred, Y_test, Y_std, train_mean, train_std, Y_train, show=False):
    """Posterior plots for observing how the model fits the data"""
    
    nlat = relative_error_mean.shape[0]
    nlon = relative_error_mean.shape[1]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=False, sharey=True)

    for i, p_idx in enumerate(idx):
        flat_idx = np.unravel_index(p_idx, relative_error_mean.shape)
        ax = axes[0, i]  

        pred_vals = Y_pred.reshape(-1)[p_idx::(nlat*nlon)]
        truth_vals = Y_test.reshape(-1)[p_idx::(nlat*nlon)]
        std_vals = Y_std.reshape(-1)[p_idx::(nlat*nlon)]

        ax.plot(pred_vals, marker='x', label='Predicted mean', color='tab:orange')
        ax.plot(truth_vals, marker='x', label='Truth', color='tab:blue')
        ax.fill_between(np.arange(len(pred_vals)),
                        pred_vals - 2 * std_vals,
                        pred_vals + 2 * std_vals,
                        alpha=0.3, color='tab:gray', label='Confidence interval')
        ax.set_xlabel(f"Time step (per {time_subsample} hrs)")
        ax.set_ylabel("Normalised z500 value")
        ax.set_title(f"GP Fit at ({int(flat_idx[0])},{int(flat_idx[1])}) on Test Set - Relative Error = {val[i]:.3f} %")

    for i, p_idx in enumerate(idx):
        flat_idx = np.unravel_index(p_idx, relative_error_mean.shape)
        ax = axes[1, i]  

        pred_vals = train_mean.reshape(-1)[p_idx::(nlat*nlon)]
        truth_vals = Y_train.reshape(-1)[p_idx::(nlat*nlon)]
        std_vals = train_std.reshape(-1)[p_idx::(nlat*nlon)]

        ax.plot(pred_vals, marker='x', label='Predicted mean', color='tab:orange')
        ax.plot(truth_vals, marker='x', label='Truth', color='tab:blue')
        ax.fill_between(np.arange(len(pred_vals)),
                        pred_vals - 2 * std_vals,
                        pred_vals + 2 * std_vals,
                        alpha=0.3, color='tab:gray', label='Confidence interval')
        ax.set_xlabel(f"Time step (per {time_subsample} hrs)")
        ax.set_ylabel("Normalised z500 value")
        ax.set_title(f"GP Fit at ({int(flat_idx[0])},{int(flat_idx[1])}) on Train Set")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    plt.savefig("tmp/posterior plots (relative err).png")
    if show:
        plt.show()

    return 


def plot_rmse_posterior(idx, val, time_subsample, rmse_error_mean, Y_pred, Y_test, Y_std, train_mean, train_std, Y_train, show=False):
    """Posterior plots for observing how the model fits the data"""

    nlat = rmse_error_mean.shape[0]
    nlon = rmse_error_mean.shape[1]

    fig, axes = plt.subplots(2, 2, figsize=(18, 8), sharex=False, sharey=True)

    for i, p_idx in enumerate(idx):
        flat_idx = np.unravel_index(p_idx, rmse_error_mean.shape)
        ax = axes[0, i]  

        pred_vals = Y_pred.reshape(-1)[p_idx::(nlat*nlon)]
        truth_vals = Y_test.reshape(-1)[p_idx::(nlat*nlon)]
        std_vals = Y_std.reshape(-1)[p_idx::(nlat*nlon)]

        ax.plot(pred_vals, marker='x', label='Predicted mean', color='tab:orange')
        ax.plot(truth_vals, marker='x', label='Truth', color='tab:blue')
        ax.fill_between(np.arange(len(pred_vals)),
                        pred_vals - 2 * std_vals,
                        pred_vals + 2 * std_vals,
                        alpha=0.3, color='tab:gray', label='Confidence interval')
        ax.set_xlabel(f"Time step (per {time_subsample} hrs)")
        ax.set_ylabel("Normalised z500 value")
        ax.set_title(f"GP Fit at ({int(flat_idx[0])},{int(flat_idx[1])}) on Test Set - RMSE on actual values = {val[i]:.3f}")

    for i, p_idx in enumerate(idx):
        flat_idx = np.unravel_index(p_idx, rmse_error_mean.shape)
        ax = axes[1, i]  

        pred_vals = train_mean.reshape(-1)[p_idx::(nlat*nlon)]
        truth_vals = Y_train.reshape(-1)[p_idx::(nlat*nlon)]
        std_vals = train_std.reshape(-1)[p_idx::(nlat*nlon)]

        ax.plot(pred_vals, marker='x', label='Predicted mean', color='tab:orange')
        ax.plot(truth_vals, marker='x', label='Truth', color='tab:blue')
        ax.fill_between(np.arange(len(pred_vals)),
                        pred_vals - 2 * std_vals,
                        pred_vals + 2 * std_vals,
                        alpha=0.3, color='tab:gray', label='Confidence interval')
        ax.set_xlabel(f"Time step (per {time_subsample} hrs)")
        ax.set_ylabel("Normalised z500 value")
        ax.set_title(f"GP Fit at ({int(flat_idx[0])},{int(flat_idx[1])}) on Train Set")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    plt.savefig("tmp/posterior plot (rmse).png")
    if show:
        plt.show()

    return 