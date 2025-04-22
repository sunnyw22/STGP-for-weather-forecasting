"""Custom Kernel on a sphere."""

import math
import torch
import gpytorch
from gpytorch.kernels import Kernel, MaternKernel
from gpytorch.functions import MaternCovariance
from gpytorch.settings import trace_mode

class SphericalMaternKernel(Kernel):
    r"""
    Basically same as gpytorch matern kernel but uses covar_dist that accounts for 
    distances on a sphere.

    This kernel computes distances on the sphere using the following approximation:

        ds^2 = d\theta^2 + \sin^2(\theta)d\phi^2
        d = sqrt((delta_theta)^2 + (sin(theta_mean) * delta_phi)^2)

    where delta_theta = theta2 - theta1, delta_phi = phi2 - phi1, and
    theta_mean is the average of theta1 and theta2. The kernel then uses these distances
    in a Matern-like covariance function.

    We assume input data shape = (ntime*nlat*nlon, 3) where 3 is (lat, lon, time).
    """
    has_lengthscale = True

    def __init__(self, nu=1.5, active_dims=(0, 1), **kwargs):
        if nu not in {0.5, 1.5, 2.5}:
            raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
        super(SphericalMaternKernel, self).__init__(active_dims=active_dims, **kwargs)
        self.nu = nu

    def covar_dist(self, x1, x2, diag=False, **params):
        """
        Computes the pairwise spherical distances between x1 and x2.
        Assumes x1 and x2 have already been sliced to only contain the active
        dimensions (latitude and longitude) and that the angles are in degrees.
        """

        lat1 = torch.deg2rad(x1[..., 0])
        lon1 = torch.deg2rad(x1[..., 1])
        lat2 = torch.deg2rad(x2[..., 0])
        lon2 = torch.deg2rad(x2[..., 1])
        
        if diag:
            return torch.zeros(lat1.size(-1), device=lat1.device, dtype=lat1.dtype)
        
        dlat = lat1.unsqueeze(1) - lat2.unsqueeze(0)  
        dlon = lon1.unsqueeze(1) - lon2.unsqueeze(0)  
        
        lat_mean = 0.5 * (lat1.unsqueeze(1) + lat2.unsqueeze(0))
        distance = torch.sqrt(dlat**2 + (torch.sin(lat_mean) * dlon)**2)
        return distance

    def forward(self, x1, x2, diag=False, **params):

        distance = self.covar_dist(x1, x2, diag=diag, **params) / self.lengthscale
        exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)

        if self.nu == 0.5:
            constant_component = 1
        elif self.nu == 1.5:
            constant_component = (math.sqrt(3) * distance).add(1)
        elif self.nu == 2.5:
            constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance**2)
        k = constant_component * exp_component
        if diag: # simple fix on stddev "not equipped to handle and diag problem" 
            k = k[0]
        return k