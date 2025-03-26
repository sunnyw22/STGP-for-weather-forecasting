"""Custom Kernel on a sphere."""

import math
import torch
import gpytorch

class SphericalMaternKernel(gpytorch.kernels.Kernel):
    """
    A custom kernel that accounts for spherical geometry.
    This kernel computes distances on the sphere using the following approximation:

        ds^2 = d\theta^2 + \sin^2(\theta)d\phi^2
           d = sqrt((delta_theta)^2 + (sin(theta_mean) * delta_phi)^2)

    where delta_theta = theta2 - theta1, delta_phi = phi2 - phi1, and
    theta_mean is the average of theta1 and theta2. The kernel then uses these distances
    in a Matern-like covariance function.

    Parameters
    ----------
    nu : float, optional (default: 1.5)
        The smoothness parameter of the Matern kernel.
    """

    def __init__(self, nu=1.5, **kwargs):
        super().__init__(**kwargs)
        self.nu = nu
        self.register_parameter(name="raw_lengthscale", parameter=torch.nn.Parameter(torch.tensor(1.0)))
        self.register_constraint("raw_lengthscale", gpytorch.constraints.Positive())

    @property
    def lengthscale(self):
        return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)

    @lengthscale.setter
    def lengthscale(self, value):
        self._set_lengthscale(value)

    def forward(self, x1, x2, diag=False, **params):
        """
        Computes the covariance matrix between inputs x1 and x2.
        Assumes that the last dimension of x contains [theta, phi] in radians.
        """
        if diag:
            return torch.zeros(x1.size(-2), device=x1.device)

        # Extract spherical coordinates
        theta1, phi1 = x1[..., 0], x1[..., 1]
        theta2, phi2 = x2[..., 0], x2[..., 1]

        # Compute differences and mean theta
        dtheta = theta1.unsqueeze(-2) - theta2.unsqueeze(-3)
        dphi = phi1.unsqueeze(-2) - phi2.unsqueeze(-3)
        theta_mean = 0.5 * (theta1.unsqueeze(-2) + theta2.unsqueeze(-3))
        
        # Approximate spherical distance
        d = torch.sqrt(dtheta**2 + (torch.sin(theta_mean) * dphi)**2)
        
        if self.nu == 1.5:
            sqrt3 = math.sqrt(3.0)
            scaled_d = sqrt3 * d / self.lengthscale
            covar = (1.0 + scaled_d) * torch.exp(-scaled_d)
        elif self.nu == 0.5:
            scaled_d = d / self.lengthscale
            covar = torch.exp(-scaled_d)
        else:
            raise NotImplementedError("Use nu=0.5 and nu=1.5 only")
        
        return covar