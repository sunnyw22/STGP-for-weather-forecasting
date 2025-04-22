"""Custom Mean functions for GP regression model."""
import torch
import torch.nn as nn
import gpytorch
from gpytorch.means.mean import Mean
## Mean function parametrised by theta, phi
class latlonmean(Mean):
    r"""
    A custom linear mean function for spatial data parameterized by (lat, lon).
    
    This mean is defined as:
    
        m(lat, lon) = constant + lat_coef * lat + lon_coef * lon.
    
    It is assumed that inputs are of shape (..., 2), where the last dimension
    corresponds to (lat, lon). 
    """
    def __init__(self, batch_shape=torch.Size()):
        super(latlonmean, self).__init__()
        self.batch_shape = batch_shape
        self.register_parameter(
            name="raw_constant", 
            parameter=torch.nn.Parameter(torch.zeros(batch_shape))
        )
        coef_shape = batch_shape + torch.Size([2])
        self.register_parameter(
            name="raw_coefs",
            parameter=torch.nn.Parameter(torch.zeros(coef_shape))
        )
        # Priors and constraints?
    
    @property
    def constant(self):
        return self.raw_constant

    @property
    def coefs(self):
        return self.raw_coefs

    def forward(self, x):

        lat = torch.deg2rad(x[...,0])
        lon = torch.deg2rad(x[...,1])

        lat_part = self.coefs[..., 0] * lat.unsqueeze(-1)
        lon_part = self.coefs[..., 1] * lon.unsqueeze(-1)
        out = self.constant.unsqueeze(-1) + lat_part + lon_part
        return out.squeeze(-1)
    

## Mean function parametrised by a neural network
class MLPmean(Mean):
    r"""
    A neural network mean function.
    """
    def __init__(self, input_dim=4, hidden_dim=16):
        super(MLPmean, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.mlp(x).squeeze(-1)


