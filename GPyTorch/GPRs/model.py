"""GP Model (exact and stochastic variational approach) with GPyTorch implementation."""

import gpytorch
from .mean import latlonmean, MLPmean
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

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
        #self.mean_module = gpytorch.means.ConstantMean() 
        #self.mean_module = gpytorch.means.LinearMean()
        self.mean_module =  latlonmean()
        #self.mean_module =  MLPmean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class VarGP(ApproximateGP):
    """Stochastic Variational GP Regression Model from GPyTorch.
    We use inducing points with Cholesky Variational Distribution (other strategies 
    such as the Delta and Mean Field approach are available in GPyTorch).

    Parameters
    ----------
    inducing_points : torch.Tensor
        Tensor containing a set of inducing points to use for variational inference.
        Expected tensor of shape (M, 4(x,y,z,t)) or (M, 3(lat,lon,t)) where M is the
        total number of inducing points.
    kernel : gpytorch.kernels.Kernel
        The covariance function (kernel).

    TODO: Implement mean_module parametrised by lat, lon.
    """
    def __init__(self, inducing_points, kernel):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(VarGP, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        #self.mean_module = latlonmean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)