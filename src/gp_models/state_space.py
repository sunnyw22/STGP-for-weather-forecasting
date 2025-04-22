"""
Adds the StateSpaceGaussianProcess model for exact inference in the MarkovGP model.
"""

import jax.numpy as jnp
from bayesnewton.utils import diag, transpose
from bayesnewton.basemodels import MarkovGaussianProcess
from bayesnewton.kernels import SpatioTemporalKernel as BaseSpatioTemporalKernel


class SpatioTemporalKernel(BaseSpatioTemporalKernel):
    """
    Modified to include the spatial kernel during inference.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def stationary_covariance(self):
        Kzz = self.spatial_kernel(self.z.value, self.z.value)
        Pinf_time = self.temporal_kernel.stationary_covariance()
        Pinf = jnp.kron(Kzz, Pinf_time)
        return Pinf


class StateSpaceGaussianProcess(MarkovGaussianProcess):
    """
    The original MarkovGaussianProcess implemented in BayesNewton was intended only to be
    used as a base class for doing variational Gaussian processes. This class fills in the
    missing pieces to do exact Gaussian processes, which is what we want to do.
    """

    def __init__(self, *args, **kwargs):
        """
        State-space GP model with a spatiotemporal kernel.

        Args:
            kernel (Kernel): spatiotemporal kernel.
            likelihood (Likelihood): likelihood function.
            X (ndarray): (n_temporal,) array of training time steps.
            R (ndarray): (n_temporal, n_spatial, n_dim) array of spatial locations.
            Y (ndarray): (n_temporal, n_spatial) array of observations e.g. weather variables.
        """
        super().__init__(*args, **kwargs)
        var = (
            self.likelihood.variance
        )  # Set to small value to approximate zero measurement noise
        n_spatial = self.R.shape[1]
        n_temporal = self.X.shape[0]

        # self.noise_cov is a diagonal (n_temporal, n_spatial, n_spatial) matrix specifying the homoscedatic measurement noise
        self.noise_cov = jnp.tile(jnp.eye(n_spatial) * var, (n_temporal, 1, 1))
        self.H = self.kernel.measurement_model()

        self.filter_mean = None
        self.filter_cov = None
        self.smoother_mean = None
        self.smoother_cov = None
        self.gain = None
        self.parallel = kwargs.get("parallel", False)

    def compute_full_pseudo_lik(self):
        return self.Y[..., None], self.noise_cov

    def fit(self, X=None, R=None):
        """
        Args:
            X (ndarray): (n_temporal,) array of temporal inputs.
            R (ndarray): (n_temporal, n_spatial, n_dim) array of spatial inputs.
            Y (ndarray): (n_temporal, n_spatial) array of observations.
        """
        _, (filter_mean, filter_cov) = self.filter(
            self.dt,
            self.kernel,
            self.Y[..., None],
            self.noise_cov,
            mask=None,
            parallel=self.parallel,
        )
        self.filter_mean = filter_mean
        self.filter_cov = filter_cov

        dt = jnp.concatenate([self.dt[1:], jnp.array([0.0])], axis=0)
        smoother_mean, smoother_cov, gain = self.smoother(
            dt,
            self.kernel,
            filter_mean,
            filter_cov,
            return_full=True,
            parallel=self.parallel,
        )
        self.smoother_mean = smoother_mean
        self.smoother_cov = smoother_cov
        self.gain = gain

    def predict(self, X=None, R=None, pseudo_lik_params=None):
        """
        Args:
            X (ndarray): array of new time points to predict at.

        Returns:
            test_mean (ndarray): mean of the predictions.
            test_std (ndarray): standard deviation of the predictions.
        """
        t_test = X
        if len(t_test.shape) < 2:
            t_test = t_test[..., None]

        t_train = self.X[:, :1]
        inf = 1e10 * jnp.ones_like(t_train[0])
        t_train_padded = jnp.block([[-inf], [t_train], [inf]])

        _, (filter_mean, filter_cov) = self.filter(
            self.dt,
            self.kernel,
            self.Y[..., None],
            self.noise_cov,
            mask=None,
            parallel=self.parallel,
        )
        self.filter_mean = filter_mean
        self.filter_cov = filter_cov

        dt = jnp.concatenate([self.dt[1:], jnp.array([0.0])], axis=0)
        smoother_mean, smoother_cov, gain = self.smoother(
            dt,
            self.kernel,
            filter_mean,
            filter_cov,
            return_full=True,
            parallel=self.parallel,
        )
        self.smoother_mean = smoother_mean
        self.smoother_cov = smoother_cov
        self.gain = gain

        # Predict the state distribution at the test time steps:
        state_mean, state_cov = self.temporal_conditional(
            t_train_padded,
            t_test,
            self.smoother_mean,
            self.smoother_cov,
            self.gain,
            self.kernel,
        )
        test_mean = (self.H @ state_mean).squeeze()
        test_std = jnp.sqrt(diag(self.H @ state_cov @ transpose(self.H)))
        return test_mean, test_std
