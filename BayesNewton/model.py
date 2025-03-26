"""
Adds the ExactMarkovGaussianProcess model for exact inference in the MarkovGP model.
"""

import jax.numpy as jnp
from bayesnewton.utils import diag, transpose
from bayesnewton.basemodels import MarkovGaussianProcess


class ExactMarkovGaussianProcess(MarkovGaussianProcess):
    """
    Modifies the MarkovGP model to do exact instead of variational inference.
    """

    def __init__(self, *args, **kwargs):
        """
        MarkovGP model with a spatiotemporal kernel.

        Args:
            kernel (Kernel): spatiotemporal kernel.
            likelihood (Likelihood): likelihood function.
            X (ndarray): (n_temporal,) array of temporal inputs.
            R (ndarray): (n_temporal, n_spatial, N_dim) array of spatial inputs.
            Y (ndarray): (n_temporal, n_spatial) array of observations.
        """
        super().__init__(*args, **kwargs)
        var = (
            self.likelihood.variance
        )  # Set to small value, approximates zero measurement noise
        n_spatial = self.R.shape[1]
        n_temporal = self.X.shape[0]

        # self.noise_cov is a diagonal (n_temporal, n_spatial, n_spatial) matrix specifying the homoscedatic measurement noise
        self.noise_cov = jnp.tile(jnp.eye(n_spatial) * var, (n_temporal, 1, 1))
        self.H = self.kernel.measurement_model()

    def compute_full_pseudo_lik(self):
        return self.Y[..., None], self.noise_cov

    def predict(self, X=None, R=None, pseudo_lik_params=None):
        """
        Args:
            X (ndarray): array of new time points to predict at.

        Returns:
            test_mean (ndarray): mean of the predictions.
            test_std (ndarray): standard deviation of the predictions.
        """
        t_test = X
        if X is None:
            t_test = self.X[:, 0]

        if len(t_test.shape) < 2:
            t_test = t_test[..., None]

        _, (filter_mean, filter_cov) = self.filter(
            self.dt,
            self.kernel,
            self.Y[..., None],
            self.noise_cov,
            mask=None,
            parallel=self.parallel,
        )
        dt = jnp.concatenate([self.dt[1:], jnp.array([0.0])], axis=0)
        smoother_mean, smoother_cov, gain = self.smoother(
            dt,
            self.kernel,
            filter_mean,
            filter_cov,
            return_full=True,
            parallel=self.parallel,
        )

        t_train = self.X[:, :1]
        inf = 1e10 * jnp.ones_like(t_train[0])
        t_train_padded = jnp.block([[-inf], [t_train], [inf]])

        # predict the state distribution at the test time steps:
        state_mean, state_cov = self.temporal_conditional(
            t_train_padded, t_test, smoother_mean, smoother_cov, gain, self.kernel
        )
        test_mean = (self.H @ state_mean).squeeze()
        test_std = jnp.sqrt(diag(self.H @ state_cov @ transpose(self.H)))
        return test_mean, test_std


ExactMarkovGP = ExactMarkovGaussianProcess
