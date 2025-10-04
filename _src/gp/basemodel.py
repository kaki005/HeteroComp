import logging
from typing import NamedTuple

import equinox as eqx
import jax.numpy as np

from .bayesnewton_utils import (
    gaussian_expected_log_lik,
)
from equinox.nn import StateIndex, State
from beartype import beartype as typechecker
from jax import vmap
from jax.lib import xla_bridge
from jaxtyping import Float, jaxtyped
from .gaussian import GaussianDistribution
from jaxtyping import Array
from .Kernel import StationaryKernel,Independent
from .likelihood import Poisson


class BaseState(NamedTuple):
    posterior_mean: Float[Array, "N func_dim 1"]
    posterior_cov: Float[Array, "N func_dim func_dim"]
    X: Float[Array, "N 1"]
    dx: Float[Array, "N 1"]


@jaxtyped(typechecker=typechecker)
class BaseModel(eqx.Module):
    """
    The stochastic differential equation (SDE) form of a Gaussian process (GP) model.
    Implements methods for inference and learning using state space methods, i.e. Kalman filtering and smoothing.
    Constructs a linear time-invariant (LTI) stochastic differential equation (SDE) of the following form:
        dx(t)/dt = F x(t) + L w(t)
              yₙ ~ p(yₙ | f(t_n)=H x(t_n))
    where w(t) is a white noise process and where the state x(t) is Gaussian distributed with initial
    state distribution x(t)~𝓝(0,Pinf).
    """

    kernel: StationaryKernel
    """kernel"""
    parallel: bool = eqx.field(static=True)
    """flag to switch between parallel and sequential implementation of Kalman filter"""
    logger: logging.Logger = eqx.field(static=True)
    likelihood: Poisson = eqx.field(static=True)
    pseudo_likelihood: GaussianDistribution
    """pseudo likelihood"""
    index: StateIndex
    """key of model state"""
    func_dim: int = eqx.field(static=True)
    obs_dim: int  = eqx.field(static=True)

    def __init__(
        self,
        kernel: StationaryKernel,
        likelihood: Poisson,
        X: Float[Array, "N 1"],
        num_data: int,
        func_dim=1,
        obs_dim = 1,
    ):
        H = kernel.measurement_model()
        self.kernel = kernel
        self.func_dim = func_dim
        """number of latent dimensions"""
        self.obs_dim = obs_dim
        self.likelihood = likelihood
        self.parallel = bool(xla_bridge.get_backend().platform == "gpu")
        self.logger: logging.Logger = logging.getLogger(__name__)
        dx = np.concatenate([np.zeros((1, 1)), np.diff(X, axis=0)])
        posterior_mean = np.zeros([num_data, self.func_dim, 1])
        posterior_var = np.tile(np.eye(self.func_dim), [num_data, 1, 1])
        self.index = eqx.nn.StateIndex(BaseState(posterior_mean, posterior_var, X, dx))
        if isinstance(self.kernel, Independent):
            pseudo_lik_size = self.func_dim  # the multi-latent case
        else:
            pseudo_lik_size = self.obs_dim
        self.pseudo_likelihood = GaussianDistribution(
            mean=np.zeros([num_data, pseudo_lik_size, 1]),
            covariance=np.tile(np.eye(pseudo_lik_size), [num_data, 1, 1]),
        )
        # inf = np.array([1e10])
        # self.pseudo_likelihood = GaussianDistribution(np.copy(zeros), inv_vmap(nat2))

    def compute_full_pseudo_lik(self, state: State):
        mean, cov, nat1, nat2 = self.pseudo_likelihood.state(state)
        return mean, cov

    def prior_sample(self, num_samps=1, X=None, seed=0):
        raise NotImplementedError

    def update_posterior(self, state: State) -> State:
        raise NotImplementedError

    def compute_log_lik(self, pseudo_y=None, pseudo_var=None):
        """Compute the log likelihood of the pseudo model, i.e. the log normaliser of the approximate posterior"""
        raise NotImplementedError

    def predict(self, state: State, X: Array, R=None, *args):
        raise NotImplementedError

    def compute_kl(self, state: State):
        raise NotImplementedError

    def group_natural_params(self, state: State, nat1_n, nat2_n, batch_ind=None):
        if batch_ind is not None:
            _, _, nat1, nat2 = self.pseudo_likelihood.state(state)
            nat1 = nat1.at[batch_ind].set(nat1_n)
            nat2 = nat2.at[batch_ind].set(nat2_n)
            return nat1, nat2
        else:
            return nat1_n, nat2_n

    def negative_log_predictive_density(
        self, state: State, X: np.ndarray, Y: np.ndarray, R=None, cubature=None
    ):
        predict_mean, predict_var = self.predict(state, X, R)
        if Y.ndim < 2:
            Y = Y.reshape(-1, 1)
        if (predict_mean.ndim > 1) and (
            predict_mean.shape[1] != Y.shape[1]
        ):  # multi-latent case
            predict_mean, predict_var = (
                predict_mean[..., None],
                predict_var[..., None] * np.eye(predict_var.shape[1]),
            )
        else:
            predict_mean, predict_var = (
                predict_mean.reshape(-1, 1, 1),
                predict_var.reshape(-1, 1, 1),
            )
        log_density = vmap(self.likelihood.log_density, (0, 0, 0, 0, None))(
            Y.reshape(predict_mean.shape[0], -1, 1),
            predict_mean,
            predict_var,
            np.arange(predict_mean.shape[0]),
            cubature,
        )
        return -np.nanmean(log_density)

    def expected_density_pseudo(self, state: State):
        """compute E_q[log N(pseudo_y_n | u, pseudo_var_n)]

        Returns:
            _type_:  E_q[log N(pseudo_y_n | u, pseudo_var_n)]
        """
        pseudo_mean, pseudo_cov, old_nat1, old_nat2 = self.pseudo_likelihood.state(
            state
        )
        posterior_mean, posterior_var, _, _ = state.get(self.index)
        expected_density = vmap(gaussian_expected_log_lik, in_axes=(0, 0, 0, 0, None))(
            pseudo_mean, posterior_mean, posterior_var, pseudo_cov, None
        )
        return np.sum(expected_density)


    def conditional_posterior_to_data(
        self, state: State, post_mean=None, post_cov=None
    ):
        """compute q(f) = int p(f | u) q(u) du
            where q(u) = N(u | post_mean, post_cov)

        Args:
            state (State): _description_

        Returns:
            _type_: _description_
        """
        if post_mean is None:
            post_mean, _, _, _ = state.get(self.index)
        if post_cov is None:
            _, post_cov, _, _ = state.get(self.index)
        return post_mean, post_cov, np.eye(post_mean.shape[1])

    def update_state(self, state: State, X: Array, new_mean, new_cov) -> State:
        dx = np.concatenate([np.zeros((1, 1)), np.diff(X, axis=0)])
        state = state.set(self.index, BaseState(new_mean, new_cov, X, dx))
        return state
