import jax.numpy as np
import jax
from jax.scipy.linalg import cho_factor, cho_solve
import math
import equinox as eqx
from equinox.nn import StateIndex, State

jax.config.update("jax_enable_x64", True)

LOG2PI = math.log(2 * math.pi)
from typing import NamedTuple


class GaussianState(NamedTuple):
    mean: np.ndarray
    covariance: np.ndarray
    nat1: np.ndarray
    nat2: np.ndarray


# ==================================
# region(GaussianDistribution)
# ==================================
class GaussianDistribution(eqx.Module):
    """
    A small class defined to handle the fact that we often need access to both the mean / cov parameterisation
    of a Gaussian and its natural parameterisation.
    Important note: for simplicity we let nat2 = inv(cov) rather than nat2 = -0.5inv(cov). The latter is the proper
    natural parameter, but for Gaussian distributions we need not worry about the -0.5 (it cancels out anyway).
    """

    index: StateIndex
    """key of model state"""

    def __init__(self, mean, covariance):
        nat1, nat2 = self.reparametrise(mean, covariance)
        self.index = StateIndex(GaussianState(mean, covariance, nat1, nat2))

    @staticmethod
    def reparametrise(param1, param2):
        """convert natural paramter to mean, covariance"""
        chol = cho_factor(param2, lower=True)
        reparam1 = cho_solve(chol, param1)
        reparam2 = cho_solve(
            chol, np.tile(np.eye(param2.shape[1]), [param2.shape[0], 1, 1])
        )
        return reparam1, reparam2

    def state(self, state: State) -> GaussianState:
        return state.get(self.index)

    def update_mean_cov(self, state: State, mean, covariance):
        nat1, nat2 = self.reparametrise(mean, covariance)
        return state.set(self.index, GaussianState(mean, covariance, nat1, nat2))

    def update_nat_params(self, state: State, nat1, nat2):
        mean, covariance = self.reparametrise(nat1, nat2)
        return state.set(self.index, GaussianState(mean, covariance, nat1, nat2))


# endregion(GaussianDistribution)
