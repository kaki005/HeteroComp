import math
import pathlib
from collections import deque
from typing import override

import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import digamma

from _src.configs import ModelConfig

from .base import Base
from .util import _multi_digamma, log_multi_beta, log_s, save_sparse_matrix

ZERO = 1.0e-8


# ========================================
# region (A_Polya)
# ========================================
#
class A_Polya(Base):
    def __init__(self, mode: int, mode_dim: int, k: int, alpha: float, config: ModelConfig, mode_idx):
        super(A_Polya, self).__init__(k, mode_dim, config)
        self.mode: int = mode
        """attribute index"""
        self.mode_idx = mode_idx.replace(" ", "_").replace("/", "_")
        self.mode_dim: int = mode_dim
        self.alpha: np.ndarray = np.full(self.mode_dim, alpha)
        """(dim)"""
        self.alpha_init: float = alpha
        self.prev_terms = np.zeros((mode_dim, k), dtype=float)
        """buffer (dim, K)"""
        self.prev_distributions = deque()

    @override
    def setL(self, L: int):
        super(A_Polya, self).setL(L)

    @override
    def reset(self):
        super(A_Polya, self).reset()
        self.alpha: np.ndarray = np.full(self.mode_dim, self.alpha_init)

    def compute_posterior(self, counter_M: np.ndarray, counter_K: np.ndarray) -> np.ndarray:
        """
        Args:
            counter_M (np.ndarray): (dim, K)
            counter_K (np.ndarray):  counter(K)

        """
        factor = (counter_M + self.prev_terms) / (counter_K + self.L * np.sum(self.alpha) + ZERO)
        factor = _normalize_factor(factor)
        return factor

    def init_gibbs_batch(self, l: int, tensor: pd.DataFrame):
        return tensor

    def post_gibbs_batch(self, l: int, counterM, counterK):
        """
        Args:
            l (int) : tensor index
            counter_M (np.ndarray): (dim, K)
            counter_K (np.ndarray):  counter(K)
        """

    def init_gibbs(self, tensor: pd.DataFrame):
        """Initial processing for Gibbs sampling"""
        self.prev_terms = self.alpha[:, np.newaxis] * np.sum(list(self.prev_distributions), axis=0)
        return tensor

    def init_prev_dist(self, l: int, counterM: np.ndarray, counterK: np.ndarray):
        """Calculate the parameters of the posterior distribution in initialize.
        Specifically, we calculate the likelihood (unmarginalized) for (K, index) when using the Dirichlet prior distribution for the parameter alpha.

        Args:
            counterM (np.ndarray): counter (dim, K)
            counterK (np.ndarray): counter (K)
        """
        factor = (counterM + self.alpha[:, np.newaxis]) / (counterK + np.sum(self.alpha))
        self.prev_distributions.append(_normalize_factor(factor))

    def update_prev_dist(self, counterM: np.ndarray, counterK: np.ndarray):
        if len(self.prev_distributions) >= self.L:  # pop old param
            self.prev_distributions.popleft()
        self.prev_distributions.append(self.compute_posterior(counterM, counterK))  # append new param

        return 0

    def update_hyperparam(self, counterM, counterK):
        self.alpha = _new_alpha(counterM, counterK, self.alpha)  # fixed-point iteration method

    def log_likelihood_init(self, counterM: np.ndarray, *args):
        """log likelihood in the initialize
        Args:
            counterM: (mode, K)
        """
        llh = 0
        # Likelihood when the alpha is used as a prior parameter
        for k in range(self.k):
            llh += log_multi_beta(counterM[:, k] + self.alpha)
        llh -= self.k * log_multi_beta(self.alpha)
        return llh

    def log_likelihood(self, counterM: np.ndarray, *args):
        """Marginal likelihood when the prior distribution is a Dirichlet distribution and the likelihood is a multinomial distribution
        Args:
            counterM: (dim, K)
        """
        llh = 0
        for i in range(self.k):
            llh += log_multi_beta(counterM[:, i] + self.prev_terms[:, i])
            llh -= log_multi_beta(self.prev_terms[:, i])
        return llh

    def save(self, out_dir: pathlib.Path, counterM: np.ndarray, labels: list[str]):
        save_sparse_matrix(out_dir / f"counter_{self.mode_idx}.csv", counterM, "dim", "component", labels)

    def save_online(self, out_dir: pathlib.Path, counterM, labels: list[str]):
        save_sparse_matrix(out_dir / f"counter_{self.mode_idx}.csv", counterM, "dim", "component", labels)

    def plot(self, out_dir: pathlib.Path, counterM: np.ndarray):
        return


# endregion (A_Polya)


@numba.njit
def _normalize_factor(factor: np.ndarray) -> np.ndarray:
    """Normalize so that the sum of the attribute directions equals 1"""
    sum_factor = np.sum(factor, axis=0)
    return factor / (sum_factor.reshape(1, -1) + ZERO)


# @numba.njit
def _new_alpha(counterM, counterK, alpha):
    ZERO = 1.0e-8
    K = counterK.shape[0]
    new_alpha = np.zeros_like(alpha)
    for i in range(alpha.shape[0]):
        new_alpha[i] = (
            alpha[i] * (np.sum(_multi_digamma(counterM[i, :] + alpha[i])) - K * digamma(alpha[i] + ZERO)) / (np.sum(_multi_digamma(counterK + np.sum(alpha))) - K * digamma(np.sum(alpha)) + ZERO)
        )
    return new_alpha
