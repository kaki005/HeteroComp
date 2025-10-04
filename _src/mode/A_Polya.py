from typing import override


import numba
import numpy as np
from _src.configs import ModelConfig
from .util import log_multi_beta, log_s, _multi_digamma, save_sparse_matrix
import pathlib
from collections import deque
from .base import Base
import seaborn as sns
import matplotlib.pyplot as plt
import math
from scipy.special import digamma # ガンマ関数とダイガンマ関数
import pandas as pd
ZERO = 1.0e-8



# ========================================
# region (A_Polya)
# ========================================
#
class A_Polya(Base):
    """
    alphaを固定したバージョンです。
    """

    def __init__(
        self, mode: int, mode_dim: int, k: int, alpha: float, config: ModelConfig,mode_idx
    ):
        super(A_Polya, self).__init__(k, mode_dim, config)
        self.mode: int = mode
        """attribute index"""
        self.mode_idx = mode_idx.replace(' ', '_').replace('/', '_')
        self.mode_dim: int = mode_dim
        self.alpha: np.ndarray = np.full(self.mode_dim, alpha)
        """(dim)"""
        self.alpha_init: float = alpha
        self.prev_terms = np.zeros((mode_dim, k), dtype=float)
        """トピックトラッキングにおけるディレクレの事前パラメータ(dim, K)"""
        self.prev_distributions = deque()


    @override
    def setL(self, L: int):
        super(A_Polya, self).setL(L)

    @override
    def reset(self):
        super(A_Polya, self).reset()
        self.alpha: np.ndarray = np.full(self.mode_dim, self.alpha_init)


    def compute_posterior(
        self, counter_M: np.ndarray, counter_K: np.ndarray
    ) -> np.ndarray:
        """
        Args:
            counter_M (np.ndarray): (dim, K)
            counter_K (np.ndarray):  counter(K)

        """
        factor = (counter_M + self.prev_terms) / (
            counter_K + self.L * np.sum(self.alpha) + ZERO
        )
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
        factor = (counterM + self.alpha[:, np.newaxis]) / (
            counterK + np.sum(self.alpha)
        )
        self.prev_distributions.append(_normalize_factor(factor))

    def update_prev_dist(self, counterM: np.ndarray, counterK: np.ndarray):
        if len(self.prev_distributions) >= self.L: # pop old param
            self.prev_distributions.popleft()
        self.prev_distributions.append(self.compute_posterior(counterM, counterK))# append new param

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
            # ↓ wrong ver
            # llh += log_multi_beta(counterM[:, i] + self.alpha[i])
            # llh -= self.mode_dim * gammaln(self.alpha[i]) - gammaln(self.mode_dim * self.alpha[i])
            # ↓  topic tracking model (5)
            llh += log_multi_beta(counterM[:, i] + self.prev_terms[:, i])
            llh -= log_multi_beta(self.prev_terms[:, i])
        return llh

    def save(self, out_dir: pathlib.Path, counterM:np.ndarray, labels: list[str]):
        save_sparse_matrix(out_dir / f"counter_{self.mode_idx}.csv", counterM, "dim", "topic", labels)

    def save_online(
        self,
        out_dir: pathlib.Path,
        counterM,
        labels: list[str]
    ):
        save_sparse_matrix(
            out_dir / f"counter_{self.mode_idx}.csv", counterM, "dim", "topic", labels
        )

    def plot(
        self,
        out_dir: pathlib.Path,
        counterM: np.ndarray
    ):
        return


# endregion (A_Polya)


@numba.njit
def _normalize_factor(factor: np.ndarray) -> np.ndarray:
    """モード方向の和が1になるように正規化"""
    sum_factor = np.sum(factor, axis=0)
    sum_factor = sum_factor.reshape(1, -1)  # 2次元配列に戻す
    return factor / (sum_factor + ZERO)


@numba.njit
def _polya_costM(factor, mode_dim: int, k: int, tol_r: float, FB: float):
    non_zeros = np.sum(factor > tol_r, axis=1)
    non_zeros[np.argmax(non_zeros)] = 0
    non_zero_count = non_zeros.sum()
    mode_refined = mode_dim
    if mode_refined < 2:
        mode_refined = 2  # avoid log(0)
    cost = non_zero_count * (np.log2(k * (mode_refined - 1)) + FB)
    cost += log_s(non_zero_count)
    return cost


# @numba.njit
def _new_alpha(counterM, counterK, alpha):
    ZERO = 1.0e-8
    K = counterK.shape[0]
    new_alpha = np.zeros_like(alpha)
    for i in range(alpha.shape[0]):
        new_alpha[i] = (
            alpha[i]
            * (
                np.sum(_multi_digamma(counterM[i, :] + alpha[i]))
                - K * digamma(alpha[i] + ZERO)
            )
            / (
                np.sum(_multi_digamma(counterK + np.sum(alpha)))
                - K * digamma(np.sum(alpha))
                + ZERO
            )
        )
    return new_alpha

# @numba.njit
# def kl_divergence_polya(alpha1: np.ndarray, alpha2: np.ndarray) -> float:
#     """
#     Polya (Dirichlet-Multinomial) distributions間のKLダイバージェンスを計算します。

#     パラメータ:
#     alpha1 (np.ndarray): 最初のPolya分布のパラメータベクトル (alpha_1, ..., alpha_K)。
#                          全ての要素は正である必要があります。
#     alpha2 (np.ndarray): 2番目のPolya分布のパラメータベクトル (beta_1, ..., beta_K)。
#                          全ての要素は正である必要があります。

#     戻り値:
#     float: 2つのPolya分布間のKLダイバージェンスの値。
#     """
#     if not (np.all(alpha1 >= 0) and np.all(alpha2 >= 0)):
#         raise ValueError("alpha1 and alpha2 must contain only positive values.")
#     if alpha1.shape != alpha2.shape:
#         raise ValueError("alpha1 and alpha2 must have the same shape.")

#     K = alpha1.shape[0]

#     # B(alpha) = product(Gamma(alpha_i)) / Gamma(sum(alpha_i))
#     # log(B(alpha)) = sum(log(Gamma(alpha_i))) - log(Gamma(sum(alpha_i)))

#     sum_alpha1 = np.sum(alpha1)
#     sum_alpha2 = np.sum(alpha2)
#     # log B(alpha1) の計算
#     log_B_alpha1 = np.sum(np.log(_multi_gamma(alpha1))) - np.log(gamma(sum_alpha1))
#     # log B(alpha2) の計算
#     log_B_alpha2 = np.sum(np.log(_multi_gamma(alpha2))) - np.log(gamma(sum_alpha2))
#     # KLダイバージェンスの式の各項を計算
#     term1 = log_B_alpha2 - log_B_alpha1
#     # ダイガンマ関数の計算
#     digamma_alpha1 = _multi_digamma(alpha1)
#     digamma_sum_alpha1 = digamma(sum_alpha1)
#     term2 = np.sum((alpha1 - alpha2) * (digamma_alpha1 - digamma_sum_alpha1))
#     kl_div = term1 + term2
#     return kl_div


# @numba.njit
# def _multi_gamma(value):
#     """多変量対応番のdigamma"""
#     return np.array([gamma(val + ZERO) for val in value])
