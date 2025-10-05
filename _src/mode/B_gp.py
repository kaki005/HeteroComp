import copy
import logging
import pathlib
from typing import override

import equinox as eqx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import scipy
import seaborn as sns
from jaxtyping import Array, Float

from _src.configs import ModelConfig
from _src.configs.Config import DataConfig
from _src.gp import MarkovGP, Matern32, Trainer
from _src.plots import set_major_tick_per_day, set_major_tick_per_year, set_minor_tick_per_month

from .base import Base
from .util import _logsumexp, _softmax, softmax_multi_posterior

ZERO = 1.0e-8


# ========================================
# region (B_gp)
# ========================================


class B_gp(Base):
    def __init__(
        self,
        time_size,
        k: int,
        config: ModelConfig,
        dataConfig: DataConfig,
        colors: sns.palettes._ColorPalette,
    ):
        self.func_dim: int = 2
        super(B_gp, self).__init__(k, self.func_dim + self.func_dim * self.func_dim, config)
        self.time_size = time_size
        self.colors: sns.palettes._ColorPalette = colors
        self.gp: tuple[MarkovGP] = _createGP(np.arange(k))
        # self.gibbs: Gibbs_B_GP = Gibbs_B_GP(time_size, k)
        self.L: int = 1
        self.means = np.zeros((time_size, k))
        """(T_c+1, K)"""
        self.variances = np.ones((time_size, k))
        """(T_c+1, K)"""
        self.dts = jnp.ndarray = jnp.ones(self.time_size) * 0.1 + 0.1
        """differences of timestamps"""
        self.trainers: Trainer = Trainer(optax.adam(config.learning_rate), self.gp)
        """Kernel Hyperparameter Optimization Class"""
        self.last_date: np.ndarray | None = None
        self.mean_history: list[np.ndarray] = []
        self.var_history: list[np.ndarray] = []
        self.dataConfig: DataConfig = dataConfig
        self.prev_dist: np.ndarray
        """buffer"""
        self.prior_mean: np.ndarray
        self.prior_vars: np.ndarray

    def compute_posterior(self, counterM: Float[np.ndarray, "T k"], counterA: Float[np.ndarray, "T"]) -> None:
        """

        Args:
            counterM (np.ndarray): (T_c, K)
            counterA (np.ndarray): counter (T_c)
        """
        for t in range(self.means.shape[0]):
            self.means[t], self.variances[t] = softmax_multi_posterior(self.means[t], self.variances[t], counterM[t])
        self.means, self.variances, latent_means, latent_covs = _compute_posterior(self.gp, self.means, self.variances, self.m0, self.P0, self.dts)
        self.means = np.array(self.means)
        self.variances = np.array(self.variances)
        self.prev_dist = np.ascontiguousarray(np.vstack([np.array(latent_means), np.array(latent_covs)]))

    def init_gibbs_batch(self, l: int, tensor: pd.DataFrame, timestamps: np.ndarray):
        return self.init_gibbs(tensor, timestamps)

    def post_gibbs_batch(self, l: int, counterM, counterA):
        self.post_gibbs(counterM, counterA)

    def init_gibbs(self, tensor: pd.DataFrame, timestamps: np.ndarray):
        """Initial processing for Gibbs sampling"""
        del self.dts
        if self.last_date is None:
            self.logger.info("init")
            self.means = np.zeros((timestamps.shape[0], self.k))
            """(T_c, K)"""
            self.variances = np.ones((timestamps.shape[0], self.k))
            """(T_c, K)"""
            diff = datetime_diff(timestamps, self.dataConfig.freq, self.dataConfig.time_scale)
            self.dts = np.concatenate([[self.dataConfig.time_scale], diff])
            self.m0 = _init_m0(self.gp)
            self.P0 = _init_P0(self.gp)
        else:
            self.dts = datetime_diff(
                np.concatenate([[self.last_date], timestamps]),
                self.dataConfig.freq,
                self.dataConfig.time_scale,
            )
            latent_mean = self.prev_dist[: self.func_dim]
            latent_cov = self.prev_dist[self.func_dim :].reshape((self.func_dim, self.func_dim, self.k))
            self.means, self.variances = _compute_prior(
                self.gp,
                latent_mean,
                latent_cov,
                datetime_base(
                    timestamps,
                    self.last_date,
                    self.dataConfig.freq,
                    self.dataConfig.time_scale,
                ),
            )
            self.m0 = latent_mean.T
            self.P0 = jnp.transpose(latent_cov, axes=(2, 0, 1))
            self.means = np.array(self.means)
            self.variances = np.array(self.variances)
        self.prior_mean = self.means.squeeze().copy()
        self.prior_vars = self.variances.squeeze().copy()
        assert np.all(self.dts > 0)
        self.last_date = timestamps[-1]
        return tensor

    def update_hyperparam(self):
        negative_llh, grad = _energy(self.gp, self.means, self.variances, self.m0, self.P0, self.dts)
        self.gp = self.trainers.update(grad)  # update hyper param

    def update_prev_dist(self, *args):
        return 0

    def post_gibbs(self, counterM: Float[np.ndarray, "T k"], counterA: Float[np.ndarray, "T"]):
        self.compute_posterior(counterM, counterA)  # udpate posterior

    def anomaly_score(self):
        anomaly_scores = _anomaly_score_vmap(self.gp, self.means, self.variances, self.m0, self.P0, self.dts)
        anomaly_scores = np.array(anomaly_scores).squeeze()
        return anomaly_scores

    def save_history(self):
        self.mean_history.append(self.means)
        self.var_history.append(self.variances)

    def save(self, out_dir: pathlib.Path):
        if len(self.mean_history) > 0:
            np.savetxt(out_dir / "Bmeans.txt", np.vstack(self.mean_history))
        if len(self.var_history) > 0:
            np.savetxt(out_dir / "Bvars.txt", np.vstack(self.var_history))

    def plot(self, out_dir: pathlib.Path, time_labels: pd.DatetimeIndex):
        mean_series = np.vstack(self.mean_history)
        out_dir.mkdir(exist_ok=True)
        y = scipy.special.softmax(mean_series, axis=1).T
        labels = [f"component {k + 1}" for k in range(self.k)]
        fig, ax = plt.subplots(figsize=(30, 20))
        ax.stackplot(time_labels, y, labels=labels, colors=self.colors)
        if self.dataConfig.freq == "H":
            set_major_tick_per_day(ax, time_labels, rotation=45)
        elif self.dataConfig.freq == "D":
            set_major_tick_per_year(ax, time_labels)
            set_minor_tick_per_month(ax, time_labels)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "B_component_prob.png")
        plt.close(fig)

        for i in range(0, self.k, 10):
            fig, ax = plt.subplots(figsize=(30, 20))
            for j in range(i, min(i + 10, self.k)):
                ax.plot(
                    time_labels,
                    mean_series[:, j],
                    label=f"component {j + 1}",
                )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)  # rotate label
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / f"B_{i}.png")
            plt.close(fig)

    def log_likelihood_init(self, counterM: Float[np.ndarray, "T_c k"], counterA: Float[np.ndarray, "T_c"]):
        """log likelihood in the initialization

        Args:
            counterM (np.ndarray): (T_c, K)
        """
        return self.log_likelihood(counterM, counterA)

    def log_likelihood(self, counterM: Float[np.ndarray, "T_c k"], counterA: Float[np.ndarray, "T_c"]) -> np.ndarray:
        """compute log likelihood

        Args:
            counterM (np.ndarray): (T_c, K)

        Returns:
            np.ndarray: log likelihood
        """
        llh = 0
        for t in range(self.means.shape[0]):
            llh += counterM[t] @ self.means[t] - counterA[t] * _logsumexp(self.means[t])
        return np.array(llh)


# endregion (B_gp)


@eqx.filter_jit
@eqx.filter_vmap(in_axes=(0, 1, 1, 0, 0, None))
@eqx.filter_value_and_grad
def _energy(gp: MarkovGP, means, variances, m0, P0, dts):
    return -gp.log_likelihood(means, variances, m0, P0, dts)


@eqx.filter_vmap
def _createGP(index) -> MarkovGP:
    kernel = Matern32()
    return MarkovGP(kernel)


@eqx.filter_jit
@eqx.filter_vmap(in_axes=(0, 1, 1, 0, 0, None), out_axes=(1, 1, 1, 1))
def _compute_posterior(gp: MarkovGP, means, variances, m0, P0, dts):
    means, variances, latent_means, latent_covs = gp.compute_posterior(means, variances, m0, P0, dts)
    return means, variances, latent_means[-1], jnp.ravel(latent_covs[-1])


@eqx.filter_vmap()
def _init_P0(gp: MarkovGP):
    return gp.kernel.stationary_covariance()


@eqx.filter_vmap()
def _init_m0(gp: MarkovGP):
    return np.zeros((gp.kernel.state_dim, 1))


@eqx.filter_jit
@eqx.filter_vmap(in_axes=(0, 1, 1, 0, 0, None))
def _log_likelihood(gp: MarkovGP, means, variandes, m0, P0, dts):
    return gp.log_likelihood(means, variandes, m0, P0, dts)


@eqx.filter_jit
@eqx.filter_vmap(in_axes=(0, 1, 2, None), out_axes=(1, 1))
def _compute_prior(gp: MarkovGP, m0, P0, dts):
    return gp.compute_prior(m0, P0, dts)


@eqx.filter_jit
@eqx.filter_vmap(in_axes=(0, 1, 1, 0, 0, None))
def _anomaly_score_vmap(gp: MarkovGP, means, variandes, m0, P0, dts):
    return jnp.squeeze(gp.anomaly_score(means, variandes, m0, P0, dts))


# @eqx.filter_jit
# @eqx.filter_vmap(in_axes=(1, 1, 1, 1))
# def _kl_divergence_gp(
#     prior_means:Float[Array, "num_bins 1"],
#     pre_vars: Float[Array, "num_bins 1 1"],
#     post_means: Float[Array, "num_bins 1"],
#     post_vars: Float[Array, "num_bins 1 1"]):
#     def _kl_gauss(mean1, mean2, var1, var2):
#         return 0.5 *(jnp.log(var2/ var1) + (var1+ (mean1-mean2)**2)/ var2 -1)
#     return jnp.sum(jax.vmap(_kl_gauss)(prior_means, post_means, pre_vars, post_vars))


def datetime_base(timestamps: np.ndarray, base_time: pd.Timestamp, freq: str, time_scale: float = 1.0):
    diff = timestamps - np.array(base_time)
    diff = np.array(diff).astype("timedelta64[us]")
    return time_scale * (diff / freq_to_timedelta64(freq))


def datetime_diff(timestamps: np.ndarray, freq: str, time_scale: float = 1.0):
    diff = np.diff(timestamps).astype("timedelta64[us]")
    convert_diff = time_scale * (diff / freq_to_timedelta64(freq))
    return convert_diff


def freq_to_timedelta64(freq_str):
    """convart pandas.freq to np.timedelta64."""
    offset = pd.tseries.frequencies.to_offset(freq_str)
    return np.timedelta64(offset.delta)
