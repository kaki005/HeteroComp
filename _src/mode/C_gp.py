import pathlib
from sre_constants import CATEGORY_UNI_NOT_LINEBREAK
from typing import override

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import vmap
from _src.gp import MarkovGP, Matern32, Trainer, weighted_logsumexp, LogDensityMarkovGP, BaseModel
import jax
import optax
from _src.gp import Poisson
from jaxtyping import Float, Array, Int
from equinox.nn import State
from _src.configs import ModelConfig
from _src.configs.Config import DataConfig
from .base import Base
import matplotlib.pyplot as plt
from collections import deque
from seaborn.palettes import _ColorPalette
from .util import save_sparse_matrix

LR_NEWTON = 1.0
ZERO = 1.0e-8


# ========================================
# region (C_gp)
# ========================================


class C_gp(Base):
    def __init__(
        self,
        mode_idx: str,
        k: int,
        config: ModelConfig,
        dataConfig: DataConfig,
        bounds: np.ndarray,
        colors: _ColorPalette,
    ):
        self.func_dim: int = 2
        super(C_gp, self).__init__(
            k, self.func_dim + self.func_dim * self.func_dim, config
        )
        self.mode_idx: str = mode_idx.replace(' ', '_').replace('/', '_')
        """column index of dataframe"""
        self.num_bins: int = bounds.shape[0]
        """num of mode discretizion"""
        self.logger.info(F"{self.mode_idx=}")
        self.L: int = 1
        self.gp: LogDensityMarkovGP
        self.states: State
        self.trainers: Trainer
        """class for kernel hyperparameter update"""
        self.dataConfig: DataConfig = dataConfig
        self.bin_x: Float[np.ndarray, "num_bins"] = np.mean(bounds, axis=1).reshape(
            -1, 1
        )
        self.leggaussX, self.leggaussW = np.polynomial.legendre.leggauss(
            300
        )
        self.leggaussX = self.leggaussX.reshape(-1, 1)
        self.grid_bounds: Float[np.ndarray, "num_bins 2"] = bounds
        self.grid_size: Float[np.ndarray, "num_bins"] = np.diff(
            bounds, axis=1
        ).squeeze()
        # self.Lambda: Float[np.ndarray, "num_bins k"]
        self.mode_prob: Float[np.ndarray, "num_bins k"]
        """(K, G_m)"""
        self.post_mean: Float[np.ndarray, "k num_bins 1"]
        """(K, G_m, 1)"""
        self.post_var: Float[np.ndarray, "k num_bins 1 1"]
        """(K, G_m, 1, 1)"""
        self.colors: _ColorPalette = colors
        self.grid_labels = [f"{x.item():.2e}" for x in self.bin_x]
        self.means_history = []
        self.vars_history = []
        self.counterQueue = deque()

    def compute_posterior(
        self, counterM: Float[np.ndarray, "mode k"], counterK: Float[np.ndarray, "K"]
    ) -> None:
        """

        Args:
            counterM (np.ndarray): counter (G_m, K)
            counterK (np.ndarray): counter (K)
        """
        self.states, self.post_mean, self.post_var = (
            _compute_posterior(self.gp, self.states, counterM)
        )
        # self.Lambda = np.array(
        #     _lambda_LGCP(self.post_mean, self.post_var, self.grid_bounds)
        # )
        self.mode_prob = np.array(
            _mode_prob_LDGP(self.post_mean, self.post_var, self.grid_size)
        )

    @override
    def reset(self):
        return


    def init_gibbs_batch(self, l: int, tensor: pd.DataFrame, *args):
        return self.init_gibbs(tensor, True)

    def post_gibbs_batch(self, l: int, counterM, counterA):
        self.post_gibbs(counterM, counterA)

    def init_gibbs(self, tensor: pd.DataFrame, init_gp: bool = False):
        """gibbs sampling in the initialize"""
        if init_gp:
            self.num_bins = self.bin_x.shape[0]
            self.post_mean = np.zeros((self.k, self.num_bins, 1))
            self.post_var = np.ones((self.k, self.num_bins))
            gp, states = _createGP(
                np.arange(self.k), self.bin_x, self.grid_size, self.config.C_lengthscale
            )
            self.gp = gp
            self.states = states
            self.trainers = Trainer(optax.adam(self.config.learning_rate), self.gp)
            self.mode_prob = np.array(
                _mode_prob_LDGP(self.post_mean, self.post_var, self.grid_size)
            )
        return tensor

    def update_hyperparam(self, counterM, counterK):
        negative_llh, grad = _energy(self.gp, self.states, counterM)
        self.logger.info(F"{self.mode_idx}: nllh={np.sum(negative_llh)}")
        self.gp = self.trainers.update(grad)  # ハイパーパラメータ更新



    def update_prev_dist(self, counterM: np.ndarray, counterK: np.ndarray):
        self.compute_posterior(counterM, counterK)  # 事後分布の計算
        return 0


    def save(self, out_dir: pathlib.Path, counterM:np.ndarray):
        np.savetxt(
            out_dir / f"final_mean.txt", self.post_mean.squeeze().T
        )
        np.savetxt(out_dir / f"final_var.txt", self.post_var.squeeze().T)
        np.savetxt(out_dir / f"final_prob.txt",  self.mode_prob.squeeze())
        if len(self.means_history) > 0:
            np.savetxt(
                out_dir / f"mean_{self.mode_idx}_hist.txt", np.vstack(self.means_history)
            )
        if len(self.vars_history) > 0:
            np.savetxt(
                out_dir / f"var_{self.mode_idx}_hist.txt", np.vstack(self.vars_history)
            )
        save_sparse_matrix(
            out_dir / "counter.csv",
            counterM,
            "bin",
            "topic",
        )
        np.savetxt(out_dir / "grid.txt", self.grid_bounds)

    def save_online(
        self, output_path: pathlib.Path, counterM: Int[np.ndarray, "dim k"]
    ):
        np.savetxt(
            output_path / f"mean_{self.mode_idx}.txt", self.post_mean.squeeze().T
        )
        np.savetxt(output_path / f"var_{self.mode_idx}.txt", self.post_var.squeeze().T)
        np.savetxt(output_path / f"prob_{self.mode_idx}.txt",  self.mode_prob.squeeze())
        save_sparse_matrix(
            output_path / f"counter_{self.mode_idx}.csv", counterM, "grid", "topic", self.grid_labels
        )

    def plot(
        self,
        out_dir: pathlib.Path,
        counterM: np.ndarray
    ):
        # plot latent gp
        _plot_latent_gp(out_dir/ f"latent_gp.png", self.bin_x, self.colors, self.post_mean, self.post_var, self.grid_bounds)
        # counter
        _plot_histgram(out_dir/ f"counter.png", counterM, self.bin_x, self.colors, self.mode_idx)

    def plot_online(
        self,
        out_dir: pathlib.Path,
        tensor: pd.DataFrame,
        assignment: np.ndarray,
        counterM: Int[np.ndarray, "dim K"],
    ):
        # plot latent gp
        _plot_latent_gp(out_dir/ f"{self.mode_idx}_gp.png", self.bin_x, self.colors, self.post_mean, self.post_var, self.grid_bounds)
        # counter
        _plot_histgram(out_dir/ f"{self.mode_idx}_counter.png", counterM, self.bin_x, self.colors, self.mode_idx)


    def log_likelihood_init(
        self, counterM: Int[np.ndarray, "dim k"], counterK: Float[np.ndarray, "K"]
    ):
        """initializeにおけるバッチの尤度

        Args:
            counterM (np.ndarray): (G_m, K)
        """
        llh = self.log_likelihood(counterM, counterK)
        return llh

    def log_likelihood(
        self, counterM: Int[np.ndarray, "dim k"], counterK: Int[np.ndarray, "K"]
    ) -> np.ndarray:
        """対数尤度を計算します

        Args:
            counterM (np.ndarray): (G_m, K)

        Returns:
            np.ndarray: log likelihood
        """
        llh = 0
        llhs = _log_likelihood(self.gp, self.states, self.bin_x, counterM + ZERO)
        llh += np.sum(llhs)
        # self.logger.info(f"{self.mode_idx} : {llh=}")
        return np.array(llh)

    # region (private method)

    @override
    def _compute_prev_dist(
        self, counterM: Int[np.ndarray, "dim k"], counterK: Int[np.ndarray, "k"]
    ) -> Float[np.ndarray, "k"]:
        return np.zeros(1)  # 何もなし

    # endregion (private method)


# endregion (C_gp)


@eqx.filter_jit
@eqx.filter_vmap(in_axes=(0, 0, 1))
@eqx.filter_value_and_grad()
def _energy(
    gp: LogDensityMarkovGP,
    state: eqx.nn.State,
    y: np.ndarray,
):
    return gp.energy(state, y)


@eqx.filter_vmap(in_axes=(0, None, None, None))
def _createGP(
    index,
    x: np.ndarray,
    bin_sizes: Float[Array, "num_bins"],
    lengthscale:int
) -> tuple[LogDensityMarkovGP, eqx.nn.State]:
    kernel = Matern32(lengthscale=lengthscale)
    likelihood = Poisson(binsizes=bin_sizes)
    return eqx.nn.make_with_state(LogDensityMarkovGP)(kernel, likelihood, jnp.copy(x))


@eqx.filter_jit
@eqx.filter_vmap(in_axes=(0, 0, 1))
def _compute_posterior(gp: LogDensityMarkovGP, state: State, Y: np.ndarray):
    def _not_update(state):
        post_mean, post_var, _, _ = state.get(gp.index)
        return state, post_mean.reshape(-1, 1), post_var
    def _update(state):
        (mean, var, jacobian, hessian, state), (diff1, diff2) = gp.inference(
            state, LR_NEWTON, Y=Y
        )
        return state, mean.reshape(-1, 1), var
    return jax.lax.cond(Y.sum() <= 0, _not_update, _update, state)


@eqx.filter_jit
@eqx.filter_vmap()
def _kl_divergence_gp(
    prior_means:Float[Array, "num_bins 1"],
    pre_vars: Float[Array, "num_bins 1 1"],
    post_means: Float[Array, "num_bins 1"],
    post_vars: Float[Array, "num_bins 1 1"]):
    def _kl_gauss(mean1, mean2, var1, var2):
        return 0.5 *(jnp.log(var2/ var1) + (var1+ (mean1-mean2)**2)/ var2 -1)
    return jnp.sum(vmap(_kl_gauss)(prior_means, post_means, pre_vars, post_vars))


@eqx.filter_jit
@eqx.filter_vmap(in_axes=(0, 0, None), out_axes=(1))
def _mode_prob_LDGP(
    means: Float[Array, "num_bins 1"],
    vars: Float[Array, "num_bins 1 1"],
    grid_sizes: Float[Array, "num_bins 1"],
) -> Float[Array, "num_bins topic"]:
    # intensity = jnp.exp(
    #     means.squeeze() + 0.5 * vars.squeeze()
    # )  # \lambda_{nk} = exp(m_{nk} + 0.5*\sigma_{nk}^2)
    # return weighted_softmax(means.squeeze(), grid_sizes).squeeze()
    log_normalizer = weighted_logsumexp(
        means.squeeze(), grid_sizes.squeeze(), axis=0
    )  # \log [\sum_n^grid exp(\lambda_{nk}) grid_sizes_n]
    return jnp.exp(means.squeeze() - log_normalizer)

@eqx.filter_jit
@eqx.filter_vmap(in_axes=(0, 0, None, 1))
def _log_likelihood(gp: LogDensityMarkovGP, state: State, x: np.ndarray, y: np.ndarray):
    return gp.log_likelihood(state, y)


@eqx.filter_jit
@eqx.filter_vmap(in_axes=(0, 0, None))
def _predict(gp: LogDensityMarkovGP, state: State, x: np.ndarray):
    return gp.predict(state, x)

@eqx.filter_vmap()
def _f(mean, var):
    return jnp.squeeze(jnp.exp(mean + -0.5 * var))



def _plot_latent_gp(path:pathlib.Path, bin_x:np.ndarray, colors, post_mean, post_var, grid_bounds):
    max_x = bin_x.max()
    test_x = bin_x.squeeze()
    K = post_mean.shape[0]
    predict_mean, predict_var = post_mean.squeeze(), post_var.squeeze()
    lb_lgcp = (predict_mean - np.sqrt(predict_var) * 1.645).squeeze()
    ub_lgcp = (predict_mean + np.sqrt(predict_var) * 1.645).squeeze()
    fig, ax = plt.subplots(K, 2, figsize=(50, 4 * K))
    for k in range(K):
        for i in range(2):
            for bound in grid_bounds:
                ax[k, i].axvline(
                    x=bound[0], linestyle="--", color="gray", linewidth=1
                )
            ax[k, i].plot(test_x, predict_mean[k], color=colors[k])
            ax[k, i].fill_between(
                test_x,
                lb_lgcp[k],
                ub_lgcp[k],
                color=colors[k],
                alpha=0.1,
                label="95% confidence",
            )
        # ax[k, i].set_ylim(-5, np.max(counterM))
        # for x, r in zip(tensor[self.mode_idx], assignment):
        #     ax[r].scatter(x, np.zeros(1))
        ax[k, 0].set_xlim(0, max_x * 1.05)
        ax[k, 1].set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_histgram(path:pathlib.Path, counterM, bin_x:np.ndarray, colors, mode_idx:str):
    fig, ax = plt.subplots(2, 1, figsize=(60, 10))
    num_bins = bin_x.shape[0]
    offset = np.zeros(num_bins)
    K= counterM.shape[1]
    for k in range(K):
        ax[0].bar(
            np.arange(num_bins),
            counterM[:, k],
            color=colors[k],
            bottom=offset,
            label=f"topic {k + 1}"
        )
        offset += counterM[:, k]
    ax[0].set_xticks(np.arange(num_bins)[counterM.sum(axis=1) > 0])
    ax[0].set_xlim(-1, bin_x.shape[0] + 2)
    ax[0].set_xticklabels(bin_x.squeeze()[counterM.sum(axis=1) > 0])
    ax[0].legend()
    ax[0].set_xlabel(mode_idx)
    ax[0].set_ylabel("count")
    ax[1].scatter(bin_x.squeeze(), np.arange(num_bins))
    for i in range(2):
        for label in ax[i].get_xticklabels():  # ラベルごとに
            label.set_rotation(90)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

# region(old)


@eqx.filter_jit
@eqx.filter_vmap(in_axes=(0, 0, None))
def _new_train_data(gp: BaseModel, state: State, x: np.ndarray):
    new_mean, new_cov = gp.predict(state, x)
    # obs_mean, obs_var = latent2measure(
    #     gp.kernel, new_mean, new_cov, jnp.full(new_cov.shape, 1e-10)
    # )
    return (
        gp.update_state(
            state, x, new_mean.reshape(-1, 1, 1), new_cov.reshape(-1, 1, 1)
        ),
        new_mean,
        new_cov,
    )


@eqx.filter_jit
@eqx.filter_vmap(in_axes=(0, 0, None, None, None, None))
def _integral_LGCP(
    gp: BaseModel,
    state: State,
    X: Array,
    W: Array,
    a: float,
    b: float,
):
    S = (b - a) * X / 2 + (b + a) / 2  # Sigma point
    mean, var = gp.predict(state, S)
    return (b - a) * 0.5 * (W @ _f(mean, var))


@eqx.filter_jit
def _lambda_LGCP(
    mean: Float[Array, "topic mode"],
    variances: Float[Array, "topic mode"],
    grid_bounds: Float[Array, "mode 2"],
) -> Float[Array, "mode topic"]:
    bound_diff = jnp.diff(grid_bounds, axis=1).squeeze()
    intensity = (
        jnp.exp(mean.squeeze() + 0.5 * variances.squeeze()) * bound_diff[jnp.newaxis, :]
    )  # \lambda_{nk} = exp(m_{nk} + 0.5*\sigma_{nk}^2) Δ_{n}
    return intensity.T


# endregion (old)
