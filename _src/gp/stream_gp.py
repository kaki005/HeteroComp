import logging

import equinox as eqx
import jax
import jax.numpy as jnp
from .bayesnewton_ops import rauch_tung_striebel_smoother, kalman_filter
from .bayesnewton_utils import (
    process_noise_covariance,
    temporal_conditional,
    mvn_logpdf, transpose
)
from beartype import beartype as typechecker
from jax.lib import xla_bridge
from jaxtyping import Float, jaxtyped, Scalar, Array
from .Kernel import StationaryKernel
from .ops import kalman_filter_online


@jaxtyped(typechecker=typechecker)
class MarkovGP(eqx.Module):
    """
    The stochastic differential equation (SDE) form of a Gaussian process (GP) model.
    Implements methods for inference and learning using state space methods, i.e. Kalman filtering and smoothing.
    Constructs a linear time-invariant (LTI) stochastic differential equation (SDE) of the following form:
        dx(t)/dt = F x(t) + L w(t)
              yₙ ~ p(yₙ | f(t_n)=H x(t_n))
    where w(t) is a white noise process and where the state x(t) is Gaussian distributed with initial
    state distribution x(t)~𝓝(0,Pinf).
    """

    func_dim: int = eqx.field(static=True)
    """dim of latent dimensions"""
    kernel: StationaryKernel
    """kernel"""
    parallel: bool = eqx.field(static=True)
    """flag to switch between parallel and sequential implementation of Kalman filter"""
    logger: logging.Logger = eqx.field(static=True)

    def __init__(self, kernel: StationaryKernel):
        H = kernel.measurement_model()
        self.func_dim = int(H.shape[1])
        self.kernel = kernel
        self.parallel = bool(xla_bridge.get_backend().platform == "gpu")
        self.logger = logging.getLogger(__name__)

    def predict(
        self, trainT, testT, m0, P0, obs_means, obs_vars, R=None, return_latent=False
    ):
        """
        predict at new test locations X
        """
        dts = jnp.concatenate([jnp.zeros((1, 1)), jnp.diff(trainT, axis=0)])
        smooth_mean, smoother_cov, gains = self._forward_backward(
            m0, P0, obs_means, obs_vars, dts
        )
        # add dummy states at either edge
        inf = 1e10 * jnp.ones_like(trainT[0, :1])
        T_aug = jnp.block([[-inf], [trainT[:, :1]], [inf]])
        # predict the state distribution at the test time steps:
        state_mean, state_cov = temporal_conditional(
            T_aug, testT, smooth_mean, smoother_cov, gains, self.kernel
        )
        if return_latent:
            return state_mean.squeeze(), state_cov.squeeze()
        H = self.kernel.measurement_model()
        # if self.spatio_temporal:
        #     # TODO: if R is fixed, only compute B, C once
        #     B, C = self.kernel.spatial_conditional(R, predict=True)
        #     W = B @ H
        #     test_mean = W @ state_mean
        #     test_var = W @ state_cov @ transpose(W) + C
        # else:
        test_mean, test_var = H @ state_mean, H @ state_cov @ transpose(H)

        return test_mean.squeeze(), test_var.squeeze()

    def compute_prior(self, mean, variance, dts):
        return self._predict_vmap(dts, mean, variance)

    def compute_posterior(
        self,
        obs_mean: Float[Array, "N 1"],
        obs_variances: Float[Array, "N 1"],
        m0: Float[Array, "N 1"],
        P0: Float[Array, "N 1"],
        dts: Float[jax.numpy.ndarray, "N"],
    ):
        ZERO = 1e-8  # 分散がnanになるのを防ぐ閾値
        post_means, post_covs, gains = self._forward_backward(
            m0, P0, obs_mean, obs_variances, dts
        )
        means, covs = self._latent2Obs(post_means, post_covs, obs_variances)
        covs = jnp.where(covs < ZERO, ZERO, covs)  # 分散が小さくならないようにする
        return (
            means.reshape(-1),
            covs.reshape(-1),
            post_means.reshape(-1, self.func_dim),
            post_covs.reshape(-1, self.func_dim, self.func_dim),
        )

    def log_likelihood(
        self,
        obs_mean: Float[Array, "N 1"],
        obs_variances: Float[Array, "N 1"],
        m0: Float[Array, "N 1"],
        P0: Float[Array, "N 1"],
        dts: Float[Array, "N"],
    ) -> Float[Scalar, "1"]:
        ell, Ss, vs, (filt_means, filt_covs) = kalman_filter_online(
            dts,
            self.kernel,
            obs_mean.reshape(-1, 1),
            obs_variances.reshape(-1, 1, 1),
            m0,
            P0,
            parallel=self.parallel,
        )  # predict and filter
        # log p(y_{1:T}) = \sum_t^T[\log |S_t| + (y_t-\hat y_t)S_t^{-1}(y_t-\hat y_t) + \log 2\pi]
        # S_t は予測分散
        return jnp.sum(ell)

    def anomaly_score(
        self,
        obs_mean: Float[Array, "N 1"],
        obs_variances: Float[Array, "N 1"],
        m0: Float[Array, "N 1"],
        P0: Float[Array, "N 1"],
        dts: Float[Array, "N"],
    ):
        # TODO: 未完成
        post_means, post_covs, gains = self._forward_backward(
            m0, P0, obs_mean, obs_variances, dts
        )
        pred_mean, pred_variances = self._latent2Obs(
            post_means, post_covs, obs_variances
        )
        return _anomaly_vmap(obs_mean, pred_mean, pred_variances)

    # region (private method)

    def _forward(
        self,
        m0: Float[Array, "N 1"],
        P0: Float[Array, "N 1"],
        obs_mean: Float[Array, "N 1"],
        obs_variances: Float[Array, "N 1"],
        dts: Float[Array, "N"],
    ):
        ZERO = 1e-8  # 分散がnanになるのを防ぐ閾値
        obs_variances = jnp.where(
            obs_variances < ZERO, ZERO, obs_variances
        )  # 分散が小さくならないようにする
        ell, Ss, vs, (filt_means, filt_covs) = kalman_filter_online(
            dts,
            self.kernel,
            obs_mean.reshape(-1, 1),
            obs_variances.reshape(-1, 1, 1),
            m0,
            P0,
            parallel=self.parallel,
        )  # predict and filter
        return ell, Ss, vs, (filt_means, filt_covs)

    def _forward_backward(
        self,
        m0: Float[Array, "N 1"],
        P0: Float[Array, "N 1"],
        obs_mean: Float[Array, "N 1"],
        obs_variances: Float[Array, "N 1"],
        dts: Float[Array, "N"],
    ):
        ell, Ss, vs, (filt_means, filt_covs) = self._forward(
            m0, P0, obs_mean, obs_variances, dts
        )
        latent_mean, latent_cov, gains = rauch_tung_striebel_smoother(
            dts,
            self.kernel,
            filt_means,
            filt_covs,
            return_full=True,
            parallel=self.parallel,
        )
        return latent_mean, latent_cov, gains

    def _latent2Obs(self, latent_means, latent_cov, obs_variances):
        return latent2measure(self.kernel, latent_means, latent_cov, obs_variances)

    @eqx.filter_vmap(in_axes=(None, 0, None, None))
    def _predict_vmap(
        self, dt, m0, P0
    ) -> tuple[Float[Scalar, "1"], Float[Array, "1 1"]]:
        """
        predict at new test locations.
        """
        A = self.kernel.state_transition(dt)
        Q = process_noise_covariance(A, self.kernel.stationary_covariance())
        H = self.kernel.measurement_model()
        pred_mean = A @ m0
        pred_cov = A @ P0 @ transpose(A) + Q
        jitter = 1e-6  # 観測ノイズ
        obs_mean, obs_var = H @ pred_mean, H @ pred_cov @ transpose(H) + jitter
        return obs_mean.squeeze(), obs_var.squeeze()

    # endregion (private method)


@eqx.filter_vmap(in_axes=(None, 0, 0, 0))
def latent2measure(kernel: StationaryKernel, latent_means, latent_cov, obs_variance):
    H = kernel.measurement_model()
    return H @ latent_means, H @ latent_cov @ transpose(H) + obs_variance


@eqx.filter_vmap
def vmap_llh_transition(z_t, z_t_minus1, A, Q):
    return mvn_logpdf(z_t, A @ z_t_minus1, Q)


@eqx.filter_vmap(in_axes=(0, None))
def noise_covariance_vmap(A, Pinf):
    return process_noise_covariance(A, Pinf)


@eqx.filter_vmap
def _anomaly_vmap(obs_mean, pred_mean, out_var):
    # return jnp.squeeze((obs_mean - pred_mean)*(obs_mean - pred_mean) /out_var + jnp.log(jnp.abs(2 * jnp.pi *out_var)))
    return jnp.squeeze((obs_mean - pred_mean) * (obs_mean - pred_mean) / out_var)
