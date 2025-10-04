from typing import override

import equinox as eqx
import jax.numpy as np
from .inference import VariationalGaussNewton, VariationalInference

# from bayesnewton.kernels import StationaryKernel
from .bayesnewton_utils import (
    process_noise_covariance,
    gaussian_expected_log_lik,
    diag,
    temporal_conditional,
    transpose,
)
from .bayesnewton_ops import rauch_tung_striebel_smoother, kalman_filter
import jax.random as random
from jax.scipy.linalg import cholesky
from jax.lax import scan
from equinox.nn import State
from jax import vmap
from jaxtyping import Float
from jaxopt import LBFGS
from jaxtyping import Array, PRNGKeyArray, Int, Scalar
from .likelihood import Poisson
from .Kernel import StationaryKernel
from .basemodel import BaseModel, BaseState
from .ops import weighted_logsumexp, weighted_softmax, kalman_filter_online


# ==================================
# region(MarkovGaussianProcess)
# =================================
class MarkovGaussianProcess(BaseModel):
    """
    The stochastic differential equation (SDE) form of a Gaussian process (GP) model.
    Implements methods for inference and learning using state space methods, i.e. Kalman filtering and smoothing.
    Constructs a linear time-invariant (LTI) stochastic differential equation (SDE) of the following form:
        dx(t)/dt = F x(t) + L w(t)
              yₙ ~ p(yₙ | f(t_n)=H x(t_n))
    where w(t) is a white noise process and where the state x(t) is Gaussian distributed with initial
    state distribution x(t)~𝓝(0,Pinf).
    """

    spatio_temporal: bool = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)
    """dimension of kernel latent space"""
    minf: Array = eqx.field(static=True)
    """stationary state mean"""

    def __init__(
        self,
        kernel: StationaryKernel,
        likelihood: Poisson,
        X: Float[Array, "N 1"],
        #  Y,
        #  R=None,
    ):
        N = X.shape[0]
        super(MarkovGaussianProcess, self).__init__(kernel, likelihood, X, N)
        self.state_dim = kernel.stationary_covariance().shape[0]
        self.minf = np.zeros((self.state_dim, 1))
        # self.spatio_temporal = np.any(~np.isnan(self.R))
        self.spatio_temporal = False

    def compute_log_lik(
        self, state: State, pseudo_y=None, pseudo_var=None
    ) -> Float[Scalar, "1"]:
        """
        compute log ∫ p(f) N(pseudo_y | f, pseudo_var) df.
        """
        if pseudo_y is None:
            pseudo_y, pseudo_var = self.compute_full_pseudo_lik(state)
        _, _, X, dx = state.get(self.index)
        log_lik, (filter_mean, filter_cov) = kalman_filter(
            dx, self.kernel, pseudo_y, pseudo_var, parallel=self.parallel
        )
        return log_lik  # log p(y_{1:T}) = \sum_t^T[\log |S_t| + (y_t-\hat y_t)S_t^{-1}(y_t-\hat y_t) + \log 2\pi]

    def predict(
        self,
        state: State,
        X=None,
        R=None,
        pseudo_mean=None,
        pseudo_cov=None,
        return_latent=False,
    ):
        """
        predict at new test locations X
        """
        _, _, trainX, dx = state.get(self.index)
        if X is None:
            X = trainX
        elif len(X.shape) < 2:
            X = X[:, None]
        # if R is None:
        #     R = X[:, 1:]
        X = X[:, :1]  # take only the temporal component
        if pseudo_mean is None or pseudo_cov is None:
            pseudo_y, pseudo_var = self.compute_full_pseudo_lik(state)

        (smooth_mean, smoother_cov, gain, loglik, Ss, vs) = self._forward_backward(
            state, pseudo_mean, pseudo_cov, return_latent=True
        )
        # add dummy states at either edge
        inf = 1e10 * np.ones_like(trainX[0, :1])
        X_aug = np.block([[-inf], [trainX[:, :1]], [inf]])
        # predict the state distribution at the test time steps:
        state_mean, state_cov = temporal_conditional(
            X_aug, X, smooth_mean, smoother_cov, gain, self.kernel
        )
        if return_latent:
            return state_mean.squeeze(), state_cov.squeeze()

        # extract function values from the state:
        H = self.kernel.measurement_model()
        if self.spatio_temporal:
            # TODO: if R is fixed, only compute B, C once
            B, C = self.kernel.spatial_conditional(R, predict=True)
            W = B @ H
            test_mean = W @ state_mean
            test_var = W @ state_cov @ transpose(W) + C
        else:
            test_mean, test_var = H @ state_mean, H @ state_cov @ transpose(H)

        # if np.squeeze(test_var).ndim > 2:  # deal with spatio-temporal case (discard spatial covariance)
        if self.spatio_temporal:
            test_var = diag(
                test_var
            )  # deal with spatio-temporal case (discard spatial covariance)
        return np.squeeze(test_mean), np.squeeze(test_var)

    @override
    def update_posterior(self, state: State):
        """
        Compute the posterior via filtering and smoothing
        """
        _, _, X, dx = state.get(self.index)
        pseudo_y, pseudo_var = self.compute_full_pseudo_lik(state)
        (smooth_mean, smoother_cov, gain, loglik, Ss, vs) = self._forward_backward(
            state, pseudo_y, pseudo_var, return_latent=False
        )
        state = state.set(self.index, BaseState(smooth_mean, smoother_cov, X, dx))
        return state

    @override
    def prior_sample(self, key: PRNGKeyArray, X: Array, num_samps=1):
        """
        Sample from the model prior f~N(0,K) multiple times using a nested loop.
        :param num_samps: the number of samples to draw [scalar]
        :param X: the input locations at which to sample (defaults to training inputs) [N, 1]
        :param seed: the random seed for sampling
        :return:
            f_samples: the prior samples [num_samps, N, 1]
            latent_samples: the prior samples [num_samps, N, func_dim]
        """
        dt = np.concatenate([np.array([0.0]), np.diff(np.sort(X))])
        sd = self.state_dim
        H = self.kernel.measurement_model()
        Pinf = self.kernel.stationary_covariance()
        As = vmap(self.kernel.state_transition)(dt)
        Qs = vmap(process_noise_covariance, [0, None])(As, Pinf)
        jitter = 1e-8 * np.eye(sd)
        f0 = np.zeros([dt.shape[0], sd, 1])

        def draw_full_sample(carry_, _):
            f_sample_i, i, key = carry_
            key, subkey = random.split(key)
            m0 = cholesky(Pinf, lower=True) @ random.normal(subkey, shape=(sd, 1))

            def sample_one_time_step(carry, inputs):
                m, k, key = carry
                key, subkey = random.split(key)
                A, Q = inputs
                chol_Q = cholesky(Q + jitter, lower=True)  # <--- can be a bit unstable
                q_samp = chol_Q @ random.normal(subkey, shape=(sd, 1))
                m = A @ m + q_samp
                f = H @ m
                return (m, k + 1, key), (f, m)

            (_, _, key), (f_sample, latent_sample) = scan(
                f=sample_one_time_step, init=(m0, 0, key), xs=(As, Qs)
            )

            return (f_sample, i + 1, key), (f_sample, latent_sample)

        (_, _, key), (f_samples, latent_samples) = scan(
            f=draw_full_sample, init=(f0, 0, key), xs=np.zeros(num_samps)
        )
        return f_samples, latent_samples

    @override
    def compute_kl(self, state: State) -> Float[Scalar, "1"]:
        """
        KL divergence between the approximate posterior q(u) and the prior p(u)
        """
        pseudo_y, pseudo_var = self.compute_full_pseudo_lik(state)
        log_lik_pseudo = self.compute_log_lik(state, pseudo_y, pseudo_var)
        post_mean, post_var, X, dx = state.get(self.index)
        expected_density_pseudo = vmap(gaussian_expected_log_lik)(  # parallel operation
            pseudo_y,
            post_mean,
            post_var,
            pseudo_var,
        )
        kl = (
            np.sum(expected_density_pseudo) - log_lik_pseudo
        )  # KL[approx_post || prior]
        return kl

    def posterior_sample(
        self,
        state: State,
        key: PRNGKeyArray,
        X: Array | None = None,
        num_samps=1,
        seed=0,
    ):
        """
        Sample from the posterior at the test locations.
        Posterior sampling works by smoothing samples from the prior using the approximate Gaussian likelihood
        model given by the pseudo-likelihood, 𝓝(f|μ*,σ²*), computed during training.
         - draw samples (f*) from the prior
         - add Gaussian noise to the prior samples using auxillary model p(y*|f*) = 𝓝(y*|f*,σ²*)
         - smooth the samples by computing the posterior p(f*|y*)
         - posterior samples = prior samples + smoothed samples + posterior mean
                             = f* - E[p(f*|y*)] + E[p(f|y)]
        See Arnaud Doucet's note "A Note on Efficient Conditional Simulation of Gaussian Distributions" for details.

        :param X: the sampling input locations [N, 1]
        :param num_samps: the number of samples to draw [scalar]
        :param seed: the random seed for sampling
        :return:
            the posterior samples [N_test, num_samps]
        """
        post_mean, post_var, train_x, dx = state.get(self.index)
        pseudo_mean, pseudo_cov, _, _ = self.pseudo_likelihood.state(state)
        key, subkey1, subkey2 = random.split(key, 3)
        N = train_x.shape[0]
        if X is None:
            train_ind = np.arange(N)
            test_ind = train_ind
        else:
            if X.ndim < 2:
                X = X[:, None]
            X = np.concatenate([train_x, X])
            X, ind = np.unique(X, return_inverse=True)
            train_ind, test_ind = ind[:N], ind[N:]
        post_mean, _ = self.predict(state, X)  # p(f|y)
        prior_samp, _ = self.prior_sample(
            subkey1, X, num_samps=num_samps
        )  # f* ~ N(0, K)
        lik_chol = np.tile(cholesky(pseudo_cov, lower=True), [num_samps, 1, 1, 1])
        prior_samp_train = prior_samp[:, train_ind]
        prior_samp_y = prior_samp_train + lik_chol @ random.normal(
            subkey2, shape=prior_samp_train.shape
        )  # p(y*|f*) = 𝓝(y*|f*,σ²*)

        def smooth_prior_sample(i, prior_samp_y_i):
            smoothed_sample, _ = self.predict(
                state, X, pseudo_mean=prior_samp_y_i, pseudo_cov=pseudo_cov
            )
            return i + 1, smoothed_sample

        _, smoothed_samples = scan(
            f=smooth_prior_sample, init=0, xs=prior_samp_y
        )  # p(f*|y*)
        # posterior samples = f* - E[p(f*|y*)] + E[p(f|y)]
        return (prior_samp[..., 0, 0] - smoothed_samples + post_mean[None])[:, test_ind]

    # region (private method)
    def _forward(self, state: State, pseudo_mean=None, pseudo_cov=None):
        if pseudo_mean is None or pseudo_cov is None:
            pseudo_mean, pseudo_cov = self.compute_full_pseudo_lik(state)
        _, _, X, dx = state.get(self.index)
        Pinf = self.kernel.stationary_covariance()
        minf = np.zeros([Pinf.shape[0], 1])
        return kalman_filter_online(
            dx, self.kernel, pseudo_mean, pseudo_cov, minf, Pinf, parallel=self.parallel
        )

    def _forward_backward(
        self,
        state: State,
        pseudo_mean=None,
        pseudo_cov=None,
        return_latent: bool = False,
    ):
        _, _, X, dx = state.get(self.index)
        log_lik, Ss, vs, (filter_mean, filter_cov) = self._forward(
            state, pseudo_mean, pseudo_cov
        )
        dx = np.concatenate([dx[1:], np.zeros((1, 1))], axis=0)
        smooth_mean, smoother_cov, gain = rauch_tung_striebel_smoother(
            dx,
            self.kernel,
            filter_mean,
            filter_cov,
            parallel=self.parallel,
            return_full=return_latent,
        )
        return (
            smooth_mean,
            smoother_cov,
            gain,
            log_lik,
            Ss,
            vs,
        )

    # endregion (private method)


# endregion(MarkovGaussianProcess)


class VariationalMarkovianGP(MarkovGaussianProcess, VariationalInference):
    def __init__(
        self,
        kernel: StationaryKernel,
        likelihood,
        X: Float[Array, "N 1"],
        means: Float[Array, "N 1"],
    ):
        super(VariationalMarkovianGP, self).__init__(kernel, likelihood, X, means)


class VariationalGaussNewtonMarkovianGP(MarkovGaussianProcess, VariationalGaussNewton):
    def __init__(
        self,
        kernel: StationaryKernel,
        likelihood,
        X: Float[Array, "N 1"],
        means: Float[Array, "N 1"],
    ):
        super(VariationalGaussNewtonMarkovianGP, self).__init__(
            kernel, likelihood, X, means
        )


class LogDensityMarkovGP(MarkovGaussianProcess):
    def __init__(
        self,
        kernel: StationaryKernel,
        likelihood,
        X: Float[Array, "N 1"],
    ):
        super().__init__(kernel, likelihood, X)

    def inference(self, state: State, lr, Y: Int[Array, "num_bins"]):
        mean_f, cov_f, _ = self.conditional_posterior_to_data(state)
        mean_f, cov_f = mean_f.squeeze(), cov_f.squeeze()
        _, _, X, dx = state.get(self.index)
        N = np.sum(Y)  # data num
        # region (↓ Naive: Gaussian Processes for Machine Learning)
        # new_mean, n_iter, L, a, jacobian, W, R = newton_loop(
        #     X, Y, N, K, self.likelihood.binsize, mean_f
        # )
        # newK = K - K @ R @ cho_solve((L, False), R @ K)
        # endregion()

        _, pseudo_var = self.compute_full_pseudo_lik(state)
        Y = Y.squeeze()
        binsize = self.likelihood.binsize.squeeze()

        def _grad_prior_loglik(S: Float[Scalar, "1 1"], v: Float[Scalar, "1"]):
            return (v / S).squeeze()

        def objective_and_grad(f):
            # smoothing result ：m_smooths = E[z_n|f], V_smooths = Var[z_n|f]
            s = weighted_softmax(f, binsize).squeeze()

            prior_log_lik, Ss, vs, (filter_mean, filter_cov) = self._forward(
                state, f.reshape(-1, 1, 1), pseudo_var
            )
            loglik = _weighted_softmax_llh(Y, binsize, f, N)
            loss = -loglik - prior_log_lik  # - log (y|f) + 1/2fK^{-1}f

            grad_prior = -vmap(_grad_prior_loglik)(Ss, vs)  # ∇log (f)
            grad_log_lik = Y - N * s  # ∇log (y|f)  + ∇log (f)
            grad_loss = - grad_prior - grad_log_lik
            return loss, grad_loss

        lbfgs = LBFGS(objective_and_grad, value_and_grad=True, maxiter=8, tol=0.1)
        solution = lbfgs.run(mean_f.squeeze())
        new_mean = solution.params
        # s = weighted_softmax(new_mean, binsize).squeeze()
        # W = N * (np.diag(s) - np.outer(s, s))
        newvar = pseudo_var
        # update pseudo_likelihood
        old_mean, old_var, nat1, nat2 = self.pseudo_likelihood.state(state)
        state = self.pseudo_likelihood.update_mean_cov(
            state, (1 - lr) * old_mean + lr * new_mean.reshape(-1, 1, 1), newvar
        )
        diff1 = np.mean(np.abs(new_mean - old_mean))
        diff2 = np.zeros(1)
        state = self.update_posterior(state)  # kalman filter
        post_mean, post_var, _, _ = state.get(self.index)
        jacobian = np.zeros(1)  # TODO. temporary value
        W = np.zeros(1)
        return (post_mean, post_var, jacobian, -W, state), (diff1, diff2)

    def log_likelihood(self, state: State, Y: Array, cubature=None, **kwargs):
        """compoute log likelihood."""
        mean_f, cov_f, W = self.conditional_posterior_to_data(state)
        N = Y.sum()
        ell = _weighted_softmax_llh(
            Y.squeeze(), self.likelihood.binsize.squeeze(), mean_f.squeeze(), N
        )
        # _, _, X, dx = state.get(self.index)
        # pseudo_mean, pseudo_cov = self.compute_full_pseudo_lik(state)
        # log_lik, (filter_mean, filter_cov) = kalman_filter(
        #     dx, self.kernel, pseudo_mean, pseudo_cov, parallel=self.parallel
        # )
        # return np.sum(ell) + np.sum(log_lik)
        return np.sum(ell)

    def energy(self, state: State, Y: Array, cubature=None, **kwargs):
        """compoute log likelihood."""
        _, _, X, dx = state.get(self.index)
        pseudo_y, pseudo_var = self.compute_full_pseudo_lik(state)
        (smooth_mean, smoother_cov, gain, loglik, Ss, vs) = self._forward_backward(
            state, pseudo_y, pseudo_var, return_latent=False
        )
        N = Y.sum()
        ell = _weighted_softmax_llh(
            Y.squeeze(), self.likelihood.binsize.squeeze(), smooth_mean.squeeze(), N
        )
        return -np.sum(ell) - np.sum(loglik)


def _weighted_softmax_llh(Y: Array, binsize: Array, f: Array, N: int):
    ZERO = 1e-8
    f = f.squeeze()
    return np.dot(Y, f + np.log(binsize + ZERO)) - N * weighted_logsumexp(f, binsize)
