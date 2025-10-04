import math

import jax.numpy as np
import equinox as eqx
from jax import jacrev, vmap
from jax.scipy.linalg import cholesky, inv
from jax.scipy.special import gammaln
from jaxtyping import Array
from .bayesnewton_cubature import gauss_hermite, cho_factor
from .bayesnewton_utils import sigmoid, sigmoid_diff, softplus
from jax.lax import stop_gradient

LOG2PI = math.log(2 * math.pi)


def log_density_cubature(likelihood, y, mean, cov, bin_index, cubature=None):
    """
    logZₙ = log ∫ p(yₙ|fₙ) N(fₙ|mₙ,vₙ) dfₙ
    :param likelihood: the likelihood model
    :param y: observed data (yₙ) [scalar]
    :param mean: cavity mean (mₙ) [scalar]
    :param cov: cavity covariance (cₙ) [scalar]
    :param cubature: the function to compute sigma points and weights to use during cubature
    :return:
        lZ: the log density, logZₙ  [scalar]
    """
    if cubature is None:
        x, w = gauss_hermite(mean.shape[0])  # Gauss-Hermite sigma points and weights
    else:
        x, w = cubature(mean.shape[0])
    cov = (cov + cov.T) / 2
    cav_cho, low = cho_factor(cov, lower=True)
    # fsigᵢ=xᵢ√cₙ + mₙ: scale locations according to cavity dist.
    sigma_points = cav_cho @ np.atleast_2d(x) + mean
    # pre-compute wᵢ p(yₙ|xᵢ√(2vₙ) + mₙ)
    weighted_likelihood_eval = w * likelihood.evaluate_likelihood(
        y, sigma_points, bin_index
    )
    # Compute partition function via cubature:
    # Zₙ = ∫ p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ ≈ ∑ᵢ wᵢ p(yₙ|fsigᵢ)
    Z = np.sum(weighted_likelihood_eval)
    lZ = np.log(np.maximum(Z, 1e-8))
    return lZ


# region (link function)
class LinkFunc(eqx.Module):
    def __call__(self, x):
        return self.link_fn(x)

    def link_fn(self, x):
        return 0

    def dlink_fn(self, x):
        return 0

    def d2link_fn(self, x):
        return 0


class LinkExp(LinkFunc):
    def link_fn(self, x):
        return np.exp(x)

    def dlink_fn(self, x):
        return np.exp(x)

    def d2link_fn(self, x):
        return np.exp(x)


class LinkLogistic(LinkFunc):
    def link_fn(self, x):
        return softplus(x)

    def dlink_fn(self, x):
        return sigmoid(x)

    def d2link_fn(self, x):
        return sigmoid_diff(x)


# endregion (link function)


# ================================
# region (Likelihood)
# ================================
class Likelihood(eqx.Module):
    def evaluate_likelihood(self, y, f, bin_index: int):
        """calc likelihood.

        Args:
            y (Array): observation
            f (Array): prior

        Returns:
            float: likelihood
        """
        return np.exp(self.evaluate_log_likelihood(y, f, bin_index))

    def observation_model(self, f, sigma, bin_index: int):
        """
        TODO: sort out broadcasting so we don't need this additional function (only difference is the transpose)
        The implicit observation model is:
            h(fₙ,rₙ) = E[yₙ|fₙ] + √Cov[yₙ|fₙ] σₙ
        """
        conditional_expectation, conditional_covariance = self.conditional_moments(
            f, bin_index
        )
        obs_model = (
            conditional_expectation
            + cholesky(conditional_covariance.T, lower=True) @ sigma
        )
        return np.squeeze(obs_model)

    def gauss_newton(self, y, f, bin_index: int):
        """
        The generalised Gauss-Newton method.
        This is equivalent to the approximation made by the extended Kalman smoother.
        When the gradient of the normaliser and the conditional expectation are both zero, this
        method will match partial_gauss_newton_normalised() and gauss_newton_normalised() exactly.
        """
        y = y.reshape(-1, 1)
        E, C = self.conditional_moments(f, bin_index)
        C = C.reshape(C.shape[0], C.shape[0])

        # --- apply mask ---
        mask = np.squeeze(np.isnan(y))
        maskv = mask.reshape(-1, 1)
        # build a mask
        y = np.where(maskv, E, y)
        C_masked = np.where(
            maskv + maskv.T, 0.0, C
        )  # ensure masked entries are independent
        C = np.where(np.diag(mask.reshape(-1)), 1, C_masked)  # ensure cholesky passes

        cholC = cholesky(C, lower=True)
        V = inv(cholC) @ (y - E)  # cannot use a solve here since cholC is triangular
        J = self.generalised_gauss_newton_residual_jacobian(
            f, cholC, bin_index
        )  # inv(cholC) @ gradE  # residual Jacobian
        # H = self.generalised_gauss_newton_residual_hessian(f, cholC)  # inv(cholC) @ hessianE  # residual Hessian
        log_target = -0.5 * V.T @ V
        jacobian = J.T @ V
        hessian_approx = -J.T @ J
        # second_order_term = -H.T * V
        return log_target, jacobian, hessian_approx  # , second_order_term

    def generalised_gauss_newton_residual_jacobian(self, f, cholC, bin_idx: int):
        return inv(cholC) @ np.squeeze(
            jacrev(self.conditional_moments)(f, bin_idx)[0], axis=(1, -1)
        )  # TODO: is this correct?

    def log_density(self, y, mean, cov, bin_index, cubature=None):
        """
        calc logZₙ = log ∫ p(yₙ|fₙ) N(fₙ|mₙ,vₙ) dfₙ
        :param likelihood: the likelihood model
        :param y: observed data (yₙ) [scalar]
        :param mean: cavity mean (mₙ) [scalar]
        :param cov: cavity covariance (cₙ) [scalar]
        :param cubature: the function to compute sigma points and weights to use during cubature
        :return:
        lZ: the log density, logZₙ  [scalar]
        """
        return log_density_cubature(self, y, mean, cov, bin_index, cubature)

    def variational_expectation(self, y, m, v, bin_index: int, cubature=None):
        """
        Most likelihoods factorise across data points. For multi-latent models, a custom method must be implemented.
        """

        # align shapes and compute mask
        y = y.reshape(-1, 1, 1)
        m = m.reshape(-1, 1, 1)
        v = np.diag(v).reshape(-1, 1, 1)
        mask = np.isnan(y)
        y = np.where(mask, m, y)

        # compute variational expectations and their derivatives
        var_exp, dE_dm, d2E_dm2 = vmap(
            self.variational_expectation_, (0, 0, 0, None, None)
        )(y, m, v, bin_index, cubature)

        # apply mask
        var_exp = np.where(np.squeeze(mask), 0.0, np.squeeze(var_exp))
        dE_dm = np.where(mask, np.nan, dE_dm)
        d2E_dm2 = np.where(mask, np.nan, d2E_dm2)

        return (
            var_exp,
            np.squeeze(dE_dm, axis=2),
            np.diag(np.squeeze(d2E_dm2, axis=(1, 2))),
        )

    def variational_gauss_newton(self, y, mean, cov, bin_index: int, cubature=None):
        if cubature is None:
            x, w = gauss_hermite(
                mean.shape[0]
            )  # Gauss-Hermite sigma points and weights
        else:
            x, w = cubature(mean.shape[0])
        w = w[:, None, None]
        sigma_points = cholesky(cov, lower=True) @ np.atleast_2d(x) + mean
        log_target, jacobian, hessian_approx = vmap(
            self.gauss_newton, in_axes=(None, 1, None)
        )(y, sigma_points[..., None], bin_index)
        return (
            np.sum(w * log_target),
            np.sum(w * jacobian, axis=0),
            np.sum(w * hessian_approx, axis=0),
            # np.sum(w * second_order_term, axis=0)
        )

    def evaluate_log_likelihood(self, y, f, bin_index: int):
        raise NotImplementedError

    def variational_expectation_(
        self, y, post_mean, post_cov, bin_index: int, cubature=None
    ):
        raise NotImplementedError

    def conditional_moments(self, f, bin_index: int):
        raise NotImplementedError


# endregion (Likelihood)


# ================================
# region (Poisson)
# ================================
class Poisson(Likelihood):
    """
    TODO: tidy docstring
    The Poisson likelihood:
        p(yₙ|fₙ) = Poisson(fₙ) = μʸ exp(-μ) / yₙ!
    where μ = g(fₙ) = mean = variance is the Poisson intensity.
    yₙ is non-negative integer count data.
    No closed form moment matching is available, so we default to using cubature.

    Letting Zy = gamma(yₙ+1) = yₙ!, we get log p(yₙ|fₙ) = log(g(fₙ))yₙ - g(fₙ) - log(Zy)
    The larger the intensity μ, the stronger the likelihood resembles a Gaussian
    since skewness = 1/sqrt(μ) and kurtosis = 1/μ.
    Two possible link functions:
    'exp':      link(fₙ) = exp(fₙ),         we have p(yₙ|fₙ) = exp(fₙyₙ-exp(fₙ))           / Zy.
    'logistic': link(fₙ) = log(1+exp(fₙ))), we have p(yₙ|fₙ) = logʸ(1+exp(fₙ)))(1+exp(fₙ)) / Zy.
    """

    link_fn: LinkFunc
    _binsizes: Array
    # name: str  = eqx.field(static=True)

    def __init__(self, binsizes: np.ndarray, link="exp"):
        """
        :param link: link function, either 'exp' or 'logistic'
        """
        super().__init__()
        if link == "exp":
            self.link_fn = LinkExp()
        elif link == "logistic":
            self.link_fn = LinkLogistic()
        else:
            raise NotImplementedError("link function not implemented")
        self._binsizes = np.array(binsizes)
        # self.name = "Poisson"

    @property
    def binsize(self):
        return stop_gradient(self._binsizes)

    def evaluate_log_likelihood(self, y, f, bin_index: int):
        """
        Evaluate the Poisson log-likelihood:
            log p(yₙ|fₙ) = log Poisson(fₙ) = log(μʸ exp(-μ) / yₙ!)
        for μ = g(fₙ), where g() is the link function (exponential or logistic).
        We use the gamma function to evaluate yₙ! = gamma(yₙ + 1).
        Can be used to evaluate Q cubature points when performing moment matching.
        :param y: observed data (yₙ) [scalar]
        :param f: latent function value (fₙ) [Q, 1]
        :return:
            log Poisson(fₙ) = log(μʸ exp(-μ) / yₙ!) [Q, 1]
        """
        mu = self.link_fn(f) * self.binsize[bin_index]
        return np.squeeze(y * np.log(mu) - mu - gammaln(y + 1))

    def conditional_moments(self, f, bin_index: int):
        """
        The first two conditional moments of a Poisson distribution are equal to the intensity:
            E[yₙ|fₙ] = link(fₙ)
            Var[yₙ|fₙ] = link(fₙ)
        """
        # TODO: multi-dim case
        return self.link_fn(f) * self.binsize[bin_index], self.link_fn(
            f
        ) * self.binsize[bin_index]
        # return self.link_fn(f) * self.binsize, vmap(np.diag, 1, 2)(self.link_fn(f) * self.binsize)

    def analytical_linearisation(self, m, bin_index: int, sigma):
        """
        Compute the Jacobian of the state space observation model w.r.t. the
        function fₙ and the noise term σₙ.
        """
        link_fm = self.link_fn(m) * self.binsize[bin_index]
        dlink_fm = self.link_fn.dlink_fn(m) * self.binsize[bin_index]
        d2link_fm = self.link_fn.d2link_fn(m) * self.binsize[bin_index]
        Jf = np.diag(
            np.squeeze(
                dlink_fm + 0.5 * link_fm**-0.5 * dlink_fm * sigma.reshape(-1, 1),
                axis=-1,
            )
        )
        Hf = np.diag(
            np.squeeze(
                d2link_fm
                - 0.25 * link_fm**-1.5 * dlink_fm**2 * sigma.reshape(-1, 1)
                + 0.5 * link_fm**-0.5 * d2link_fm * sigma.reshape(-1, 1),
                axis=-1,
            )
        )
        Jsigma = np.diag(np.squeeze(link_fm**0.5, axis=-1))
        Hsigma = np.zeros_like(Jsigma)
        return Jf, Hf, Jsigma, Hsigma

    def variational_expectation_(
        self, y, post_mean, post_cov, bin_index: int, cubature=None
    ):
        """
        Computes the "variational expectation", i.e. the
        expected log-likelihood, and its derivatives w.r.t. the posterior mean
        Let a = E[f] = m and b = E[exp(f)] = exp(m+v/2) then
        E[log Poisson(y | exp(f)*binsize)] = Y log binsize  + E[Y * log exp(f)] - E[binsize * exp(f)] - log Y!
                                           = Y log binsize + Y * m - binsize * exp(m + v/2) - log Y!
        :param y: observed data (yₙ) [scalar]
        :param post_mean: posterior mean (mₙ) [scalar]
        :param post_cov: posterior variance (vₙ) [scalar]
        :param cubature: the function to compute sigma points and weights to use during cubature
        :return:
            exp_log_lik: the expected log likelihood, E[log p(yₙ|fₙ)]  [scalar]
            dE_dm: derivative of E[log p(yₙ|fₙ)] w.r.t. mₙ  [scalar]
            d2E_dm2: 2nd derivative of E[log p(yₙ|fₙ)] w.r.t. mₙ  [scalar]
        """
        # TODO: multi-dim case
        exp_mean_cov = self.binsize[bin_index] * np.exp(
            post_mean + post_cov / 2
        )  # λの期待値
        # Compute expected log likelihood:
        exp_log_lik = (
            y * np.log(self.binsize[bin_index])
            + y * post_mean
            - exp_mean_cov
            - gammaln(y + 1.0)
        )  # E[log p(y_n|f_n)]
        # Compute first derivative:
        dE_dm = y - exp_mean_cov  # E[log p(y_n|f_n)]の平均に対する微分
        # Compute second derivative:
        d2E_dm2 = -exp_mean_cov  # E[log p(y_n|f_n)]の平均に対する2階微分
        return exp_log_lik, dE_dm, d2E_dm2.reshape(-1, 1)


# endregion (Poisson)
