import jax.numpy as np
from jax import vmap
from jax.lax import scan, fori_loop, select
from .bayesnewton_ops import _parallel_kf
from .bayesnewton_utils import mvn_logpdf, solve, process_noise_covariance
from jaxtyping import Array, Float
from jax.scipy.special import logsumexp
import equinox as eqx


def cholesky_tridiagonal(
    diag: Float[Array, "N"], subdiag: Float[Array, "N-1"]
) -> tuple[Float[Array, "N"], Float[Array, "N-1"]]:
    """
    Computes Cholesky decomposition of symmetric tridiagonal matrix:
        A = diag + subdiag + subdiag.T = LL^T
    Args:
        diag: (N,) diagonal elements
        subdiag: (N-1,) subdiagonal elements
    Returns:
        l: (N,) diagonal of L
        d: (N-1,) subdiagonal of L (lower triangle)
    """

    def body_fn(carry, i):
        l_prev = carry
        d_i = subdiag[i] / l_prev
        l_i = np.sqrt(diag[i + 1] - d_i**2)
        return l_i, (d_i, l_i)

    l0 = np.sqrt(diag[0])
    _, (D, l_rest) = scan(body_fn, l0, np.arange(diag.shape[0] - 1))
    L = np.concatenate([np.array([l0]), l_rest])
    return L, D


def weighted_logsumexp(x, w, axis=None, keepdims=False):
    # 数値安定化のための補正
    x_max = np.max(x, axis=axis, keepdims=True)
    sum_exp = np.sum(w * np.exp(x - x_max), axis=axis, keepdims=keepdims)
    return np.log(sum_exp) + np.squeeze(x_max, axis=axis if not keepdims else None)


def weighted_softmax(x, w, axis=-1):
    ZERO = 1e-8
    x = np.asarray(x)
    w = np.asarray(w)

    x_max = np.max(x, axis=axis, keepdims=True)
    x_stable = x - x_max
    log_weighted_exp = np.log(w) + x_stable
    log_sum = logsumexp(log_weighted_exp, axis=axis, keepdims=True)
    log_probs = log_weighted_exp - log_sum
    probs = np.exp(log_probs)
    return probs + ZERO


def pivoted_chol(K: Float[Array, "N N"], M: int):
    """
    A simple python function which computes the Pivoted Cholesky decomposition/approximation of positive
    semi-definite operator.
    半正定値行列の近似的なコレスキー分解を行います。
    特に、近似分解において数値的に安定するよう、行のピボット選択（交換）を行いながら分解を進めます。

    Args:
        - K : カーネル共分散行列
        - M: The maximum rank of the approximate decomposition; an integer.
        - err_tol: 許容誤差の閾値です。デフォルトでは 1e-6 となっていますが、現在この引数はコード内で使われていないため、分解終了の基準としては考慮されていません。

    Returns:
        - R, an upper triangular matrix of column dimension equal to the target matrix.
        - pi, the index of the pivots.
    """
    d0 = np.diag(K)  # 対角成分
    N = d0.shape[0]
    used = np.zeros(N, dtype=int)
    pi0 = np.arange(N)
    R0 = np.zeros((M, N))
    print(f"{pi0.shape=}")

    def body_fn(carry, m):
        d, pi, R, used = carry
        # 最大ピボットのインデックスを選択
        pivot = np.argmax(select(used, np.zeros(N), d))
        used = used.at[pivot].set(1)
        pi = pi.at[np.array([m, pivot])].set(
            pi[np.array([pivot, m])]
        )  # ピボットの入れ替え
        # R 行列の更新
        Rmm = np.sqrt(d[pivot])
        R = R.at[m, pivot].set(Rmm)
        Apim = K[pivot, :]

        def update_fn(i, r_d):
            R, d = r_d
            ip = select(m > 0, np.dot(R[:, pi[m]], R[:, pi[i]]), 0.0)
            Rmi = (Apim[pi[i]] - ip) / Rmm
            R = R.at[m, pi[i]].set(Rmi)
            d = d.at[pi[i]].subtract(Rmi**2)
            return R, d

        R, d = fori_loop(m + 1, N, update_fn, (R, d))
        return (d, pi, R, used), None

    # scanでループ展開
    (d_final, pi_final, R_final, used), _ = scan(
        body_fn, (d0, pi0, R0, used), np.arange(M)
    )
    print(f"{pi_final.shape=}")
    return R_final, pi_final[
        :M
    ]  # 分解が終了した時点での R 行列を返します。また、ピボット選択の結果も返す


def kalman_filter_online(
    dt, kernel, y, noise_cov, m0, P0, mask=None, parallel=False, return_predict=False
):
    """
    Run the Kalman filter to get p(fₙ|y₁,...,yₙ).
    Assumes a heteroscedastic Gaussian observation model, i.e. var is vector valued
    :param dt: step sizes [N, 1]
    :param kernel: an instantiation of the kernel class, used to determine the state space model
    :param y: observations [N, D, 1]
    :param noise_cov: observation noise covariances [N, D, D]
    :param mask: boolean mask for the observations (to indicate missing data locations) [N, D, 1]
    :param parallel: flag to switch between parallel and sequential implementation of Kalman filter
    :param return_predict: flag whether to return predicted state, rather than updated state
    :return:
        ell: the log-marginal likelihood log p(y), for hyperparameter optimisation (learning) [scalar]
        means: intermediate filtering means [N, state_dim, 1]
        covs: intermediate filtering covariances [N, state_dim, state_dim]
    """
    if mask is None:
        mask = np.zeros_like(y, dtype=bool)
    Pinf = kernel.stationary_covariance()

    As = vmap(kernel.state_transition)(dt)
    Qs = vmap(process_noise_covariance, [0, None])(As, Pinf)
    H = kernel.measurement_model()

    if parallel:
        ell, means, covs = _parallel_kf(
            As, Qs, H, y, noise_cov, m0, P0, mask, return_predict=return_predict
        )
    else:
        ell, means, covs, Ss, vs = _sequential_kf(
            As, Qs, H, y, noise_cov, m0, P0, mask, return_predict=return_predict
        )
    return ell, Ss, vs, (means, covs)


def _sequential_kf(As, Qs, H, ys, noise_covs, m0, P0, masks, return_predict=False):
    def body(carry, inputs):
        y, A, Q, obs_cov, mask = inputs
        m, P, ell = carry
        m_ = A @ m
        P_ = A @ P @ A.T + Q  # predict共分散

        obs_mean = H @ m_
        HP = H @ P_
        S = HP @ H.T + obs_cov  # predict 観測分散 (innvoationの分散)
        v = y - obs_mean  # innvoation の平均

        ell_n = mvn_logpdf(y, obs_mean, S, mask)
        ell = ell + ell_n

        K = solve(S, HP).T
        m = m_ + K @ v
        P = P_ - K @ HP

        if return_predict:
            return (m, P, ell), (m_, P_, S, v)
        else:
            return (m, P, ell), (m, P, S, v)

    (_, _, loglik), (fms, fPs, Ss, vs) = scan(
        f=body, init=(m0, P0, 0.0), xs=(ys, As, Qs, noise_covs, masks)
    )
    return loglik, fms, fPs, Ss, vs
