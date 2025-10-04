import numba
import numpy as np
from jaxtyping import Float, Int
from numba import float64, int32
from polyagamma import random_polyagamma
from scipy.special import gammaln, digamma
from pathlib import Path
from scipy import sparse
import pandas as pd
import math

ZERO = 1.0e-8


def save_sparse_matrix(path: Path, matrix: np.ndarray, row_name: str, col_name: str, row_labels:list[str] |None = None):
    sparse_matrix = sparse.coo_matrix(matrix)
    if row_labels is None:
        df = pd.DataFrame(
            {
                row_name: sparse_matrix.row,
                col_name: sparse_matrix.col,
                "value": sparse_matrix.data,
            }
        )
    else:
        df = pd.DataFrame(
            {
                row_name: sparse_matrix.row,
                col_name: sparse_matrix.col,
                f"{row_name}_labels": np.array(row_labels)[sparse_matrix.row],
                "value": sparse_matrix.data,
            }
        )
    df.to_csv(path, index=False)


def _softmax(x):
    # オーバーフロー対策として最大値を引く
    max_x = np.max(x)
    exps = np.exp(x - max_x)
    sum_exps = np.sum(exps)
    return exps / (sum_exps + ZERO)


def _multi_digamma(value):
    """多変量対応番のdigamma"""
    return np.array([digamma(val + ZERO) for val in value])


@numba.njit(float64(float64[:]))
def _logsumexp(x):
    a = np.max(x)
    sum_exp = np.sum(np.exp(x - a))
    return a + np.log(sum_exp)


# @numba.njit(float64[:](float64[:, :]))
# def logsumexp_batch(X):
#     n, d = X.shape
#     result = np.empty(n, dtype=np.float64)
#     for i in range(n):
#         a = np.max(X[i])
#         s = np.sum(np.exp(X[i] - a))
#         result[i] = a + np.log(s)
#     return result

# @numba.njit
# def log_multi_beta(param: np.ndarray):
#     """
#     Logarithm of the multivariate beta function.
#     """
#     ZERO = 1.0e-8
#     # param is assumed to be a vector
#     return np.sum(np.array([gammaln(a + ZERO) for a in param])) - gammaln(
#         np.sum(param) + ZERO
#     )


def log_multi_beta(param: np.ndarray):
    """
    Logarithm of the multivariate beta function.
    """
    ZERO = 1.0e-8
    return np.sum(gammaln(param +ZERO)) - gammaln(np.sum(param)+ ZERO)

def softmax_multi_posterior(
    mean: Float[np.ndarray, "K"],
    var_diag: Float[np.ndarray, "K"],
    count: Int[np.ndarray, "K"],
) -> tuple[Float[np.ndarray, "K"], Float[np.ndarray, "K"]]:
    # Scalable Inference for Logistic-Normal Topic Modelsより
    N = count.sum()
    if N == 0:
        N += 3e-4
    K = mean.shape[0]
    # xi = np.array([logsumexp_exclude_k(mean, k) for k in range(K)])
    xi = logsumexp_exclude_self(mean)
    rho = mean - xi
    omega = random_polyagamma(N, rho)
    inv_var_diag = 1.0 / (var_diag + ZERO)
    new_var_diag = 1.0 / (inv_var_diag + omega + ZERO)
    new_mean = np.diag(new_var_diag) @ (
        np.diag(inv_var_diag) @ mean + count - N / 2 + np.diag(omega) @ xi
    )
    return new_mean, new_var_diag


@numba.njit(float64(int32))
def log_s(x: int):
    """calc universal code length log*(x)"""
    if x == 0:
        return 0
    return 2.0 * np.log2(x) + 1


def logsumexp_exclude_k(vec, k):
    """要素kを除いたlogsumexpを計算"""
    max_val = -np.inf
    for i in range(vec.shape[0]):
        if i != k and vec[i] > max_val:
            max_val = vec[i]

    total = 0.0
    for i in range(vec.shape[0]):
        if i != k:
            total += np.exp(vec[i] - max_val)

    return max_val + np.log(total)


def logsumexp_exclude_self(vec):
    """Calculate the logsumexp excluding itself for each element"""
    new_vec = np.zeros_like(vec)
    for k in range(vec.shape[0]):
        new_vec[k] = logsumexp_exclude_k(vec, k)
    return new_vec
