# from bayes newton
from jax import vmap, grad, jacrev
import jax.numpy as np
from jax.scipy.linalg import cholesky, cho_factor
from numpy.polynomial.hermite import hermgauss
import itertools

def mvhermgauss(H: int, D: int):
    """
    This function is adapted from GPflow: https://github.com/GPflow/GPflow

    Return the evaluation locations 'xn', and weights 'wn' for a multivariate
    Gauss-Hermite quadrature.

    The outputs can be used to approximate the following type of integral:
    int exp(-x)*f(x) dx ~ sum_i w[i,:]*f(x[i,:])

    :param H: Number of Gauss-Hermite evaluation points.
    :param D: Number of input dimensions. Needs to be known at call-time.
    :return: eval_locations 'x' (H**DxD), weights 'w' (H**D)
    """
    gh_x, gh_w = hermgauss(H)
    x = np.array(list(itertools.product(*(gh_x,) * D)))  # H**DxD
    w = np.prod(np.array(list(itertools.product(*(gh_w,) * D))), 1)  # H**D
    return x, w


def gauss_hermite(dim=1, num_quad_pts=20):
    """
    Return weights and sigma-points for Gauss-Hermite cubature
    """
    # sigma_pts, weights = hermgauss(num_quad_pts)  # Gauss-Hermite sigma points and weights
    sigma_pts, weights = mvhermgauss(num_quad_pts, dim)
    sigma_pts = np.sqrt(2) * sigma_pts.T
    weights = weights.T * np.pi ** (-0.5 * dim)  # scale weights by 1/√π
    return sigma_pts, weights
