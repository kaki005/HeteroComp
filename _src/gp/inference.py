import jax.numpy as np

from equinox.nn import State
from .bayesnewton_utils import ensure_diagonal_positive_precision, transpose, set_z_stats
from .likelihood import Poisson
from jax import vmap
from jaxtyping import Array
from .gaussian import GaussianDistribution
import equinox as eqx


def newton_update(mean, jacobian, hessian):
    """
    Applies one step of Newton's method to update the pseudo_likelihood parameters.
    Note that this is the natural parameter form of a Newton step.
    """

    # deal with missing data
    hessian = np.where(np.isnan(hessian), -1e-6, hessian)
    jacobian = np.where(np.isnan(jacobian), hessian @ mean, jacobian)

    # Newton update
    pseudo_likelihood_nat1 = (
        jacobian - hessian @ mean
    )
    pseudo_likelihood_nat2 = (
        -hessian
    )

    return pseudo_likelihood_nat1, pseudo_likelihood_nat2



# ======================================
# region (InferenceMixin)
# ======================================
class InferenceMixin(eqx.Module):
    """
    The approximate inference class. To be used as a Mixin, to add inference functionality to the model classƒ.
    Each approximate inference scheme implements an 'update_()' method which is called during
    inference in order to update the local likelihood approximation (the sites).
    TODO: improve code sharing between classes
    TODO: re-derive and re-implement QuasiNewton methods
    TODO: move as much of the generic functionality as possible from the base model class to this class.
    """

    pseudo_likelihood: GaussianDistribution
    likelihood: Poisson
    update_posterior: classmethod
    group_natural_params: classmethod
    conditional_posterior_to_data: classmethod

    def inference(self, state: State, lr=1.0, **kwargs):
        state = self.update_posterior(state)  # make sure the posterior is up to date
        # use the chosen inference method (VI, EP, ...) to compute the necessary terms for the parameter update
        mean, jacobian, hessian = self.update_variational_params(state, lr, **kwargs)
        # ---- Newton update ----
        nat1_new, nat2_new = newton_update(
            mean, jacobian, hessian
        )  # compute narural param
        # only required for SparseMarkov models
        _, _, old_nat1, old_nat2 = self.pseudo_likelihood.state(state)
        nat1, nat2 = self.group_natural_params(state, nat1_new, nat2_new)
        diff1 = np.mean(np.abs(nat1 - old_nat1))
        diff2 = np.mean(np.abs(nat2 - old_nat2))
        # update the model variational parameters
        state = self.pseudo_likelihood.update_nat_params(
            state,
            nat1=(1 - lr) * old_nat1 + lr * nat1,
            nat2=(1 - lr) * old_nat2 + lr * nat2,
        )
        state = self.update_posterior(state)  # recompute posterior with new params
        # output state to be used in linesearch methods
        return (mean, jacobian, hessian, state), (diff1, diff2)

    def update_variational_params(self, state: State, lr=1.0, **kwargs):
        """use the chosen inference method (VI, EP, ...) to compute the necessary terms for the parameter update.

        Args:
            batch_ind (_type_, optional): batch index. Defaults to None.
            lr (float, optional): learning rate. Defaults to 1.0.

        Returns:
            _type_: (mean, jacobian, hessian)
        """
        raise NotImplementedError

    def energy(self, batch_ind=None, **kwargs):
        raise NotImplementedError


# endregion (InferenceMixin)


# ======================================
# region (VariationalInference)
# ======================================
class VariationalInference(InferenceMixin):
    """
    Natural gradient VI (using the conjugate-computation VI approach)
    Refs:
        Khan & Lin 2017 "Conugate-computation variational inference - converting inference
                         in non-conjugate models in to inference in conjugate models"
        Chang, Wilkinson, Khan & Solin 2020 "Fast variational learning in state space Gaussian process models"
    """

    index: eqx.nn.StateIndex
    compute_kl: classmethod

    def update_variational_params(
        self,
        state: State,
        lr,
        Y: Array,
        cubature=None,
        ensure_psd=True,
        **kwargs,
    ):
        """_summary_

        Args:
            batch_ind (_type_, optional): batch index. Defaults to None.
            lr (float, optional): learning rate. Defaults to 1..
            cubature (_type_, optional): _description_. Defaults to None.
            ensure_psd (bool, optional): wheter to avoid non-PSD precision. Defaults to True.

        Returns:
            _type_: (mean, variances, jacobian, hessian)
        """

        mean_f, cov_f, W = self.conditional_posterior_to_data(state)
        # VI expected density is expected log-likelihood: E_q[log p(y|f)]
        ell, dell_dm, d2ell_dm2 = vmap(
            self.likelihood.variational_expectation, (0, 0, 0, 0, None)
        )(Y, mean_f, cov_f, np.arange(Y.shape[0]), cubature)
        if ensure_psd:  # manual fix to avoid non-PSD precision
            d2ell_dm2 = -ensure_diagonal_positive_precision(-d2ell_dm2)
        jacobian = transpose(W) @ dell_dm
        hessian = transpose(W) @ d2ell_dm2 @ W
        if mean_f.shape[1] == jacobian.shape[1]:
            return mean_f, jacobian, hessian
        else:  # sparse Markov case
            post_mean, post_var, Z, dz = state.get(self.index)
            ind, num_neighbours = set_z_stats(X, Z)
            return (post_mean[ind], jacobian, hessian)

    def energy(self, state: State, Y: Array, cubature=None, **kwargs):
        """ """
        scale = 1
        mean_f, cov_f, W = self.conditional_posterior_to_data(state)

        # VI expected density is expected log-likelihood: E_q[log p(y|f)]
        ell, _, _ = vmap(self.likelihood.variational_expectation, (0, 0, 0, 0, None))(
            Y, mean_f, cov_f, np.arange(Y.shape[0]), cubature
        )
        KL = self.compute_kl(state)  # KL[q(f)|p(f)]
        variational_free_energy = -(  # the variational free energy, i.e., the negative ELBO
            scale * np.nansum(ell)  # nansum accounts for missing data
            - KL
        )
        return variational_free_energy


# endregion (VariationalInference)


# ======================================
# region (VariationalGaussNewton)
# ======================================
class VariationalGaussNewton(VariationalInference):
    """
    Variational Gauss-Newton
    """

    def update_variational_params(
        self,
        state: State,
        lr,
        Y: Array,
        cubature=None,
        ensure_psd=True,
        **kwargs,
    ):
        mean_f, cov_f, W = self.conditional_posterior_to_data(state)
        log_target, jacobian, hessian = vmap(
            self.likelihood.variational_gauss_newton, (0, 0, 0, 0, None)
        )(Y, mean_f, cov_f, np.arange(Y.shape[0]), cubature)
        jacobian = transpose(W) @ jacobian  # (50)
        hessian = transpose(W) @ hessian @ W  # (50)
        hessian = -ensure_diagonal_positive_precision(-hessian)  # 負の固有値を消す
        if mean_f.shape[1] == jacobian.shape[1]:
            return mean_f, jacobian, hessian
        else:  # sparse markovian case
            post_mean, post_var, Z, dz = state.get(self.index)
            ind, num_neighbours = set_z_stats(X, Z)
            return (post_mean[ind], jacobian, hessian)


# endregion (VariationalGaussNewton)
