import equinox as eqx
import jax.numpy as jnp
from .bayesnewton_utils import (
    rotation_matrix,
    softplus,
    softplus_inv,
    scaled_squared_euclid_dist,
)
from jax import vmap
from jax.lax import stop_gradient
from jax.scipy.linalg import block_diag, expm
from jaxtyping import Array, Float, Scalar
from tensorflow_probability.substrates.jax.math import bessel_ive


class Kernel(eqx.Module):
    def __call__(self, X, X2):
        return self.K(X, X2)

    @property
    def F(self):
        return self.feedback_matrix()

    @property
    def H(self):
        return self.measurement_model()

    def K(self, X, X2):
        raise NotImplementedError("kernel function not implemented")

    def measurement_model(self) -> Float[jnp.ndarray, "1 state_dim"]:
        raise NotImplementedError

    def latent2measure(
        self,
        latent_mean: Float[jnp.ndarray, "state_dim 1"],
        latent_cov: Float[jnp.ndarray, "state_dim 1 1"],
        t: float,
    ) -> tuple[Float[jnp.ndarray, "1"], Float[jnp.ndarray, "1 1"]]:
        raise NotImplementedError

    def inducing_precision(self):
        return None, None

    def kernel_to_state_space(
        self, R=None
    ) -> tuple[
        Float[jnp.ndarray, "state_dim state_dim"],
        Float[jnp.ndarray, "state_dim 1"],
        Float[jnp.ndarray, "1 1"],
        Float[jnp.ndarray, "1 state_dim"],
        Float[jnp.ndarray, "state_dim state_dim"],
    ]:
        raise NotImplementedError

    def spatial_conditional(self, R=None, predict: bool = False):
        """ """
        return None, None

    def get_meanfield_block_index(self) -> jnp.ndarray:
        raise Exception(
            "Either the mean-field method is not applicable to this kernel, "
            "or this kernel's get_meanfield_block_index() method has not been implemented"
        )

    def feedback_matrix(self) -> Float[jnp.ndarray, "state_dim state_dim"]:
        """return F"""
        raise NotImplementedError

    def state_transition(
        self, dt: float | Float[Array, "1"]
    ) -> Float[jnp.ndarray, "state_dim state_dim"]:
        """return A"""
        # TODO(32): fix prediction when using expm to compute the state transition.
        F = self.feedback_matrix()
        A = expm(F * dt)
        return A


# ===========================
# region(Stationary Kernel)
# ===========================
class StationaryKernel(Kernel):
    _transformed_lengthscale: jnp.ndarray
    fix_variance: bool = eqx.field(static=True)
    _transformed_variance: jnp.ndarray
    fix_lengthscale: bool = eqx.field(static=True)

    def __init__(
        self, variance=1.0, lengthscale=1.0, fix_variance=False, fix_lengthscale=False
    ):
        # check whether the parameters are to be optimised
        self.fix_variance = fix_variance
        self.fix_lengthscale = fix_lengthscale
        self._transformed_variance = softplus_inv(variance)
        self._transformed_lengthscale = softplus_inv(lengthscale)

    @property
    def transformed_lengthscale(self) -> Float[Scalar, "1"]:
        if self.fix_lengthscale:
            return stop_gradient(self._transformed_lengthscale)
        return self._transformed_lengthscale

    @property
    def transformed_variance(self) -> Float[Scalar, "1"]:
        if self.fix_variance:
            return stop_gradient(self._transformed_variance)
        return self._transformed_variance

    @property
    def variance(self) -> Float[Scalar, "1"]:
        return jnp.clip(
            softplus(self.transformed_variance), min=1e-10
        )  # 2025/05/12 S.Kakio add clip

    @property
    def Pinf(self):
        return self.stationary_covariance()

    @property
    def lengthscale(self) -> Float[Scalar, "1"]:
        return jnp.clip(
            softplus(self.transformed_lengthscale), min=1e-10
        )  # 2025/05/12 S.Kakio add clip

    def K(self, X: Float[Array, "N D"], Y: Float[Array, "M D"]) -> Float[Array, "N M"]:
        r2 = scaled_squared_euclid_dist(X, Y, self.lengthscale)
        return self.K_r2(r2)

    def K_r2(self, r2: Float[jnp.ndarray, "1"]) -> Float[Scalar, "1"]:
        # Clipping around the (single) float precision which is ~1e-45.
        r = jnp.sqrt(jnp.maximum(r2, 1e-36))
        return self.K_r(r)

    def K_r(self, r: Float[jnp.ndarray, "1"]) -> Float[Scalar, "1"]:
        raise NotImplementedError("kernel not implemented")

    def stationary_covariance(self) -> Float[jnp.ndarray, "state_dim state_dim"]:
        raise NotImplementedError

    @property
    def L(self) -> Float[jnp.ndarray, "1 state_dim"]:
        raise NotImplementedError

    @property
    def Qc(self) -> Float[Scalar, "1 1"]:
        raise NotImplementedError

    def latent2measure(
        self,
        latent_mean: Float[jnp.ndarray, "state_dim 1"],
        latent_cov: Float[jnp.ndarray, "state_dim 1 1"],
    ) -> tuple[Float[Array, "1"], Float[Array, "1 1"]]:
        H = self.measurement_model()
        return H @ latent_mean, H @ latent_cov @ H.T

    def kernel_to_state_space(
        self, R=None
    ) -> tuple[
        Float[jnp.ndarray, "state_dim state_dim"],
        Float[jnp.ndarray, "state_dim 1"],
        Float[jnp.ndarray, "1 1"],
        Float[jnp.ndarray, "1 state_dim"],
        Float[jnp.ndarray, "state_dim state_dim"],
    ]:
        F = self.feedback_matrix()
        Pinf = self.stationary_covariance()
        H = self.measurement_model()
        return F, self.L, self.Qc, H, Pinf


class Matern12(StationaryKernel):
    """
    The Matern 1/2 kernel. Functions drawn from a GP with this kernel are not
    differentiable anywhere. The kernel equation is

    k(r) = σ² exp{-r}

    where:
    r  is the Euclidean distance between the input points, scaled by the lengthscales parameter ℓ.
    σ² is the variance parameter
    """

    @property
    def state_dim(self):
        return 1

    def K_r(self, r):
        return self.variance * jnp.exp(-r)

    @property
    def L(self) -> Float[jnp.ndarray, "1 state_dim"]:
        return jnp.array([[1.0]])

    @property
    def Qc(self) -> Float[Scalar, "1 1"]:
        return jnp.array([[2.0 * self.variance / self.lengthscale]])

    def stationary_covariance(self):
        Pinf = jnp.array([[self.variance]])
        return Pinf

    def measurement_model(self):
        H = jnp.array([[1.0]])
        return H

    def state_transition(self, dt):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for the exponential prior.
        :param dt: step size(s), Δtₙ = tₙ - tₙ₋₁ [scalar]
        :return: state transition matrix A [1, 1]
        """
        A = jnp.broadcast_to(jnp.exp(-dt / self.lengthscale), [1, 1])
        return A

    def feedback_matrix(self):
        F = jnp.array([[-1.0 / self.lengthscale]])
        return F


class Matern32(StationaryKernel):
    """
    The Matern 3/2 kernel. Functions drawn from a GP with this kernel are once
    differentiable. The kernel equation is

    k(r) = σ² (1 + √3r) exp{-√3 r}

    where:
    r  is the Euclidean distance between the input points, scaled by the lengthscales parameter,
    σ² is the variance parameter.
    """

    @property
    def state_dim(self) -> int:
        return 2

    def K_r(self, r) -> Float[jnp.ndarray, "1"]:
        sqrt3 = jnp.sqrt(3.0)
        return self.variance * (1.0 + sqrt3 * r) * jnp.exp(-sqrt3 * r)

    def stationary_covariance(self):
        Pinf = jnp.array(
            [[self.variance, 0.0], [0.0, 3.0 * self.variance / self.lengthscale**2.0]]
        )
        return Pinf

    @property
    def L(self) -> Float[jnp.ndarray, "1 state_dim"]:
        return jnp.array([[0], [1]])

    @property
    def Qc(self) -> Float[Scalar, "1 1"]:
        return jnp.array([[12.0 * 3.0**0.5 / self.lengthscale**3.0 * self.variance]])

    @staticmethod
    def measurement_model():
        H = jnp.array([[1.0, 0.0]])
        return H

    def state_transition(self, dt):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for the Matern-3/2 prior.
        :param dt: step size(s), Δtₙ = tₙ - tₙ₋₁ [scalar]
        :return: state transition matrix A [2, 2]
        """
        # A = expm(self.feedback_matrix() * dt)
        # return A
        lam = jnp.sqrt(3.0) / self.lengthscale
        A = jnp.exp(-dt * lam) * (
            dt * jnp.array([[lam, 1.0], [-(lam**2.0), -lam]]) + jnp.eye(2)
        )
        return A

    def feedback_matrix(self):
        lam = 3.0**0.5 / self.lengthscale
        F = jnp.array([[0.0, 1.0], [-(lam**2), -2 * lam]])
        return F


class Matern52(StationaryKernel):
    """
    The Matern 5/2 kernel. Functions drawn from a GP with this kernel are twice
    differentiable. The kernel equation is

    k(r) = σ² (1 + √5r + 5/3r²) exp{-√5 r}

    where:
    r  is the Euclidean distance between the input points, scaled by the lengthscales parameter ℓ,
    σ² is the variance parameter.
    """

    @property
    def state_dim(self) -> int:
        return 3

    def K_r(self, r):
        sqrt5 = jnp.sqrt(5.0)
        return (
            self.variance
            * (1.0 + sqrt5 * r + 5.0 / 3.0 * jnp.square(r))
            * jnp.exp(-sqrt5 * r)
        )

    @property
    def L(self) -> Float[jnp.ndarray, "1 state_dim"]:
        return jnp.array([[0.0], [0.0], [1.0]])

    @property
    def Qc(self) -> Float[Scalar, "1 1"]:
        return jnp.array(
            [[self.variance * 400.0 * 5.0**0.5 / 3.0 / self.lengthscale**5.0]]
        )

    def measurement_model(self):
        H = jnp.array([[1.0, 0.0, 0.0]])
        return H

    def state_transition(self, dt):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for the Matern-5/2 prior.
        :param dt: step size(s), Δtₙ = tₙ - tₙ₋₁ [scalar]
        :return: state transition matrix A [3, 3]
        """
        lam = jnp.sqrt(5.0) / self.lengthscale
        dtlam = dt * lam
        A = jnp.exp(-dtlam) * (
            dt
            * jnp.array(
                [
                    [lam * (0.5 * dtlam + 1.0), dtlam + 1.0, 0.5 * dt],
                    [-0.5 * dtlam * lam**2, lam * (1.0 - dtlam), 1.0 - 0.5 * dtlam],
                    [
                        lam**3 * (0.5 * dtlam - 1.0),
                        lam**2 * (dtlam - 3),
                        lam * (0.5 * dtlam - 2.0),
                    ],
                ]
            )
            + jnp.eye(3)
        )
        return A

    def stationary_covariance(self):
        kappa = 5.0 / 3.0 * self.variance / self.lengthscale**2.0
        Pinf = jnp.array(
            [
                [self.variance, 0.0, -kappa],
                [0.0, kappa, 0.0],
                [-kappa, 0.0, 25.0 * self.variance / self.lengthscale**4.0],
            ]
        )
        return Pinf

    def feedback_matrix(self):
        lam = 5.0**0.5 / self.lengthscale
        F = jnp.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-(lam**3.0), -3.0 * lam**2.0, -3.0 * lam],
            ]
        )
        return F


class Matern72(StationaryKernel):
    """
    The Matern 7/2 kernel. Functions drawn from a GP with this kernel are three times differentiable.

    where:
    r  is the Euclidean distance between the input points, scaled by the lengthscales parameter ℓ,
    σ² is the variance parameter.
    """

    @property
    def state_dim(self) -> int:
        return 4

    def K_r(self, r):
        sqrt7 = jnp.sqrt(7.0)
        return (
            self.variance
            * (1.0 + sqrt7 * r + 14.0 / 5.0 * jnp.square(r) + 7.0 * sqrt7 / 15.0 * r**3)
            * jnp.exp(-sqrt7 * r)
        )

    @property
    def L(self) -> Float[jnp.ndarray, "1 state_dim"]:
        return jnp.array([[0.0], [0.0], [0.0], [1.0]])

    @property
    def Qc(self) -> Float[Scalar, "1 1"]:
        return jnp.array(
            [[self.variance * 10976.0 * 7.0**0.5 / 5.0 / self.lengthscale**7.0]]
        )

    def measurement_model(self):
        H = jnp.array([[1.0, 0.0, 0.0, 0.0]])
        return H

    def state_transition(self, dt):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for the Matern-7/2 prior.
        :param dt: step size(s), Δtₙ = tₙ - tₙ₋₁ [scalar]
        :return: state transition matrix A [4, 4]
        """
        lam = jnp.sqrt(7.0) / self.lengthscale
        lam2 = lam * lam
        lam3 = lam2 * lam
        dtlam = dt * lam
        dtlam2 = dtlam**2
        A = jnp.exp(-dtlam) * (
            dt
            * jnp.array(
                [
                    [
                        lam * (1.0 + 0.5 * dtlam + dtlam2 / 6.0),
                        1.0 + dtlam + 0.5 * dtlam2,
                        0.5 * dt * (1.0 + dtlam),
                        dt**2 / 6,
                    ],
                    [
                        -dtlam2 * lam**2.0 / 6.0,
                        lam * (1.0 + 0.5 * dtlam - 0.5 * dtlam2),
                        1.0 + dtlam - 0.5 * dtlam2,
                        dt * (0.5 - dtlam / 6.0),
                    ],
                    [
                        lam3 * dtlam * (dtlam / 6.0 - 0.5),
                        dtlam * lam2 * (0.5 * dtlam - 2.0),
                        lam * (1.0 - 2.5 * dtlam + 0.5 * dtlam2),
                        1.0 - dtlam + dtlam2 / 6.0,
                    ],
                    [
                        lam2**2 * (dtlam - 1.0 - dtlam2 / 6.0),
                        lam3 * (3.5 * dtlam - 4.0 - 0.5 * dtlam2),
                        lam2 * (4.0 * dtlam - 6.0 - 0.5 * dtlam2),
                        lam * (1.5 * dtlam - 3.0 - dtlam2 / 6.0),
                    ],
                ]
            )
            + jnp.eye(4)
        )
        return A

    def stationary_covariance(self):
        kappa = 7.0 / 5.0 * self.variance / self.lengthscale**2.0
        kappa2 = 9.8 * self.variance / self.lengthscale**4.0
        Pinf = jnp.array(
            [
                [self.variance, 0.0, -kappa, 0.0],
                [0.0, kappa, 0.0, -kappa2],
                [-kappa, 0.0, kappa2, 0.0],
                [0.0, -kappa2, 0.0, 343.0 * self.variance / self.lengthscale**6.0],
            ]
        )
        return Pinf

    def feedback_matrix(self):
        lam = 7.0**0.5 / self.lengthscale
        F = jnp.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [-(lam**4.0), -4.0 * lam**3.0, -6.0 * lam**2.0, -4.0 * lam],
            ]
        )
        return F


# ===========================
# endregion(Stationary Kernel)
# ===========================
# ===========================
# region(Periodic)
# ===========================


class Periodic(Kernel):
    order: int = eqx.field(static=True)
    _transformed_period: Float[Scalar, "1"]

    def __init__(self, period=1.0, order=6):
        self.order = order
        self._transformed_period = softplus_inv(period)

    @property
    def period(self):
        return softplus(self._transformed_period)


class StationaryPeriodic(StationaryKernel, Periodic):
    """
    Periodic kernel in SDE form.
    Hyperparameters:
        variance, σ²
        lengthscale, l
        period, p
    The associated continuous-time state space model matrices are constructed via
    a sum of cosines.
    """

    def __init__(
        self,
        variance=1.0,
        lengthscale=1.0,
        period: float = 1.0,
        order: int = 6,
        fix_variance: bool = False,
        fix_lengthscale: bool = False,
    ):
        StationaryKernel.__init__(
            self,
            variance,
            lengthscale,
            fix_variance=fix_variance,
            fix_lengthscale=fix_lengthscale,
        )
        Periodic.__init__(self, period=period, order=order)

    @property
    def L(self) -> Float[jnp.ndarray, "2*order+2 2*order+2"]:
        return jnp.eye(2 * (self.order + 1))

    @property
    def Qc(self) -> Float[Array, "2*order"]:
        return jnp.zeros(2 * (self.order + 1))

    def stationary_covariance(self):
        q2 = (
            jnp.array([1, *[2] * self.order])
            * self.variance
            * bessel_ive([*range(self.order + 1)], self.lengthscale ** (-2))
        )
        Pinf = jnp.kron(jnp.diag(q2), jnp.eye(2))
        return Pinf

    def measurement_model(self):
        H = jnp.kron(jnp.ones([1, self.order + 1]), jnp.array([1.0, 0.0]))
        return H

    def state_transition(self, dt: Float[Scalar, "1"]):
        """
        Calculation of the closed form discrete-time state
        transition matrix A = expm(FΔt) for the Periodic prior
        :param dt: step size(s), Δt = tₙ - tₙ₋₁ [1]
        :return: state transition matrix A [2(N+1), 2(N+1)]
        """
        omega = 2 * jnp.pi / self.period  # The angular frequency
        harmonics = jnp.arange(self.order + 1) * omega
        A = block_diag(*vmap(rotation_matrix, [None, 0])(dt.squeeze(), harmonics))
        return A

    def feedback_matrix(self):
        omega = 2 * jnp.pi / self.period  # The angular frequency
        F = jnp.kron(
            jnp.diag(jnp.arange(self.order + 1)),
            jnp.array([[0.0, -omega], [omega, 0.0]]),
        )  # The model
        return F

    def K_r(self, r: Float[Scalar, "1"]) -> Float[Scalar, "1"]:
        omega = 2 * jnp.pi / self.period  # The angular frequency
        return self.variance * jnp.exp(
            -2 * jnp.sin(omega * r / 2) ** 2 / (self.lengthscale**2)
        )


# ===========================
# endregion(Periodic)
# ===========================


# ===========================
# region(Independent)
# ===========================
class Independent(Kernel):
    """
    A stack of independent GP priors. 'kernels' is a list of GP kernels, and this class stacks
    the state space models such that each component is fed to the likelihood.
    This class differs from Sum only in the measurement model.
    """

    def __init__(self, kernels: list[StationaryKernel]):
        self.num_kernels: int = len(kernels)
        self.kernels: list[StationaryKernel] = kernels
        self.name = "Independent"
        self.kernel0: StationaryKernel = kernels[0]

    def K(self, X, X2):
        zeros = jnp.zeros(self.num_kernels)
        K0 = self.kernel0.K(X, X2)
        index_vector = zeros.at[0].add(1.0)
        Kstack = jnp.kron(K0, jnp.diag(index_vector))
        for i in range(1, self.num_kernels):
            kerneli = self.kernels[i]
            index_vector = zeros.at[i].add(1.0)
            Kstack += jnp.kron(kerneli.K(X, X2), jnp.diag(index_vector))
        return Kstack

    def kernel_to_state_space(self, R=None):
        F, L, Qc, H, Pinf = self.kernel0.kernel_to_state_space(R)
        for i in range(1, self.num_kernels):
            kerneli = self.kernels[i]
            F_, L_, Qc_, H_, Pinf_ = kerneli.kernel_to_state_space(R)
            F = block_diag(F, F_)
            L = block_diag(L, L_)
            Qc = block_diag(Qc, Qc_)
            H = block_diag(H, H_)
            Pinf = block_diag(Pinf, Pinf_)
        return F, L, Qc, H, Pinf

    def measurement_model(self):
        H = self.kernel0.measurement_model()
        for i in range(1, self.num_kernels):
            kerneli = self.kernels[i]
            H_ = kerneli.measurement_model()
            H = block_diag(H, H_)
        return H

    def stationary_covariance(self):
        Pinf = self.kernel0.stationary_covariance()
        for i in range(1, self.num_kernels):
            kerneli = self.kernels[i]
            Pinf_ = kerneli.stationary_covariance()
            Pinf = block_diag(Pinf, Pinf_)
        return Pinf

    def stationary_covariance_meanfield(self):
        """
        Stationary covariance as a tensor of blocks, as required when using a mean-field assumption
        """
        raise NotImplementedError("blocks are not all the same shape")
        # Pinf = self.kernel0.stationary_covariance()[None]
        # for i in range(1, self.num_kernels):
        #     kerneli = eval("self.kernel" + str(i))
        #     Pinf_ = kerneli.stationary_covariance()[None]
        #     Pinf = jnp.concatenate([Pinf, Pinf_])
        # return Pinf

    def inducing_precision(self):
        Qzz0, Lzz0 = self.kernel0.inducing_precision()
        Qzz, Lzz = [Qzz0], [Lzz0]
        for i in range(1, self.num_kernels):
            kerneli = self.kernels[i]
            Qzz_, Lzz_ = kerneli.inducing_precision()
            Qzz, Lzz = Qzz + [Qzz_], Lzz + [Lzz_]
        return Qzz, Lzz

    def state_transition(self, dt):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for a set of stacked of GPs
        :param dt: step size(s), Δt = tₙ - tₙ₋₁ [1]
        :return: state transition matrix A [D, D]
        """
        A = self.kernel0.state_transition(dt)
        for i in range(1, self.num_kernels):
            kerneli = self.kernels[i]
            A_ = kerneli.state_transition(dt)
            A = block_diag(A, A_)
        return A

    def state_transition_meanfield(self, dt):
        """
        State transition matrix in the form required for mean-field inference.
        :param dt: step size(s), Δtₙ = tₙ - tₙ₋₁ [scalar]
        :return: state transition matrix A
        """
        raise NotImplementedError("blocks are not all the same shape")
        # A = self.kernel0.state_transition(dt)[None]
        # for i in range(1, self.num_kernels):
        #     kerneli = eval("self.kernel" + str(i))
        #     A_ = kerneli.state_transition(dt)[None]
        #     A = jnp.concatenate([A, A_])
        # return A

    def get_meanfield_block_index(self):
        raise NotImplementedError("blocks are not all the same shape")
        # Pinf = self.stationary_covariance_meanfield()
        # num_latents = Pinf.shape[0]
        # sub_state_dim = Pinf.shape[1]
        # state = jnp.ones([sub_state_dim, sub_state_dim])
        # for i in range(1, num_latents):
        #     state = block_diag(state, jnp.ones([sub_state_dim, sub_state_dim]))
        # block_index = jnp.where(np.array(state, dtype=bool))
        # return block_index

    def feedback_matrix(self):
        F = self.kernel0.feedback_matrix()
        for i in range(1, self.num_kernels):
            kerneli = self.kernels[i]
            F_ = kerneli.feedback_matrix()
            F = block_diag(F, F_)
        return F


class Sum(Independent):
    """
    A sum of GP priors. 'components' is a list of GP kernels, and this class stacks
    the state space models to produce their sum.
    This class differs from Independent only in the measurement model.
    """

    def __init__(self, kernels):
        super().__init__(kernels=kernels)
        self.name = "Sum"

    def K(self, X, X2):
        Ksum = self.kernel0.K(X, X2)
        for i in range(1, self.num_kernels):
            kerneli = self.kernels[i]
            Ksum = Ksum + kerneli.K(X, X2)
        return Ksum

    def kernel_to_state_space(self, R=None):
        F, L, Qc, H, Pinf = self.kernel0.kernel_to_state_space(R)
        for i in range(1, self.num_kernels):
            kerneli = self.kernels[i]
            F_, L_, Qc_, H_, Pinf_ = kerneli.kernel_to_state_space(R)
            F = block_diag(F, F_)
            L = block_diag(L, L_)
            Qc = block_diag(Qc, Qc_)
            H = jnp.block([H, H_])
            Pinf = block_diag(Pinf, Pinf_)
        return F, L, Qc, H, Pinf

    def measurement_model(self):
        H = self.kernel0.measurement_model()
        for i in range(1, self.num_kernels):
            kerneli = self.kernels[i]
            H_ = kerneli.measurement_model()
            H = jnp.block([H, H_])
        return H


class Separable(Independent):
    """
    A product of separable GP priors. 'components' is a list of GP kernels, and this class stacks
    the state space models to produce their product.
    This class differs from Independent only in the measurement model.
    TODO: this assumes that each kernel acts on a different dimension. Generalise.
    TODO: implement state space form of product kernels
    """

    def __init__(self, kernels):
        super().__init__(kernels=kernels)
        self.name = "Product"

    def K(self, X, X2):
        Kprod = self.kernel0.K(X[:, :1], X2[:, :1])
        for i in range(1, self.num_kernels):
            kerneli = self.kernels[i]
            Kprod = Kprod * kerneli.K(X[:, i : i + 1], X2[:, i : i + 1])
        return Kprod

    # def measurement_model(self):
    #     H = self.kernel0.measurement_model()
    #     for i in range(1, self.num_kernels):
    #         kerneli = eval("self.kernel" + str(i))
    #         H_ = kerneli.measurement_model()
    #         H = jnp.block([
    #             H, H_
    #         ])
    #     return H


# ===========================
# endregion(Independent)
# ===========================
