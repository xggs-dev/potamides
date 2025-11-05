"""Curvature analysis functions."""

__all__ = (
    "combine_ln_likelihoods",
    "compute_ln_likelihood",
)

import functools as ft
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Int, Real

from .custom_types import BoolSzGamma, Sz0, SzGamma2

log2pi = jnp.log(2 * jnp.pi)


@ft.partial(jax.jit)
def compute_ln_likelihood(
    kappa_hat: SzGamma2,
    acc_xy_unit: SzGamma2,
    where_straight: BoolSzGamma | None = None,
    *,
    sigma_theta: float = jnp.deg2rad(10.0),
) -> Sz0:
    r"""Compute the log-likelihood of accelerations given track curvature.

    This function calculates the likelihood that observed gravitational
    accelerations are consistent with the curvature of a stellar stream track.
    It implements the method from Nibauer et al. (2023) for assessing the
    goodness of fit between a gravitational potential model and stream
    observations.

    The likelihood is based on the alignment between unit curvature vectors
    (principal normal directions) and the local acceleration field. Compatible
    alignments indicate that the acceleration points in the direction of
    curvature, as expected for streams shaped by gravitational forces.

    Parameters
    ----------
    kappa_hat : Array[float, (N, 2)]
        Unit curvature vectors (principal normal vectors) at N positions along
        the stream track. These point in the direction of maximum curvature.
    acc_xy_unit : Array[float, (N, 2)]
        Unit acceleration vectors in the x-y plane at N positions. These
        represent the direction of the gravitational acceleration from the
        potential model.
    where_straight : Array[bool, (N,)]
        Boolean mask indicating positions where the stream is locally straight
        (has negligible curvature). If `None` (default), all positions are
        assumed to be curved.
    sigma_theta : float
        Standard deviation of the angle distribution between acceleration and
        curvature vectors for straight segments, given in radians. Only used
        when `where_straight` contains `True` values. Default is 10 degrees in
        radians.

    Returns
    -------
    Array[float, ()]
        The log-likelihood value. Higher values indicate better agreement
        between the acceleration field and track curvature. Returns -∞ if
        the majority of curved segments are incompatible.

    Notes
    -----
    The algorithm computes three fractions (Nibauer et al. 2023, Eq. 18):

    - f1: fraction of positions with compatible curvature-acceleration alignment
    - f2: fraction of positions with incompatible alignment
    - f3: fraction of positions with undefined curvature

    These fractions define a trinomial distribution where f1 + f2 + f3 = 1.
    The log-likelihood for curved segments is computed using the trinomial
    likelihood (see `compute_ln_lik_curved`), while straight segments use
    a Gaussian likelihood for angular alignment.

    The likelihood is only computed if f1 > f2 (more compatible than
    incompatible alignments), otherwise returns -∞. This breaks the
    degeneracy in the trinomial parameters (Nibauer et al. 2023, Eq. 20).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import potamides as ptd

    >>> # Simple example: 3 points with perfect alignment
    >>> kappa_hat = jnp.array([
    ...     [1.0, 0.0],   # pointing right
    ...     [0.0, 1.0],   # pointing up
    ...     [-1.0, 0.0]   # pointing left
    ... ])
    >>>
    >>> # Perfectly aligned accelerations
    >>> acc_xy_unit = jnp.array([
    ...     [1.0, 0.0],   # perfectly aligned
    ...     [0.0, 1.0],   # perfectly aligned
    ...     [-1.0, 0.0]   # perfectly aligned
    ... ])
    >>>
    >>> ln_lik = ptd.compute_ln_likelihood(kappa_hat, acc_xy_unit)
    >>> print(f"Perfect alignment: {ln_lik:.2f}")
    Perfect alignment: 2.48

    >>> # Anti-aligned case (bad fit)
    >>> acc_xy_unit_bad = jnp.array([
    ...     [-1.0, 0.0],  # opposite direction
    ...     [0.0, -1.0],  # opposite direction
    ...     [1.0, 0.0]    # opposite direction
    ... ])
    >>>
    >>> ln_lik_bad = ptd.compute_ln_likelihood(kappa_hat, acc_xy_unit_bad)
    >>> print(f"Anti-aligned: {ln_lik_bad}")
    Anti-aligned: -inf

    """
    # ---------------------------------------------------
    # Compute the 'fractions' f1, f2, f3 (Eq. 18 of Nibauer et al. 2023)

    # - n1: number of eval points with compatible curvature vectors and planar
    #   accelerations, where compatible means that theta -- the angle between
    #   the unit curvature vector and the planar acceleration vector -- is less
    #   than pi/2.
    N = len(kappa_hat)  # Number of gamma points
    where_curved = jnp.ones(N, bool) if where_straight is None else ~where_straight
    acc_curv_align: SzGamma2 = jnp.where(
        where_curved[:, None], acc_xy_unit * kappa_hat, jnp.zeros_like(kappa_hat)
    )
    n1 = jnp.sum(jnp.abs(1 + jnp.sign(jnp.sum(acc_curv_align, axis=1))) / 2)

    # - n2: number of eval points with incompatible curvature vectors and
    #   planar accelerations.
    n2 = jnp.sum(where_curved) - n1

    # - n3: is the number of evaluation points with undefined curvature
    #   vectors. This is fixed for each stream track and therefore doesn't
    #   really matter since the likelihoods are ultimately divided by the
    #   maximum likelihood, so this term will cancel out.
    n3 = N - n1 - n2

    # ---------------------------------------------------

    # The likelihood is degenerate with the "n" parameters. To break the degeneracy we require n1 > n2 (Nibauer et al. 2023, Eq. 20).
    # Note the inversion on `where_curved` in the operands.
    mostly_good = n1 > n2
    operands = (kappa_hat, acc_xy_unit, ~where_curved, n1, n2, n3, sigma_theta)
    ln_lik = lax.cond(mostly_good, compute_lnlik_good, compute_lnlik_bad, *operands)

    return ln_lik


@ft.partial(jax.jit)
def compute_lnlik_bad(*_: Any) -> Sz0:
    """Log-Likelihood when the majority of the curved segments are incompatible.

    Returns -∞ to indicate a poor fit when incompatible alignments dominate.
    """
    return -jnp.inf


@ft.partial(jax.jit)
def compute_lnlik_good(
    kappa_hat: SzGamma2,
    acc_xy_unit: SzGamma2,
    where_straight: BoolSzGamma,
    n1: Sz0,
    n2: Sz0,
    n3: Sz0,
    sigma_theta: float,
) -> Sz0:
    """Log-Likelihood when the majority of the curved segments are compatible.

    Parameters
    ----------
    kappa_hat : SzGamma2
        Unit curvature vectors (principal normal vectors) at N positions along
        the stream track. These point in the direction of maximum curvature.
    acc_xy_unit : SzGamma2
        Unit acceleration vectors in the x-y plane at N positions. These
        represent the direction of the gravitational acceleration from the
        potential model.
    where_straight : BoolSzGamma
        Boolean mask indicating positions where the stream is locally straight
        (has negligible curvature).
    n1 : Sz0
        Number of points with compatible curvature-acceleration alignment.
    n2 : Sz0
        Number of points with incompatible curvature-acceleration alignment.
    n3 : Sz0
        Number of points with undefined curvature.
    sigma_theta : float
        Standard deviation of the angle distribution for straight segments.

    Returns
    -------
    Sz0
        Log-likelihood of the good fit case.

    """
    # Log-likelihood of the curved part of the stream
    lnlik_curved = compute_ln_lik_curved(len(kappa_hat), n1, n2, n3)

    # TODO: it is more efficient to lax cond on where_straight having any True.
    lnlik_straight = compute_lnlik_straight(
        kappa_hat, acc_xy_unit, where_straight, sigma_theta
    )

    # Return the total log-likelihood
    return lnlik_curved + lnlik_straight


@ft.partial(jax.jit)
def stirling_ln_factorial_k_ln_k(k: Sz0) -> Sz0:
    """Compute k*ln(k) term from Stirling's approximation.

    Uses only the k*ln(k) term from ln(k!) ≈ k*ln(k) - k. This is useful
    for computing multinomial coefficients where the linear terms cancel.
    Returns 0 for k = 0 (using the convention 0*ln(0) = 0).

    Parameters
    ----------
    k : Sz0
        The count for which to compute k*ln(k).

    Returns
    -------
    Array[float, ()]
        The value k*ln(k), or 0 if k = 0.

    Notes
    -----
    When computing multinomial coefficients ln(N!/(n1!...!nk!)) where
    n1 + ... + nk = N, the linear terms (-N + n1 + ... + nk) cancel,
    leaving only N*ln(N) - n1*ln(n1) - ... - nk*ln(nk). This function
    computes just the k*ln(k) terms needed for this calculation.

    """
    return lax.select(k > 0, k * jnp.log(k), jnp.array(0.0))


def klogk(k: Sz0) -> Sz0:
    """Compute k*log(k) with the convention that 0*log(0) = 0.

    Parameters
    ----------
    k : Sz0
        Input value.

    Returns
    -------
    Sz0
        Computed value of k*log(k), or 0 if k = 0.

    """
    return lax.select(jnp.isclose(k, 0.0), jnp.array(0.0), k * jnp.log(k))


@ft.partial(jax.jit)
def compute_ln_lik_curved(ngamma: Sz0, n1: Sz0, n2: Sz0, n3: Sz0) -> Sz0:
    """Log-Likelihood of the curved part of the stream based on trinomial distribution.

    This function computes the log-likelihood for a trinomial distribution where
    each evaluation point along the stream can fall into one of three categories:

    1. Compatible alignment (f1): curvature and acceleration are aligned
    2. Incompatible alignment (f2): curvature and acceleration are misaligned
    3. Undefined curvature (f3): curvature cannot be computed

    The trinomial log-likelihood for observing counts (n1, n2, n3) out of N trials
    with probabilities (p1, p2, p3) is:

        ln L = ln(N!/(n1!n2!n3!)) + n1*ln(p1) + n2*ln(p2) + n3*ln(p3)

    Using maximum likelihood estimators p_i = n_i/N = f_i (sample fractions) and
    including the multinomial coefficient using Stirling's approximation:

        ln(N!) ≈ N*ln(N) - N
        ln(n_i!) ≈ n_i*ln(n_i) - n_i

    We get:

        ln L = N*ln(N) - n1*ln(n1) - n2*ln(n2) - n3*ln(n3) + n1*ln(p1) + n2*ln(p2) + n3*ln(p3)
             = N*ln(N) + N*(f1*ln(p1/f1) + f2*ln(p2/f2) + f3*ln(p3/f3))

    With p_i = f_i (MLE), this simplifies to:

        ln L = N*ln(N) + N*(f1*ln(1) + f2*ln(1) + f3*ln(1)) = N*ln(N)

    But we want to allow p_i ≠ f_i, so using the general form with p_i = f_i:

        ln L = N*(f1*ln(f1) + f2*ln(f2) + f3*ln(f3)) + ln(N!/(n1!n2!n3!))

    The multinomial coefficient is included for proper comparison across tracks
    with different numbers of points or different category distributions.

    Parameters
    ----------
    ngamma : int
        Total number of gamma evaluation points (N in the trinomial).
    n1 : Array[float, ()]
        Number of points with compatible curvature-acceleration alignment.
    n2 : Array[float, ()]
        Number of points with incompatible curvature-acceleration alignment.
    n3 : Array[float, ()]
        Number of points with undefined curvature.

    Returns
    -------
    Array[float, ()]
        Log-likelihood of the curved part of the stream including the multinomial
        coefficient normalization.

    Notes
    -----
    The multinomial coefficient is computed using Stirling's approximation for
    factorial terms. This is necessary when comparing tracks with different
    geometries or numbers of evaluation points, as the coefficient will not
    cancel out in likelihood ratios.

    References
    ----------
    - Multinomial distribution: https://webspace.maths.qmul.ac.uk/i.goldsheid/MTH5118/Notes6-09.pdf
    - Nibauer et al. (2023): Stream track likelihood method

    """
    # Entropy contribution: n1*ln(p1) + n2*ln(p2) + n3*ln(p3)
    # With MLE p_i = f_i = n_i/N, this becomes: N*(f1*ln(f1) + f2*ln(f2) + f3*ln(f3))
    entropy_term = klogk(n1) + klogk(n2) + klogk(n3) - (n1 + n2 + n3) * jnp.log(ngamma)

    return ln_multinom_coef + entropy_term


@ft.partial(jax.jit)
def compute_lnlik_straight(
    kappa_hat: SzGamma2,
    acc_xy_unit: SzGamma2,
    where_straight: BoolSzGamma,
    sigma_theta: float,
) -> Sz0:
    r"""Log-Likelihood of the straight part of the stream.

    This function computes the log-likelihood for stream segments that are
    locally straight (have negligible curvature). The likelihood is based on the
    angular alignment between the planar acceleration vectors and the stream
    track direction.

    For straight segments, we expect the acceleration to be aligned with the
    stream track. The angle theta_T between the acceleration vector and the
    track direction is used to compute a Gaussian likelihood:

        $$ ln L = -0.5 * [ln(2π) + 2ln(\sigma_θ) + (θ_T - 0)² / \sigma_θ²] $$

    where $\sigma_θ$ is the standard deviation of the angle distribution,
    representing the expected spread of angles for straight segments.

    Parameters
    ----------
    kappa_hat : SzGamma2
        Unit curvature vectors (principal normal vectors) at N positions along
        the stream track. These point in the direction of maximum curvature.
    acc_xy_unit : SzGamma2
        Unit acceleration vectors in the x-y plane at N positions. These
        represent the direction of the gravitational acceleration from the
        potential model.
    where_straight : BoolSzGamma
        Boolean mask indicating positions where the stream is locally straight
        (has negligible curvature).
    sigma_theta : float
        Standard deviation of the angle distribution for straight segments.

    Returns
    -------
    Sz0
        Log-likelihood of the straight part of the stream.

    """
    # Log-likelihood of the straight part of the stream
    # If no part is straight then `acc_linear_align` is all zeros
    acc_linear_align = jnp.where(
        where_straight[:, None],
        acc_xy_unit * kappa_hat,
        jnp.zeros_like(kappa_hat),
    )
    # Angle between planar acceleration and stream track (Nibauer et al. 2023,
    # Eq. 15). acc_linear_align = 0 => theta_T = 0
    theta_T = jnp.pi / 2 - jnp.arccos(jnp.sum(acc_linear_align, axis=1))
    # The likelihoods of the straight segment (Nibauer et al. 2023, Eq. 16)
    ln_normal = -0.5 * (
        log2pi + 2 * jnp.log(sigma_theta) + (theta_T - 0) ** 2 / sigma_theta**2
    )

    return jnp.sum(ln_normal)  # sum to get the total likelihood


##############################################################################


@ft.partial(jnp.vectorize, signature="(n),(n),(n)->()")
@ft.partial(jax.jit)
def combine_ln_likelihoods(
    lnliks: Real[Array, "S"],
    /,
    ngammas: Int[Array, "S"],
    arclengths: Real[Array, "S"],
) -> Sz0:
    r"""Combine likelihoods from different stream segments with density weighting.

    This function combines log-likelihoods from multiple stream segments by
    applying density-based weighting. Stream segments with lower measurement
    density (fewer gamma points per unit arc-length) are up-weighted, while
    segments with higher measurement density are down-weighted. This ensures
    fair contribution from all segments regardless of their sampling density.

    The function is vectorized using JAX's `jnp.vectorize` with signature
    `(n),(n),(n)->()`, allowing it to process multiple sets of stream segments
    in parallel. When given 2D input arrays, it processes each row independently
    and returns a 1D array of combined likelihoods.

    Parameters
    ----------
    lnliks : Array[float, (S,)]
        The log-likelihoods of S stream segments. For vectorized operation,
        this can be a 2D array where each row represents a different set
        of stream segments.
    ngammas : Array[int, (S,)]
        The number of gamma points in each of the S stream segments.
        Must have the same shape as `lnliks`.
    arclengths : Array[float, (S,)]
        The total arc-lengths of the S stream segments.
        Must have the same shape as `lnliks`.

    Returns
    -------
    Array[float, ()]
        The combined weighted log-likelihood. For vectorized inputs,
        returns an array with one combined likelihood per input set.

    Notes
    -----
    The weighting scheme computes the mean measurement density across all
    segments and uses this to normalize individual segment contributions:

    .. math::

        \bar{\rho} = \frac{\sum_i n_i}{\sum_i L_i}

        w_i = \frac{\bar{\rho}}{\rho_i} = \frac{\bar{\rho} L_i}{n_i}

        \mathcal{L}_{combined} = \sum_i w_i \mathcal{L}_i

    where $n_i$ is the number of gamma points, $L_i$ is the arc-length, and
    $\mathcal{L}_i$ is the log-likelihood for segment $i$.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import potamides as ptd

    >>> # Scalar inputs - single set of stream segments
    >>> lnliks = jnp.array([0.5, 1.0, 1.5])
    >>> ngammas = jnp.array([100, 100, 100])
    >>> arclengths = jnp.array([1.0, 1.0, 1.0])
    >>> combined = ptd.combine_ln_likelihoods(lnliks, ngammas, arclengths)
    >>> print(combined)
    3.0

    >>> # Vector inputs - multiple sets of stream segments
    >>> lnliks = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    >>> ngammas = jnp.array([[100, 200, 300], [150, 250, 350]])
    >>> arclengths = jnp.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])
    >>> combined = ptd.combine_ln_likelihoods(lnliks, ngammas, arclengths)
    >>> print(combined.round(1))
    [0.6 1.5]

    """
    # Compute the mean measurement density of the stream segments. This is the
    # ratio of the total number of gamma points to the total arclength.
    mean_gamma_density = jnp.sum(ngammas) / jnp.sum(arclengths)

    # Compute the weights for each segment. This is the ratio of the total
    # measurement density to the measurement density of each segment. For
    # streams with lower measurement density the likelihood is up-weighted and
    # vice versa for streams with higher measurement density.
    gamma_densities = ngammas / arclengths
    weights = mean_gamma_density / gamma_densities

    # Compute the weighted log-likelihoods
    lnliks_weighted = weights * lnliks
    # TODO: should this be normalized by the sum of the weights?

    # Return the total log-likelihood
    return jnp.sum(lnliks_weighted)
