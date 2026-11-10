"""Monte Carlo propagation of the per-joint pose error into angular uncertainty.

Error model (an assumption, see CLAUDE.md): the error of a joint is an isotropic 3D
Gaussian. SkillCorner publishes a radius ``p90_mae_cm`` that contains 90% of the estimates,
so ``sigma = r90 / 2.50`` per axis, where 2.50 is the square root of the 0.90 quantile of a
chi-square distribution with 3 degrees of freedom. The marginal of an isotropic 3D Gaussian
in the ground plane is an isotropic 2D Gaussian with the same sigma, so z is never sampled.

Every orientation in this package is the direction of a difference of joint positions
(right shoulder minus left shoulder, nose minus ear midpoint, ...). The difference of
Gaussian joints is Gaussian, so the sampler draws the difference vector directly. This is
exact, halves the number of random draws, and makes the correlation between the two joint
errors an explicit parameter instead of a hidden assumption.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from blindside.geometry.circular import circular_mean, circular_std, wrap_angle
from blindside.geometry.orientation import FACING_ROTATION

FloatArray = NDArray[np.floating]

#: sqrt(chi2.ppf(0.90, df=3)) = 2.5003. Converts a 90% radius into a per-axis sigma.
P90_TO_SIGMA = 2.50
#: Conversion factors of the sensitivity check in CLAUDE.md.
P90_TO_SIGMA_SENSITIVITY = (2.0, 2.5, 3.0)
DEFAULT_N_SAMPLES = 200

#: Correlation between the errors of the left and the right joint of a segment under the
#: two bounds of the angular uncertainty. ``upper`` is independence. ``lower`` holds the
#: correlations at which the Monte Carlo SD equals the frame-to-frame jitter of the angle
#: (0.2 s window, both pose matches pooled, docs/RELIABILITY.md section 1). The true
#: angular error lies between the two and cannot be identified without ground truth, so
#: every quantity that uses angular uncertainty is computed under both.
#: The ``lower`` values are defaults for this dataset. Recalibrate on other matches with
#: :func:`blindside.analysis.bracket.calibrate_bound_correlation`.
BOUND_CORRELATION: dict[str, dict[str, float]] = {
    "upper": {"torso": 0.0, "hips": 0.0, "head": 0.0},
    "lower": {"torso": 0.935, "hips": 0.975, "head": 0.979},
}


def bound_correlation(segment: str, bound: str) -> float:
    """Left/right error correlation of a segment under the ``"lower"`` or ``"upper"`` bound."""
    if bound not in BOUND_CORRELATION:
        raise ValueError(f"bound must be one of {sorted(BOUND_CORRELATION)}, got {bound!r}")
    if segment not in BOUND_CORRELATION[bound]:
        raise ValueError(f"unknown segment {segment!r}")
    return BOUND_CORRELATION[bound][segment]


@dataclass(frozen=True)
class AngularEstimate:
    """Circular mean and circular standard deviation of an angle, both in radians."""

    mean: FloatArray
    std: FloatArray


def sigma_from_p90(p90_cm: ArrayLike, p90_to_sigma: float = P90_TO_SIGMA) -> FloatArray:
    """Per-axis standard deviation in metres from a 90% error radius in centimetres."""
    if p90_to_sigma <= 0:
        raise ValueError("p90_to_sigma must be positive")
    return np.asarray(p90_cm, dtype=float) / 100.0 / p90_to_sigma


def difference_sigma(
    sigma_a: ArrayLike, sigma_b: ArrayLike, correlation: float = 0.0
) -> FloatArray:
    """Per-axis sigma of ``a - b`` for two joints with per-axis sigmas and a given correlation.

    ``correlation`` is the correlation between the errors of the two joints along the same
    axis. 0 is the independence assumption. Positive values (both joints displaced together,
    for example by a shifted bounding box) shrink the noise of the difference.
    """
    if not -1.0 <= correlation <= 1.0:
        raise ValueError("correlation must lie in [-1, 1]")
    sigma_a = np.asarray(sigma_a, dtype=float)
    sigma_b = np.asarray(sigma_b, dtype=float)
    variance = sigma_a**2 + sigma_b**2 - 2.0 * correlation * sigma_a * sigma_b
    return np.sqrt(np.maximum(variance, 0.0))


def sample_vector_angles(
    vector_xy: ArrayLike,
    sigma: ArrayLike,
    n_samples: int = DEFAULT_N_SAMPLES,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """Directions of a ground-plane vector under isotropic Gaussian noise.

    Parameters
    ----------
    vector_xy
        Array of shape ``(..., 2)``.
    sigma
        Per-axis standard deviation of the vector, shape ``(...)`` or scalar, same unit.
    n_samples
        Number of Monte Carlo draws.
    rng
        Random generator. A fresh unseeded generator is used when omitted.

    Returns
    -------
    Array of shape ``(n_samples, ...)`` with angles in (-pi, pi]. NaN inputs give NaN.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")
    rng = rng if rng is not None else np.random.default_rng()
    vector_xy = np.asarray(vector_xy, dtype=float)
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), vector_xy.shape[:-1])
    noise = rng.standard_normal((n_samples, *vector_xy.shape), dtype=np.float32)
    noise *= sigma[..., None].astype(np.float32)
    noise += vector_xy.astype(np.float32)
    return np.arctan2(noise[..., 1], noise[..., 0]).astype(float)


def sample_facing_angles(
    left_xy: ArrayLike,
    right_xy: ArrayLike,
    left_p90_cm: ArrayLike,
    right_p90_cm: ArrayLike,
    *,
    n_samples: int = DEFAULT_N_SAMPLES,
    p90_to_sigma: float = P90_TO_SIGMA,
    correlation: float = 0.0,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """Monte Carlo samples of the facing angle of a left-right joint pair.

    Returns an array of shape ``(n_samples, ...)``. See :func:`facing_uncertainty`.
    """
    left_xy = np.asarray(left_xy, dtype=float)
    right_xy = np.asarray(right_xy, dtype=float)
    sigma = difference_sigma(
        sigma_from_p90(left_p90_cm, p90_to_sigma),
        sigma_from_p90(right_p90_cm, p90_to_sigma),
        correlation,
    )
    samples = sample_vector_angles(right_xy - left_xy, sigma, n_samples, rng)
    return wrap_angle(samples + FACING_ROTATION)


def summarise_samples(samples: ArrayLike) -> AngularEstimate:
    """Circular mean and circular standard deviation over the first axis."""
    return AngularEstimate(mean=circular_mean(samples, axis=0), std=circular_std(samples, axis=0))


def vector_uncertainty(
    vector_xy: ArrayLike,
    sigma: ArrayLike,
    *,
    n_samples: int = DEFAULT_N_SAMPLES,
    rng: np.random.Generator | None = None,
) -> AngularEstimate:
    """Angular estimate for the direction of a noisy ground-plane vector."""
    return summarise_samples(sample_vector_angles(vector_xy, sigma, n_samples, rng))


def facing_uncertainty(
    left_xy: ArrayLike,
    right_xy: ArrayLike,
    left_p90_cm: ArrayLike,
    right_p90_cm: ArrayLike,
    *,
    n_samples: int = DEFAULT_N_SAMPLES,
    p90_to_sigma: float = P90_TO_SIGMA,
    correlation: float = 0.0,
    rng: np.random.Generator | None = None,
) -> AngularEstimate:
    """Facing angle of a left-right joint pair with its angular uncertainty.

    Parameters
    ----------
    left_xy, right_xy
        Ground-plane joint positions in metres, shape ``(..., 2)``.
    left_p90_cm, right_p90_cm
        ``p90_mae_cm`` of the two joints, shape ``(...)``.
    n_samples
        Monte Carlo draws per pose.
    p90_to_sigma
        Conversion factor from the 90% radius to sigma. 2.50 for an isotropic 3D Gaussian.
    correlation
        Assumed correlation between the errors of the two joints, default 0 (independence).
        Independence overstates the angular noise if the true correlation is positive.
    rng
        Random generator, pass a seeded one for reproducible output.

    Returns
    -------
    AngularEstimate
        Circular mean and circular standard deviation in radians, each of shape ``(...)``.
    """
    return summarise_samples(
        sample_facing_angles(
            left_xy,
            right_xy,
            left_p90_cm,
            right_p90_cm,
            n_samples=n_samples,
            p90_to_sigma=p90_to_sigma,
            correlation=correlation,
            rng=rng,
        )
    )


def small_angle_std(length_m: ArrayLike, sigma_m: ArrayLike) -> FloatArray:
    """Analytic first-order angular standard deviation ``sigma / length`` in radians.

    ``sigma_m`` is the per-axis sigma of the difference vector, see :func:`difference_sigma`.
    Valid while ``sigma`` is small against ``length``. Used to cross-check the Monte Carlo.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(sigma_m, dtype=float) / np.asarray(length_m, dtype=float)
