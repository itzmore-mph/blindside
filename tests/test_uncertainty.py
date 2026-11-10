"""Tests for the Monte Carlo error propagation."""

import numpy as np
import pytest

from blindside.geometry.circular import angular_difference
from blindside.geometry.uncertainty import (
    P90_TO_SIGMA,
    difference_sigma,
    facing_uncertainty,
    sample_facing_angles,
    sample_vector_angles,
    sigma_from_p90,
    small_angle_std,
    vector_uncertainty,
)
from skeleton import HIP_WIDTH, SHOULDER_WIDTH, make_skeleton

DEG = np.pi / 180.0


def test_conversion_factor_is_the_90_percent_radius_of_a_3d_gaussian() -> None:
    rng = np.random.default_rng(0)
    radius = np.linalg.norm(rng.standard_normal((400_000, 3)), axis=1)
    assert np.mean(radius <= P90_TO_SIGMA) == pytest.approx(0.90, abs=0.002)


def test_sigma_from_p90_units_and_factor() -> None:
    assert sigma_from_p90(5.0) == pytest.approx(0.02)
    np.testing.assert_allclose(sigma_from_p90([4.0, 8.0], p90_to_sigma=2.0), [0.02, 0.04])
    with pytest.raises(ValueError):
        sigma_from_p90(5.0, p90_to_sigma=0.0)


def test_difference_sigma_and_correlation() -> None:
    assert difference_sigma(0.03, 0.04) == pytest.approx(0.05)
    assert difference_sigma(0.02, 0.02, correlation=1.0) == pytest.approx(0.0)
    assert difference_sigma(0.02, 0.02, correlation=0.5) == pytest.approx(0.02)
    with pytest.raises(ValueError):
        difference_sigma(0.02, 0.02, correlation=1.5)


def test_monte_carlo_matches_the_small_angle_approximation_for_two_joints() -> None:
    # Two joints 0.40 m apart, each with a 2.5 cm 90% radius, so sigma = 1 cm per axis.
    width, p90 = 0.40, 2.5
    left, right = np.array([0.0, width / 2]), np.array([0.0, -width / 2])
    estimate = facing_uncertainty(
        left, right, p90, p90, n_samples=200_000, rng=np.random.default_rng(1)
    )
    sigma = difference_sigma(sigma_from_p90(p90), sigma_from_p90(p90))
    analytic = small_angle_std(width, sigma)
    assert analytic == pytest.approx(np.sqrt(2) * 0.01 / 0.40)
    assert estimate.std == pytest.approx(analytic, rel=0.01)
    assert abs(angular_difference(estimate.mean, 0.0)) < 0.02 * DEG


@pytest.mark.parametrize(("width", "expected_deg"), [(SHOULDER_WIDTH, 4.9), (HIP_WIDTH, 10.7)])
def test_baseline_width_sets_the_noise_floor(width: float, expected_deg: float) -> None:
    # Shoulder p90 of 5.6 cm gives about 5 degrees. Hip p90 of 7.3 cm on the narrower hip
    # axis gives more than twice that (CLAUDE.md, "Baseline widths matter").
    p90 = 5.6 if width == SHOULDER_WIDTH else 7.3
    estimate = facing_uncertainty(
        [0.0, width / 2], [0.0, -width / 2], p90, p90, n_samples=50_000,
        rng=np.random.default_rng(2),
    )  # fmt: skip
    assert estimate.std / DEG == pytest.approx(expected_deg, abs=0.3)


def test_conversion_factor_scales_the_angular_uncertainty() -> None:
    kwargs = {"n_samples": 100_000, "rng": np.random.default_rng(3)}
    left, right = [0.0, 0.2], [0.0, -0.2]
    wide = facing_uncertainty(left, right, 5.0, 5.0, p90_to_sigma=2.0, **kwargs)
    narrow = facing_uncertainty(left, right, 5.0, 5.0, p90_to_sigma=3.0, **kwargs)
    assert wide.std / narrow.std == pytest.approx(1.5, rel=0.03)


def test_positive_correlation_reduces_the_uncertainty() -> None:
    left, right = [0.0, 0.2], [0.0, -0.2]
    independent = facing_uncertainty(
        left, right, 5.0, 5.0, n_samples=100_000, rng=np.random.default_rng(4)
    )
    correlated = facing_uncertainty(
        left, right, 5.0, 5.0, n_samples=100_000, correlation=0.5, rng=np.random.default_rng(4)
    )
    assert correlated.std / independent.std == pytest.approx(np.sqrt(0.5), rel=0.03)
    perfectly = facing_uncertainty(left, right, 5.0, 5.0, correlation=1.0)
    assert perfectly.std == pytest.approx(0.0, abs=1e-6)


def test_vectorised_over_frames_and_players_and_recovers_the_facing() -> None:
    rng = np.random.default_rng(5)
    truth = rng.uniform(-np.pi, np.pi, size=(30, 22))
    joints = make_skeleton(truth, rng.uniform(-50, 50, size=(30, 22, 2)))
    p90 = rng.uniform(5.0, 9.0, size=(30, 22))
    samples = sample_facing_angles(
        joints["lShoulder"], joints["rShoulder"], p90, p90, n_samples=200, rng=rng
    )
    assert samples.shape == (200, 30, 22)
    estimate = facing_uncertainty(
        joints["lShoulder"], joints["rShoulder"], p90, p90, n_samples=200, rng=rng
    )
    assert estimate.mean.shape == estimate.std.shape == (30, 22)
    # The mean of 200 draws is within a few standard errors of the true facing.
    assert np.max(np.abs(angular_difference(estimate.mean, truth))) < 3 * DEG
    # A larger joint error gives a larger angular uncertainty.
    assert np.corrcoef(p90.ravel(), estimate.std.ravel())[0, 1] > 0.9


def test_seeded_generator_is_reproducible() -> None:
    args = ([0.0, 0.1], [0.0, -0.1], 7.0, 7.0)
    a = facing_uncertainty(*args, rng=np.random.default_rng(7))
    b = facing_uncertainty(*args, rng=np.random.default_rng(7))
    c = facing_uncertainty(*args, rng=np.random.default_rng(8))
    assert a.std == b.std and a.mean == b.mean
    assert a.std != c.std


def test_noise_larger_than_the_baseline_gives_a_uniform_direction() -> None:
    estimate = vector_uncertainty([0.01, 0.0], 1.0, n_samples=20_000, rng=np.random.default_rng(9))
    assert estimate.std / DEG > 100.0


def test_nan_and_argument_checks() -> None:
    samples = sample_vector_angles([[np.nan, 0.0], [1.0, 0.0]], 0.01, n_samples=10)
    assert np.isnan(samples[:, 0]).all() and not np.isnan(samples[:, 1]).any()
    with pytest.raises(ValueError):
        sample_vector_angles([1.0, 0.0], 0.01, n_samples=0)
