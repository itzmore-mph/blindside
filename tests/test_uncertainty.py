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


def test_coincident_joints_have_no_direction() -> None:
    # facing_angle returns NaN for coincident joints. The samples must not invent a direction:
    # without the guard they are uniform on the circle, or exactly 90 degrees at zero error.
    same = [[1.0, 2.0], [0.0, 0.2]]
    other = [[1.0, 2.0], [0.0, -0.2]]
    for p90 in (0.0, 6.0):
        samples = sample_facing_angles(same, other, p90, p90, rng=np.random.default_rng(0))
        assert np.isnan(samples[:, 0]).all() and not np.isnan(samples[:, 1]).any()
    estimate = facing_uncertainty(same, other, 6.0, 6.0, rng=np.random.default_rng(0))
    assert np.isnan(estimate.mean[0]) and np.isnan(estimate.std[0])
    assert np.isnan(sample_vector_angles([0.0, 0.0], 0.01, n_samples=5)).all()


def test_missing_joint_coordinates_give_nan_for_that_pose_only() -> None:
    left = [[np.nan, 0.2], [0.0, 0.2], [0.0, 0.2]]
    right = [[0.0, -0.2], [0.0, np.nan], [0.0, -0.2]]
    samples = sample_facing_angles(left, right, 6.0, 6.0, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(np.isnan(samples).all(axis=0), [True, True, False])
    np.testing.assert_array_equal(np.isnan(samples).any(axis=0), [True, True, False])


def test_invalid_p90_values_give_nan() -> None:
    np.testing.assert_array_equal(
        np.isnan(sigma_from_p90([-1.0, np.nan, np.inf, 0.0, 5.0])), [True, True, True, False, False]
    )
    args = ([[0.0, 0.2]] * 4, [[0.0, -0.2]] * 4, [-6.0, np.nan, np.inf, 6.0], 6.0)
    for correlation in (0.0, 0.9):
        samples = sample_facing_angles(*args, correlation=correlation, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(np.isnan(samples).all(axis=0), [True, True, True, False])
    with pytest.raises(ValueError):
        sigma_from_p90(5.0, p90_to_sigma=np.nan)


def test_extreme_p90_values() -> None:
    left, right = [0.0, 0.09], [0.0, -0.09]
    # No error: every sample is the point estimate.
    exact = sample_facing_angles(left, right, 0.0, 0.0, rng=np.random.default_rng(0))
    np.testing.assert_allclose(exact, 0.0, atol=1e-6)
    # An error far larger than the ear width: the direction is unknown, never NaN.
    huge = facing_uncertainty(left, right, 1e4, 1e4, n_samples=20_000, rng=np.random.default_rng(0))
    assert np.isfinite(huge.std) and huge.std / DEG > 100.0
    # The uncertainty grows monotonically with the published error.
    stds = [
        facing_uncertainty(left, right, p, p, n_samples=20_000, rng=np.random.default_rng(1)).std
        for p in (1.0, 3.0, 6.0, 12.0, 24.0)
    ]
    assert np.all(np.diff(stds) > 0)


@pytest.mark.parametrize("facing_deg", [180.0, -180.0, 179.9, -179.9, 540.0])
def test_samples_stay_wrapped_and_concentrated_across_the_wrap(facing_deg: float) -> None:
    joints = make_skeleton(facing_deg * DEG)
    samples = sample_facing_angles(
        joints["lShoulder"], joints["rShoulder"], 5.6, 5.6, n_samples=2000,
        rng=np.random.default_rng(0),
    )  # fmt: skip
    assert np.all(samples > -np.pi) and np.all(samples <= np.pi)
    # Samples fall on both sides of the wrap, and the circular summary is not disturbed.
    assert (samples > 0).any() and (samples < 0).any()
    estimate = facing_uncertainty(
        joints["lShoulder"], joints["rShoulder"], 5.6, 5.6, n_samples=2000,
        rng=np.random.default_rng(0),
    )  # fmt: skip
    assert abs(angular_difference(estimate.mean, np.pi)) < 1.0 * DEG
    assert estimate.std / DEG == pytest.approx(4.9, abs=0.4)


def test_sample_facing_angles_is_deterministic_only_with_a_seed() -> None:
    args = ([0.0, 0.1], [0.0, -0.1], 7.0, 7.0)
    a = sample_facing_angles(*args, rng=np.random.default_rng(7))
    b = sample_facing_angles(*args, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(sample_facing_angles(*args), sample_facing_angles(*args))
    # One generator, two calls: the stream advances, so the second draw differs.
    rng = np.random.default_rng(7)
    np.testing.assert_array_equal(sample_facing_angles(*args, rng=rng), a)
    assert not np.array_equal(sample_facing_angles(*args, rng=rng), a)


def test_nan_and_argument_checks() -> None:
    samples = sample_vector_angles([[np.nan, 0.0], [1.0, 0.0]], 0.01, n_samples=10)
    assert np.isnan(samples[:, 0]).all() and not np.isnan(samples[:, 1]).any()
    with pytest.raises(ValueError):
        sample_vector_angles([1.0, 0.0], 0.01, n_samples=0)
