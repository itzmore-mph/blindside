"""Tests for the circular statistics."""

import numpy as np
import pytest

from blindside.geometry.circular import (
    angular_difference,
    circular_mean,
    circular_std,
    resultant_length,
    wrap_angle,
)

DEG = np.pi / 180.0


def test_wrap_angle_interval_is_half_open_at_minus_pi() -> None:
    wrapped = wrap_angle([0.0, np.pi, -np.pi, 3 * np.pi, 2 * np.pi, -0.5, 7.0])
    np.testing.assert_allclose(wrapped, [0.0, np.pi, np.pi, np.pi, 0.0, -0.5, 7.0 - 2 * np.pi])
    assert np.all(wrapped > -np.pi) and np.all(wrapped <= np.pi)


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [(10, 350, 20), (350, 10, -20), (170, -170, -20), (-170, 170, 20), (90, 0, 90), (0, 180, 180)],
)
def test_angular_difference_takes_the_short_way(a: float, b: float, expected: float) -> None:
    assert angular_difference(a * DEG, b * DEG) == pytest.approx(expected * DEG)


def test_circular_mean_across_the_wrap() -> None:
    angles = np.array([350.0, 10.0]) * DEG
    assert circular_mean(angles) == pytest.approx(0.0, abs=1e-12)
    # The arithmetic mean would be 180 degrees, the opposite direction.
    assert abs(np.mean(angles) - np.pi) < 1e-12


def test_circular_mean_near_180_degrees() -> None:
    angles = np.array([175.0, -175.0, 179.0, -179.0]) * DEG
    assert abs(angular_difference(circular_mean(angles), np.pi)) < 1e-9


def test_circular_mean_along_axis_and_weights() -> None:
    angles = np.array([[0.0, 90.0], [350.0, 10.0]]) * DEG
    np.testing.assert_allclose(circular_mean(angles, axis=1), [45 * DEG, 0.0], atol=1e-12)
    weighted = circular_mean(np.array([0.0, 90.0]) * DEG, weights=[1.0, 0.0])
    assert weighted == pytest.approx(0.0, abs=1e-12)


def test_circular_std_matches_linear_std_for_concentrated_data() -> None:
    rng = np.random.default_rng(1)
    angles = 3.0 + rng.normal(0.0, 5 * DEG, size=200_000)  # centred near the wrap
    assert circular_std(angles) == pytest.approx(5 * DEG, rel=0.01)


def test_circular_std_limits() -> None:
    assert circular_std(np.full(10, 1.3)) == pytest.approx(0.0, abs=1e-7)
    uniform = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    assert resultant_length(uniform) == pytest.approx(0.0, abs=1e-12)
    assert circular_std(uniform) > 5.0


def test_circular_std_along_axis() -> None:
    angles = np.array([[0.0, 0.0, 0.0], [-10.0, 0.0, 10.0]]) * DEG
    std = circular_std(angles, axis=1)
    assert std.shape == (2,)
    assert std[0] == pytest.approx(0.0, abs=1e-7)
    assert std[1] == pytest.approx(np.std([-10, 0, 10]) * DEG, rel=0.01)


def test_nan_propagates() -> None:
    assert np.isnan(circular_mean([0.1, np.nan]))
    assert np.isnan(circular_std([0.1, np.nan]))
