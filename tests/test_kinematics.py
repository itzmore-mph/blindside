"""Tests for the smoothed velocity."""

import numpy as np
import pytest

from blindside.geometry.kinematics import central_velocity


def test_constant_velocity_is_recovered() -> None:
    t = np.arange(40) / 25.0
    xy = np.stack([3.0 * t, -4.0 * t], axis=-1)[:, None, :]  # one player, 5 m/s
    velocity = central_velocity(xy, fps=25, half_window=5)
    assert velocity.shape == xy.shape
    np.testing.assert_allclose(velocity[5:-5, 0], [[3.0, -4.0]] * 30)
    assert np.isnan(velocity[:5]).all() and np.isnan(velocity[-5:]).all()


def test_missing_positions_and_segment_boundaries_give_nan() -> None:
    xy = np.stack([np.arange(20.0), np.zeros(20)], axis=-1)
    xy[10] = np.nan
    velocity = central_velocity(xy, fps=10, half_window=2)
    assert np.isnan(velocity[8]).all() and np.isnan(velocity[12]).all()
    assert velocity[5, 0] == pytest.approx(10.0)

    segment = np.array([1] * 10 + [2] * 10)
    xy = np.stack([np.arange(20.0), np.zeros(20)], axis=-1)
    velocity = central_velocity(xy, fps=10, half_window=2, segment=segment)
    assert np.isnan(velocity[8:12]).all()
    assert not np.isnan(velocity[[7, 12]]).any()


def test_short_series_and_invalid_window() -> None:
    assert np.isnan(central_velocity(np.zeros((4, 2)), fps=25, half_window=2)).all()
    with pytest.raises(ValueError):
        central_velocity(np.zeros((4, 2)), fps=25, half_window=0)
