"""Tests for the jitter lower bound and the uncertainty bracket."""

import numpy as np
import pandas as pd
import pytest

from blindside.analysis.bracket import (
    add_residuals,
    bound_ratio,
    bracket_summary,
    jitter_sd,
    local_quadratic_residual,
    quadratic_smoother,
    reconciling_correlation,
    residual_autocorrelation,
    residual_variance_factor,
    sort_for_time_series,
    white_noise_autocorrelation,
)
from blindside.geometry.uncertainty import BOUND_CORRELATION, bound_correlation, facing_uncertainty


def _series(angle: np.ndarray, player_id: int = 1, first_frame: int = 0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": 1,
            "player_id": player_id,
            "frame": first_frame + np.arange(len(angle)),
            "a_deg": np.mod(angle + 180.0, 360.0) - 180.0,
        }
    )


def test_quadratic_smoother_matches_the_savitzky_golay_coefficients() -> None:
    np.testing.assert_allclose(quadratic_smoother(2), np.array([-3, 12, 17, 12, -3]) / 35)
    assert residual_variance_factor(2) == pytest.approx(18 / 35)
    assert residual_variance_factor(5) == pytest.approx(1 - 89 / 429)
    with pytest.raises(ValueError):
        quadratic_smoother(1)


@pytest.mark.parametrize("half_window", [2, 5])
def test_quadratic_motion_leaves_no_residual_even_across_the_wrap(half_window: int) -> None:
    t = np.arange(60, dtype=float)
    angle = 150.0 + 2.0 * t - 0.03 * t**2  # crosses +180 degrees and comes back
    residual = local_quadratic_residual(_series(angle), "a_deg", half_window)
    inside = residual[half_window:-half_window]
    assert np.isnan(residual[:half_window]).all() and np.isnan(residual[-half_window:]).all()
    np.testing.assert_allclose(inside, 0.0, atol=1e-9)


def test_gaps_and_player_changes_break_the_window() -> None:
    a = _series(np.zeros(10), player_id=1)
    b = _series(np.zeros(10), player_id=2)
    gap = _series(np.zeros(10), player_id=3).query("frame not in (3, 6)")
    ordered = sort_for_time_series(pd.concat([b, gap, a]))
    residual = local_quadratic_residual(ordered, "a_deg", 2)
    valid = ordered.assign(ok=np.isfinite(residual)).groupby("player_id")["ok"].sum()
    # 10 frames give 6 complete windows. The gaps leave no run of 5 consecutive frames.
    assert valid.to_dict() == {1: 6, 2: 6, 3: 0}


@pytest.mark.parametrize("half_window", [2, 5])
def test_white_noise_sd_is_recovered_on_top_of_smooth_motion(half_window: int) -> None:
    rng = np.random.default_rng(0)
    t = np.arange(200_000) / 25.0
    motion = 120.0 * np.sin(2 * np.pi * t / 20.0)  # a slow turn back and forth
    residual = local_quadratic_residual(
        _series(motion + rng.normal(0.0, 2.0, len(t))), "a_deg", half_window
    )
    assert jitter_sd(residual, half_window) == pytest.approx(2.0, rel=0.03)


def test_error_that_is_smooth_in_time_is_invisible_which_makes_this_a_lower_bound() -> None:
    rng = np.random.default_rng(1)
    n = 100_000
    white = rng.normal(0.0, 2.0, n)
    # A slowly drifting error with an SD of 10 degrees (periods of 3 to 11 seconds).
    t = np.arange(n) / 25.0
    drift = sum(np.sin(2 * np.pi * t / period + period) for period in (3.0, 5.0, 7.0, 11.0))
    drift *= 10.0 / drift.std()
    residual = local_quadratic_residual(_series(white + drift), "a_deg", 2)
    total_sd = np.std(white + drift)
    assert total_sd > 9.0
    assert jitter_sd(residual, 2) == pytest.approx(2.0, rel=0.05)


def test_bound_ratio_and_reconciling_correlation() -> None:
    rng = np.random.default_rng(2)
    n = 100_000
    upper = np.repeat(rng.uniform(4.0, 12.0, n // 1000), 1000)  # Monte Carlo SD per pose
    residual = local_quadratic_residual(_series(rng.normal(0.0, 0.25 * upper)), "a_deg", 2)
    ratio = bound_ratio(residual, upper, 2)
    assert ratio == pytest.approx(0.25, rel=0.03)
    rho = reconciling_correlation(ratio)
    assert rho == pytest.approx(1 - 0.25**2, abs=0.01)
    assert reconciling_correlation(1.3) == 0.0

    # The Monte Carlo with that correlation reproduces the ratio (small-angle regime).
    left, right = [0.0, 0.185], [0.0, -0.185]
    independent = facing_uncertainty(
        left, right, 5.6, 5.6, n_samples=50_000, rng=np.random.default_rng(3)
    )
    reconciled = facing_uncertainty(
        left, right, 5.6, 5.6, n_samples=50_000, correlation=rho, rng=np.random.default_rng(3)
    )
    assert reconciled.std / independent.std == pytest.approx(ratio, rel=0.05)


def test_white_noise_autocorrelation_reference() -> None:
    assert white_noise_autocorrelation(2, 1) == pytest.approx(-0.8, abs=0.01)
    assert white_noise_autocorrelation(2, 5) == 0.0
    assert white_noise_autocorrelation(5, 25) == 0.0
    rng = np.random.default_rng(4)
    table = pd.DataFrame(
        {
            "match_id": 1,
            "player_id": 1,
            "frame": np.arange(100_000),
            **{f"{s}_deg": rng.normal(0.0, 3.0, 100_000) for s in ("torso", "hips", "head")},
            **{f"{s}_sd_deg": 3.0 for s in ("torso", "hips", "head")},
        }
    )
    with_residuals = add_residuals(table)
    acf = residual_autocorrelation(with_residuals)
    np.testing.assert_allclose(acf["autocorrelation"], acf["white_noise"], atol=0.02)

    summary = bracket_summary(with_residuals)
    np.testing.assert_allclose(summary["lower_sd_deg (0.2 s)"], 3.0, rtol=0.03)
    np.testing.assert_allclose(summary["lower_sd_deg (0.4 s)"], 3.0, rtol=0.03)
    np.testing.assert_allclose(summary["ratio"], 1.0, rtol=0.03)
    np.testing.assert_allclose(summary["reconciling_correlation"], 0.0, atol=0.06)


def test_bracket_summary_by_band() -> None:
    rng = np.random.default_rng(5)
    n = 60_000
    speed = np.repeat([1.0, 5.0], n // 2)
    noise = np.where(speed < 2, 1.0, 2.0)
    table = pd.DataFrame(
        {
            "match_id": 1,
            "player_id": np.repeat([1, 2], n // 2),
            "frame": np.tile(np.arange(n // 2), 2),
            "speed": speed,
            **{f"{s}_deg": rng.normal(0.0, noise) for s in ("torso", "hips", "head")},
            **{f"{s}_sd_deg": 8.0 for s in ("torso", "hips", "head")},
        }
    )
    summary = bracket_summary(add_residuals(table), "speed", (0.0, 2.0, 4.0, np.inf))
    torso = summary[summary["segment"] == "torso"].set_index("band")
    assert torso.loc["0 to 2 m/s", "lower_sd_deg (0.2 s)"] == pytest.approx(1.0, rel=0.05)
    assert torso.loc["above 4 m/s", "lower_sd_deg (0.2 s)"] == pytest.approx(2.0, rel=0.05)
    assert torso.loc["2 to 4 m/s", "n"] == 0
    assert torso.loc["above 4 m/s", "ratio"] == pytest.approx(0.25, rel=0.05)


def test_bound_correlation_lookup() -> None:
    assert bound_correlation("hips", "upper") == 0.0
    assert bound_correlation("hips", "lower") == BOUND_CORRELATION["lower"]["hips"]
    assert all(0.0 < value < 1.0 for value in BOUND_CORRELATION["lower"].values())
    with pytest.raises(ValueError):
        bound_correlation("hips", "middle")
    with pytest.raises(ValueError):
        bound_correlation("feet", "lower")
