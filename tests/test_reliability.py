"""Tests for the reliability map on synthetic orientation tables with known answers."""

import numpy as np
import pandas as pd
import pytest

from blindside.analysis.bracket import HALF_WINDOWS, PRIMARY_WINDOW, add_residuals
from blindside.analysis.consistency import SEGMENTS
from blindside.analysis.reliability import (
    GROSS_MISALIGNMENT_M,
    misalignment_summary,
    pitch_zone_summary,
    standard_xy,
    touchline_distance,
    usability_sensitivity,
    usability_share,
    velocity_agreement_by_band,
    zone_grid,
    zone_index,
)

FRAMES = 4000


def _player(player_id: int, x: float, y: float, noise_deg: float, rng: np.random.Generator):
    """A standing player whose three facing angles carry white noise of a known SD."""
    data = {
        "match_id": 1,
        "player_id": player_id,
        "frame": np.arange(FRAMES),
        "x": x,
        "y": y,
        "speed": 1.0,
        "heading_deg": 0.0,
    }
    for segment in SEGMENTS:
        data[f"{segment}_deg"] = 20.0 + rng.normal(0.0, noise_deg, FRAMES)
        data[f"{segment}_sd_deg"] = 4.0 * noise_deg
        data[f"{segment}_sd_low_deg"] = noise_deg
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def two_players() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    # Player 1 stands in the corner at negative x and y, player 2 in the opposite one.
    return pd.concat(
        [_player(1, -50.0, -30.0, 1.0, rng), _player(2, 50.0, 30.0, 3.0, rng)], ignore_index=True
    )


def test_standard_xy_rescales_each_match_to_the_standard_pitch() -> None:
    table = pd.DataFrame({"match_id": [1, 1, 2], "x": [50.0, -25.0, 55.0], "y": [32.0, 0.0, -35.0]})
    x, y = standard_xy(table, {1: (100.0, 64.0), 2: (110.0, 70.0)})
    np.testing.assert_allclose(x, [52.5, -26.25, 52.5])
    np.testing.assert_allclose(y, [34.0, 0.0, -34.0])


def test_touchline_distance_depends_on_the_camera_side() -> None:
    y = np.array([-34.0, 0.0, 34.0])
    np.testing.assert_allclose(touchline_distance(y, 68.0, "-y"), [0.0, 34.0, 68.0])
    np.testing.assert_allclose(touchline_distance(y, 68.0, "+y"), [68.0, 34.0, 0.0])
    with pytest.raises(ValueError):
        touchline_distance(y, 68.0, "left")


def test_zone_index_covers_the_pitch_and_clips_outside_positions() -> None:
    x = np.array([-52.5, -0.01, 0.0, 52.4, 60.0, np.nan])
    y = np.array([-34.0, -0.01, 0.0, 33.9, -40.0, 0.0])
    ix, iy = zone_index(x, y, bins=(12, 8))
    np.testing.assert_array_equal(ix, [0, 5, 6, 11, 11, -1])
    np.testing.assert_array_equal(iy, [0, 3, 4, 7, 0, -1])


def test_pitch_zone_summary_recovers_noise_per_zone(two_players: pd.DataFrame) -> None:
    with_residuals = add_residuals(two_players)
    summary = pitch_zone_summary(
        with_residuals,
        with_residuals["x"].to_numpy(),
        with_residuals["y"].to_numpy(),
        bins=(12, 8),
        min_poses=100,
        threshold=10.0,
    )
    assert len(summary) == len(SEGMENTS) * 12 * 8
    occupied = summary[summary["n"] > 0]
    assert set(zip(occupied["ix"], occupied["iy"], strict=True)) == {(0, 0), (11, 7)}

    for segment in SEGMENTS:
        lower = zone_grid(summary, segment, "lower_sd_deg")
        upper = zone_grid(summary, segment, "upper_sd_deg")
        usable = zone_grid(summary, segment, "share_usable")
        assert lower.shape == (8, 12)
        # The jitter estimate returns the SD of the white noise that was put in.
        assert lower[0, 0] == pytest.approx(1.0, rel=0.1)
        assert lower[7, 11] == pytest.approx(3.0, rel=0.1)
        assert upper[0, 0] == pytest.approx(4.0)
        assert upper[7, 11] == pytest.approx(12.0)
        assert usable[0, 0] == 1.0 and usable[7, 11] == 0.0
        assert np.isnan(lower[3, 3])

    corner = summary[(summary["ix"] == 0) & (summary["iy"] == 0)].iloc[0]
    assert corner["x_centre"] == pytest.approx(-52.5 + 105.0 / 24)
    assert corner["y_centre"] == pytest.approx(-34.0 + 68.0 / 16)


def test_pitch_zone_summary_masks_sparse_zones(two_players: pd.DataFrame) -> None:
    with_residuals = add_residuals(two_players)
    summary = pitch_zone_summary(
        with_residuals,
        with_residuals["x"].to_numpy(),
        with_residuals["y"].to_numpy(),
        min_poses=FRAMES + 1,
    )
    assert summary["upper_sd_deg"].isna().all()
    assert summary["n"].sum() == len(SEGMENTS) * 2 * FRAMES


def test_usability_share_under_both_bounds(two_players: pd.DataFrame) -> None:
    share = usability_share(two_players, thresholds=(2.0, 10.0)).set_index(["segment", "bound"])
    for segment in SEGMENTS:
        # Upper bound: 4 and 12 degrees. Lower bound: 1 and 3 degrees.
        assert share.loc[(segment, "upper"), "below_10"] == 0.5
        assert share.loc[(segment, "upper"), "below_2"] == 0.0
        assert share.loc[(segment, "lower"), "below_10"] == 1.0
        assert share.loc[(segment, "lower"), "below_2"] == 0.5
        assert share.loc[(segment, "upper"), "n"] == 2 * FRAMES


def test_usability_sensitivity_orders_with_the_factor(two_players: pd.DataFrame) -> None:
    def scaled(factor: float) -> pd.DataFrame:
        out = two_players.copy()
        for segment in SEGMENTS:
            out[f"{segment}_sd_deg"] *= 2.5 / factor
        return out

    result = usability_sensitivity({f: scaled(f) for f in (3.0, 2.0, 2.5)}, thresholds=(4.5,))
    assert list(result["p90_to_sigma"].unique()) == [2.0, 2.5, 3.0]
    torso = result[result["segment"] == "torso"].set_index("p90_to_sigma")
    # A smaller factor means a larger sigma and a larger angular SD.
    assert torso["sd_median_deg"].is_monotonic_decreasing
    # 4 degrees at factor 2.5 becomes 5 degrees at factor 2.0 and crosses the threshold.
    assert torso.loc[2.0, "below_4.5"] == 0.0
    assert torso.loc[2.5, "below_4.5"] == 0.5
    assert torso.loc[3.0, "below_4.5"] == 0.5


def test_velocity_agreement_by_band_has_pooled_and_per_match_rows() -> None:
    rng = np.random.default_rng(2)
    n = 2000
    table = pd.DataFrame(
        {
            "match_id": np.repeat([1, 2], n // 2),
            "speed": rng.uniform(0.0, 8.0, n),
            "heading_deg": rng.uniform(-180.0, 180.0, n),
        }
    )
    for segment, offset in zip(SEGMENTS, (0.0, 10.0, -30.0), strict=True):
        table[f"{segment}_deg"] = table["heading_deg"] + offset
    result = velocity_agreement_by_band(table)
    assert set(result["match"]) == {"all", "1", "2"}
    assert len(result) == 3 * 3 * 4
    pooled = result[result["match"] == "all"].groupby("segment")["median_abs_deg"].first()
    assert pooled["torso"] == pytest.approx(0.0, abs=1e-6)
    assert pooled["hips"] == pytest.approx(10.0, abs=1e-4)
    assert pooled["head"] == pytest.approx(30.0, abs=1e-4)
    assert result.loc[result["match"] == "all", "n"].sum() == 3 * n


def test_misalignment_summary_distance_and_lead() -> None:
    n = 1000
    heading = np.linspace(-170.0, 170.0, n)
    lead = np.where(np.arange(n) < 100, 1.0, 0.2)  # 10% gross offsets
    table = pd.DataFrame(
        {
            # The pose centroid lies ahead of the tracking position along the heading.
            "centroid_dx_m": lead * np.cos(np.radians(heading)),
            "centroid_dy_m": lead * np.sin(np.radians(heading)),
            "speed": np.where(np.arange(n) % 2 == 0, 1.0, 5.0),
            "heading_deg": heading,
        }
    )
    summary = misalignment_summary(table).set_index("band")
    overall = summary.loc["all"]
    assert overall["n"] == n
    assert overall["median_m"] == pytest.approx(0.2)
    assert overall["p99_m"] == pytest.approx(1.0)
    assert overall["share_gross"] == pytest.approx(0.1)
    assert overall["along_median_m"] == pytest.approx(0.2)
    assert summary.loc["0 to 2 m/s", "n"] == n // 2
    assert summary.loc["4 to 6 m/s", "n"] == n // 2
    assert summary.loc["above 6 m/s", "n"] == 0
    assert GROSS_MISALIGNMENT_M == 0.5

    behind = table.assign(
        centroid_dx_m=-table["centroid_dx_m"], centroid_dy_m=-table["centroid_dy_m"]
    )
    assert misalignment_summary(behind).set_index("band").loc["all", "along_median_m"] < 0


def test_primary_window_is_the_one_used_for_zones() -> None:
    assert HALF_WINDOWS[PRIMARY_WINDOW] == 2


def test_figures_draw_from_the_tidy_tables(two_players: pd.DataFrame) -> None:
    from blindside.analysis.bracket import bracket_summary
    from blindside.analysis.consistency import SPEED_EDGES
    from blindside.viz.reliability import (
        plot_misalignment,
        plot_reliability_composite,
        plot_usability_sensitivity,
        plot_zone_maps,
    )

    with_residuals = add_residuals(two_players)
    zones = pitch_zone_summary(
        with_residuals,
        with_residuals["x"].to_numpy(),
        with_residuals["y"].to_numpy(),
        min_poses=100,
    )
    composite = plot_reliability_composite(
        bracket_summary(with_residuals, "speed", SPEED_EDGES),
        zones,
        velocity_agreement_by_band(two_players),
        caption="synthetic",
    )
    # Three panels and the colour bar of the pitch map.
    assert len(composite.axes) == 4
    # Every segment is drawn in the bracket panel and in the velocity panel.
    assert len(composite.axes[0].patches) == len(SEGMENTS) * 4
    assert len(composite.axes[2].lines) == len(SEGMENTS)

    maps = plot_zone_maps(zones, camera_side=None)
    assert len(maps.axes) == 2 * (len(SEGMENTS) * 2)

    sensitivity = usability_sensitivity({f: two_players for f in (2.0, 2.5, 3.0)})
    figure = plot_usability_sensitivity(sensitivity)
    assert len(figure.axes) == len(SEGMENTS)
    assert all(len(axis.lines) == 3 for axis in figure.axes)

    offsets = two_players.assign(centroid_dx_m=0.05, centroid_dy_m=-0.02)
    assert len(plot_misalignment(offsets).axes) == 2
