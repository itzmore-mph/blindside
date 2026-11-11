"""Tests for the pipeline steps that need no data files, and for the figures drawn from them."""

import numpy as np
import pandas as pd
import pytest

from blindside.analysis.baseline import cluster_bootstrap_weights
from blindside.analysis.consistency import SPEED_EDGES, speed_band_labels
from blindside.analysis.pipeline import (
    BLIND_FEATURES,
    CONFIDENT_ROW,
    decision,
    engagement_geometry,
    inclusion_flow,
    proxy_agreement,
)
from blindside.data.tracking import position_grid
from blindside.features.engagements import PRIMARY_OUTCOME
from blindside.features.pose_features import BOUNDS, FOV_HALF_ANGLE_DEG
from blindside.features.xy_baseline import LOOKBACK_FRAMES
from blindside.viz.agreement import plot_engagement_geometry, plot_proxy_agreement


def _engagements(n: int = 800, seed: int = 0) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Engagements whose pose label follows the XY proxy more closely at higher carrier speed."""
    rng = np.random.default_rng(seed)
    speed = rng.uniform(0.0, 8.0, n)
    heading = rng.uniform(0.0, 180.0, n)
    proxy = heading > FOV_HALF_ANGLE_DEG
    follows = rng.random(n) < speed / 8.0
    blind = np.where(follows, proxy, rng.random(n) < 0.2)
    p = np.where(blind, 0.97, 0.02)
    test = pd.DataFrame(
        {
            "approach_angle_heading_deg": heading,
            "carrier_speed": speed,
            "cluster": np.arange(n) // 2,
            "match_id": np.where(np.arange(n) < n // 2, 1, 2),
        }
    )
    features = {
        bound: pd.DataFrame({"p_blind_side_head": p, "p_blind_side_torso": p}) for bound in BOUNDS
    }
    return test, features


def _agreement() -> pd.DataFrame:
    test, features = _engagements()
    weights = cluster_bootstrap_weights(
        test["cluster"].to_numpy(), test["match_id"].to_numpy(), 200, seed=0
    )
    return proxy_agreement(test, features, weights)


def test_inclusion_flow_applies_the_criteria_in_order():
    table = pd.DataFrame(
        {
            "has_pose": [True, True, True, False, True],
            PRIMARY_OUTCOME: [0.0, 1.0, np.nan, 1.0, 0.0],
            "has_tracking": [True, False, True, True, True],
            "cluster": ["a", "a", "b", "c", "d"],
        }
    )
    included, flow = inclusion_flow(table, with_pose=True)
    assert flow["Engagements"].tolist() == [5, 4, 3, 2]
    assert flow["Possessions"].tolist() == [4, 3, 2, 2]
    assert included["cluster"].tolist() == ["a", "d"]
    _, without_pose = inclusion_flow(table, with_pose=False)
    assert without_pose["Engagements"].tolist() == [5, 4, 3]


def test_proxy_agreement_has_one_row_per_feature_bound_and_group():
    agreement = _agreement()
    bands = speed_band_labels(SPEED_EDGES)
    assert len(agreement) == len(BLIND_FEATURES) * len(BOUNDS) * (len(bands) + 2)
    assert set(agreement["group"]) == {"all", *bands, CONFIDENT_ROW}
    head = agreement[
        (agreement["feature"] == "p_blind_side_head") & (agreement["bound"] == "lower")
    ].set_index("group")
    assert head.loc[bands, "n"].sum() == head.loc["all", "n"] == 800
    # The synthetic pose label follows the proxy more closely the faster the carrier runs.
    assert head.loc[bands, "kappa"].is_monotonic_increasing
    assert (head["kappa_low"] <= head["kappa"]).all() and (
        head["kappa"] <= head["kappa_high"]
    ).all()


def test_decision_wording_follows_the_intervals():
    rows = pd.DataFrame({"bound": ["lower", "upper"], "low": [-0.0004, -0.0003]})
    assert decision(rows).startswith("No detectable increment")
    assert decision(rows.assign(low=[0.001, 0.002])).startswith("H1 is supported")
    assert "upper bound only" in decision(rows.assign(low=[-0.001, 0.002]))


def test_plot_proxy_agreement_draws_the_table():
    agreement = _agreement()
    figure = plot_proxy_agreement(agreement, caption="caption")
    kappa_axis, share_axis = figure.axes
    bands = speed_band_labels(SPEED_EDGES)
    head = agreement[
        (agreement["feature"] == "p_blind_side_head") & (agreement["bound"] == "lower")
    ].set_index("group")
    drawn = kappa_axis.lines[0].get_ydata()
    assert np.allclose(drawn, head.loc[bands, "kappa"].to_numpy())
    # Three labels for every speed band and for all engagements.
    assert len(share_axis.patches) == 3 * (len(bands) + 1)
    heights = [patch.get_height() for patch in share_axis.patches]
    assert heights[-1] == pytest.approx(100.0 * head.loc["all", "xy_blind"])
    legend = [text.get_text() for text in share_axis.get_legend().get_texts()]
    assert legend == ["Pose, head", "Pose, torso", "XY proxy"]
    with pytest.raises(KeyError):
        plot_proxy_agreement(agreement[agreement["group"] != "all"])


def test_engagement_geometry_and_its_figure():
    t0 = 100
    positions = pd.DataFrame(
        {
            "frame": [t0 - LOOKBACK_FRAMES, t0, t0 - LOOKBACK_FRAMES, t0],
            "period": [1, 1, 1, 1],
            "player_id": [7, 7, 9, 9],
            "x": [0.0, 2.0, 10.0, 8.0],
            "y": [0.0, 0.0, 0.0, 0.0],
        }
    )
    engagements = pd.DataFrame(
        {
            "pose_frame": [250],
            "frame_start": [t0],
            "carrier_id": [7],
            "presser_id": [9],
            "carrier_speed": [4.0],
            "approach_angle_heading_deg": [0.0],
        }
    )
    poses = pd.DataFrame(
        {"frame": [250, 250], "player_id": [7, 9], "x": [2.0, 8.0], "y": [0.1, 0.0]}
    )
    features = {
        bound: pd.DataFrame(
            {
                "carrier_head_deg": [10.0],
                "carrier_torso_deg": [5.0],
                "carrier_head_sd_deg": [sd],
                "carrier_torso_sd_deg": [sd / 2],
                "p_blind_side_head": [0.0],
                "p_blind_side_torso": [0.0],
            }
        )
        for bound, sd in zip(BOUNDS, (2.0, 12.0), strict=True)
    }
    geometry = engagement_geometry(engagements, poses, features, position_grid(positions))
    row = geometry.iloc[0]
    assert (row["carrier_x"], row["presser_x"]) == (2.0, 8.0)
    assert (row["travel_dx"], row["travel_dy"]) == (2.0, 0.0)
    assert (row["head_sd_lower_deg"], row["head_sd_upper_deg"]) == (2.0, 12.0)

    figure = plot_engagement_geometry(geometry.assign(time_s=1.0))
    assert len(figure.axes) == 1
    assert "in view" in figure.axes[0].get_xlabel()
    empty = plot_engagement_geometry(geometry.iloc[:0].assign(time_s=[]))
    assert "no on-ball engagement" in empty.axes[0].texts[0].get_text()
