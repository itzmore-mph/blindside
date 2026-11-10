"""Tests for the XY baseline features on a hand-built scene."""

import numpy as np
import pandas as pd
import pytest

from blindside.data.tracking import position_grid
from blindside.features.xy_baseline import (
    BASELINE_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    angle_between_deg,
    design_frame,
    tracking_features,
)

PRESSER, CARRIER, MATE_NEAR, MATE_FAR, OPPONENT = 1, 2, 3, 4, 5
TEAM_OF = {PRESSER: 10, MATE_NEAR: 10, MATE_FAR: 10, CARRIER: 20, OPPONENT: 20}


def _positions() -> pd.DataFrame:
    """Frames 100 to 110. The carrier runs along +x at 2 m/s, the presser closes from behind."""
    rows = []
    for frame in range(100, 111):
        t = (frame - 110) / 10.0  # seconds relative to frame 110
        scene = {
            CARRIER: (10.0 + 2.0 * t, 0.0),
            PRESSER: (6.0 + 6.0 * t, 0.0),  # 4 m behind at frame 110, 6 m behind at frame 105
            MATE_NEAR: (10.0, 8.0),
            MATE_FAR: (10.0, 30.0),
            OPPONENT: (11.0, 1.0),
        }
        for player, (x, y) in scene.items():
            rows.append({"match_id": 7, "frame": frame, "period": 1 if frame >= 103 else 2,
                         "player_id": player, "x": x, "y": y, "is_detected": True})  # fmt: skip
    return pd.DataFrame(rows)


def _engagements(frames: list[int], side: str = "left_to_right") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame_start": frames,
            "presser_id": PRESSER,
            "carrier_id": CARRIER,
            "presser_team_id": 10,
            "attacking_side": side,
        }
    )


def test_angle_between_deg() -> None:
    a = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    b = np.array([[2.0, 0.0], [0.0, -3.0], [-1.0, 0.0], [1.0, 0.0]])
    np.testing.assert_allclose(angle_between_deg(a, b)[:3], [0.0, 90.0, 180.0], atol=1e-9)
    assert np.isnan(angle_between_deg(a, b)[3])


def test_tracking_features_of_a_press_from_behind() -> None:
    grid = position_grid(_positions())
    features = tracking_features(_engagements([110]), grid, TEAM_OF, pitch_length=105.0).iloc[0]
    assert features["has_tracking"]
    assert features["closing_speed"] == pytest.approx(4.0, abs=1e-5)  # 6 m to 4 m in 0.5 s
    assert features["presser_speed"] == pytest.approx(6.0, abs=1e-5)
    assert features["carrier_speed"] == pytest.approx(2.0, abs=1e-5)
    # The carrier runs along +x, the presser is straight behind.
    assert features["approach_angle_heading_deg"] == pytest.approx(180.0, abs=1e-3)
    # The pressing team attacks +x, so the carrier attacks the goal at -x.
    # The presser stands on that line.
    assert features["approach_angle_goal_deg"] == pytest.approx(0.0, abs=1e-3)
    # Presser (4 m) and one team mate (8 m) are within 10 m.
    # The carrier's own team mate is not counted.
    assert features["n_defenders_10m"] == 2.0


def test_attacking_side_flips_the_goal() -> None:
    grid = position_grid(_positions())
    features = tracking_features(
        _engagements([110], side="right_to_left"), grid, TEAM_OF, pitch_length=105.0
    ).iloc[0]
    assert features["approach_angle_goal_deg"] == pytest.approx(180.0, abs=1e-3)


def test_engagements_without_a_full_lookback_have_no_tracking_features() -> None:
    grid = position_grid(_positions())
    # Frame 104: five frames earlier lies outside the grid.
    # Frame 106: the lookback crosses a period. Frame 300: outside the grid.
    features = tracking_features(_engagements([104, 106, 300, 108]), grid, TEAM_OF, 105.0)
    np.testing.assert_array_equal(features["has_tracking"], [False, False, False, True])
    assert features.loc[~features["has_tracking"], "closing_speed"].isna().all()
    assert features.loc[~features["has_tracking"], "n_defenders_10m"].isna().all()


def test_no_feature_uses_positions_after_t0() -> None:
    positions = _positions()
    later = positions["frame"] > 108
    scrambled = positions.copy()
    scrambled.loc[later, ["x", "y"]] = 99.0
    before = tracking_features(_engagements([108]), position_grid(positions), TEAM_OF, 105.0)
    after = tracking_features(_engagements([108]), position_grid(scrambled), TEAM_OF, 105.0)
    pd.testing.assert_frame_equal(before, after)


def test_design_frame_types_and_missing_level() -> None:
    table = pd.DataFrame({name: [1.0, np.nan] for name in NUMERIC_FEATURES})
    for name in CATEGORICAL_FEATURES:
        table[name] = pd.Series(["a", None], dtype=object)
    frame = design_frame(table)
    assert list(frame.columns) == list(BASELINE_FEATURES)
    assert all(frame[name].dtype == float for name in NUMERIC_FEATURES)
    assert list(frame[CATEGORICAL_FEATURES[0]]) == ["a", "missing"]
    assert np.isnan(frame[NUMERIC_FEATURES[0]].iloc[1])
