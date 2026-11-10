"""Tests for the engagement table on a small synthetic events file."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from blindside.features.engagements import (
    EVENT_BOOLEAN,
    EVENT_CATEGORICAL,
    EVENT_NUMERIC,
    POSSESSION_LABELS,
    PRIMARY_OUTCOME,
    engagement_table,
    pose_available,
)

MATCH = 42


def _possession(event_id: str, end_type: str, pass_outcome: str | None, pressure: str | None):
    return {
        "event_id": event_id,
        "event_type": "player_possession",
        "frame_start": 100,
        "period": 1,
        "player_id": 9,
        "team_id": 2,
        "attacking_side": "right_to_left",
        "x_start": 0.0,
        "y_start": 0.0,
        "pass_outcome": pass_outcome,
        "possession_epv_total": 0.01,
        "overall_pressure_start": pressure,
        "reception_difficulty_start": "easy",
        "end_type_dummy": end_type,
    }


def _engagement(event_id: str, possession: str, end_type: str, side: str, frame: int, **extra):
    row = {
        "event_id": event_id,
        "event_type": "on_ball_engagement",
        "frame_start": frame,
        "period": 1,
        "player_id": 5,
        "player_in_possession_id": 9,
        "team_id": 1,
        "attacking_side": side,
        "x_start": 10.0,
        "y_start": -4.0,
        "associated_player_possession_event_id": possession,
        "associated_player_possession_frame_start": 100,
        "associated_player_possession_end_type": end_type,
        "force_backward": False,
        "goal_side_start": True,
        "penalty_area_start": False,
    }
    row |= {name: 1.5 for name in EVENT_NUMERIC}
    row |= {name: "x" for name in EVENT_CATEGORICAL}
    return row | extra


@pytest.fixture
def opendata(tmp_path: Path) -> Path:
    rows = [
        _possession("p1", "possession_loss", None, "high_pressure"),
        _possession("p2", "pass", "unsuccessful", None),
        _possession("p3", "pass", "successful", "low_pressure"),
        _possession("p4", "unknown", None, "low_pressure"),
        _engagement("e1", "p1", "possession_loss", "left_to_right", 98),
        _engagement("e2", "p1", "possession_loss", "right_to_left", 104, force_backward=True),
        _engagement("e3", "p2", "pass", "left_to_right", 100),
        _engagement("e4", "p3", "pass", "left_to_right", 101),
        _engagement("e5", "p4", "unknown", "left_to_right", 101),
    ]
    directory = tmp_path / "data" / "matches" / str(MATCH)
    directory.mkdir(parents=True)
    pd.DataFrame(rows).drop(columns="end_type_dummy").to_csv(
        directory / f"{MATCH}_dynamic_events.csv", index=False
    )
    return tmp_path


def test_table_has_one_row_per_engagement_with_clusters(opendata: Path) -> None:
    table = engagement_table(MATCH, opendata)
    assert list(table["event_id"]) == ["e1", "e2", "e3", "e4", "e5"]
    assert list(table["cluster"]) == ["42_p1", "42_p1", "42_p2", "42_p3", "42_p4"]
    assert list(table["presser_id"].unique()) == [5]
    assert list(table["carrier_id"].unique()) == [9]
    assert list(table["starts_before_possession"]) == [True, False, False, False, False]
    for name in (*EVENT_NUMERIC, *EVENT_BOOLEAN):
        assert table[name].dtype == float
    assert table["goal_side_start"].eq(1.0).all()


def test_event_coordinates_are_mirrored_to_pitch_coordinates(opendata: Path) -> None:
    table = engagement_table(MATCH, opendata).set_index("event_id")
    assert (table.loc["e1", "presser_x"], table.loc["e1", "presser_y"]) == (10.0, -4.0)
    assert (table.loc["e2", "presser_x"], table.loc["e2", "presser_y"]) == (-10.0, 4.0)


def test_outcomes_follow_the_preregistration(opendata: Path) -> None:
    table = engagement_table(MATCH, opendata).set_index("event_id")
    np.testing.assert_array_equal(table[PRIMARY_OUTCOME].to_numpy()[:4], [1.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(table["y_turnover_wide"].to_numpy()[:4], [1.0, 1.0, 1.0, 0.0])
    # An unknown end type has no outcome.
    assert np.isnan(table.loc["e5", PRIMARY_OUTCOME])
    assert np.isnan(table.loc["e5", "y_turnover_wide"])
    assert list(table["y_force_backward"]) == [0.0, 1.0, 0.0, 0.0, 0.0]
    assert table["possession_epv_total"].eq(0.01).all()


def test_possession_labels_come_from_the_linked_possession(opendata: Path) -> None:
    table = engagement_table(MATCH, opendata).set_index("event_id")
    assert table.loc["e1", "overall_pressure_start"] == "high_pressure"
    assert pd.isna(table.loc["e3", "overall_pressure_start"])
    assert table.loc["e4", "reception_difficulty_start"] == "easy"


def test_without_outcomes_no_outcome_column_is_present(opendata: Path) -> None:
    table = engagement_table(MATCH, opendata, with_outcomes=False)
    assert not [c for c in table.columns if c.startswith("y_") or "end_type" in c or "epv" in c]
    assert set(POSSESSION_LABELS) <= set(table.columns)


def test_pose_available_needs_both_players_in_one_frame_of_the_window() -> None:
    engagements = pd.DataFrame(
        {
            "frame_start": [100, 100, 101, 200],
            "presser_id": [5, 5, 5, 5],
            "carrier_id": [9, 8, 9, 9],
        }
    )
    pose_keys = pd.DataFrame(
        {
            # Target of frame 100 is pose frame 250, of frame 101 it is 252.5.
            "frame": [248, 248, 250, 252, 255, 255, 497, 503],
            "player_id": [5, 9, 5, 8, 5, 9, 5, 9],
        }
    )
    available = pose_available(engagements, pose_keys)
    # Row 0: both in frame 248. Row 1: presser and carrier 8 never share a frame.
    # Row 2: the window of 252.5 is 251 to 254, frame 255 is outside. Row 3: different frames.
    np.testing.assert_array_equal(available, [True, False, False, False])
    np.testing.assert_array_equal(
        pose_available(engagements, pose_keys, window=3), [True, False, True, False]
    )
