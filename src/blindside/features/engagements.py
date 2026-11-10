"""Engagement table: one row per on-ball engagement, as fixed in docs/PREREGISTRATION.md.

The table carries the identifiers, the cluster (the pressed possession), the features that
come straight from the dynamic events and, on request, the outcomes. Event coordinates are
direction-normalised in the source (the acting team attacks towards +x). They are mirrored
back to pitch coordinates here, so that they can be joined with tracking and pose.

Feature columns and outcome columns are read through separate lists. A caller that must not
see outcomes passes ``with_outcomes=False`` and the outcome columns are never loaded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from blindside.data.events import ON_BALL_ENGAGEMENT, load_dynamic_events

PLAYER_POSSESSION = "player_possession"
PRIMARY_OUTCOME = "y_possession_loss"

#: Numeric features taken from the engagement row, all measured at engagement start.
EVENT_NUMERIC: tuple[str, ...] = (
    "interplayer_distance_start",
    "player_targeted_distance_to_goal_start",
    "player_targeted_angle_to_goal_start",
    "delta_to_last_defensive_line_start",
    "n_player_targeted_opponents_within_5m_start",
    "n_player_targeted_teammates_within_5m_start",
)
EVENT_BOOLEAN: tuple[str, ...] = ("goal_side_start", "penalty_area_start")
EVENT_CATEGORICAL: tuple[str, ...] = (
    "third_start",
    "channel_start",
    "event_subtype",
    "team_out_of_possession_phase_type",
    "game_state",
)
#: SkillCorner labels of the pressed possession, measured at possession start.
POSSESSION_LABELS: tuple[str, ...] = ("overall_pressure_start", "reception_difficulty_start")

_ID_COLUMNS: tuple[str, ...] = (
    "event_id",
    "event_type",
    "frame_start",
    "period",
    "player_id",
    "player_in_possession_id",
    "team_id",
    "attacking_side",
    "x_start",
    "y_start",
    "associated_player_possession_event_id",
    "associated_player_possession_frame_start",
)
_OUTCOME_COLUMNS: tuple[str, ...] = (
    "associated_player_possession_end_type",
    "force_backward",
    "pass_outcome",
    "possession_epv_total",
)


def _to_float(series: pd.Series) -> np.ndarray:
    """Numeric or boolean column as float64 with NaN for missing values."""
    return pd.to_numeric(series.astype(object), errors="coerce").to_numpy(dtype=float)


def _to_label(series: pd.Series) -> np.ndarray:
    """Categorical column as Python strings with None for missing values."""
    values = series.astype(object)
    return values.where(values.notna(), None).to_numpy(dtype=object)


def engagement_table(
    match_id: int, root: Path | None = None, *, with_outcomes: bool = True
) -> pd.DataFrame:
    """On-ball engagements of one match with event features and, optionally, outcomes.

    Parameters
    ----------
    match_id
        SkillCorner match id.
    root
        Path of the opendata clone, default :func:`blindside.data.paths.opendata_dir`.
    with_outcomes
        If False, no outcome column is read from the file.

    Returns
    -------
    One row per engagement, in file order:

    ``match_id``, ``event_id``, ``possession_id``, ``cluster``
        Identifiers. ``cluster`` is ``"{match_id}_{possession_id}"``.
    ``frame_start``, ``period``, ``possession_frame_start``, ``starts_before_possession``
        Engagement start t0 on the 10 fps clock and its relation to the possession start.
    ``presser_id``, ``carrier_id``, ``presser_team_id``, ``attacking_side``
        ``attacking_side`` is the direction of the pressing team.
    ``presser_x``, ``presser_y``
        Presser position at t0 in pitch coordinates (mirrored for ``right_to_left``).
    Event features
        :data:`EVENT_NUMERIC` and :data:`EVENT_BOOLEAN` as float, :data:`EVENT_CATEGORICAL`
        and :data:`POSSESSION_LABELS` as strings (missing where the source is empty).
    Outcomes, only with ``with_outcomes``
        ``possession_end_type``, ``y_possession_loss`` (primary, NaN where the end type is
        unknown), ``y_turnover_wide`` (loss or unsuccessful pass), ``y_force_backward``
        and ``possession_epv_total``.
    """
    columns = [
        *_ID_COLUMNS,
        *EVENT_NUMERIC,
        *EVENT_BOOLEAN,
        *EVENT_CATEGORICAL,
        *POSSESSION_LABELS,
    ]
    if with_outcomes:
        columns += _OUTCOME_COLUMNS
    events = load_dynamic_events(match_id, root, columns=columns)
    rows = events.loc[events["event_type"] == ON_BALL_ENGAGEMENT].reset_index(drop=True)
    possessions = events.loc[events["event_type"] == PLAYER_POSSESSION].set_index("event_id")
    linked = possessions.reindex(rows["associated_player_possession_event_id"].to_numpy())

    mirror = np.where(rows["attacking_side"].to_numpy(dtype=object) == "right_to_left", -1.0, 1.0)
    frame_start = rows["frame_start"].to_numpy(dtype=np.int64)
    possession_start = _to_float(rows["associated_player_possession_frame_start"])
    possession_id = rows["associated_player_possession_event_id"].to_numpy(dtype=object)
    table: dict[str, np.ndarray] = {
        "match_id": np.full(len(rows), match_id, dtype=np.int64),
        "event_id": rows["event_id"].to_numpy(dtype=object),
        "possession_id": possession_id,
        "cluster": np.array([f"{match_id}_{p}" for p in possession_id], dtype=object),
        "frame_start": frame_start,
        "period": rows["period"].to_numpy(dtype=np.int64),
        "possession_frame_start": possession_start,
        "starts_before_possession": frame_start < possession_start,
        "presser_id": rows["player_id"].to_numpy(dtype=np.int64),
        "carrier_id": rows["player_in_possession_id"].to_numpy(dtype=np.int64),
        "presser_team_id": rows["team_id"].to_numpy(dtype=np.int64),
        "attacking_side": _to_label(rows["attacking_side"]),
        "presser_x": mirror * _to_float(rows["x_start"]),
        "presser_y": mirror * _to_float(rows["y_start"]),
    }
    for name in (*EVENT_NUMERIC, *EVENT_BOOLEAN):
        table[name] = _to_float(rows[name])
    for name in EVENT_CATEGORICAL:
        table[name] = _to_label(rows[name])
    for name in POSSESSION_LABELS:
        table[name] = _to_label(linked[name])

    if with_outcomes:
        end_type = _to_label(rows["associated_player_possession_end_type"])
        known = np.array([value is not None and value != "unknown" for value in end_type])
        loss = end_type == "possession_loss"
        failed_pass = (end_type == "pass") & (_to_label(linked["pass_outcome"]) == "unsuccessful")
        table["possession_end_type"] = end_type
        table[PRIMARY_OUTCOME] = np.where(known, loss.astype(float), np.nan)
        table["y_turnover_wide"] = np.where(known, (loss | failed_pass).astype(float), np.nan)
        table["y_force_backward"] = _to_float(rows["force_backward"])
        table["possession_epv_total"] = _to_float(linked["possession_epv_total"])
    return pd.DataFrame(table)


def pose_available(
    engagements: pd.DataFrame, pose_keys: pd.DataFrame, window: int = 2
) -> np.ndarray:
    """Inclusion criterion 1: presser and carrier have joints in the same pose frame near t0.

    Parameters
    ----------
    engagements
        Engagements of one match with ``frame_start``, ``presser_id`` and ``carrier_id``.
    pose_keys
        ``frame`` (25 fps pose clock) and ``player_id`` of every pose of that match, for
        example the two columns of the orientation table.
    window
        Pose frames within this distance of ``2.5 * frame_start`` are searched. The target
        is a half frame for odd tracking frames, so the window then holds four frames.

    Returns
    -------
    Boolean array: True where at least one pose frame in the window has joints for both.
    """
    frames = pose_keys["frame"].to_numpy(dtype=np.int64)
    players = pose_keys["player_id"].to_numpy(dtype=np.int64)
    present = set(zip(frames.tolist(), players.tolist(), strict=True))
    target = 2.5 * engagements["frame_start"].to_numpy(dtype=float)
    first = np.ceil(target - window).astype(np.int64)
    last = np.floor(target + window).astype(np.int64)
    presser = engagements["presser_id"].to_numpy(dtype=np.int64)
    carrier = engagements["carrier_id"].to_numpy(dtype=np.int64)
    out = np.zeros(len(engagements), dtype=bool)
    for i, (lo, hi, a, b) in enumerate(
        zip(first.tolist(), last.tolist(), presser.tolist(), carrier.tolist(), strict=True)
    ):
        out[i] = any((f, a) in present and (f, b) in present for f in range(lo, hi + 1))
    return out
