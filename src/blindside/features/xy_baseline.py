"""XY-only baseline features of an on-ball engagement (docs/PREREGISTRATION.md section 5).

Features from tracking use positions up to the engagement start t0 and nothing after it.
Speeds are mean velocities over the 0.5 s that end at t0, which is the smoothing of
:func:`blindside.geometry.kinematics.central_velocity` shifted so that it needs no future frame.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from blindside.data.events import load_match_metadata
from blindside.data.manifest import TRACKING_FPS
from blindside.data.tracking import PositionGrid, cached_positions, position_grid
from blindside.features.engagements import (
    EVENT_BOOLEAN,
    EVENT_CATEGORICAL,
    EVENT_NUMERIC,
    POSSESSION_LABELS,
    engagement_table,
)

#: Tracking frames between t0 and the earlier position used for speeds (0.5 s at 10 fps).
LOOKBACK_FRAMES = 5
#: Radius in metres within which defenders around the carrier are counted.
DEFENDER_RADIUS_M = 10.0

TRACKING_NUMERIC: tuple[str, ...] = (
    "closing_speed",
    "presser_speed",
    "carrier_speed",
    "approach_angle_goal_deg",
    "approach_angle_heading_deg",
    "n_defenders_10m",
)
NUMERIC_FEATURES: tuple[str, ...] = (*EVENT_NUMERIC, *EVENT_BOOLEAN, *TRACKING_NUMERIC)
CATEGORICAL_FEATURES: tuple[str, ...] = (*EVENT_CATEGORICAL, *POSSESSION_LABELS)
BASELINE_FEATURES: tuple[str, ...] = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)
MISSING_LABEL = "missing"


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Unsigned angle in degrees, in [0, 180], between vectors ``(n, 2)``. NaN for a zero vector."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    dot = (a * b).sum(axis=1)
    angle = np.degrees(np.arctan2(np.abs(cross), dot))
    zero = (np.hypot(a[:, 0], a[:, 1]) == 0.0) | (np.hypot(b[:, 0], b[:, 1]) == 0.0)
    return np.where(zero, np.nan, angle)


def tracking_features(
    engagements: pd.DataFrame,
    grid: PositionGrid,
    team_of: Mapping[int, int],
    pitch_length: float,
    lookback: int = LOOKBACK_FRAMES,
    fps: float = TRACKING_FPS,
) -> pd.DataFrame:
    """Features of each engagement from XY tracking up to its start.

    Parameters
    ----------
    engagements
        Table of one match from :func:`blindside.features.engagements.engagement_table`.
    grid
        Positions of that match, see :func:`blindside.data.tracking.position_grid`.
    team_of
        ``{player_id: team_id}`` for the players of the match.
    pitch_length
        Length of the pitch in metres, for the position of the goals.
    lookback
        Frames between t0 and the earlier position. Both must lie in the same period.

    Returns
    -------
    One row per engagement, same order:

    ``has_tracking``
        Inclusion criterion 3: presser and carrier have a position at t0 and ``lookback``
        frames earlier, in the same period.
    ``closing_speed``
        Decrease of the presser to carrier distance over the lookback, m/s. Positive when
        the presser closes in.
    ``presser_speed``, ``carrier_speed``
        Mean speed over the lookback, m/s.
    ``approach_angle_goal_deg``
        Angle at the carrier between the direction to the presser and the direction to the
        centre of the goal the carrier attacks. 0 means the presser stands on that line.
    ``approach_angle_heading_deg``
        Angle between the carrier's direction of travel and the direction to the presser.
        0 means the carrier runs at the presser, 180 means the presser comes from behind.
        NaN for a carrier who did not move.
    ``n_defenders_10m``
        Players of the pressing team within 10 m of the carrier at t0, presser included.
    """
    t0 = engagements["frame_start"].to_numpy(dtype=np.int64)
    presser = engagements["presser_id"].to_numpy(dtype=np.int64)
    carrier = engagements["carrier_id"].to_numpy(dtype=np.int64)
    seconds = lookback / fps

    presser_now, presser_before = grid.at(t0, presser), grid.at(t0 - lookback, presser)
    carrier_now, carrier_before = grid.at(t0, carrier), grid.at(t0 - lookback, carrier)
    period_now, period_before = grid.period_at(t0), grid.period_at(t0 - lookback)
    has_tracking = (
        np.isfinite(
            np.column_stack([presser_now, presser_before, carrier_now, carrier_before])
        ).all(axis=1)
        & (period_now > 0)
        & (period_now == period_before)
    )

    to_presser = presser_now - carrier_now
    distance_now = np.hypot(to_presser[:, 0], to_presser[:, 1])
    gap_before = presser_before - carrier_before
    distance_before = np.hypot(gap_before[:, 0], gap_before[:, 1])
    presser_step = presser_now - presser_before
    carrier_step = carrier_now - carrier_before

    # ``attacking_side`` is the direction of the pressing team. The carrier attacks the other goal.
    presser_attacks_right = engagements["attacking_side"].to_numpy(dtype=object) == "left_to_right"
    goal = np.column_stack(
        [
            np.where(presser_attacks_right, -pitch_length / 2.0, pitch_length / 2.0),
            np.zeros(len(t0)),
        ]
    )

    teams = np.array([team_of.get(int(player), -1) for player in grid.players])
    row = grid.row(t0)
    everyone = grid.xy[np.maximum(row, 0)].astype(float)  # (engagements, players, 2)
    to_carrier = everyone - carrier_now[:, None, :]
    near = np.hypot(to_carrier[..., 0], to_carrier[..., 1]) <= DEFENDER_RADIUS_M
    pressing_team = teams[None, :] == engagements["presser_team_id"].to_numpy()[:, None]
    n_defenders = (near & pressing_team).sum(axis=1).astype(float)

    out = pd.DataFrame(
        {
            "has_tracking": has_tracking,
            "closing_speed": (distance_before - distance_now) / seconds,
            "presser_speed": np.hypot(presser_step[:, 0], presser_step[:, 1]) / seconds,
            "carrier_speed": np.hypot(carrier_step[:, 0], carrier_step[:, 1]) / seconds,
            "approach_angle_goal_deg": angle_between_deg(to_presser, goal - carrier_now),
            "approach_angle_heading_deg": angle_between_deg(carrier_step, to_presser),
            "n_defenders_10m": n_defenders,
        },
        index=engagements.index,
    )
    out.loc[~has_tracking, list(TRACKING_NUMERIC)] = np.nan
    return out


def design_frame(
    table: pd.DataFrame,
    numeric: tuple[str, ...] = NUMERIC_FEATURES,
    categorical: tuple[str, ...] = CATEGORICAL_FEATURES,
) -> pd.DataFrame:
    """Model input: numeric features as float, categorical features as strings.

    Missing categories become the level ``"missing"``. Missing numeric values stay NaN and
    are imputed inside the model pipeline with training statistics.
    """
    frame = pd.DataFrame(index=table.index)
    for name in numeric:
        frame[name] = table[name].to_numpy(dtype=float)
    for name in categorical:
        values = table[name].astype(object)
        frame[name] = values.where(values.notna(), MISSING_LABEL).astype(str)
    return frame


def baseline_table(
    match_id: int, root: Path | None = None, *, with_outcomes: bool = True
) -> pd.DataFrame:
    """Engagements of one match with every XY baseline feature.

    Joins :func:`blindside.features.engagements.engagement_table` with
    :func:`tracking_features`. The tracking file is downloaded on first use.

    Parameters
    ----------
    match_id
        A match of the open data.
    root
        Path of the opendata clone (events and match metadata).
    with_outcomes
        If False, no outcome column is read.
    """
    engagements = engagement_table(match_id, root, with_outcomes=with_outcomes)
    metadata = load_match_metadata(match_id, root)
    team_of = {int(player["id"]): int(player["team_id"]) for player in metadata["players"]}
    grid = position_grid(cached_positions(match_id))
    features = tracking_features(engagements, grid, team_of, float(metadata["pitch_length"]))
    return pd.concat([engagements, features], axis=1)
