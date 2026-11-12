"""One engagement prepared for a drawing on the pitch: geometry, run paths, context.

Shared by ``scripts/plot_scene_example.py``, ``scripts/tactical_case_study.py`` and
``blindside reproduce --with-scene``. The engagement is picked by the fixed rule of
:func:`blindside.analysis.case_study.case_study_candidates`, which reads no outcome column.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from blindside.analysis.case_study import DISAGREE, case_study_candidates
from blindside.analysis.pipeline import engagement_geometry
from blindside.data.events import load_dynamic_events, load_match_metadata
from blindside.data.manifest import TRACKING_FPS
from blindside.data.tracking import PositionGrid, cached_positions, position_grid

#: Seconds of the run before the engagement start that are drawn.
PATH_SECONDS = 2.0


@dataclass(frozen=True)
class Scene:
    """One engagement with everything :func:`blindside.viz.case_study.plot_scene_cone` reads."""

    geometry: pd.Series
    others: pd.DataFrame
    pitch_dims: tuple[float, float]
    paths: dict[str, np.ndarray]
    n_candidates: int


def other_players(engagement: pd.Series, grid: PositionGrid | None = None) -> pd.DataFrame:
    """Everyone but presser and carrier at the engagement start, from the 10 fps tracking.

    Returns
    -------
    ``player_id``, ``x``, ``y`` in metres and the boolean ``pressing_team``.
    """
    match_id = int(engagement["match_id"])
    if grid is None:
        grid = position_grid(cached_positions(match_id))
    xy = grid.xy[int(grid.row(int(engagement["frame_start"])))].astype(float)
    team_of = {int(p["id"]): int(p["team_id"]) for p in load_match_metadata(match_id)["players"]}
    table = pd.DataFrame({"player_id": grid.players, "x": xy[:, 0], "y": xy[:, 1]})
    table["pressing_team"] = table["player_id"].map(team_of) == int(engagement["presser_team_id"])
    involved = table["player_id"].isin(
        [int(engagement["presser_id"]), int(engagement["carrier_id"])]
    )
    return table[~involved & table["x"].notna()].reset_index(drop=True)


def source_line(match_id: int, event_id: str) -> str:
    """Teams, clock time and match id of one engagement."""
    metadata = load_match_metadata(match_id)
    events = load_dynamic_events(match_id, columns=["event_id", "time_start", "period"])
    event = events.loc[events["event_id"] == event_id].iloc[0]
    teams = f"{metadata['home_team']['short_name']} against {metadata['away_team']['short_name']}"
    return f"{teams}, period {int(event['period'])} at {event['time_start']} (match {match_id})"


def run_path(
    grid: PositionGrid, frame_start: int, player_id: int, seconds: float = PATH_SECONDS
) -> np.ndarray:
    """Tracking positions ``(n, 2)`` of a player over the seconds before the engagement start.

    The path ends at ``frame_start`` and holds only frames of the same period with a
    position, so it never uses information from after the engagement start.
    """
    frames = np.arange(frame_start - int(round(seconds * TRACKING_FPS)), frame_start + 1)
    xy = grid.at(frames, np.full(len(frames), player_id))
    same_period = grid.period_at(frames) == grid.period_at(np.array([frame_start]))[0]
    return xy[same_period & np.isfinite(xy).all(axis=1)]


def player_label(match_id: int, player_id: int) -> str:
    """Short name and team of a player, for example ``"J. Smith (Sydney FC)"``."""
    metadata = load_match_metadata(match_id)
    teams = {
        int(metadata[side]["id"]): str(metadata[side]["short_name"])
        for side in ("home_team", "away_team")
    }
    for player in metadata["players"]:
        if int(player["id"]) == int(player_id):
            return f"{player['short_name']} ({teams.get(int(player['team_id']), 'unknown team')})"
    return f"player {player_id}"


def scene_example(
    test: pd.DataFrame,
    poses: pd.DataFrame,
    features: Mapping[str, pd.DataFrame],
    *,
    kind: str = DISAGREE,
    rank: int = 0,
) -> Scene:
    """The engagement of a kind at a rank of the fixed selection rule, ready to draw.

    Parameters
    ----------
    test, poses, features
        Included engagements, their poses and ``{bound: pose_feature_table(...)}``, as in
        :func:`blindside.analysis.case_study.case_study_candidates`.
    kind
        ``"disagree"`` (slow carrier, XY proxy blind, pose label in view) or
        ``"agree_blind"``.
    rank
        0 is the clearest candidate of the kind, 1 the next one.

    Raises
    ------
    ValueError
        If the rule leaves fewer than ``rank + 1`` candidates.
    """
    candidates = case_study_candidates(test, features)[kind]
    if not 0 <= rank < len(candidates):
        raise ValueError(f"rank {rank} is outside the {len(candidates)} candidates of {kind!r}")
    chosen = candidates.iloc[[rank]]
    engagement = chosen.iloc[0]
    match_id = int(engagement["match_id"])
    grid = position_grid(cached_positions(match_id))
    geometry = engagement_geometry(
        chosen, poses[poses["match_id"] == match_id], features, grid
    ).iloc[0]
    geometry = geometry.copy()
    geometry["match_id"], geometry["event_id"] = match_id, engagement["event_id"]
    geometry["presser_off_head_deg"] = float(
        features["upper"].loc[chosen.index[0], "presser_off_head_deg"]
    )
    geometry["source"] = source_line(match_id, str(engagement["event_id"]))
    for player in ("carrier", "presser"):
        geometry[f"{player}_name"] = player_label(match_id, int(engagement[f"{player}_id"]))
    frame_start = int(engagement["frame_start"])
    paths = {
        player: run_path(grid, frame_start, int(engagement[f"{player}_id"]))
        for player in ("carrier", "presser")
    }
    metadata = load_match_metadata(match_id)
    return Scene(
        geometry=geometry,
        others=other_players(engagement, grid),
        pitch_dims=(float(metadata["pitch_length"]), float(metadata["pitch_width"])),
        paths=paths,
        n_candidates=len(candidates),
    )
