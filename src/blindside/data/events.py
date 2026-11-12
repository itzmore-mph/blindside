"""Loaders for dynamic events and match metadata from a local SkillCorner/opendata clone."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from blindside.data.opendata import (
    events_relative,
    metadata_relative,
    opendata_file,
    phases_relative,
)

ON_BALL_ENGAGEMENT = "on_ball_engagement"


def load_dynamic_events(
    match_id: int,
    root: Path | None = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read the dynamic events of one match.

    Parameters
    ----------
    match_id
        SkillCorner match id.
    root
        Path of an opendata clone. By default a clone in ``$BLINDSIDE_OPENDATA_DIR`` or
        ``../opendata`` is used, and without one the file is downloaded
        (:func:`blindside.data.opendata.opendata_file`).
    columns
        Columns to read. Pass a list to keep outcome columns out of an analysis step.
    """
    path = opendata_file(events_relative(match_id), root)
    return pd.read_csv(
        path,
        usecols=list(columns) if columns is not None else None,
        dtype_backend="pyarrow",
    )


def on_ball_engagements(events: pd.DataFrame) -> pd.DataFrame:
    """Rows of type ``on_ball_engagement``, index reset."""
    return events.loc[events["event_type"] == ON_BALL_ENGAGEMENT].reset_index(drop=True)


def load_phases_of_play(match_id: int, root: Path | None = None) -> pd.DataFrame:
    """Read ``{match_id}_phases_of_play.csv``: one row per team phase with its frame range.

    Without a local clone the file is downloaded, which works for the matches whose phases
    are in the manifest (the two pose matches).
    """
    return pd.read_csv(opendata_file(phases_relative(match_id), root), dtype_backend="pyarrow")


def phase_at(phases: pd.DataFrame, frame: np.ndarray, column: str) -> np.ndarray:
    """Value of a phase column at tracking frames.

    A frame belongs to the phase with ``frame_start <= frame < frame_end``.

    Parameters
    ----------
    phases
        Table of :func:`load_phases_of_play`. Phases do not overlap, and gaps lie between them.
    frame
        Tracking frames (10 fps), shape ``(n,)``.
    column
        Column to read, for example ``"team_out_of_possession_phase_type"``.

    Returns
    -------
    Object array ``(n,)``, None for a frame outside every phase.
    """
    ordered = phases.sort_values("frame_start")
    start = ordered["frame_start"].to_numpy(dtype=np.int64)
    end = ordered["frame_end"].to_numpy(dtype=np.int64)
    values = ordered[column].astype(object).to_numpy(dtype=object)
    frame = np.asarray(frame, dtype=np.int64)
    out = np.full(frame.shape, None, dtype=object)
    if len(start) == 0:
        return out
    index = np.searchsorted(start, frame, side="right") - 1
    inside = (index >= 0) & (frame < end[np.maximum(index, 0)])
    out[inside] = values[index[inside]]
    return out


def load_match_metadata(match_id: int, root: Path | None = None) -> dict[str, Any]:
    """Read ``{match_id}_match.json`` (periods, pitch size, teams, players)."""
    path = opendata_file(metadata_relative(match_id), root)
    return json.loads(path.read_text("utf-8"))
