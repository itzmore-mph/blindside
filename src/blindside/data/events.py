"""Loaders for dynamic events and match metadata from a local SkillCorner/opendata clone."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from blindside.data.opendata import events_relative, metadata_relative, opendata_file

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


def load_match_metadata(match_id: int, root: Path | None = None) -> dict[str, Any]:
    """Read ``{match_id}_match.json`` (periods, pitch size, teams, players)."""
    path = opendata_file(metadata_relative(match_id), root)
    return json.loads(path.read_text("utf-8"))
