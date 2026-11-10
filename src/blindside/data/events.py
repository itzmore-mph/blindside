"""Loaders for dynamic events and match metadata from a local SkillCorner/opendata clone."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from blindside.data.paths import opendata_dir

ON_BALL_ENGAGEMENT = "on_ball_engagement"


def _match_dir(match_id: int, root: Path | None) -> Path:
    directory = Path(root or opendata_dir()) / "data" / "matches" / str(match_id)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"{directory} not found. Clone github.com/SkillCorner/opendata and pass its path, "
            "or set BLINDSIDE_OPENDATA_DIR."
        )
    return directory


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
        Path of the opendata clone, default :func:`blindside.data.paths.opendata_dir`.
    columns
        Columns to read. Pass a list to keep outcome columns out of an analysis step.
    """
    path = _match_dir(match_id, root) / f"{match_id}_dynamic_events.csv"
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
    path = _match_dir(match_id, root) / f"{match_id}_match.json"
    return json.loads(path.read_text("utf-8"))
