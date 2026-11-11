"""Download, streaming reader and cached position table for the 10 fps XY tracking.

The tracking files are Git LFS objects of ``github.com/SkillCorner/opendata``. They are
fetched from the GitHub LFS media endpoint and verified against the size and sha256 of the
pointer files, which are packaged in ``tracking_manifest.json``. ``git-lfs`` is not needed.

Usage: ``python -m blindside.data.tracking`` fetches the tracking of every match into
``data/raw/tracking`` and verifies it. Files that are already verified are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from array import array
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from blindside.data.download import (
    DownloadError,
    ProgressCallback,
    download_file,
    print_progress,
)
from blindside.data.paths import cache_dir, tracking_raw_dir


@dataclass(frozen=True)
class TrackingFile:
    """One downloadable tracking file with the values needed to verify it."""

    match_id: int
    file: str
    size_bytes: int
    sha256: str
    url: str


@cache
def load_tracking_manifest() -> dict[str, Any]:
    """Return the packaged tracking manifest as a dictionary."""
    text = resources.files("blindside.data").joinpath("tracking_manifest.json").read_text("utf-8")
    return json.loads(text)


def tracking_match_ids() -> tuple[int, ...]:
    """Match ids for which tracking exists (all matches of the open data), ascending."""
    return tuple(sorted(int(match_id) for match_id in load_tracking_manifest()["matches"]))


def tracking_file(match_id: int) -> TrackingFile:
    """Look up the tracking file of one match.

    Raises
    ------
    ValueError
        If the match is not part of the open data.
    """
    manifest = load_tracking_manifest()
    entry = manifest["matches"].get(str(match_id))
    if entry is None:
        raise ValueError(
            f"Match {match_id} is not in the open data. Available: {list(tracking_match_ids())}"
        )
    return TrackingFile(
        match_id=int(match_id),
        file=entry["file"],
        size_bytes=entry["size_bytes"],
        sha256=entry["sha256"],
        url=f"{manifest['base_url']}/{entry['file']}",
    )


def download_tracking(
    match_id: int,
    dest_dir: Path | None = None,
    *,
    progress: ProgressCallback | None = None,
    **kwargs: float,
) -> Path:
    """Download and verify the tracking file of one match.

    Returns
    -------
    Path
        The verified ``{match_id}_tracking_extrapolated.jsonl`` in ``dest_dir``, default
        ``data/raw/tracking``.
    """
    entry = tracking_file(match_id)
    dest = Path(dest_dir or tracking_raw_dir()) / Path(entry.file).name
    return download_file(
        entry.url,
        dest,
        size_bytes=entry.size_bytes,
        sha256=entry.sha256,
        progress=progress,
        **kwargs,
    )


def iter_tracking_frames(path: Path | str) -> Iterator[dict[str, Any]]:
    """Yield the frames of a tracking ``.jsonl`` file one at a time."""
    with Path(path).open("rb") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def read_positions(path: Path | str, match_id: int) -> pd.DataFrame:
    """Player positions of one match as a long table.

    One row per player and in-period frame: ``match_id``, ``frame`` (10 fps tracking
    clock), ``period``, ``player_id``, ``x``, ``y`` (metres, origin at the pitch centre)
    and ``is_detected`` (False where the position is extrapolated). The file is streamed,
    only the position columns are kept.
    """
    frames, periods, players = array("i"), array("b"), array("i")
    xy, detected = array("f"), array("b")
    for record in iter_tracking_frames(path):
        period = record.get("period")
        if period is None:
            continue
        frame = record["frame"]
        for player in record.get("player_data") or ():
            x, y = player.get("x"), player.get("y")
            if x is None or y is None:
                continue
            frames.append(frame)
            periods.append(period)
            players.append(player["player_id"])
            xy.append(x)
            xy.append(y)
            detected.append(bool(player.get("is_detected")))
    positions = np.frombuffer(xy, dtype=np.float32).reshape(-1, 2)
    return pd.DataFrame(
        {
            "match_id": np.full(len(frames), match_id, dtype=np.int32),
            "frame": np.frombuffer(frames, dtype=np.int32),
            "period": np.frombuffer(periods, dtype=np.int8),
            "player_id": np.frombuffer(players, dtype=np.int32),
            "x": positions[:, 0],
            "y": positions[:, 1],
            "is_detected": np.frombuffer(detected, dtype=np.int8).astype(bool),
        }
    )


def positions_cache_path(match_id: int) -> Path:
    """Parquet path of the cached position table of one match."""
    return cache_dir() / "tracking" / f"{match_id}_positions.parquet"


def cached_positions(
    match_id: int, path: Path | str | None = None, *, refresh: bool = False
) -> pd.DataFrame:
    """Position table of one match, downloading the tracking and caching the table on first use.

    Parameters
    ----------
    match_id
        A match of the open data.
    path
        Tracking file to read. By default it is downloaded to ``data/raw/tracking``.
    refresh
        Rebuild the cached table.
    """
    target = positions_cache_path(match_id)
    if target.exists() and not refresh:
        return pd.read_parquet(target)
    table = read_positions(path or download_tracking(match_id), match_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".parquet.tmp")
    table.to_parquet(temporary, index=False)
    temporary.replace(target)
    return table


@dataclass(frozen=True)
class PositionGrid:
    """Positions of one match on a dense frame grid.

    ``xy`` has shape ``(frames, players, 2)`` and is NaN where a player has no position.
    ``period`` has one entry per frame, 0 outside the periods.
    """

    first_frame: int
    players: np.ndarray
    xy: np.ndarray
    period: np.ndarray

    def row(self, frame: np.ndarray | int) -> np.ndarray:
        """Index along the frame axis, -1 for frames outside the grid."""
        index = np.asarray(frame, dtype=np.int64) - self.first_frame
        return np.where((index >= 0) & (index < len(self.xy)), index, -1)

    def column(self, player_id: np.ndarray | int) -> np.ndarray:
        """Index along the player axis, -1 for players without any position."""
        player_id = np.asarray(player_id, dtype=np.int64)
        index = np.clip(np.searchsorted(self.players, player_id), 0, len(self.players) - 1)
        return np.where(self.players[index] == player_id, index, -1)

    def at(self, frame: np.ndarray, player_id: np.ndarray) -> np.ndarray:
        """Positions ``(n, 2)`` of players at frames, NaN where either is unknown."""
        row, column = self.row(frame), self.column(player_id)
        out = self.xy[np.maximum(row, 0), np.maximum(column, 0)].astype(float)
        out[(row < 0) | (column < 0)] = np.nan
        return out

    def period_at(self, frame: np.ndarray) -> np.ndarray:
        """Period of each frame, 0 outside the grid and outside the periods."""
        row = self.row(frame)
        return np.where(row >= 0, self.period[np.maximum(row, 0)], 0)


def position_grid(positions: pd.DataFrame) -> PositionGrid:
    """Dense grid from the long position table of one match."""
    frame = positions["frame"].to_numpy()
    first, last = int(frame.min()), int(frame.max())
    players = np.unique(positions["player_id"].to_numpy())
    xy = np.full((last - first + 1, len(players), 2), np.nan, dtype=np.float32)
    column = np.searchsorted(players, positions["player_id"].to_numpy())
    xy[frame - first, column] = positions[["x", "y"]].to_numpy(dtype=np.float32)
    period = np.zeros(last - first + 1, dtype=np.int8)
    period[frame - first] = positions["period"].to_numpy()
    return PositionGrid(first_frame=first, players=players, xy=xy, period=period)


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description="Download and verify the tracking files.")
    parser.add_argument("--match", type=int, action="append", help="match id, repeatable")
    parser.add_argument("--dest-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    for match_id in args.match or tracking_match_ids():
        try:
            path = download_tracking(
                match_id, args.dest_dir, progress=print_progress(str(match_id))
            )
        except (DownloadError, ValueError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        print(f"{match_id}: verified {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
