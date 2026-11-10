"""Streaming reader for SkillCorner body pose files.

A full match is about 3.3 GB of JSON lines. The reader decodes one frame at a time
straight from the archive, so memory use is one frame regardless of file size.
"""

from __future__ import annotations

import gzip
import json
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any

from blindside.data.manifest import POSE_FPS, TRACKING_FPS


def tracking_to_pose_frame(tracking_frame: float) -> float:
    """Convert a 10 fps tracking frame to the 25 fps pose clock.

    Odd tracking frames fall halfway between two pose frames.
    """
    return tracking_frame * POSE_FPS / TRACKING_FPS


def pose_to_tracking_frame(pose_frame: float) -> float:
    """Convert a 25 fps pose frame to the 10 fps tracking clock."""
    return pose_frame * TRACKING_FPS / POSE_FPS


def _is_pose_member(member: str) -> bool:
    """True for a ``.jsonl`` zip member that is not macOS metadata.

    The published archives were zipped on macOS and carry ``__MACOSX/._*.jsonl`` entries.
    """
    parts = member.split("/")
    is_metadata = "__MACOSX" in parts or parts[-1].startswith("._")
    return member.lower().endswith(".jsonl") and not is_metadata


def _decode_lines(stream: IO[bytes]) -> Iterator[dict[str, Any]]:
    for line in stream:
        if line.strip():
            yield json.loads(line)


def iter_pose_frames(path: Path | str) -> Iterator[dict[str, Any]]:
    """Yield pose frames one by one from a ``.jsonl.zip``, ``.jsonl.gz`` or ``.jsonl`` file.

    Each frame is the decoded record as published: ``frame``, ``period``, ``ball_data``,
    ``player_data`` and so on. ``player_data[i]["joints"]`` is ``None`` when pose was not
    resolved, otherwise ``{joint: {"xyz": [x, y, z], "p90_mae_cm": float}}``.

    Raises
    ------
    ValueError
        For an unsupported file type or a zip that does not hold exactly one ``.jsonl``.
    """
    path = Path(path)
    name = path.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            members = [m for m in archive.namelist() if _is_pose_member(m)]
            if len(members) != 1:
                raise ValueError(
                    f"{path.name} must contain exactly one .jsonl file, found {members}"
                )
            with archive.open(members[0]) as stream:
                yield from _decode_lines(stream)
    elif name.endswith(".gz"):
        with gzip.open(path, "rb") as stream:
            yield from _decode_lines(stream)
    elif name.endswith(".jsonl"):
        with path.open("rb") as stream:
            yield from _decode_lines(stream)
    else:
        raise ValueError(f"Unsupported pose file type: {path.name}")
