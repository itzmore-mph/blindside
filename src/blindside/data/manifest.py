"""Packaged manifest of the SkillCorner body pose archives and format constants."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Any

POSE_FPS = 25
TRACKING_FPS = 10

#: The 29 joints of a resolved pose, in the order SkillCorner documents them.
JOINTS: tuple[str, ...] = (
    "nose",
    "neck",
    "lEye",
    "rEye",
    "lEar",
    "rEar",
    "lShoulder",
    "rShoulder",
    "lElbow",
    "rElbow",
    "lWrist",
    "rWrist",
    "lThumb",
    "rThumb",
    "lPinky",
    "rPinky",
    "midHip",
    "lHip",
    "rHip",
    "lKnee",
    "rKnee",
    "lAnkle",
    "rAnkle",
    "lHeel",
    "rHeel",
    "lBigToe",
    "rBigToe",
    "lSmallToe",
    "rSmallToe",
)


@dataclass(frozen=True)
class PoseArchive:
    """One downloadable pose archive with the values needed to verify it."""

    match_id: int
    fixture: str
    file: str
    size_bytes: int
    sha256: str
    url: str


@cache
def load_manifest() -> dict[str, Any]:
    """Return the packaged manifest as a dictionary."""
    text = resources.files("blindside.data").joinpath("pose_manifest.json").read_text("utf-8")
    return json.loads(text)


def pose_match_ids() -> tuple[int, ...]:
    """Match ids for which body pose exists, ascending."""
    return tuple(sorted(int(match_id) for match_id in load_manifest()["matches"]))


def pose_archive(match_id: int) -> PoseArchive:
    """Look up the archive of one match.

    Raises
    ------
    ValueError
        If the match has no body pose.
    """
    manifest = load_manifest()
    entry = manifest["matches"].get(str(match_id))
    if entry is None:
        raise ValueError(f"Match {match_id} has no body pose. Available: {list(pose_match_ids())}")
    return PoseArchive(
        match_id=int(match_id),
        fixture=entry["fixture"],
        file=entry["file"],
        size_bytes=entry["size_bytes"],
        sha256=entry["sha256"],
        url=f"{manifest['base_url']}/{entry['file']}",
    )
