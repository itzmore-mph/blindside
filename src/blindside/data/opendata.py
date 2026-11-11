"""Files of ``github.com/SkillCorner/opendata`` that are not Git LFS objects.

Dynamic events, match metadata and the body pose sample phase are small files in the
opendata repository. They are read from a local clone when one is found and are fetched
from ``raw.githubusercontent.com`` at the pinned commit otherwise, verified against the
size and sha256 in ``opendata_manifest.json``. A clean checkout of this package therefore
needs no clone of the data repository.
"""

from __future__ import annotations

import json
import os
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

from blindside.data.paths import (
    OPENDATA_DIR_ENV,
    SAMPLE_POSE_RELATIVE,
    opendata_download_dir,
    opendata_path,
)


@cache
def load_opendata_manifest() -> dict[str, Any]:
    """Return the packaged manifest of the non-LFS opendata files as a dictionary."""
    text = resources.files("blindside.data").joinpath("opendata_manifest.json").read_text("utf-8")
    return json.loads(text)


def events_relative(match_id: int) -> str:
    """Repository path of the dynamic events of one match."""
    return f"data/matches/{match_id}/{match_id}_dynamic_events.csv"


def metadata_relative(match_id: int) -> str:
    """Repository path of the match metadata of one match."""
    return f"data/matches/{match_id}/{match_id}_match.json"


def opendata_file(relative: str, root: Path | str | None = None) -> Path:
    """Return a local path of one opendata file, downloading it if no clone has it.

    Parameters
    ----------
    relative
        Path inside the repository, for example ``data/matches/1925299/1925299_match.json``.
    root
        An opendata clone to read from. With a root, or with ``$BLINDSIDE_OPENDATA_DIR``
        set, nothing is downloaded and a missing file is an error.

    Raises
    ------
    FileNotFoundError
        If an explicit clone does not hold the file, or the file is not in the manifest.
    blindside.data.download.DownloadError
        If the download fails or the file does not match the manifest.
    """
    if root is not None or OPENDATA_DIR_ENV in os.environ:
        path = Path(root) / relative if root is not None else opendata_path(relative)
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found. Point to a clone of github.com/SkillCorner/opendata, or "
                f"unset {OPENDATA_DIR_ENV} to let blindside download the file."
            )
        return path

    path = opendata_path(relative)
    if path.is_file() and opendata_download_dir() not in path.parents:
        return path  # a file of a local clone, trusted as is
    manifest = load_opendata_manifest()
    entry = manifest["files"].get(relative)
    if entry is None:
        raise FileNotFoundError(
            f"{relative} is not in the opendata manifest and no local clone holds it."
        )
    # Imported here so that ``python -m blindside.data.download`` runs without a double import.
    from blindside.data.download import download_file

    return download_file(
        f"{manifest['base_url']}/{relative}",
        path,
        size_bytes=entry["size_bytes"],
        sha256=entry["sha256"],
    )


def sample_pose_file() -> Path:
    """Local path of the pose sample phase (309 frames, 12.3 s, 1.9 MB), fetched if needed."""
    return opendata_file(SAMPLE_POSE_RELATIVE)
