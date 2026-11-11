"""Default locations for downloaded and cached data.

Nothing below ``data/`` is committed. Both locations can be redirected with
environment variables so the package also works outside a repository checkout.

Files of ``github.com/SkillCorner/opendata`` (events, match metadata, the pose sample phase)
are read from a local clone if there is one and are downloaded otherwise, see
:mod:`blindside.data.opendata`.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR_ENV = "BLINDSIDE_DATA_DIR"
OPENDATA_DIR_ENV = "BLINDSIDE_OPENDATA_DIR"


def data_dir() -> Path:
    """Root for raw downloads and caches: ``$BLINDSIDE_DATA_DIR`` or ``./data``."""
    return Path(os.environ.get(DATA_DIR_ENV, Path.cwd() / "data"))


def pose_raw_dir() -> Path:
    """Directory holding the downloaded pose archives."""
    return data_dir() / "raw" / "bodypose"


def tracking_raw_dir() -> Path:
    """Directory holding the downloaded tracking files."""
    return data_dir() / "raw" / "tracking"


def cache_dir() -> Path:
    """Directory for derived intermediate tables."""
    return data_dir() / "cache"


def opendata_dir() -> Path:
    """Local clone of SkillCorner/opendata: ``$BLINDSIDE_OPENDATA_DIR`` or ``../opendata``."""
    return Path(os.environ.get(OPENDATA_DIR_ENV, Path.cwd().parent / "opendata"))


def opendata_download_dir() -> Path:
    """Directory that mirrors the opendata repository for files fetched without a clone."""
    return data_dir() / "raw" / "opendata"


def opendata_path(relative: str) -> Path:
    """Where one file of the opendata repository is, or where it will be downloaded to.

    ``$BLINDSIDE_OPENDATA_DIR`` wins if set. Otherwise the file in ``../opendata`` if it
    exists there, else the path below :func:`opendata_download_dir`.

    Parameters
    ----------
    relative
        Path inside the repository, for example ``data/matches/1925299/1925299_match.json``.
    """
    if OPENDATA_DIR_ENV in os.environ:
        return Path(os.environ[OPENDATA_DIR_ENV]) / relative
    clone = Path.cwd().parent / "opendata" / relative
    return clone if clone.exists() else opendata_download_dir() / relative


SAMPLE_POSE_FILE = "sample_1925299_phase406.jsonl.gz"
SAMPLE_POSE_RELATIVE = f"data/bodypose/{SAMPLE_POSE_FILE}"
SAMPLE_MATCH_ID = 1925299


def sample_pose_path() -> Path:
    """The pose sample phase of the opendata repository (309 frames, 12.3 s).

    The path may not exist yet. :func:`blindside.data.opendata.sample_pose_file` downloads it.
    """
    return opendata_path(SAMPLE_POSE_RELATIVE)
