"""Default locations for downloaded and cached data.

Nothing below ``data/`` is committed. Both locations can be redirected with
environment variables so the package also works outside a repository checkout.
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


SAMPLE_POSE_FILE = "sample_1925299_phase406.jsonl.gz"


def sample_pose_path() -> Path:
    """The pose sample phase committed in the opendata repository (309 frames, 12.3 s)."""
    return opendata_dir() / "data" / "bodypose" / SAMPLE_POSE_FILE
