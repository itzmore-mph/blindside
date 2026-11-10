"""Download, caching and loaders for pose, events and match metadata.

The downloader lives in :mod:`blindside.data.download` and is not re-exported here,
so that ``python -m blindside.data.download`` runs without a double import.
"""

from blindside.data.events import load_dynamic_events, load_match_metadata, on_ball_engagements
from blindside.data.manifest import (
    JOINTS,
    POSE_FPS,
    TRACKING_FPS,
    PoseArchive,
    pose_archive,
    pose_match_ids,
)
from blindside.data.pose import iter_pose_frames, pose_to_tracking_frame, tracking_to_pose_frame

__all__ = [
    "JOINTS",
    "POSE_FPS",
    "TRACKING_FPS",
    "PoseArchive",
    "iter_pose_frames",
    "load_dynamic_events",
    "load_match_metadata",
    "on_ball_engagements",
    "pose_archive",
    "pose_match_ids",
    "pose_to_tracking_frame",
    "tracking_to_pose_frame",
]
