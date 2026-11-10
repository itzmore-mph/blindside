"""Infer the position of the broadcast camera from the projected image corners.

The data does not say on which side of the pitch the main camera stands. Every pose frame
carries ``image_corners_projection``: the four corners of the broadcast image projected
onto the pitch plane. Three properties of that quadrilateral identify the camera side:

1. The bottom edge of the image shows the ground closest to the camera.
2. The near edge is shorter on the ground than the far edge.
3. The left and right image edges, extended towards the camera, meet at a point on the
   ground next to the camera. For a main camera this point lies on the halfway line.

The meeting point of cue 3 is used as the camera position. It lies somewhat behind the
true camera (by camera height times the tangent of the downward tilt), so distances to it
overstate the true ground distance by a roughly constant offset. They rank players by
distance to the camera correctly and are used for banding only.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from blindside.data.paths import cache_dir
from blindside.data.pose import iter_pose_frames

CORNER_KEYS: tuple[str, ...] = (
    "x_top_left",
    "y_top_left",
    "x_bottom_left",
    "y_bottom_left",
    "x_bottom_right",
    "y_bottom_right",
    "x_top_right",
    "y_top_right",
)
#: Share of frames in which a cue must point to the same side for the inference to count.
MIN_CUE_AGREEMENT = 0.99
#: A main camera stands at the halfway line. Largest accepted |x| of the estimate in metres.
MAX_HALFWAY_OFFSET_M = 5.0


def camera_ground_points(corners: np.ndarray) -> np.ndarray:
    """Meeting point of the left and right image edges on the pitch plane, per frame.

    Parameters
    ----------
    corners
        Array ``(frames, 8)`` in the order of :data:`CORNER_KEYS`.

    Returns
    -------
    Array ``(frames, 2)``. NaN where the two edges are parallel.
    """
    corners = np.asarray(corners, dtype=float)
    top_left, bottom_left = corners[:, 0:2], corners[:, 2:4]
    bottom_right, top_right = corners[:, 4:6], corners[:, 6:8]
    d_left, d_right = bottom_left - top_left, bottom_right - top_right
    denominator = d_left[:, 0] * d_right[:, 1] - d_left[:, 1] * d_right[:, 0]
    offset = top_right - top_left
    denominator = np.where(denominator == 0.0, np.nan, denominator)
    t = (offset[:, 0] * d_right[:, 1] - offset[:, 1] * d_right[:, 0]) / denominator
    return top_left + t[:, None] * d_left


@dataclass(frozen=True)
class CameraEstimate:
    """Inferred camera position of one match with the evidence behind it.

    ``side`` is ``"-y"`` or ``"+y"``: the touchline behind which the camera stands.
    ``x``, ``y`` is the median meeting point of the image edges. The three ``share_*``
    fields are the shares of frames in which each cue points to ``side``.
    """

    match_id: int
    side: str
    x: float
    y: float
    n_frames: int
    share_bottom_edge: float
    share_near_edge_shorter: float
    share_meeting_point: float
    x_iqr: float
    y_iqr: float
    per_period_xy: dict[str, tuple[float, float]]

    @property
    def reliable(self) -> bool:
        """True if all cues agree in at least 99% of frames and the point is on the halfway line."""
        shares = (self.share_bottom_edge, self.share_near_edge_shorter, self.share_meeting_point)
        return min(shares) >= MIN_CUE_AGREEMENT and abs(self.x) <= MAX_HALFWAY_OFFSET_M

    def distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Ground distance in metres from pitch positions to the inferred camera point."""
        return np.hypot(np.asarray(x, dtype=float) - self.x, np.asarray(y, dtype=float) - self.y)


def estimate_camera(corners: np.ndarray, periods: np.ndarray, match_id: int) -> CameraEstimate:
    """Infer the camera side and position from image corners of many frames.

    Parameters
    ----------
    corners
        Array ``(frames, 8)`` in the order of :data:`CORNER_KEYS`, without NaN rows.
    periods
        Period of each frame, used to confirm that the estimate does not change at half time.
    """
    corners = np.asarray(corners, dtype=float)
    periods = np.asarray(periods)
    y_top = (corners[:, 1] + corners[:, 7]) / 2.0
    y_bottom = (corners[:, 3] + corners[:, 5]) / 2.0
    top_width = np.hypot(corners[:, 6] - corners[:, 0], corners[:, 7] - corners[:, 1])
    bottom_width = np.hypot(corners[:, 4] - corners[:, 2], corners[:, 5] - corners[:, 3])
    points = camera_ground_points(corners)
    finite = np.isfinite(points).all(axis=1)

    negative = np.median(y_bottom - y_top) < 0  # bottom of the image lies towards -y
    sign = -1.0 if negative else 1.0
    centre_y = (y_top + y_bottom) / 2.0
    x_q = np.percentile(points[finite, 0], [25, 50, 75])
    y_q = np.percentile(points[finite, 1], [25, 50, 75])
    return CameraEstimate(
        match_id=match_id,
        side="-y" if negative else "+y",
        x=float(x_q[1]),
        y=float(y_q[1]),
        n_frames=len(corners),
        share_bottom_edge=float(np.mean(sign * (y_bottom - y_top) > 0)),
        share_near_edge_shorter=float(np.mean(bottom_width < top_width)),
        share_meeting_point=float(np.mean(sign * (points[finite, 1] - centre_y[finite]) > 0)),
        x_iqr=float(x_q[2] - x_q[0]),
        y_iqr=float(y_q[2] - y_q[0]),
        per_period_xy={
            str(int(period)): tuple(
                float(v) for v in np.median(points[finite & (periods == period)], axis=0)
            )
            for period in np.unique(periods)
        },
    )


def read_corners(path: Path | str, step: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frames, periods and image corners of every ``step``-th in-period frame that has corners."""
    frames, periods, corners = [], [], []
    for record in iter_pose_frames(path, step=step):
        projection = record.get("image_corners_projection")
        if record.get("period") is None or not projection:
            continue
        values = [projection.get(key) for key in CORNER_KEYS]
        if any(value is None for value in values):
            continue
        frames.append(record["frame"])
        periods.append(record["period"])
        corners.append(values)
    return np.array(frames), np.array(periods), np.array(corners, dtype=float).reshape(-1, 8)


def cached_camera_estimate(
    path: Path | str, match_id: int, step: int = 50, refresh: bool = False
) -> CameraEstimate:
    """Camera estimate of a pose file, cached as JSON in ``data/cache/camera/``."""
    target = cache_dir() / "camera" / f"{Path(path).name.split('.')[0]}__step{step}.json"
    if target.exists() and not refresh:
        stored = json.loads(target.read_text())
        stored["per_period_xy"] = {k: tuple(v) for k, v in stored["per_period_xy"].items()}
        return CameraEstimate(**stored)
    _, periods, corners = read_corners(path, step)
    estimate = estimate_camera(corners, periods, match_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(estimate), indent=2) + "\n")
    return estimate
