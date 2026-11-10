"""Velocity from XY positions on a regular frame grid."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.floating]


def central_velocity(
    xy: ArrayLike,
    fps: float,
    half_window: int,
    segment: ArrayLike | None = None,
) -> FloatArray:
    """Smoothed velocity by a central difference over ``2 * half_window`` frames.

    ``v[t] = (xy[t + h] - xy[t - h]) / (2 h / fps)``. This is the mean velocity over the
    window, which is the smoothing that raw frame-to-frame speeds need.

    Parameters
    ----------
    xy
        Positions of shape ``(frames, ..., 2)`` on a gap-free frame grid along axis 0.
        Use NaN for missing positions.
    fps
        Frames per second of the grid.
    half_window
        ``h`` in frames, at least 1.
    segment
        Optional label per frame, shape ``(frames,)`` (for example the period). The velocity
        is NaN wherever the window is not fully inside one segment.

    Returns
    -------
    Velocities in units per second, same shape as ``xy``. NaN within ``half_window`` frames
    of either end and wherever one of the two positions is missing.
    """
    if half_window < 1:
        raise ValueError("half_window must be at least 1")
    xy = np.asarray(xy, dtype=float)
    velocity = np.full_like(xy, np.nan)
    h = half_window
    if xy.shape[0] <= 2 * h:
        return velocity
    velocity[h:-h] = (xy[2 * h :] - xy[: -2 * h]) * (fps / (2.0 * h))
    if segment is not None:
        segment = np.asarray(segment)
        same = (segment[2 * h :] == segment[h:-h]) & (segment[: -2 * h] == segment[h:-h])
        velocity[h:-h][~same] = np.nan
    return velocity
