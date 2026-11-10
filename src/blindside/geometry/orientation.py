"""Facing angles in the pitch plane from body pose joints.

Convention (verified against the velocity heading, docs/FEASIBILITY.md section 5):
the facing direction of a left-right joint pair is the left-to-right axis rotated by
+90 degrees. Angles are radians in (-pi, pi], 0 points along +x, counter-clockwise positive.
Only x and y are used. The z axis of the pose data is not accurate in pitch coordinates.

Front/back ambiguity
--------------------
The rule above depends on the left and right labels. If the pose model swaps them, the
result points backwards. Two cues do not depend on those labels: the nose lies in front of
the ear midpoint, and the toes lie in front of the heels. :func:`anatomical_forward` builds
such a cue and :func:`resolve_front_back` compares a facing angle against it. The decision
is returned as a flag, never applied silently.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from blindside.geometry.circular import angular_difference, wrap_angle

FloatArray = NDArray[np.floating]
BoolArray = NDArray[np.bool_]

#: Rotation from the left-to-right axis to the facing direction.
FACING_ROTATION = np.pi / 2.0

#: Joint pair (left, right) of each body segment.
SEGMENT_JOINTS: dict[str, tuple[str, str]] = {
    "torso": ("lShoulder", "rShoulder"),
    "hips": ("lHip", "rHip"),
    "head": ("lEar", "rEar"),
}

#: Below this length in metres a direction is undefined and NaN is returned.
MIN_LENGTH_M = 1e-6


def segment_width(left_xy: ArrayLike, right_xy: ArrayLike) -> FloatArray:
    """Ground-plane distance between two joints, shape ``(...)`` for inputs ``(..., 2)``."""
    delta = np.asarray(right_xy, dtype=float) - np.asarray(left_xy, dtype=float)
    return np.hypot(delta[..., 0], delta[..., 1])


def vector_angle(vector_xy: ArrayLike) -> FloatArray:
    """Direction of ground-plane vectors of shape ``(..., 2)``. NaN for a zero vector."""
    vector_xy = np.asarray(vector_xy, dtype=float)
    x, y = vector_xy[..., 0], vector_xy[..., 1]
    angle = np.arctan2(y, x)
    return np.where(np.hypot(x, y) < MIN_LENGTH_M, np.nan, angle)


def facing_angle(left_xy: ArrayLike, right_xy: ArrayLike) -> FloatArray:
    """Facing angle of a left-right joint pair.

    Parameters
    ----------
    left_xy, right_xy
        Ground-plane positions ``(..., 2)`` of the left and right joint, any leading shape
        (for example frames by players).

    Returns
    -------
    Angles of shape ``(...)`` in (-pi, pi]. NaN where the joints coincide or are NaN.
    """
    axis = np.asarray(right_xy, dtype=float) - np.asarray(left_xy, dtype=float)
    return wrap_angle(vector_angle(axis) + FACING_ROTATION)


def nose_direction(
    nose_xy: ArrayLike, left_ear_xy: ArrayLike, right_ear_xy: ArrayLike
) -> FloatArray:
    """Head direction from the ear midpoint to the nose. Independent of left/right labels."""
    midpoint = (np.asarray(left_ear_xy, dtype=float) + np.asarray(right_ear_xy, dtype=float)) / 2.0
    return vector_angle(np.asarray(nose_xy, dtype=float) - midpoint)


def anatomical_forward(
    nose_xy: ArrayLike,
    left_ear_xy: ArrayLike,
    right_ear_xy: ArrayLike,
    left_heel_xy: ArrayLike,
    right_heel_xy: ArrayLike,
    left_toe_xy: ArrayLike,
    right_toe_xy: ArrayLike,
) -> FloatArray:
    """Label-independent forward cue: nose ahead of the ears, toes ahead of the heels.

    The cue is the direction of the sum of three ground-plane vectors: ear midpoint to nose
    and heel to big toe for each foot. Swapping all left and right labels leaves the sum
    unchanged, which is what makes it usable against label swaps. It is a coarse cue for
    deciding front versus back (a 180 degree question), not an orientation estimate.
    """
    midpoint = (np.asarray(left_ear_xy, dtype=float) + np.asarray(right_ear_xy, dtype=float)) / 2.0
    head = np.asarray(nose_xy, dtype=float) - midpoint
    left_foot = np.asarray(left_toe_xy, dtype=float) - np.asarray(left_heel_xy, dtype=float)
    right_foot = np.asarray(right_toe_xy, dtype=float) - np.asarray(right_heel_xy, dtype=float)
    return vector_angle(head + left_foot + right_foot)


def resolve_front_back(facing: ArrayLike, reference: ArrayLike) -> tuple[FloatArray, BoolArray]:
    """Resolve the front/back ambiguity of a facing angle against a forward reference.

    Parameters
    ----------
    facing
        Facing angles from :func:`facing_angle`.
    reference
        Independent forward direction, for example :func:`anatomical_forward`.

    Returns
    -------
    resolved
        ``facing`` rotated by 180 degrees wherever it points more than 90 degrees away from
        the reference. Unchanged where the reference is NaN.
    flipped
        True where the rotation was applied.
    """
    facing = np.asarray(facing, dtype=float)
    with np.errstate(invalid="ignore"):
        flipped = np.abs(angular_difference(facing, reference)) > np.pi / 2.0
    return np.where(flipped, wrap_angle(facing + np.pi), facing), flipped
