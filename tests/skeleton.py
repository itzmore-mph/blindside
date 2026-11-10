"""Synthetic skeletons with a known facing angle, shared by the geometry tests."""

from __future__ import annotations

import numpy as np

SHOULDER_WIDTH = 0.37
HIP_WIDTH = 0.22
EAR_WIDTH = 0.18
NOSE_AHEAD = 0.10
FOOT_LENGTH = 0.25
STANCE_WIDTH = 0.20


def unit(angle: np.ndarray | float) -> np.ndarray:
    """Unit vectors ``(..., 2)`` for angles in radians."""
    angle = np.asarray(angle, dtype=float)
    return np.stack([np.cos(angle), np.sin(angle)], axis=-1)


def make_skeleton(
    facing: np.ndarray | float, position: np.ndarray | tuple[float, float] = (0.0, 0.0)
) -> dict[str, np.ndarray]:
    """Ground-plane joints of a player facing ``facing`` (radians) at ``position``.

    The left side of a player lies 90 degrees counter-clockwise of the facing direction.
    Shapes broadcast: ``facing`` of shape ``(...)`` gives joints of shape ``(..., 2)``.
    """
    facing = np.asarray(facing, dtype=float)
    position = np.broadcast_to(np.asarray(position, dtype=float), (*facing.shape, 2))
    forward, left = unit(facing), unit(facing + np.pi / 2.0)
    joints = {}
    for name, width in (("Shoulder", SHOULDER_WIDTH), ("Hip", HIP_WIDTH), ("Ear", EAR_WIDTH)):
        joints[f"l{name}"] = position + left * width / 2.0
        joints[f"r{name}"] = position - left * width / 2.0
    joints["nose"] = position + forward * NOSE_AHEAD
    for side, sign in (("l", 1.0), ("r", -1.0)):
        heel = position + sign * left * STANCE_WIDTH / 2.0 - forward * 0.05
        joints[f"{side}Heel"] = heel
        joints[f"{side}BigToe"] = heel + forward * FOOT_LENGTH
    return joints


def mirror_y(joints: dict[str, np.ndarray], swap_labels: bool) -> dict[str, np.ndarray]:
    """Reflect a skeleton across the x axis.

    A mirrored body has its anatomical left on the other side. With ``swap_labels`` the
    labels follow the anatomy and the result is a valid skeleton facing ``-facing``.
    Without it the labels are wrong, which is what a left/right swap of the pose model
    looks like.
    """
    reflected = {name: xy * np.array([1.0, -1.0]) for name, xy in joints.items()}
    if not swap_labels:
        return reflected
    swapped = {}
    for name, xy in reflected.items():
        if name[0] in "lr" and name[1].isupper():
            name = ("r" if name[0] == "l" else "l") + name[1:]
        swapped[name] = xy
    return swapped


def rotate(joints: dict[str, np.ndarray], angle: float) -> dict[str, np.ndarray]:
    """Rotate a skeleton about the origin by ``angle`` radians."""
    c, s = np.cos(angle), np.sin(angle)
    matrix = np.array([[c, -s], [s, c]])
    return {name: xy @ matrix.T for name, xy in joints.items()}
