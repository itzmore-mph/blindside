"""Pose features of an on-ball engagement (docs/PREREGISTRATION.md section 6).

Four features at the engagement start t0, each computed under one bound of the angular
uncertainty (``"lower"`` or ``"upper"``, see
:func:`blindside.geometry.uncertainty.bound_correlation`):

``p_blind_side_head``, ``p_blind_side_torso``
    Share of Monte Carlo samples of the carrier's head (ear axis) or torso direction for
    which the presser lies more than the field of view half-angle off that direction.
``presser_alignment``
    Circular mean over samples of the absolute angle between the presser's hip direction
    and the direction to the carrier, in degrees. 0 means the hips point at the carrier.
``carrier_openness``
    Mean over samples of the cosine of the angle between the carrier's torso direction and
    the direction in which the carrier's team attacks. 1 means facing the opponent's goal.

Positions are taken as exact. Only the orientation is sampled. Angles are used as the joint
labels give them: the ``{segment}_flip`` flag of the orientation table is reported next to
the features and never applied.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from blindside.data.paths import cache_dir
from blindside.data.pose import iter_pose_frames, tracking_to_pose_frame
from blindside.geometry.circular import (
    angular_difference,
    circular_mean,
    circular_std,
    wrap_angle,
)
from blindside.geometry.orientation import FACING_ROTATION, SEGMENT_JOINTS, vector_angle
from blindside.geometry.uncertainty import (
    DEFAULT_N_SAMPLES,
    P90_TO_SIGMA,
    bound_correlation,
    sample_facing_angles,
)

#: Joints needed for the three segments.
FEATURE_JOINTS: tuple[str, ...] = tuple(joint for pair in SEGMENT_JOINTS.values() for joint in pair)
#: The four pre-registered pose features, in model order.
POSE_FEATURES: tuple[str, ...] = (
    "p_blind_side_head",
    "p_blind_side_torso",
    "presser_alignment",
    "carrier_openness",
)
#: Half-angle of the visual field in degrees. A presser further off the facing direction is unseen.
FOV_HALF_ANGLE_DEG = 100.0
#: Pose frames at most this far from ``2.5 * frame_start`` are searched.
POSE_WINDOW = 2
BOUNDS: tuple[str, ...] = ("lower", "upper")


def select_pose_frames(
    engagements: pd.DataFrame, pose_keys: pd.DataFrame, window: int = POSE_WINDOW
) -> np.ndarray:
    """Pose frame of each engagement: the nearest to t0 in which both players have joints.

    Parameters
    ----------
    engagements
        Engagements of one match with ``frame_start`` (10 fps), ``presser_id``, ``carrier_id``.
    pose_keys
        ``frame`` (25 fps pose clock) and ``player_id`` of every pose of that match.
    window
        Largest accepted distance in pose frames from ``2.5 * frame_start``.

    Returns
    -------
    Integer array with the selected pose frame, -1 where no frame in the window has joints
    for both players. Of two equally near frames the earlier one is taken.
    """
    frames = pose_keys["frame"].to_numpy(dtype=np.int64)
    players = pose_keys["player_id"].to_numpy(dtype=np.int64)
    present = set(zip(frames.tolist(), players.tolist(), strict=True))
    target = tracking_to_pose_frame(engagements["frame_start"].to_numpy(dtype=float))
    presser = engagements["presser_id"].to_numpy(dtype=np.int64)
    carrier = engagements["carrier_id"].to_numpy(dtype=np.int64)
    out = np.full(len(engagements), -1, dtype=np.int64)
    for i, (t, a, b) in enumerate(
        zip(target.tolist(), presser.tolist(), carrier.tolist(), strict=True)
    ):
        candidates = range(int(np.ceil(t - window)), int(np.floor(t + window)) + 1)
        for frame in sorted(candidates, key=lambda f, t=t: (abs(f - t), f)):
            if (frame, a) in present and (frame, b) in present:
                out[i] = frame
                break
    return out


def read_engagement_poses(
    path: Path | str, keys: set[tuple[int, int]], match_id: int
) -> pd.DataFrame:
    """Stream a pose file and keep the joints of the requested ``(frame, player_id)`` pairs.

    Returns
    -------
    One row per pair found with joints: ``match_id``, ``frame``, ``player_id``, the position
    ``x``, ``y`` stored in the pose file and, for each joint of :data:`FEATURE_JOINTS`,
    ``{joint}_x``, ``{joint}_y`` in metres and ``{joint}_p90`` in centimetres.
    """
    wanted: dict[int, set[int]] = {}
    for frame, player in keys:
        wanted.setdefault(int(frame), set()).add(int(player))
    rows: list[dict[str, float | int]] = []
    for record in iter_pose_frames(path):
        frame = record["frame"]
        players = wanted.get(frame)
        if not players:
            continue
        for player in record.get("player_data") or ():
            joints = player.get("joints")
            if player["player_id"] not in players or not joints:
                continue
            row: dict[str, float | int] = {
                "match_id": match_id,
                "frame": frame,
                "player_id": player["player_id"],
                "x": player.get("x"),
                "y": player.get("y"),
            }
            for name in FEATURE_JOINTS:
                joint = joints[name]
                row[f"{name}_x"] = joint["xyz"][0]
                row[f"{name}_y"] = joint["xyz"][1]
                row[f"{name}_p90"] = joint["p90_mae_cm"]
            rows.append(row)
    columns = ["match_id", "frame", "player_id", "x", "y"] + [
        f"{name}_{part}" for name in FEATURE_JOINTS for part in ("x", "y", "p90")
    ]
    table = pd.DataFrame(rows, columns=columns)
    return table.astype({c: float for c in columns[3:]})


def cached_engagement_poses(
    path: Path | str, keys: set[tuple[int, int]], match_id: int, *, refresh: bool = False
) -> pd.DataFrame:
    """:func:`read_engagement_poses` with a parquet cache in ``data/cache/pose_features/``.

    The cache is reused only if it holds exactly the requested pairs.
    """
    target = cache_dir() / "pose_features" / f"{match_id}_engagement_poses.parquet"
    if target.exists() and not refresh:
        table = pd.read_parquet(target)
        stored = set(zip(table["frame"].tolist(), table["player_id"].tolist(), strict=True))
        if stored == {(int(f), int(p)) for f, p in keys}:
            return table
    table = read_engagement_poses(path, keys, match_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".parquet.tmp")
    table.to_parquet(temporary, index=False)
    temporary.replace(target)
    return table


def _direction(origin_xy: ArrayLike, target_xy: ArrayLike) -> np.ndarray:
    """Angle of the vector from ``origin_xy`` to ``target_xy``. NaN where they coincide."""
    return vector_angle(np.asarray(target_xy, dtype=float) - np.asarray(origin_xy, dtype=float))


def blind_side_probability(
    facing_samples: ArrayLike,
    carrier_xy: ArrayLike,
    presser_xy: ArrayLike,
    half_angle_deg: float = FOV_HALF_ANGLE_DEG,
) -> np.ndarray:
    """Share of facing samples for which the presser lies outside the carrier's visual field.

    Parameters
    ----------
    facing_samples
        Samples of the carrier's facing angle in radians, shape ``(n_samples, n)``, or
        ``(n_samples,)`` for a single engagement. Angles need not be wrapped.
    carrier_xy, presser_xy
        Positions in metres, shape ``(n, 2)``, or ``(2,)`` for a single engagement.
    half_angle_deg
        Half-angle of the visual field. The presser is unseen if the direction to them
        deviates from the facing direction by more than this.

    Returns
    -------
    Array ``(n,)`` in [0, 1], a scalar array for a single engagement. NaN where a sample is
    NaN or where the presser stands on the carrier, so that the direction is undefined.

    Raises
    ------
    ValueError
        If ``half_angle_deg`` is not strictly between 0 and 180 or there are no samples.
    """
    if not 0.0 < half_angle_deg < 180.0:
        raise ValueError("half_angle_deg must lie between 0 and 180")
    facing_samples = np.asarray(facing_samples, dtype=float)
    if facing_samples.shape[0] == 0:
        raise ValueError("facing_samples must hold at least one sample")
    to_presser = _direction(carrier_xy, presser_xy)
    off = np.abs(angular_difference(to_presser[None, ...], facing_samples))
    share = (off > np.radians(half_angle_deg)).mean(axis=0)
    undefined = np.isnan(off).any(axis=0)
    return np.where(undefined, np.nan, share)


def presser_alignment_deg(
    hip_samples: np.ndarray, presser_xy: np.ndarray, carrier_xy: np.ndarray
) -> np.ndarray:
    """Circular mean of the absolute angle between presser hips and the direction to the carrier.

    Returns degrees in [0, 180], shape ``(n,)``. 0 means the hips point at the carrier.
    """
    to_carrier = _direction(presser_xy, carrier_xy)
    off = np.abs(angular_difference(to_carrier[None, :], np.asarray(hip_samples, dtype=float)))
    return np.degrees(circular_mean(off, axis=0))


def carrier_openness(torso_samples: np.ndarray, attacking_angle: np.ndarray) -> np.ndarray:
    """Mean cosine between the carrier's torso direction and their attacking direction.

    ``attacking_angle`` is 0 for a carrier whose team attacks towards +x and pi otherwise.
    """
    torso_samples = np.asarray(torso_samples, dtype=float)
    return np.cos(torso_samples - np.asarray(attacking_angle, dtype=float)[None, :]).mean(axis=0)


def carrier_attacking_angle(presser_attacking_side: np.ndarray) -> np.ndarray:
    """Direction of attack of the carrier's team from the pressing team's ``attacking_side``."""
    side = np.asarray(presser_attacking_side, dtype=object)
    known = (side == "left_to_right") | (side == "right_to_left")
    angle = np.where(side == "left_to_right", np.pi, 0.0)
    return np.where(known, angle, np.nan)


def _joined(engagements: pd.DataFrame, poses: pd.DataFrame, player: str) -> pd.DataFrame:
    """Pose rows aligned with the engagements for the ``presser`` or the ``carrier``."""
    keys = engagements[["match_id", "pose_frame", f"{player}_id"]].rename(
        columns={"pose_frame": "frame", f"{player}_id": "player_id"}
    )
    return keys.merge(poses, on=["match_id", "frame", "player_id"], how="left", validate="m:1")


def _segment_samples(
    pose: pd.DataFrame,
    segment: str,
    bound: str,
    n_samples: int,
    p90_to_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Facing samples ``(n_samples, len(pose))`` of one segment under one bound."""
    left, right = SEGMENT_JOINTS[segment]
    return sample_facing_angles(
        pose[[f"{left}_x", f"{left}_y"]].to_numpy(dtype=float),
        pose[[f"{right}_x", f"{right}_y"]].to_numpy(dtype=float),
        pose[f"{left}_p90"].to_numpy(dtype=float),
        pose[f"{right}_p90"].to_numpy(dtype=float),
        n_samples=n_samples,
        p90_to_sigma=p90_to_sigma,
        correlation=bound_correlation(segment, bound),
        rng=rng,
    )


def pose_feature_table(
    engagements: pd.DataFrame,
    poses: pd.DataFrame,
    *,
    bound: str = "upper",
    half_angle_deg: float = FOV_HALF_ANGLE_DEG,
    n_samples: int = DEFAULT_N_SAMPLES,
    p90_to_sigma: float = P90_TO_SIGMA,
    seed: int = 0,
) -> pd.DataFrame:
    """The four pose features of each engagement under one uncertainty bound.

    Parameters
    ----------
    engagements
        One row per engagement with ``match_id``, ``pose_frame`` (see
        :func:`select_pose_frames`), ``presser_id``, ``carrier_id`` and ``attacking_side``
        (direction of the pressing team).
    poses
        Joints of the players in those frames, see :func:`read_engagement_poses`.
    bound
        ``"lower"`` or ``"upper"`` bound of the angular uncertainty.
    half_angle_deg
        Half-angle of the visual field.
    n_samples, p90_to_sigma, seed
        Monte Carlo settings. The same seed gives the same standard normal draws under
        both bounds, so the bounds differ only through the noise scale.

    Returns
    -------
    One row per engagement, same index: :data:`POSE_FEATURES`, the circular SD of the
    samples ``carrier_head_sd_deg``, ``carrier_torso_sd_deg``, ``presser_hips_sd_deg``, the
    point-estimate facing angles ``carrier_head_deg``, ``carrier_torso_deg`` and
    ``presser_off_head_deg``, ``presser_off_torso_deg``: the absolute angle between the
    carrier's point-estimate facing direction and the direction to the presser.
    """
    carrier = _joined(engagements, poses, "carrier")
    presser = _joined(engagements, poses, "presser")
    carrier_xy = carrier[["x", "y"]].to_numpy(dtype=float)
    presser_xy = presser[["x", "y"]].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    head = _segment_samples(carrier, "head", bound, n_samples, p90_to_sigma, rng)
    torso = _segment_samples(carrier, "torso", bound, n_samples, p90_to_sigma, rng)
    hips = _segment_samples(presser, "hips", bound, n_samples, p90_to_sigma, rng)
    to_presser = _direction(carrier_xy, presser_xy)

    out = pd.DataFrame(index=engagements.index)
    out["p_blind_side_head"] = blind_side_probability(head, carrier_xy, presser_xy, half_angle_deg)
    out["p_blind_side_torso"] = blind_side_probability(
        torso, carrier_xy, presser_xy, half_angle_deg
    )
    out["presser_alignment"] = presser_alignment_deg(hips, presser_xy, carrier_xy)
    out["carrier_openness"] = carrier_openness(
        torso, carrier_attacking_angle(engagements["attacking_side"].to_numpy(dtype=object))
    )
    out["carrier_head_sd_deg"] = np.degrees(circular_std(head, axis=0))
    out["carrier_torso_sd_deg"] = np.degrees(circular_std(torso, axis=0))
    out["presser_hips_sd_deg"] = np.degrees(circular_std(hips, axis=0))
    for segment in ("head", "torso"):
        left, right = SEGMENT_JOINTS[segment]
        axis = carrier[[f"{right}_x", f"{right}_y"]].to_numpy(dtype=float) - carrier[
            [f"{left}_x", f"{left}_y"]
        ].to_numpy(dtype=float)
        facing = wrap_angle(vector_angle(axis) + FACING_ROTATION)
        out[f"carrier_{segment}_deg"] = np.degrees(facing)
        out[f"presser_off_{segment}_deg"] = np.degrees(
            np.abs(angular_difference(to_presser, facing))
        )
    return out
