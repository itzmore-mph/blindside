"""Per-frame orientation table: facing angles with angular uncertainty for every pose.

One row per player and pose frame in which the player has joints. All angles are degrees
in (-180, 180], 0 points along +x of the pitch, counter-clockwise positive.

Columns
-------
``match_id``, ``period``, ``frame`` (25 fps pose clock), ``player_id``, ``x``, ``y``,
``is_detected``
    Identity and the tracking position stored in the pose file.
``speed``, ``heading_deg``
    Smoothed XY speed in m/s and direction of travel, see
    :func:`blindside.geometry.kinematics.central_velocity`. NaN at the edges of a period.
``{segment}_deg``
    Point estimate of the facing angle for ``torso`` (shoulders), ``hips`` and ``head``
    (ears), from the published joint positions and the left/right labels.
``{segment}_mean_deg``, ``{segment}_sd_deg``
    Circular mean and circular standard deviation of the Monte Carlo samples under
    independent joint errors. The SD is the **upper bound** of the angular uncertainty.
``{segment}_sd_low_deg``
    The **lower bound**: the Monte Carlo SD with the left/right error correlation at which
    it matches the frame-to-frame jitter of the angle (docs/RELIABILITY.md section 1).
    The true angular error lies between the two bounds.
``{segment}_width_m``, ``{segment}_p90_cm``
    Ground-plane distance of the joint pair and the larger of its two ``p90_mae_cm``.
``head_nose_deg``, ``head_nose_sd_deg``
    Alternative head direction from the ear midpoint to the nose, with its uncertainty.
``forward_deg``
    Label-independent forward cue (nose ahead of ears, toes ahead of heels).
``{segment}_flip``
    True where ``{segment}_deg`` points more than 90 degrees away from ``forward_deg``,
    which indicates swapped left/right labels. The angle columns are not altered.
"""

from __future__ import annotations

from array import array
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from blindside.data.manifest import POSE_FPS
from blindside.data.paths import cache_dir
from blindside.data.pose import iter_pose_frames
from blindside.geometry.circular import wrap_angle
from blindside.geometry.kinematics import central_velocity
from blindside.geometry.orientation import (
    FACING_ROTATION,
    SEGMENT_JOINTS,
    anatomical_forward,
    resolve_front_back,
    vector_angle,
)
from blindside.geometry.uncertainty import (
    BOUND_CORRELATION,
    DEFAULT_N_SAMPLES,
    P90_TO_SIGMA,
    difference_sigma,
    sample_vector_angles,
    sigma_from_p90,
    summarise_samples,
)

#: Joints read from every pose. Everything else is skipped while streaming.
TABLE_JOINTS: tuple[str, ...] = (
    "lShoulder",
    "rShoulder",
    "lHip",
    "rHip",
    "lEar",
    "rEar",
    "nose",
    "lHeel",
    "rHeel",
    "lBigToe",
    "rBigToe",
)
_J = {name: index for index, name in enumerate(TABLE_JOINTS)}

ProgressCallback = Callable[[int], None]


@dataclass(frozen=True)
class OrientationConfig:
    """Settings of the orientation table. Every field is part of the cache key.

    Attributes
    ----------
    n_samples
        Monte Carlo draws per pose and segment.
    p90_to_sigma
        Conversion from ``p90_mae_cm`` to a per-axis sigma (error model assumption).
    correlation
        Assumed correlation between the errors of two joints of one player for the upper
        bound (assumption, default independence).
    lower_correlation
        Left/right correlation per segment (torso, hips, head) for the lower bound.
    seed
        Seed of the random generator.
    velocity_half_window
        Half window of the central velocity difference in pose frames (5 = 0.2 s each side).
    chunk_poses
        Poses per Monte Carlo batch. Bounds memory. Changing it changes the random stream.
    """

    n_samples: int = DEFAULT_N_SAMPLES
    p90_to_sigma: float = P90_TO_SIGMA
    correlation: float = 0.0
    lower_correlation: tuple[float, float, float] = tuple(
        BOUND_CORRELATION["lower"][segment] for segment in SEGMENT_JOINTS
    )
    seed: int = 0
    velocity_half_window: int = 5
    chunk_poses: int = 20_000

    def key(self) -> str:
        """Short, readable identifier used in cache file names."""
        return (
            f"n{self.n_samples}_f{self.p90_to_sigma:g}_r{self.correlation:g}"
            f"_s{self.seed}_h{self.velocity_half_window}_c{self.chunk_poses}"
            "_l" + "-".join(f"{value:g}" for value in self.lower_correlation)
        )


def orientation_columns(
    xy: np.ndarray,
    p90: np.ndarray,
    config: OrientationConfig,
    rng: np.random.Generator,
    rng_low: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Angles and uncertainties for a batch of poses.

    Parameters
    ----------
    xy
        Ground-plane joint positions in metres, shape ``(poses, len(TABLE_JOINTS), 2)``.
    p90
        ``p90_mae_cm`` per joint, shape ``(poses, len(TABLE_JOINTS))``.
    rng, rng_low
        Generators for the upper and the lower bound. They are separate streams so that
        the upper bound does not depend on whether a lower bound is computed.

    Returns
    -------
    Mapping from column name to an array of length ``poses``.
    """
    sigma = sigma_from_p90(p90, config.p90_to_sigma)
    rng_low = rng_low if rng_low is not None else np.random.default_rng()
    lower_correlation = dict(zip(SEGMENT_JOINTS, config.lower_correlation, strict=True))
    out: dict[str, np.ndarray] = {}

    forward = anatomical_forward(
        *(
            xy[:, _J[name]]
            for name in ("nose", "lEar", "rEar", "lHeel", "rHeel", "lBigToe", "rBigToe")
        )
    )
    out["forward_deg"] = np.degrees(forward)

    for segment, (left, right) in SEGMENT_JOINTS.items():
        li, ri = _J[left], _J[right]
        axis = xy[:, ri] - xy[:, li]
        facing = wrap_angle(vector_angle(axis) + FACING_ROTATION)
        samples = sample_vector_angles(
            axis,
            difference_sigma(sigma[:, li], sigma[:, ri], config.correlation),
            config.n_samples,
            rng,
        )
        estimate = summarise_samples(samples)
        out[f"{segment}_deg"] = np.degrees(facing)
        out[f"{segment}_mean_deg"] = np.degrees(wrap_angle(estimate.mean + FACING_ROTATION))
        out[f"{segment}_sd_deg"] = np.degrees(estimate.std)
        lower = summarise_samples(
            sample_vector_angles(
                axis,
                difference_sigma(sigma[:, li], sigma[:, ri], lower_correlation[segment]),
                config.n_samples,
                rng_low,
            )
        )
        out[f"{segment}_sd_low_deg"] = np.degrees(lower.std)
        out[f"{segment}_width_m"] = np.hypot(axis[:, 0], axis[:, 1])
        out[f"{segment}_p90_cm"] = np.maximum(p90[:, li], p90[:, ri])
        out[f"{segment}_flip"] = resolve_front_back(facing, forward)[1]

    # Alternative head direction: ear midpoint to nose. The midpoint of two joints with
    # correlation rho has variance (s_l^2 + s_r^2 + 2 rho s_l s_r) / 4. The nose error is
    # taken as correlated with the midpoint error by the same rho.
    le, re, nose = _J["lEar"], _J["rEar"], _J["nose"]
    midpoint_sigma = (
        np.sqrt(
            sigma[:, le] ** 2
            + sigma[:, re] ** 2
            + 2.0 * config.correlation * sigma[:, le] * sigma[:, re]
        )
        / 2.0
    )
    nose_vector = xy[:, nose] - (xy[:, le] + xy[:, re]) / 2.0
    estimate = summarise_samples(
        sample_vector_angles(
            nose_vector,
            difference_sigma(sigma[:, nose], midpoint_sigma, config.correlation),
            config.n_samples,
            rng,
        )
    )
    out["head_nose_deg"] = np.degrees(vector_angle(nose_vector))
    out["head_nose_sd_deg"] = np.degrees(estimate.std)
    return out


def _velocity_columns(
    pos_frame: np.ndarray,
    pos_player: np.ndarray,
    pos_xy: np.ndarray,
    frame_period: dict[int, int],
    row_frame: np.ndarray,
    row_player: np.ndarray,
    half_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Speed and heading at the pose rows from the positions of all listed players."""
    first, last = int(pos_frame.min()), int(pos_frame.max())
    players = np.unique(pos_player)
    grid = np.full((last - first + 1, len(players), 2), np.nan, dtype=np.float32)
    grid[pos_frame - first, np.searchsorted(players, pos_player)] = pos_xy
    period = np.zeros(last - first + 1, dtype=np.int16)
    for frame, value in frame_period.items():
        if first <= frame <= last:
            period[frame - first] = value
    velocity = central_velocity(grid, POSE_FPS, half_window, segment=period)
    at_rows = velocity[row_frame - first, np.searchsorted(players, row_player)]
    speed = np.hypot(at_rows[:, 0], at_rows[:, 1])
    heading = np.degrees(np.arctan2(at_rows[:, 1], at_rows[:, 0]))
    return speed, np.where(np.isnan(speed), np.nan, heading)


def build_orientation_table(
    path: Path | str,
    match_id: int,
    config: OrientationConfig | None = None,
    progress: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Stream one pose file and compute the orientation table.

    The file is read one frame at a time. Joint coordinates are held only for the current
    Monte Carlo batch (``config.chunk_poses`` poses), so peak memory is set by the output
    table and the position grid, not by the 3.3 GB input.

    Parameters
    ----------
    path
        ``.jsonl.zip``, ``.jsonl.gz`` or ``.jsonl`` pose file.
    match_id
        Written to the ``match_id`` column.
    config
        Monte Carlo and velocity settings, default :class:`OrientationConfig`.
    progress
        Optional callback receiving the number of frames read so far, once per batch.
    """
    config = config or OrientationConfig()
    rng = np.random.default_rng(config.seed)
    rng_low = np.random.default_rng([config.seed, 1])
    n_joints = len(TABLE_JOINTS)

    pos_frame, pos_player = array("i"), array("i")
    pos_xy = array("f")
    frame_period: dict[int, int] = {}

    row_frame, row_period, row_player = array("i"), array("b"), array("i")
    row_xy, row_detected = array("f"), array("b")
    batch_xy, batch_p90 = array("d"), array("d")
    chunks: list[dict[str, np.ndarray]] = []

    def flush() -> None:
        if not batch_p90:
            return
        xy = np.frombuffer(batch_xy, dtype=float).reshape(-1, n_joints, 2)
        p90 = np.frombuffer(batch_p90, dtype=float).reshape(-1, n_joints)
        columns = orientation_columns(xy, p90, config, rng, rng_low)
        chunks.append(
            {k: v if v.dtype == bool else v.astype(np.float32) for k, v in columns.items()}
        )
        del xy, p90  # release the buffer exports before clearing the arrays
        del batch_xy[:], batch_p90[:]

    nan = float("nan")
    n_frames = 0
    for record in iter_pose_frames(path):
        frame = record["frame"]
        period = record.get("period") or 0
        frame_period[frame] = period
        n_frames += 1
        for player in record.get("player_data") or ():
            player_id = player["player_id"]
            x, y = player.get("x"), player.get("y")
            x = nan if x is None else x
            y = nan if y is None else y
            pos_frame.append(frame)
            pos_player.append(player_id)
            pos_xy.append(x)
            pos_xy.append(y)
            joints = player.get("joints")
            if not joints:
                continue
            for name in TABLE_JOINTS:
                joint = joints[name]
                xyz = joint["xyz"]
                batch_xy.append(xyz[0])
                batch_xy.append(xyz[1])
                batch_p90.append(joint["p90_mae_cm"])
            row_frame.append(frame)
            row_period.append(period)
            row_player.append(player_id)
            row_xy.append(x)
            row_xy.append(y)
            row_detected.append(bool(player.get("is_detected")))
        if len(batch_p90) >= config.chunk_poses * n_joints:
            flush()
            if progress is not None:
                progress(n_frames)
    flush()
    if progress is not None:
        progress(n_frames)

    frames = np.frombuffer(row_frame, dtype=np.int32)
    players = np.frombuffer(row_player, dtype=np.int32)
    xy = np.frombuffer(row_xy, dtype=np.float32).reshape(-1, 2)
    table: dict[str, np.ndarray] = {
        "match_id": np.full(len(frames), match_id, dtype=np.int32),
        "period": np.frombuffer(row_period, dtype=np.int8),
        "frame": frames,
        "player_id": players,
        "x": xy[:, 0],
        "y": xy[:, 1],
        "is_detected": np.frombuffer(row_detected, dtype=np.int8).astype(bool),
    }
    if len(frames):
        speed, heading = _velocity_columns(
            np.frombuffer(pos_frame, dtype=np.int32),
            np.frombuffer(pos_player, dtype=np.int32),
            np.frombuffer(pos_xy, dtype=np.float32).reshape(-1, 2),
            frame_period,
            frames,
            players,
            config.velocity_half_window,
        )
        table["speed"] = speed.astype(np.float32)
        table["heading_deg"] = heading.astype(np.float32)
        for name in chunks[0]:
            table[name] = np.concatenate([chunk[name] for chunk in chunks])
    return pd.DataFrame(table)


def orientation_cache_path(source: Path | str, config: OrientationConfig | None = None) -> Path:
    """Parquet path of the cached table for one pose file and configuration."""
    config = config or OrientationConfig()
    stem = Path(source).name.split(".")[0]
    return cache_dir() / "orientation" / f"{stem}__{config.key()}.parquet"


def cached_orientation_table(
    path: Path | str,
    match_id: int,
    config: OrientationConfig | None = None,
    *,
    refresh: bool = False,
    progress: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Return the orientation table of a pose file, building and caching it on first use.

    The cache lives in ``data/cache/orientation/`` (git ignored) and is keyed by the file
    name and every field of the configuration.
    """
    config = config or OrientationConfig()
    target = orientation_cache_path(path, config)
    if target.exists() and not refresh:
        return pd.read_parquet(target)
    table = build_orientation_table(path, match_id, config, progress)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".parquet.tmp")
    table.to_parquet(temporary, index=False)
    temporary.replace(target)
    return table


def config_dict(config: OrientationConfig) -> dict[str, float | int]:
    """Configuration as a plain dictionary, for reports."""
    return asdict(config)
