"""Reliability map: where and when is pose-derived orientation usable?

Every function takes an orientation table (:mod:`blindside.features.orientation_table`) and
returns a tidy table. Nothing here reads events or outcomes.

The angular uncertainty is a bracket (:mod:`blindside.analysis.bracket`): the Monte Carlo SD
under independent joint errors is the upper bound, the frame-to-frame jitter the lower bound.
The map reports both wherever it reports an uncertainty.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from blindside.analysis.bracket import (
    HALF_WINDOWS,
    MAD_TO_SD,
    PRIMARY_WINDOW,
    residual_variance_factor,
)
from blindside.analysis.consistency import (
    SEGMENTS,
    SPEED_EDGES,
    USABLE_SD_DEG,
    speed_band_labels,
    velocity_agreement,
)

#: Pitch size in metres to which positions are rescaled before matches are pooled.
STANDARD_PITCH: tuple[float, float] = (105.0, 68.0)
#: Zones along the pitch length and width.
ZONE_BINS: tuple[int, int] = (12, 8)
#: Zones with fewer poses are reported as NaN.
MIN_ZONE_POSES = 2000
#: Angular SD thresholds in degrees at which the share of usable poses is reported.
USABLE_THRESHOLDS: tuple[float, ...] = (10.0, 15.0, USABLE_SD_DEG, 30.0)
#: Distance from the touchline on the camera side, band edges in metres.
TOUCHLINE_EDGES: tuple[float, ...] = (0.0, 17.0, 34.0, 51.0, np.inf)
#: An XY to pose offset above this many metres is counted as a gross misalignment.
GROSS_MISALIGNMENT_M = 0.5


def standard_xy(
    table: pd.DataFrame,
    pitch_dims: Mapping[int, tuple[float, float]],
    standard: tuple[float, float] = STANDARD_PITCH,
) -> tuple[np.ndarray, np.ndarray]:
    """Positions rescaled to one pitch size, so that matches can be pooled by zone.

    Parameters
    ----------
    table
        Needs ``match_id``, ``x``, ``y`` (metres, origin at the pitch centre).
    pitch_dims
        ``{match_id: (pitch_length, pitch_width)}`` in metres.
    standard
        Target length and width.
    """
    match = table["match_id"].to_numpy()
    x = table["x"].to_numpy(dtype=float).copy()
    y = table["y"].to_numpy(dtype=float).copy()
    for match_id, (length, width) in pitch_dims.items():
        selected = match == match_id
        x[selected] *= standard[0] / length
        y[selected] *= standard[1] / width
    return x, y


def touchline_distance(y: np.ndarray, pitch_width: float, camera_side: str = "-y") -> np.ndarray:
    """Distance in metres from the touchline behind which the camera stands.

    Needs only the camera side, not the inferred camera point. Negative for a player who
    stands outside the pitch on the camera side.
    """
    if camera_side not in ("-y", "+y"):
        raise ValueError("camera_side must be '-y' or '+y'")
    y = np.asarray(y, dtype=float)
    return y + pitch_width / 2.0 if camera_side == "-y" else pitch_width / 2.0 - y


def zone_index(
    x: np.ndarray,
    y: np.ndarray,
    bins: tuple[int, int] = ZONE_BINS,
    pitch: tuple[float, float] = STANDARD_PITCH,
) -> tuple[np.ndarray, np.ndarray]:
    """Zone column (along x) and row (along y) of each position.

    Positions outside the pitch are counted in the nearest edge zone. NaN gives -1.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    ix = np.floor((np.where(finite, x, 0.0) / pitch[0] + 0.5) * bins[0]).astype(int)
    iy = np.floor((np.where(finite, y, 0.0) / pitch[1] + 0.5) * bins[1]).astype(int)
    ix, iy = np.clip(ix, 0, bins[0] - 1), np.clip(iy, 0, bins[1] - 1)
    return np.where(finite, ix, -1), np.where(finite, iy, -1)


def pitch_zone_summary(
    with_residuals: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    bins: tuple[int, int] = ZONE_BINS,
    pitch: tuple[float, float] = STANDARD_PITCH,
    min_poses: int = MIN_ZONE_POSES,
    threshold: float = USABLE_SD_DEG,
) -> pd.DataFrame:
    """Angular uncertainty per pitch zone and segment, under both bounds.

    Parameters
    ----------
    with_residuals
        Output of :func:`blindside.analysis.bracket.add_residuals`.
    x, y
        Positions aligned with ``with_residuals``, usually from :func:`standard_xy`.
    bins, pitch
        Number of zones along x and y, and the pitch size the positions refer to.
    min_poses
        Zones with fewer poses keep their ``n`` and get NaN for every statistic.

    Returns
    -------
    One row per segment and zone, also for empty zones: ``ix``, ``iy``, the zone centre
    ``x_centre``, ``y_centre``, ``n``, ``upper_sd_deg`` (median Monte Carlo SD),
    ``lower_sd_deg`` (jitter of the primary window) and ``share_usable`` (share of poses
    whose upper bound is below ``threshold``).
    """
    half_window = HALF_WINDOWS[PRIMARY_WINDOW]
    scale = MAD_TO_SD / np.sqrt(residual_variance_factor(half_window))
    ix, iy = zone_index(x, y, bins, pitch)
    zone = np.where(ix >= 0, ix * bins[1] + iy, -1)
    index = pd.RangeIndex(bins[0] * bins[1])
    frames = []
    for segment in SEGMENTS:
        upper = pd.Series(with_residuals[f"{segment}_sd_deg"].to_numpy(dtype=float))
        residual = pd.Series(with_residuals[f"{segment}_res_{half_window}"].to_numpy(dtype=float))
        valid = (zone >= 0) & upper.notna().to_numpy()
        keys = zone[valid]
        n = upper[valid].groupby(keys).size().reindex(index, fill_value=0)
        enough = (n >= min_poses).to_numpy()
        frame = pd.DataFrame(
            {
                "segment": segment,
                "ix": index // bins[1],
                "iy": index % bins[1],
                "n": n.to_numpy(),
                "upper_sd_deg": upper[valid].groupby(keys).median().reindex(index).to_numpy(),
                "lower_sd_deg": scale
                * residual[valid].abs().groupby(keys).median().reindex(index).to_numpy(),
                "share_usable": (upper[valid] < threshold)
                .groupby(keys)
                .mean()
                .reindex(index)
                .to_numpy(),
            }
        )
        frame.loc[~enough, ["upper_sd_deg", "lower_sd_deg", "share_usable"]] = np.nan
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out.insert(3, "x_centre", ((out["ix"] + 0.5) / bins[0] - 0.5) * pitch[0])
    out.insert(4, "y_centre", ((out["iy"] + 0.5) / bins[1] - 0.5) * pitch[1])
    return out


def zone_grid(summary: pd.DataFrame, segment: str, value: str) -> np.ndarray:
    """One statistic of :func:`pitch_zone_summary` as an array ``(rows in y, columns in x)``."""
    selected = summary[summary["segment"] == segment]
    grid = np.full((selected["iy"].max() + 1, selected["ix"].max() + 1), np.nan)
    grid[selected["iy"].to_numpy(), selected["ix"].to_numpy()] = selected[value].to_numpy()
    return grid


def velocity_agreement_by_band(
    table: pd.DataFrame, edges: tuple[float, ...] = SPEED_EDGES
) -> pd.DataFrame:
    """Facing angle versus direction of travel by speed band, for every segment.

    The no-ground-truth consistency check of
    :func:`blindside.analysis.consistency.velocity_agreement`, pooled (``match`` is
    ``"all"``) and per match.
    """
    frames = []
    groups: list[tuple[str, pd.DataFrame]] = [("all", table)]
    groups += [(str(match_id), group) for match_id, group in table.groupby("match_id", sort=True)]
    for segment in SEGMENTS:
        for label, group in groups:
            frame = velocity_agreement(group, segment, edges)
            frame.insert(0, "match", label)
            frame.insert(0, "segment", segment)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _share_row(values: pd.Series, thresholds: tuple[float, ...]) -> dict[str, float]:
    sd = values.to_numpy(dtype=float)
    sd = sd[np.isfinite(sd)]
    row: dict[str, float] = {
        "n": len(sd),
        "sd_median_deg": float(np.median(sd)) if len(sd) else np.nan,
    }
    for threshold in thresholds:
        row[f"below_{threshold:g}"] = float(np.mean(sd < threshold)) if len(sd) else np.nan
    return row


def usability_share(
    table: pd.DataFrame, thresholds: tuple[float, ...] = USABLE_THRESHOLDS
) -> pd.DataFrame:
    """Share of poses whose angular SD is below a threshold, under both bounds.

    Returns
    -------
    One row per segment and bound (``upper`` is ``{segment}_sd_deg``, ``lower`` is
    ``{segment}_sd_low_deg``) with ``n``, the median SD and one ``below_{t}`` share per
    threshold.
    """
    rows = []
    for segment in SEGMENTS:
        for bound, column in (("upper", f"{segment}_sd_deg"), ("lower", f"{segment}_sd_low_deg")):
            rows.append(
                {"segment": segment, "bound": bound, **_share_row(table[column], thresholds)}
            )
    return pd.DataFrame(rows)


def usability_sensitivity(
    tables_by_factor: Mapping[float, pd.DataFrame],
    thresholds: tuple[float, ...] = USABLE_THRESHOLDS,
) -> pd.DataFrame:
    """Sensitivity of the usable share to the conversion from ``p90_mae_cm`` to sigma.

    Parameters
    ----------
    tables_by_factor
        ``{factor: table}`` where each table was built with
        ``OrientationConfig(p90_to_sigma=factor)``. Only ``{segment}_sd_deg`` is read.

    Returns
    -------
    One row per factor and segment for the upper bound. The lower bound is anchored to the
    measured jitter and does not depend on the factor.
    """
    rows = []
    for factor in sorted(tables_by_factor):
        for segment in SEGMENTS:
            values = tables_by_factor[factor][f"{segment}_sd_deg"]
            rows.append(
                {"p90_to_sigma": factor, "segment": segment, **_share_row(values, thresholds)}
            )
    return pd.DataFrame(rows)


def misalignment_summary(
    table: pd.DataFrame, edges: tuple[float, ...] = SPEED_EDGES
) -> pd.DataFrame:
    """Offset between the XY tracking position and the pose centroid, as a distribution.

    The pose centroid is the midpoint of the two hip joints. XY tracking and body pose are
    generated separately, so the two do not coincide.

    Returns
    -------
    One row for all poses and one per speed band: ``n``, median, 90th and 99th percentile of
    the ground distance in metres, the share above :data:`GROSS_MISALIGNMENT_M`, the median
    offset along pitch x and y, and the median offset along the direction of travel
    (``along_median_m``, positive when the pose centroid is ahead of the tracking position).
    """
    dx = table["centroid_dx_m"].to_numpy(dtype=float)
    dy = table["centroid_dy_m"].to_numpy(dtype=float)
    speed = table["speed"].to_numpy(dtype=float)
    heading = np.radians(table["heading_deg"].to_numpy(dtype=float))
    distance = np.hypot(dx, dy)
    along = dx * np.cos(heading) + dy * np.sin(heading)
    finite = np.isfinite(distance)

    bands = [("all", finite)]
    for label, low, high in zip(speed_band_labels(edges), edges[:-1], edges[1:], strict=True):
        bands.append((label, finite & (speed >= low) & (speed < high)))
    rows = []
    for label, selected in bands:
        d = distance[selected]
        if not len(d):
            rows.append({"band": label, "n": 0})
            continue
        rows.append(
            {
                "band": label,
                "n": len(d),
                "median_m": float(np.median(d)),
                "p90_m": float(np.percentile(d, 90)),
                "p99_m": float(np.percentile(d, 99)),
                "share_gross": float(np.mean(d > GROSS_MISALIGNMENT_M)),
                "dx_median_m": float(np.median(dx[selected])),
                "dy_median_m": float(np.median(dy[selected])),
                "along_median_m": float(np.nanmedian(along[selected]))
                if np.isfinite(along[selected]).any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)
