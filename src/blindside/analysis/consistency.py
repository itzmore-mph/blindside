"""Consistency checks on the orientation table that need no ground truth.

All functions take the table of :mod:`blindside.features.orientation_table` (angles in
degrees) and return small summary tables. Differences between angles are always taken on
the circle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEGMENTS: tuple[str, ...] = ("torso", "hips", "head")
#: Speed bands in m/s of docs/FEASIBILITY.md section 5.
SPEED_EDGES: tuple[float, ...] = (0.0, 2.0, 4.0, 6.0, np.inf)
#: Circular standard deviation below which a pose counts as usable in the summaries.
USABLE_SD_DEG = 20.0
#: Median of |N(0, 1)|. Converts a Gaussian sigma into a median absolute deviation.
_MEDIAN_ABS_NORMAL = 0.6745


def signed_difference_deg(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> np.ndarray:
    """Signed circular difference ``a - b`` in degrees, in [-180, 180)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.mod(a - b + 180.0, 360.0) - 180.0


def abs_difference_deg(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> np.ndarray:
    """Absolute circular difference in degrees, in [0, 180]."""
    return np.abs(signed_difference_deg(a, b))


def speed_band_labels(edges: tuple[float, ...] = SPEED_EDGES) -> list[str]:
    """Readable labels such as ``0 to 2 m/s`` and ``above 6 m/s``."""
    return [
        f"above {low:g} m/s" if np.isinf(high) else f"{low:g} to {high:g} m/s"
        for low, high in zip(edges[:-1], edges[1:], strict=True)
    ]


def uncertainty_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Width and Monte Carlo angular SD per segment (docs/FEASIBILITY.md section 4)."""
    rows = []
    for segment in SEGMENTS:
        sd = table[f"{segment}_sd_deg"].to_numpy(dtype=float)
        sd = sd[np.isfinite(sd)]
        rows.append(
            {
                "segment": segment,
                "n": len(sd),
                "median_width_m": float(np.nanmedian(table[f"{segment}_width_m"])),
                "sd_median_deg": float(np.median(sd)),
                "sd_p90_deg": float(np.percentile(sd, 90)),
                "sd_p99_deg": float(np.percentile(sd, 99)),
                "share_sd_below_20": float(np.mean(sd < USABLE_SD_DEG)),
            }
        )
    return pd.DataFrame(rows)


def segment_agreement(table: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Median absolute difference between the facing angles of two segments, in degrees."""
    pairs = (("torso", "hips"), ("torso", "head"), ("hips", "head"))
    return {
        (a, b): float(np.nanmedian(abs_difference_deg(table[f"{a}_deg"], table[f"{b}_deg"])))
        for a, b in pairs
    }


def velocity_agreement(
    table: pd.DataFrame, segment: str = "hips", edges: tuple[float, ...] = SPEED_EDGES
) -> pd.DataFrame:
    """Facing angle versus direction of travel, by speed band.

    A player who runs fast faces where they run. The difference between a facing angle and
    the velocity heading should therefore shrink with speed and stay clear of 180 degrees.
    This checks the axis convention and the signal content without ground truth.

    Returns
    -------
    One row per speed band with ``n``, the median signed difference (bias), the median and
    90th percentile of the absolute difference and the share above 90 degrees.
    """
    difference = signed_difference_deg(table[f"{segment}_deg"], table["heading_deg"])
    speed = table["speed"].to_numpy(dtype=float)
    valid = np.isfinite(difference) & np.isfinite(speed)
    rows = []
    for label, low, high in zip(speed_band_labels(edges), edges[:-1], edges[1:], strict=True):
        selected = difference[valid & (speed >= low) & (speed < high)]
        absolute = np.abs(selected)
        rows.append(
            {
                "speed_band": label,
                "n": len(selected),
                "median_signed_deg": float(np.median(selected)) if len(selected) else np.nan,
                "median_abs_deg": float(np.median(absolute)) if len(selected) else np.nan,
                "p90_abs_deg": float(np.percentile(absolute, 90)) if len(selected) else np.nan,
                "share_above_90": float(np.mean(absolute > 90.0)) if len(selected) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _consecutive(table: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Table sorted by player and frame, and a mask of rows whose predecessor is one frame back."""
    ordered = table.sort_values(["match_id", "player_id", "frame"], kind="stable")
    same = (
        (ordered["frame"].diff() == 1)
        & (ordered["player_id"].diff() == 0)
        & (ordered["match_id"].diff() == 0)
    ).to_numpy()
    return ordered, same


def temporal_jitter(table: pd.DataFrame) -> pd.DataFrame:
    """Frame-to-frame change of each facing angle against what the error model predicts.

    If the Monte Carlo SD described noise that is independent from frame to frame, the
    median absolute change between consecutive frames would be at least
    ``0.6745 * sqrt(2) * SD`` even for a player who does not move at all.
    """
    ordered, same = _consecutive(table)
    rows = []
    for segment in SEGMENTS:
        angle = ordered[f"{segment}_deg"].to_numpy(dtype=float)
        change = abs_difference_deg(angle[1:], angle[:-1])[same[1:]]
        change = change[np.isfinite(change)]
        sd_median = float(np.nanmedian(ordered[f"{segment}_sd_deg"]))
        rows.append(
            {
                "segment": segment,
                "n": len(change),
                "jitter_median_deg": float(np.median(change)),
                "jitter_p90_deg": float(np.percentile(change, 90)),
                "sd_median_deg": sd_median,
                "jitter_if_independent_deg": _MEDIAN_ABS_NORMAL * np.sqrt(2.0) * sd_median,
            }
        )
    return pd.DataFrame(rows)


def head_estimator_comparison(table: pd.DataFrame) -> pd.DataFrame:
    """Ear axis versus ear-midpoint-to-nose direction as the head orientation estimator."""
    ordered, same = _consecutive(table)
    rows = []
    for label, angle_column, sd_column in (
        ("Ear axis (`head`)", "head_deg", "head_sd_deg"),
        ("Ear midpoint to nose (`head_nose`)", "head_nose_deg", "head_nose_sd_deg"),
    ):
        angle = ordered[angle_column].to_numpy(dtype=float)
        change = abs_difference_deg(angle[1:], angle[:-1])[same[1:]]
        change = change[np.isfinite(change)]
        sd = ordered[sd_column].to_numpy(dtype=float)
        sd = sd[np.isfinite(sd)]
        rows.append(
            {
                "Estimator": label,
                "SD median, deg": float(np.median(sd)),
                "SD 90th pct, deg": float(np.percentile(sd, 90)),
                "SD below 20 deg, %": 100.0 * float(np.mean(sd < USABLE_SD_DEG)),
                "Jitter median, deg": float(np.median(change)),
                "Jitter 90th pct, deg": float(np.percentile(change, 90)),
            }
        )
    return pd.DataFrame(rows)


def flip_summary(table: pd.DataFrame, min_speed: float = 4.0) -> pd.DataFrame:
    """How often the left/right rule and the forward cue disagree, and which one is right.

    Among fast players the direction of travel acts as the referee: a facing angle within
    90 degrees of the heading is taken as pointing forwards.
    """
    fast = table["speed"].to_numpy(dtype=float) >= min_speed
    rows = []
    for segment in SEGMENTS:
        flagged = table[f"{segment}_flip"].to_numpy(dtype=bool)
        difference = abs_difference_deg(table[f"{segment}_deg"], table["heading_deg"])
        valid = np.isfinite(difference)
        flagged_fast = flagged & fast & valid
        unflagged_fast = ~flagged & fast & valid
        rows.append(
            {
                "Segment": segment,
                "Flagged poses": int(flagged.sum()),
                "Share, %": 100.0 * float(flagged.mean()),
                f"Flagged at {min_speed:g} m/s or faster": int(flagged_fast.sum()),
                "Of those, left/right rule within 90 deg of heading, %": 100.0
                * float(np.mean(difference[flagged_fast] <= 90.0)),
                "Unflagged within 90 deg of heading, %": 100.0
                * float(np.mean(difference[unflagged_fast] <= 90.0)),
                "SD median, flagged, deg": float(
                    np.nanmedian(table.loc[flagged, f"{segment}_sd_deg"])
                ),
            }
        )
    return pd.DataFrame(rows)
