"""Agreement of two binary labels of the same engagements, with a cluster bootstrap.

Used to ask how well an XY proxy reproduces the pose-based blind-side label. The functions
take the two labels, the resampling weights and optional groups. No outcome enters.

All statistics come from the weighted 2x2 table of the labels, so one matrix product per
cell gives every bootstrap resample at once.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from blindside.analysis.incremental import MIN_BIN_SIZE

#: Statistics of :func:`agreement_table` that carry an interval.
STATISTICS: tuple[str, ...] = ("agreement", "kappa", "recall", "precision")


def _cells(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> dict[str, np.ndarray]:
    """Weighted counts of the 2x2 table per resample: both, ``a`` only, ``b`` only, neither."""
    weights = np.asarray(weights, dtype=float)
    return {
        "both": weights @ (a & b),
        "a_only": weights @ (a & ~b),
        "b_only": weights @ (~a & b),
        "neither": weights @ (~a & ~b),
    }


def _statistics(cells: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Shares, agreement, kappa, recall and precision from the cells. NaN where undefined."""
    both, a_only, b_only, neither = (
        cells["both"],
        cells["a_only"],
        cells["b_only"],
        cells["neither"],
    )
    total = both + a_only + b_only + neither
    with np.errstate(invalid="ignore", divide="ignore"):
        share_a = (both + a_only) / total
        share_b = (both + b_only) / total
        observed = (both + neither) / total
        expected = share_a * share_b + (1.0 - share_a) * (1.0 - share_b)
        kappa = np.where(expected < 1.0, (observed - expected) / (1.0 - expected), np.nan)
        return {
            "share_a": share_a,
            "share_b": share_b,
            "agreement": observed,
            "kappa": kappa,
            "recall": both / (both + a_only),
            "precision": both / (both + b_only),
        }


def cohen_kappa(
    a: np.ndarray, b: np.ndarray, weights: np.ndarray | None = None
) -> float | np.ndarray:
    """Cohen's kappa of two binary labels: agreement beyond what the two rates give by chance.

    Parameters
    ----------
    a, b
        Boolean labels ``(n,)``.
    weights
        Optional weights ``(n_resamples, n)``, for example from
        :func:`blindside.analysis.baseline.cluster_bootstrap_weights`.

    Returns
    -------
    A float without weights, else one kappa per resample. NaN where both labels are constant
    and equal, or where the weights are all zero.
    """
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if weights is None:
        return float(_statistics(_cells(a, b, np.ones((1, len(a)))))["kappa"][0])
    return _statistics(_cells(a, b, np.atleast_2d(weights)))["kappa"]


def agreement_table(
    pose_blind: np.ndarray,
    xy_blind: np.ndarray,
    weights: np.ndarray,
    groups: Mapping[str, np.ndarray] | None = None,
    *,
    keep: np.ndarray | None = None,
    min_size: int = MIN_BIN_SIZE,
    level: float = 0.95,
) -> pd.DataFrame:
    """How well a proxy label reproduces a reference label, overall and per group.

    Parameters
    ----------
    pose_blind
        Reference label per engagement ``(n,)``, boolean.
    xy_blind
        Proxy label per engagement ``(n,)``, boolean.
    weights
        Cluster bootstrap weights ``(n_resamples, n)``.
    groups
        ``{label: boolean mask}``. One row per group after the row ``"all"``.
    keep
        Optional subset of engagements for every row.
    min_size
        Rows with fewer engagements are reported without intervals.
    level
        Coverage of the percentile intervals.

    Returns
    -------
    One row per group: ``group``, ``n``, the shares ``pose_blind`` and ``xy_blind``, and with
    ``_low`` and ``_high`` columns each of

    ``agreement``
        Share of engagements with the same label.
    ``kappa``
        Cohen's kappa.
    ``recall``
        Share of the pose-blind engagements that the proxy also calls blind.
    ``precision``
        Share of the proxy-blind engagements that the pose label also calls blind.

    Resamples in which a statistic is undefined are left out of its interval.
    """
    a, b = np.asarray(pose_blind, dtype=bool), np.asarray(xy_blind, dtype=bool)
    weights = np.asarray(weights)
    if a.shape != b.shape or a.ndim != 1 or weights.shape[1:] != a.shape:
        raise ValueError("labels and weights must have the shape (n,) and (n_resamples, n)")
    everything = np.ones(len(a), dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    masks = {"all": everything}
    for label, mask in (groups or {}).items():
        masks[label] = everything & np.asarray(mask, dtype=bool)

    tail = 100.0 * (1.0 - level) / 2.0
    rows = []
    for label, mask in masks.items():
        n = int(mask.sum())
        point = _statistics(_cells(a[mask], b[mask], np.ones((1, n))))
        row: dict[str, float | int | str] = {
            "group": label,
            "n": n,
            "pose_blind": float(point["share_a"][0]),
            "xy_blind": float(point["share_b"][0]),
        }
        resampled = _statistics(_cells(a[mask], b[mask], weights[:, mask]))
        for name in STATISTICS:
            low, high = np.nan, np.nan
            if n >= min_size and np.isfinite(resampled[name]).any():
                low, high = np.nanpercentile(resampled[name], [tail, 100.0 - tail])
            row[name] = float(point[name][0])
            row[f"{name}_low"], row[f"{name}_high"] = float(low), float(high)
        rows.append(row)
    return pd.DataFrame(rows)
