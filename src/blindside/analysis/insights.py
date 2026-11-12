"""Descriptive profiles of the pose blind-side label, and a sample size for a rerun.

Everything here is exploratory and descriptive. No outcome column is read, the
pre-registered comparison is not touched, and ``p_blind_side`` is used as computed by
:func:`blindside.features.pose_features.pose_feature_table`.

A press counts as *confidently blind* if ``p_blind_side`` exceeds a threshold under both
bounds of the angular uncertainty. That is a statement about the pose label. There is no
ground truth for orientation, so it is not a statement about what the carrier saw.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import log
from statistics import NormalDist

import numpy as np
import pandas as pd

from blindside.analysis.incremental import MIN_BIN_SIZE
from blindside.features.pose_features import BOUNDS

#: ``p_blind_side`` above this value under both bounds counts as confidently blind.
CONFIDENT_THRESHOLD = 0.8


def confident_blind(
    features: Mapping[str, pd.DataFrame],
    feature: str = "p_blind_side_head",
    threshold: float = CONFIDENT_THRESHOLD,
) -> np.ndarray:
    """Engagements whose ``p_blind_side`` exceeds ``threshold`` under both bounds.

    Parameters
    ----------
    features
        ``{bound: pose_feature_table(...)}`` for ``"lower"`` and ``"upper"``, same index.
    feature
        ``"p_blind_side_head"`` or ``"p_blind_side_torso"``.
    threshold
        Probability that both bounds have to exceed.

    Returns
    -------
    Boolean array, one entry per engagement. NaN counts as not blind.
    """
    p = np.column_stack([features[bound][feature].to_numpy(dtype=float) for bound in BOUNDS])
    with np.errstate(invalid="ignore"):
        return (p > threshold).all(axis=1)


def group_profile(
    groups: np.ndarray | pd.Series,
    flag: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    min_interval_size: int = MIN_BIN_SIZE,
) -> pd.DataFrame:
    """Count and share of flagged engagements per group.

    Parameters
    ----------
    groups
        Group label per engagement (a player, a team, a pitch zone). Missing labels are
        left out.
    flag
        Boolean per engagement, for example :func:`confident_blind`.
    weights
        Cluster bootstrap weights ``(n_resamples, n)`` for a percentile 95% interval of the
        share. None gives no interval.
    min_interval_size
        Groups with fewer engagements carry no interval.

    Returns
    -------
    One row per group, the most flagged engagements first (ties: the larger group, then
    the label): ``group``, ``n``, ``flagged``, ``share``, ``low``, ``high``.
    """
    labels = pd.Series(np.asarray(groups, dtype=object))
    flag = np.asarray(flag, dtype=bool)
    if len(labels) != len(flag):
        raise ValueError("groups and flag need one entry per engagement")
    rows = []
    for label in labels.dropna().unique():
        members = (labels == label).to_numpy()
        n, flagged = int(members.sum()), int(flag[members].sum())
        low = high = np.nan
        if weights is not None and n >= min_interval_size:
            w = weights[:, members]
            with np.errstate(invalid="ignore", divide="ignore"):
                resampled = (w @ flag[members].astype(float)) / w.sum(axis=1)
            low, high = (float(v) for v in np.nanpercentile(resampled, [2.5, 97.5]))
        rows.append(
            {
                "group": label,
                "n": n,
                "flagged": flagged,
                "share": flagged / n,
                "low": low,
                "high": high,
            }
        )
    table = pd.DataFrame(rows, columns=["group", "n", "flagged", "share", "low", "high"])
    table["_label"] = table["group"].astype(str)
    return (
        table.sort_values(["flagged", "n", "_label"], ascending=[False, False, True])
        .drop(columns="_label")
        .reset_index(drop=True)
    )


def standard_error_from_interval(low: float, high: float, level: float = 0.95) -> float:
    """Standard error implied by a symmetric normal interval ``low`` to ``high``."""
    if not high > low:
        raise ValueError("high has to exceed low")
    return (high - low) / (2.0 * NormalDist().inv_cdf(0.5 + level / 2.0))


def required_events(
    standard_error: float,
    events: int,
    odds_ratio: float,
    *,
    power: float = 0.8,
    alpha: float = 0.05,
) -> float:
    """Events needed to tell a log odds ratio from zero, scaled from an observed standard error.

    A planning approximation, not a result: the standard error of a logistic coefficient
    is taken to shrink with the square root of the number of events, with the event rate,
    the feature distribution and the correlation between the features unchanged.

    Parameters
    ----------
    standard_error
        Standard error of the coefficient in the existing sample, on the log odds scale.
    events
        Events in the existing sample.
    odds_ratio
        Assumed true odds ratio per unit of the feature, not equal to 1.
    power
        Probability of a two-sided rejection at level ``alpha`` if the assumption holds.
    alpha
        Two-sided significance level.

    Returns
    -------
    Number of events at which the two-sided Wald test has the requested power.
    """
    if standard_error <= 0 or events <= 0:
        raise ValueError("standard_error and events have to be positive")
    if odds_ratio <= 0 or odds_ratio == 1.0:
        raise ValueError("odds_ratio has to be positive and different from 1")
    if not 0.0 < alpha < 1.0 or not 0.0 < power < 1.0:
        raise ValueError("alpha and power have to lie between 0 and 1")
    normal = NormalDist()
    z = normal.inv_cdf(1.0 - alpha / 2.0) + normal.inv_cdf(power)
    return events * (z * standard_error / abs(log(odds_ratio))) ** 2
