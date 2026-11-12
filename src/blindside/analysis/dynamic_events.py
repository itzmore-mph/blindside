"""Exploratory: ``p_blind_side`` against SkillCorner's possession labels and defending phases.

Nothing here is pre-registered (docs/PREREGISTRATION.md fixes the primary test and is not
touched). The possession columns are properties of the pressed possession, most of them
measured after the engagement start, so every table built from them is a description of
what went together in two matches. The phase tables use no outcome at all.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from blindside.analysis.agreement import agreement_table
from blindside.analysis.incremental import MIN_BIN_SIZE, bin_index, bin_table
from blindside.data.events import load_dynamic_events, load_phases_of_play, phase_at
from blindside.features.pose_features import FOV_HALF_ANGLE_DEG
from blindside.features.xy_proxy import xy_blind_side_proxy

#: Numeric column of the pressed possession: change of the expected possession value.
EPV_DELTA = "possession_epv_delta_for"
#: Graded columns of the pressed possession, five levels from ``very_easy`` to ``very_hard``.
GRADED_COLUMNS: tuple[str, ...] = (
    "passing_option_ease_end",
    "time_to_impact_end",
    "space_constraint_start",
)
POSSESSION_COLUMNS: tuple[str, ...] = (EPV_DELTA, *GRADED_COLUMNS)
GRADES: tuple[str, ...] = ("very_easy", "easy", "medium", "hard", "very_hard")
#: Grades counted as difficult for the player in possession.
HARD_GRADES: tuple[str, ...] = ("hard", "very_hard")
PHASE_COLUMN = "team_out_of_possession_phase_type"
NO_PHASE = "outside every phase"
#: Pose label used for the phase tables: blind if more than this share of the samples is blind.
LABEL_THRESHOLD = 0.5


def is_hard(grades: np.ndarray | pd.Series) -> np.ndarray:
    """1.0 for ``hard`` or ``very_hard``, 0.0 for the other grades, NaN for a missing grade.

    Raises
    ------
    ValueError
        If a value is not one of :data:`GRADES`.
    """
    values = pd.Series(grades, dtype=object)
    known = values.notna().to_numpy()
    unexpected = set(values[known]) - set(GRADES)
    if unexpected:
        raise ValueError(f"unknown grades: {sorted(unexpected)}")
    return np.where(known, values.isin(HARD_GRADES).to_numpy(dtype=float), np.nan)


def possession_context(test: pd.DataFrame) -> pd.DataFrame:
    """The possession columns of each engagement's pressed possession.

    Parameters
    ----------
    test
        Engagements with ``match_id`` and ``possession_id`` (the event id of the pressed
        possession).

    Returns
    -------
    Same index as ``test``: :data:`EPV_DELTA` as float, each of :data:`GRADED_COLUMNS` as
    its grade and ``{column}_hard`` from :func:`is_hard`. NaN or None where the possession
    row is missing or the column is empty.
    """
    out = pd.DataFrame(index=test.index)
    out[EPV_DELTA] = np.nan
    for name in GRADED_COLUMNS:
        out[name] = None
    for match_id in test["match_id"].unique():
        rows = test["match_id"] == match_id
        events = load_dynamic_events(int(match_id), columns=["event_id", *POSSESSION_COLUMNS])
        linked = events.drop_duplicates("event_id").set_index("event_id")
        linked = linked.reindex(test.loc[rows, "possession_id"].to_numpy())
        out.loc[rows, EPV_DELTA] = pd.to_numeric(
            linked[EPV_DELTA].astype(object), errors="coerce"
        ).to_numpy(dtype=float)
        for name in GRADED_COLUMNS:
            values = linked[name].astype(object)
            out.loc[rows, name] = values.where(values.notna(), None).to_numpy(dtype=object)
    for name in GRADED_COLUMNS:
        out[f"{name}_hard"] = is_hard(out[name])
    return out


def defending_phase(test: pd.DataFrame) -> np.ndarray:
    """Defending phase of the pressing team at each engagement start, from ``phases_of_play.csv``.

    Returns an object array, :data:`NO_PHASE` for an engagement that starts between phases.
    """
    out = np.full(len(test), NO_PHASE, dtype=object)
    for match_id in test["match_id"].unique():
        rows = (test["match_id"] == match_id).to_numpy()
        phase = phase_at(
            load_phases_of_play(int(match_id)),
            test.loc[rows, "frame_start"].to_numpy(),
            PHASE_COLUMN,
        )
        out[rows] = np.where(pd.isna(phase), NO_PHASE, phase)
    return out


def context_by_bin(
    p_blind_side: np.ndarray, values: np.ndarray, weights: np.ndarray
) -> tuple[pd.DataFrame, tuple[float, float, float]]:
    """Mean of a possession column per pre-registered bin of ``p_blind_side``.

    Parameters
    ----------
    p_blind_side
        Probability per engagement.
    values
        Possession column per engagement, NaN where missing. Those rows are left out.
    weights
        Cluster bootstrap weights ``(n_resamples, n)``.

    Returns
    -------
    table
        :func:`blindside.analysis.incremental.bin_table` without the Model 0 columns.
    contrast
        Blind bin minus in-view bin with its 95% interval. The interval is NaN if either
        bin holds fewer than 50 engagements.
    """
    p = np.asarray(p_blind_side, dtype=float)
    values = np.asarray(values, dtype=float)
    table = bin_table(p, values, np.full(len(values), np.nan), weights)
    index = np.where(np.isnan(values), -1, bin_index(p))
    blind, view = index == 2, index == 0
    if not blind.any() or not view.any():
        return table, (np.nan, np.nan, np.nan)
    estimate = float(values[blind].mean() - values[view].mean())
    if blind.sum() < MIN_BIN_SIZE or view.sum() < MIN_BIN_SIZE:
        return table, (estimate, np.nan, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        resampled = (weights[:, blind] @ values[blind]) / weights[:, blind].sum(axis=1) - (
            weights[:, view] @ values[view]
        ) / weights[:, view].sum(axis=1)
    low, high = np.nanpercentile(resampled, [2.5, 97.5])
    return table, (estimate, float(low), float(high))


def phase_agreement(
    test: pd.DataFrame,
    features: Mapping[str, pd.DataFrame],
    phase: np.ndarray,
    weights: np.ndarray,
    *,
    bound: str = "lower",
) -> pd.DataFrame:
    """Both blind-side labels per defending phase. No outcome column is read.

    Returns
    -------
    One row per phase after the row ``"all"``, the largest phase first: the columns of
    :func:`blindside.analysis.agreement.agreement_table` for the head label, plus
    ``torso_blind`` and ``kappa_torso`` with its interval.
    """
    heading = test["approach_angle_heading_deg"].to_numpy(dtype=float)
    proxy = xy_blind_side_proxy(heading, FOV_HALF_ANGLE_DEG)
    phase = np.asarray(phase, dtype=object)
    names = pd.Series(phase).value_counts().index.tolist()
    groups = {name: phase == name for name in names}
    tables = {}
    for segment in ("head", "torso"):
        p = features[bound][f"p_blind_side_{segment}"].to_numpy(dtype=float)
        tables[segment] = agreement_table(
            p > LABEL_THRESHOLD, proxy, weights, groups, keep=np.isfinite(p) & np.isfinite(heading)
        )
    out = tables["head"].copy()
    out["torso_blind"] = tables["torso"]["pose_blind"].to_numpy()
    for suffix in ("", "_low", "_high"):
        out[f"kappa_torso{suffix}"] = tables["torso"][f"kappa{suffix}"].to_numpy()
    return out
