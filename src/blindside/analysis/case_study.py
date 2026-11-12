"""Pick single engagements that illustrate the two blind-side labels, by a fixed rule.

The rule reads the carrier speed, the XY proxy, ``p_blind_side`` under both bounds and the
flip flags. It reads no outcome column, so the engagements are not chosen for how they ended.
They are still chosen for clarity: each is the most clear-cut case of its kind, not a
typical one.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from blindside.analysis.incremental import BIN_EDGES
from blindside.features.pose_features import BOUNDS, FOV_HALF_ANGLE_DEG
from blindside.features.xy_proxy import (
    RELIABLE,
    UNRELIABLE,
    classify_xy_reliability,
    xy_blind_side_proxy,
)

#: Slow carriers below this speed barely move, so their direction of travel is mostly noise.
MIN_SLOW_SPEED_MPS = 1.0
#: Presser and carrier at most this far apart, so that one zoomed panel holds both.
MAX_DISTANCE_M = 8.0
DISAGREE = "disagree"
AGREE_BLIND = "agree_blind"


def case_study_candidates(
    test: pd.DataFrame, features: Mapping[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """Candidates for the two panels, the clearest first.

    Parameters
    ----------
    test
        Included engagements with ``carrier_speed``, ``approach_angle_heading_deg``,
        ``interplayer_distance_start``, ``carrier_head_flip`` and ``carrier_torso_flip``.
    features
        ``{bound: pose_feature_table(...)}`` for both bounds, same index as ``test``.

    Returns
    -------
    ``"disagree"``
        Carrier at 1 to 2 m/s, XY proxy blind, head and torso ``p_blind_side`` below 0.1
        under both bounds. Sorted by the proxy angle minus the angle between head and
        presser, largest first: far behind the direction of travel and in front of the head.
    ``"agree_blind"``
        Carrier at 4 m/s or faster, XY proxy blind, head and torso ``p_blind_side`` above
        0.9 under both bounds. Sorted by the sum of the two angles, largest first.

    Both need a carrier pose without a flip flag and at most 8 m between the players. Each
    table is a subset of ``test`` with the column ``score``. Ties keep the order of ``test``.
    """
    heading = test["approach_angle_heading_deg"].to_numpy(dtype=float)
    speed = test["carrier_speed"].to_numpy(dtype=float)
    reliability = classify_xy_reliability(speed)
    off_head = features["upper"]["presser_off_head_deg"].to_numpy(dtype=float)
    p = np.column_stack(
        [
            features[bound][name].to_numpy(dtype=float)
            for bound in BOUNDS
            for name in ("p_blind_side_head", "p_blind_side_torso")
        ]
    )
    usable = (
        xy_blind_side_proxy(heading, FOV_HALF_ANGLE_DEG)
        & ~test["carrier_head_flip"].to_numpy(dtype=bool)
        & ~test["carrier_torso_flip"].to_numpy(dtype=bool)
        & (test["interplayer_distance_start"].to_numpy(dtype=float) <= MAX_DISTANCE_M)
    )
    masks = {
        DISAGREE: usable
        & (reliability == UNRELIABLE)
        & (speed >= MIN_SLOW_SPEED_MPS)
        & (p < BIN_EDGES[0]).all(axis=1),
        AGREE_BLIND: usable & (reliability == RELIABLE) & (p > BIN_EDGES[1]).all(axis=1),
    }
    scores = {DISAGREE: heading - off_head, AGREE_BLIND: heading + off_head}
    return {
        kind: test[mask]
        .assign(score=scores[kind][mask])
        .sort_values("score", ascending=False, kind="stable")
        for kind, mask in masks.items()
    }
