"""Descriptive profiles and the sample size approximation. Synthetic inputs only."""

from math import log

import numpy as np
import pandas as pd
import pytest

from blindside.analysis.insights import (
    confident_blind,
    group_profile,
    required_events,
    standard_error_from_interval,
)


def _features(lower: list[float], upper: list[float]) -> dict[str, pd.DataFrame]:
    return {
        "lower": pd.DataFrame({"p_blind_side_head": lower}),
        "upper": pd.DataFrame({"p_blind_side_head": upper}),
    }


def test_confident_blind_needs_both_bounds_above_the_threshold():
    features = _features([1.0, 0.95, 0.85, 0.2, np.nan], [0.9, 0.6, 0.8, 0.1, 1.0])
    # Row 1 fails the upper bound, row 2 sits on the threshold, row 4 has no lower bound.
    assert confident_blind(features).tolist() == [True, False, False, False, False]
    assert confident_blind(features, threshold=0.5).tolist() == [True, True, True, False, False]


def test_group_profile_counts_and_sorts_by_flagged_engagements():
    groups = np.array(["a", "a", "a", "b", "b", "c", None], dtype=object)
    flag = np.array([True, False, False, True, True, False, True])
    table = group_profile(groups, flag)
    assert table["group"].tolist() == ["b", "a", "c"]  # the missing label is left out
    assert table["n"].tolist() == [2, 3, 1]
    assert table["flagged"].tolist() == [2, 1, 0]
    assert table["share"].tolist() == pytest.approx([1.0, 1 / 3, 0.0])
    assert table[["low", "high"]].isna().all().all()


def test_group_profile_interval_covers_the_share_and_respects_the_minimum_size():
    rng = np.random.default_rng(0)
    groups = np.array(["big"] * 200 + ["small"] * 10, dtype=object)
    flag = rng.random(210) < 0.3
    weights = rng.multinomial(210, np.full(210, 1 / 210), size=500)
    table = group_profile(groups, flag, weights).set_index("group")
    assert table.loc["big", "low"] < table.loc["big", "share"] < table.loc["big", "high"]
    assert table.loc["big", "high"] - table.loc["big", "low"] < 0.2
    assert np.isnan(table.loc["small", "low"])
    with pytest.raises(ValueError, match="one entry per engagement"):
        group_profile(groups, flag[:5])


def test_standard_error_from_interval_inverts_a_normal_interval():
    assert standard_error_from_interval(-1.96, 1.96) == pytest.approx(1.0, abs=1e-4)
    with pytest.raises(ValueError):
        standard_error_from_interval(1.0, 1.0)


def test_required_events_follows_the_inverse_square_law():
    # At power 50% the effect sits on the threshold: 1.96 standard errors.
    se, events = 0.1, 400
    threshold_effect = float(np.exp(1.959964 * se))
    assert required_events(se, events, threshold_effect, power=0.5) == pytest.approx(400, rel=1e-4)
    # Half the log odds ratio needs four times the events, and the sign does not matter.
    small = float(np.exp(log(threshold_effect) / 2))
    assert required_events(se, events, small, power=0.5) == pytest.approx(1600, rel=1e-4)
    assert required_events(se, events, 1 / small, power=0.5) == pytest.approx(1600, rel=1e-4)
    # More power needs more events: (1.96 + 0.8416)^2 / 1.96^2.
    assert required_events(se, events, threshold_effect) == pytest.approx(817.5, rel=1e-3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"standard_error": 0.0, "events": 10, "odds_ratio": 1.2},
        {"standard_error": 0.1, "events": 0, "odds_ratio": 1.2},
        {"standard_error": 0.1, "events": 10, "odds_ratio": 1.0},
        {"standard_error": 0.1, "events": 10, "odds_ratio": 1.2, "power": 1.0},
    ],
)
def test_required_events_rejects_impossible_inputs(kwargs: dict[str, float]):
    with pytest.raises(ValueError):
        required_events(**kwargs)
