"""Exploratory joins and tables against possession labels and phases. Synthetic data only."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from blindside.analysis import dynamic_events
from blindside.analysis.baseline import cluster_bootstrap_weights
from blindside.analysis.dynamic_events import (
    EPV_DELTA,
    NO_PHASE,
    PHASE_COLUMN,
    context_by_bin,
    defending_phase,
    is_hard,
    phase_agreement,
    possession_context,
)
from blindside.data.events import load_phases_of_play, phase_at
from blindside.data.manifest import pose_match_ids
from blindside.data.opendata import load_opendata_manifest, phases_relative

PHASES = pd.DataFrame(
    {
        "frame_start": [10, 30, 60],
        "frame_end": [30, 50, 90],
        PHASE_COLUMN: ["low_block", "high_block", "chaotic"],
    }
)


def test_phase_at_uses_half_open_ranges_and_leaves_gaps_empty():
    frames = np.array([9, 10, 29, 30, 49, 50, 59, 60, 89, 90])
    assert phase_at(PHASES, frames, PHASE_COLUMN).tolist() == [
        None, "low_block", "low_block", "high_block", "high_block", None, None,
        "chaotic", "chaotic", None,
    ]  # fmt: skip


def test_phase_at_does_not_depend_on_the_row_order_and_handles_no_phase():
    shuffled = PHASES.iloc[[2, 0, 1]]
    assert phase_at(shuffled, np.array([35, 70]), PHASE_COLUMN).tolist() == [
        "high_block",
        "chaotic",
    ]
    assert phase_at(PHASES.iloc[:0], np.array([35]), PHASE_COLUMN).tolist() == [None]


def test_phases_of_the_pose_matches_are_in_the_manifest_and_load_from_a_clone(tmp_path: Path):
    files = load_opendata_manifest()["files"]
    for match_id in pose_match_ids():
        assert phases_relative(match_id) in files
    target = tmp_path / phases_relative(7)
    target.parent.mkdir(parents=True)
    PHASES.to_csv(target, index=False)
    assert load_phases_of_play(7, tmp_path)[PHASE_COLUMN].tolist() == PHASES[PHASE_COLUMN].tolist()


def test_is_hard_grades_and_missing_values():
    grades = ["very_easy", "easy", "medium", "hard", "very_hard", None, np.nan]
    np.testing.assert_array_equal(is_hard(grades), [0, 0, 0, 1, 1, np.nan, np.nan])
    with pytest.raises(ValueError, match="unknown grades"):
        is_hard(["hard", "impossible"])


def test_possession_context_reads_the_linked_possession_row(monkeypatch: pytest.MonkeyPatch):
    events = pd.DataFrame(
        {
            "event_id": ["8_1", "8_2", "9_1"],
            EPV_DELTA: [0.01, -0.02, None],
            "passing_option_ease_end": ["hard", "easy", None],
            "time_to_impact_end": ["very_hard", "medium", None],
            "space_constraint_start": ["easy", None, None],
        }
    )
    requested = []

    def fake(match_id: int, root=None, columns=None) -> pd.DataFrame:
        requested.append((match_id, tuple(columns)))
        return events[list(columns)]

    monkeypatch.setattr(dynamic_events, "load_dynamic_events", fake)
    test = pd.DataFrame({"match_id": [5, 5, 5, 5], "possession_id": ["8_2", "8_1", "8_1", "8_9"]})
    context = possession_context(test)
    assert requested[0][0] == 5 and "event_id" in requested[0][1]
    np.testing.assert_array_equal(context[EPV_DELTA], [-0.02, 0.01, 0.01, np.nan])
    assert context["passing_option_ease_end"].tolist() == ["easy", "hard", "hard", None]
    np.testing.assert_array_equal(context["time_to_impact_end_hard"], [0, 1, 1, np.nan])
    np.testing.assert_array_equal(context["space_constraint_start_hard"], [np.nan, 0, 0, np.nan])


def test_defending_phase_joins_per_match(monkeypatch: pytest.MonkeyPatch):
    later = PHASES.assign(**{PHASE_COLUMN: ["medium_block"] * 3})
    monkeypatch.setattr(
        dynamic_events, "load_phases_of_play", lambda match_id: PHASES if match_id == 1 else later
    )
    test = pd.DataFrame({"match_id": [1, 2, 1, 2], "frame_start": [15, 15, 55, 95]})
    assert defending_phase(test).tolist() == ["low_block", "medium_block", NO_PHASE, NO_PHASE]


def _bins(n: int = 400):
    rng = np.random.default_rng(0)
    p = np.repeat([0.0, 0.5, 1.0], [n, 20, n])
    values = np.where(p == 1.0, 1.0, 0.0) + rng.normal(0, 0.5, len(p))
    clusters = np.arange(len(p))
    return p, values, cluster_bootstrap_weights(clusters, None, 300, seed=0)


def test_context_by_bin_contrast_is_blind_minus_in_view_and_skips_missing_values():
    p, values, weights = _bins()
    values[:5] = np.nan
    table, (estimate, low, high) = context_by_bin(p, values, weights)
    assert table["n"].tolist() == [395, 20, 400]
    assert estimate == pytest.approx(values[p == 1.0].mean() - np.nanmean(values[p == 0.0]))
    assert low < estimate < high
    assert low > 0.8 and high < 1.2
    assert np.isnan(table.loc[1, "rate_low"])  # fewer than 50 engagements: no interval


def test_context_by_bin_gives_no_interval_for_a_small_bin():
    p, values, weights = _bins()
    keep = np.r_[0:400, 400:420, 420:450]
    _, (estimate, low, high) = context_by_bin(p[keep], values[keep], weights[:, keep])
    assert np.isfinite(estimate) and np.isnan(low) and np.isnan(high)
    _, contrast = context_by_bin(p[:400], values[:400], weights[:, :400])
    assert np.isnan(contrast).all()


def test_phase_agreement_has_one_row_per_phase_and_reads_no_outcome():
    rng = np.random.default_rng(1)
    n = 300
    heading = rng.uniform(0, 180, n)
    test = pd.DataFrame({"approach_angle_heading_deg": heading})
    truth = heading > 100
    p_head = np.where(truth, 0.95, 0.05)
    p_head[:150] = rng.random(150)  # unrelated to the proxy in the first phase
    feature = pd.DataFrame({"p_blind_side_head": p_head, "p_blind_side_torso": p_head})
    phase = np.repeat(np.array(["low_block", "chaotic"], dtype=object), [150, 150])
    weights = cluster_bootstrap_weights(np.arange(n), None, 200, seed=0)
    table = phase_agreement(test, {"lower": feature}, phase, weights).set_index("group")
    assert table.index.tolist() == ["all", "low_block", "chaotic"]
    assert table.loc["chaotic", "kappa"] == pytest.approx(1.0)
    assert abs(table.loc["low_block", "kappa"]) < 0.25
    assert table.loc["chaotic", "torso_blind"] == pytest.approx(truth[150:].mean())
    assert table.loc["chaotic", "kappa_torso"] == pytest.approx(1.0)
