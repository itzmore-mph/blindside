"""Tests for the counting logic of scripts/coverage.py on synthetic frames."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from blindside.data.manifest import JOINTS


def _load_script() -> ModuleType:
    # Loaded by path under a private name: "coverage" is also a well-known PyPI package.
    path = Path(__file__).parents[1] / "scripts" / "coverage.py"
    spec = importlib.util.spec_from_file_location("blindside_coverage_script", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cov = _load_script()

PRESSER, CARRIER, OTHER = 1, 2, 3


def _pose(err_cm: float = 6.0) -> dict:
    return {name: {"xyz": [0.0, 0.0, 1.0], "p90_mae_cm": err_cm} for name in JOINTS}


def _frame(number: int, poses: dict[int, dict | None], period: int | None = 1) -> dict:
    """A frame listing presser, carrier and one other player; ``poses`` maps id to joints."""
    return {
        "frame": number,
        "period": period,
        "player_data": [
            {
                "player_id": pid,
                "x": float(pid),
                "y": 0.0,
                "is_detected": pid in poses,
                "joints": poses.get(pid),
            }
            for pid in (PRESSER, CARRIER, OTHER)
        ],
    }


def _engagement(frame_start: int) -> "cov.Engagement":
    return cov.Engagement(
        event_id="9_0", presser_id=PRESSER, carrier_id=CARRIER, frame_start=frame_start, period=1
    )


def _run(frames: list[dict], frame_start: int = 100) -> "cov.MatchCoverage":
    acc = cov.MatchCoverage([_engagement(frame_start)], period_starts={1: 10})
    for frame in frames:
        acc.update(frame)
    return acc


def _flags(acc: "cov.MatchCoverage", offset_index: int = 0) -> dict[str, bool]:
    return {name: bool(values[0, offset_index]) for name, values in acc.flags.items()}


def test_window_for_even_tracking_frame_is_centred_on_one_pose_frame() -> None:
    assert cov.window_frames(100, 0.0, period_start=10) == [
        (248, False),
        (249, False),
        (250, True),
        (251, False),
        (252, False),
    ]


def test_window_for_odd_tracking_frame_has_two_nearest_pose_frames() -> None:
    assert cov.window_frames(101, 0.0, period_start=10) == [
        (251, False),
        (252, True),
        (253, True),
        (254, False),
    ]


def test_window_is_shifted_back_by_the_offset() -> None:
    assert [f for f, _ in cov.window_frames(100, 0.5, period_start=10)] == [236, 237, 238, 239]
    assert [f for f, _ in cov.window_frames(100, 1.0, period_start=10)] == [223, 224, 225, 226, 227]


def test_offset_before_period_start_has_no_window() -> None:
    assert cov.window_frames(12, 1.0, period_start=10) == []
    assert cov.window_frames(15, 0.5, period_start=10) != []


def test_both_requires_presser_and_carrier_in_the_same_frame() -> None:
    acc = _run([_frame(248, {PRESSER: _pose()}), _frame(252, {CARRIER: _pose()})])
    flags = _flags(acc)
    assert flags["presser"] and flags["carrier"]
    assert not flags["both"]
    assert not flags["both_near"]


def test_both_in_the_exact_frame_sets_every_flag() -> None:
    acc = _run([_frame(250, {PRESSER: _pose(), CARRIER: _pose()})])
    assert all(_flags(acc).values())


def test_window_edge_counts_but_is_not_nearest() -> None:
    acc = _run([_frame(254, {PRESSER: _pose(), CARRIER: _pose()})], frame_start=101)
    flags = _flags(acc)
    assert flags["both"]
    assert not flags["both_near"]


def test_frames_outside_the_window_are_ignored() -> None:
    acc = _run([_frame(247, {PRESSER: _pose(), CARRIER: _pose()}), _frame(253, {PRESSER: _pose()})])
    assert not any(_flags(acc).values())


def test_offsets_are_tracked_separately() -> None:
    acc = _run([_frame(225, {PRESSER: _pose(), CARRIER: _pose()})])
    assert not _flags(acc, 0)["both"]
    assert not _flags(acc, 1)["both"]
    assert _flags(acc, 2)["both"]


def test_error_gate_needs_all_six_orientation_joints_of_both_players() -> None:
    noisy = _pose()
    noisy["lEar"]["p90_mae_cm"] = 20.0
    acc = _run([_frame(250, {PRESSER: _pose(), CARRIER: noisy})])
    flags = _flags(acc)
    assert flags["both"]
    assert not flags["both_gated"]


def test_player_frame_counters_per_period() -> None:
    acc = _run(
        [
            _frame(250, {PRESSER: _pose(), CARRIER: _pose()}),
            _frame(251, {}),
            _frame(900, {OTHER: _pose()}, period=2),
        ]
    )
    first, second = acc.periods[1], acc.periods[2]
    assert first["frames"] == 2
    assert first["frames_with_pose"] == 1
    assert first["player_frames"] == 6
    assert first["player_frames_with_pose"] == 2
    assert first["detected"] == 2
    assert first["joint_observations"] == 2 * 29
    assert first["incomplete_poses"] == 0
    assert second["player_frames_with_pose"] == 1


def test_incomplete_pose_is_counted() -> None:
    partial = _pose()
    del partial["nose"]
    acc = _run([_frame(250, {PRESSER: partial})])
    assert acc.periods[1]["incomplete_poses"] == 1


def test_nearest_frame_positions_are_averaged_for_the_alignment_check() -> None:
    acc = _run([_frame(252, {}), _frame(253, {})], frame_start=101)
    result = acc.engagement_rows()[0]
    assert result["presser_pose_x"] == pytest.approx(float(PRESSER))
    assert result["carrier_pose_x"] == pytest.approx(float(CARRIER))


def test_histogram_quantiles_are_exact_at_two_decimals() -> None:
    hist = {560: 1, 570: 2, 900: 1}
    assert cov.hist_quantile(hist, 0.5) == pytest.approx(5.70)
    assert cov.hist_quantile(hist, 0.0) == pytest.approx(5.60)
    assert cov.hist_quantile(hist, 1.0) == pytest.approx(9.00)


def test_error_histogram_collects_every_joint() -> None:
    acc = _run([_frame(250, {PRESSER: _pose(7.25)})])
    hist = acc.p90_histograms()
    assert set(hist) == set(JOINTS)
    assert hist["lHip"] == {725: 1}


def test_replace_between_markers() -> None:
    text = f"a\n{cov.MARKER_START}\nold\n{cov.MARKER_END}\nb\n"
    assert cov.replace_between_markers(text, "new") == (
        f"a\n{cov.MARKER_START}\nnew\n{cov.MARKER_END}\nb\n"
    )
    with pytest.raises(ValueError, match="markers"):
        cov.replace_between_markers("no markers here", "new")
