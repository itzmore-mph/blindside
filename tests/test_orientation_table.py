"""Tests for the per-frame orientation table, including the sample regression."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from blindside.analysis.consistency import (
    abs_difference_deg,
    segment_agreement,
    uncertainty_summary,
    velocity_agreement,
)
from blindside.data.paths import sample_pose_path
from blindside.features.orientation_table import (
    TABLE_JOINTS,
    OrientationConfig,
    build_orientation_table,
    cached_orientation_table,
    orientation_cache_path,
)
from skeleton import make_skeleton, mirror_y

DEG = np.pi / 180.0
N_FRAMES = 40
RUNNER, SWAPPED, NO_POSE = 1, 2, 3
RUNNER_FACING = 30.0  # degrees, also the direction of travel
RUNNER_SPEED = 5.0  # m/s


def _joints(skeleton: dict[str, np.ndarray], p90: float) -> dict:
    return {
        name: {"xyz": [float(skeleton[name][0]), float(skeleton[name][1]), 1.0], "p90_mae_cm": p90}
        for name in TABLE_JOINTS
    }


@pytest.fixture
def pose_file(tmp_path: Path) -> Path:
    """Two periods of 20 frames. One runner, one player with swapped labels, one without pose."""
    direction = np.array([np.cos(RUNNER_FACING * DEG), np.sin(RUNNER_FACING * DEG)])
    lines = []
    for i in range(N_FRAMES):
        position = direction * RUNNER_SPEED * i / 25.0
        runner = make_skeleton(RUNNER_FACING * DEG, position)
        swapped = mirror_y(make_skeleton(40 * DEG, (10.0, 5.0)), swap_labels=False)
        players = [
            {"player_id": RUNNER, "x": position[0], "y": position[1], "is_detected": True,
             "joints": _joints(runner, 6.0) if i != 3 else None},
            {"player_id": SWAPPED, "x": 10.0, "y": -5.0, "is_detected": True,
             "joints": _joints(swapped, 6.0)},
            {"player_id": NO_POSE, "x": -20.0, "y": 0.0, "is_detected": False, "joints": None},
        ]  # fmt: skip
        lines.append(json.dumps({"frame": 100 + i, "period": 1 if i < 20 else 2,
                                 "player_data": players}))  # fmt: skip
    lines.append(json.dumps({"frame": 100 + N_FRAMES, "period": None, "player_data": []}))
    path = tmp_path / "synthetic.jsonl"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_table_rows_and_angles(pose_file: Path) -> None:
    table = build_orientation_table(pose_file, match_id=7)
    # One row per player-frame with joints: the runner misses one frame, player 3 has none.
    assert len(table) == (N_FRAMES - 1) + N_FRAMES
    assert set(table["player_id"]) == {RUNNER, SWAPPED}
    assert (table["match_id"] == 7).all()
    assert set(table["period"]) == {1, 2}

    runner = table[table["player_id"] == RUNNER]
    for segment in ("torso", "hips", "head"):
        assert abs_difference_deg(runner[f"{segment}_deg"], RUNNER_FACING).max() < 1e-3
        assert abs_difference_deg(runner[f"{segment}_mean_deg"], RUNNER_FACING).max() < 5.0
        assert not runner[f"{segment}_flip"].any()
    assert abs_difference_deg(runner["head_nose_deg"], RUNNER_FACING).max() < 1e-3
    assert abs_difference_deg(runner["forward_deg"], RUNNER_FACING).max() < 1e-3
    np.testing.assert_allclose(runner["torso_width_m"], 0.37, atol=1e-5)
    np.testing.assert_allclose(runner["torso_p90_cm"], 6.0)
    # Same joint error, narrower baseline, larger angular uncertainty.
    assert runner["torso_sd_deg"].median() < 0.7 * runner["hips_sd_deg"].median()
    assert runner["hips_sd_deg"].median() < runner["head_sd_deg"].median()


def test_table_velocity_respects_period_boundaries(pose_file: Path) -> None:
    table = build_orientation_table(pose_file, match_id=7)
    runner = table[table["player_id"] == RUNNER].set_index("frame")
    inside = runner.loc[[106, 110, 114, 126, 130]]
    np.testing.assert_allclose(inside["speed"], RUNNER_SPEED, rtol=1e-4)
    assert abs_difference_deg(inside["heading_deg"], RUNNER_FACING).max() < 1e-2
    # Within 5 frames of a period start or end the central difference is undefined.
    edge = runner.loc[[100, 104, 115, 119, 120, 124, 135, 139]]
    assert edge["speed"].isna().all() and edge["heading_deg"].isna().all()
    standing = table[table["player_id"] == SWAPPED].set_index("frame")
    assert standing.loc[110, "speed"] == pytest.approx(0.0, abs=1e-6)


def test_swapped_labels_are_flagged_not_altered(pose_file: Path) -> None:
    table = build_orientation_table(pose_file, match_id=7)
    swapped = table[table["player_id"] == SWAPPED]
    assert swapped[["torso_flip", "hips_flip", "head_flip"]].all().all()
    assert abs_difference_deg(swapped["forward_deg"], -40.0).max() < 1e-3
    assert abs_difference_deg(swapped["torso_deg"], 140.0).max() < 1e-3


def test_table_is_reproducible_and_config_changes_the_uncertainty(pose_file: Path) -> None:
    first = build_orientation_table(pose_file, match_id=7)
    second = build_orientation_table(pose_file, match_id=7)
    pd.testing.assert_frame_equal(first, second)
    wide = build_orientation_table(pose_file, 7, OrientationConfig(p90_to_sigma=2.0))
    narrow = build_orientation_table(pose_file, 7, OrientationConfig(p90_to_sigma=3.0))
    ratio = wide["torso_sd_deg"].median() / narrow["torso_sd_deg"].median()
    assert ratio == pytest.approx(1.5, rel=0.1)
    # Batches smaller than the file exercise the chunked path.
    chunked = build_orientation_table(pose_file, 7, OrientationConfig(chunk_poses=7))
    np.testing.assert_allclose(chunked["torso_deg"], first["torso_deg"])
    assert len(chunked) == len(first)


def test_cache_roundtrip(pose_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BLINDSIDE_DATA_DIR", str(tmp_path / "data"))
    config = OrientationConfig(n_samples=50)
    target = orientation_cache_path(pose_file, config)
    assert not target.exists()
    built = cached_orientation_table(pose_file, 7, config)
    assert target.exists() and target.parent == tmp_path / "data" / "cache" / "orientation"
    pose_file.unlink()  # the second call must not need the source
    cached = cached_orientation_table(pose_file, 7, config)
    pd.testing.assert_frame_equal(built, cached)
    assert orientation_cache_path(pose_file, OrientationConfig(n_samples=51)) != target


# --- Regression against docs/FEASIBILITY.md sections 4 and 5 (sample phase) ---------------

needs_sample = pytest.mark.skipif(
    not sample_pose_path().exists(),
    reason="pose sample not found, clone SkillCorner/opendata or set BLINDSIDE_OPENDATA_DIR",
)


@pytest.fixture(scope="module")
def sample_table() -> pd.DataFrame:
    return build_orientation_table(sample_pose_path(), match_id=1925299)


@needs_sample
def test_sample_size(sample_table: pd.DataFrame) -> None:
    assert len(sample_table) == 4953  # 143,637 joint observations / 29 joints
    assert sample_table["frame"].nunique() == 309


@needs_sample
def test_sample_reproduces_feasibility_section_4(sample_table: pd.DataFrame) -> None:
    summary = uncertainty_summary(sample_table).set_index("segment")
    # (median width m, SD median, SD 90th percentile, share with SD below 20 degrees)
    expected = {
        "torso": (0.37, 5.0, 6.5, 0.99),
        "hips": (0.22, 12.2, 17.3, 0.95),
        "head": (0.18, 12.5, 18.4, 0.92),
    }
    for segment, (width, sd_median, sd_p90, share) in expected.items():
        row = summary.loc[segment]
        assert row["median_width_m"] == pytest.approx(width, abs=0.005), segment
        # Tolerances cover the Monte Carlo noise of 200 draws with a different seed.
        assert row["sd_median_deg"] == pytest.approx(sd_median, abs=0.15), segment
        assert row["sd_p90_deg"] == pytest.approx(sd_p90, abs=0.3), segment
        assert row["share_sd_below_20"] == pytest.approx(share, abs=0.01), segment
    agreement = segment_agreement(sample_table)
    assert agreement[("torso", "hips")] == pytest.approx(12.2, abs=0.06)
    assert agreement[("torso", "head")] == pytest.approx(15.3, abs=0.06)


@needs_sample
def test_sample_reproduces_feasibility_section_5(sample_table: pd.DataFrame) -> None:
    hips = velocity_agreement(sample_table, "hips")
    torso = velocity_agreement(sample_table, "torso")
    assert hips["n"].tolist() == [1609, 2118, 903, 176]
    np.testing.assert_allclose(hips["median_abs_deg"], [34.3, 14.0, 11.4, 9.1], atol=0.06)
    np.testing.assert_allclose(torso["median_abs_deg"], [35.6, 18.9, 14.3, 9.1], atol=0.06)
    np.testing.assert_allclose(hips["share_above_90"], [0.14, 0.10, 0.03, 0.00], atol=0.005)
    # The check itself: agreement improves monotonically with speed, no flips at sprint speed.
    assert hips["median_abs_deg"].is_monotonic_decreasing
    assert hips["share_above_90"].iloc[-1] == 0.0
