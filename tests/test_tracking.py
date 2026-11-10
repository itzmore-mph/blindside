"""Tests for the tracking manifest, the streaming reader and the position grid."""

import json
from pathlib import Path

import numpy as np
import pytest

from blindside.data.tracking import (
    cached_positions,
    position_grid,
    read_positions,
    tracking_file,
    tracking_match_ids,
)


def test_manifest_lists_all_matches_with_pinned_urls() -> None:
    ids = tracking_match_ids()
    assert len(ids) == 20
    assert {1925299, 1996435} <= set(ids)
    entry = tracking_file(1925299)
    assert entry.size_bytes == 97091519
    assert len(entry.sha256) == 64
    assert entry.url.startswith("https://media.githubusercontent.com/media/SkillCorner/opendata/")
    assert entry.url.endswith("data/matches/1925299/1925299_tracking_extrapolated.jsonl")
    assert "/main/" not in entry.url
    with pytest.raises(ValueError):
        tracking_file(1)


@pytest.fixture
def tracking_path(tmp_path: Path) -> Path:
    """Frames 8 to 14. Frames 8 and 9 lie outside the periods, player 3 appears late."""
    lines = []
    for frame in range(8, 15):
        players = [
            {"player_id": 1, "x": float(frame), "y": 1.0, "is_detected": True},
            {"player_id": 2, "x": -float(frame), "y": None if frame == 12 else 2.0,
             "is_detected": False},
        ]  # fmt: skip
        if frame >= 13:
            players.append({"player_id": 3, "x": 0.5, "y": 0.5, "is_detected": True})
        lines.append(
            json.dumps(
                {
                    "frame": frame,
                    "period": None if frame < 10 else (1 if frame < 13 else 2),
                    "player_data": [] if frame < 10 else players,
                }
            )
        )
    path = tmp_path / "7_tracking_extrapolated.jsonl"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_read_positions_keeps_in_period_rows_with_a_position(tracking_path: Path) -> None:
    table = read_positions(tracking_path, match_id=7)
    # 5 in-period frames: player 1 always, player 2 misses frame 12, player 3 in two frames.
    assert len(table) == 5 + 4 + 2
    assert table["frame"].min() == 10
    assert (table["match_id"] == 7).all()
    assert table.loc[table["player_id"] == 1, "is_detected"].all()
    assert not table.loc[table["player_id"] == 2, "is_detected"].any()


def test_position_grid_lookup(tracking_path: Path) -> None:
    grid = position_grid(read_positions(tracking_path, match_id=7))
    assert grid.first_frame == 10
    assert grid.xy.shape == (5, 3, 2)
    np.testing.assert_allclose(grid.at(np.array([11]), np.array([1])), [[11.0, 1.0]])
    np.testing.assert_allclose(grid.at(np.array([14]), np.array([2])), [[-14.0, 2.0]])
    # Missing position, frame outside the grid, unknown player.
    assert np.isnan(grid.at(np.array([12]), np.array([2]))).all()
    assert np.isnan(grid.at(np.array([9]), np.array([1]))).all()
    assert np.isnan(grid.at(np.array([11]), np.array([99]))).all()
    assert np.isnan(grid.at(np.array([11]), np.array([3]))).all()
    np.testing.assert_array_equal(grid.period_at(np.array([9, 10, 12, 13, 99])), [0, 1, 1, 2, 0])


def test_cached_positions_reads_the_file_once(
    tracking_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BLINDSIDE_DATA_DIR", str(tmp_path / "data"))
    first = cached_positions(7, tracking_path)
    tracking_path.unlink()
    second = cached_positions(7)
    assert first.equals(second)
