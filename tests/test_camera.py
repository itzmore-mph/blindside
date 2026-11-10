"""Tests for the camera inference from projected image corners."""

import json
from pathlib import Path

import numpy as np
import pytest

from blindside.analysis.camera import (
    CORNER_KEYS,
    cached_camera_estimate,
    camera_ground_points,
    estimate_camera,
    read_corners,
)


def _corners(camera: tuple[float, float], pan: float, rng: np.random.Generator) -> list[float]:
    """Image footprint whose left and right edges pass through ``camera`` on the ground."""
    cx, cy = camera
    toward = 1.0 if cy < 0 else -1.0  # the view looks from the camera across the pitch
    near, far = 45.0 + rng.uniform(-5, 5), 110.0 + rng.uniform(-5, 5)
    left, right = pan - 0.25, pan + 0.25  # edge directions as slopes dx/dy

    def point(slope: float, distance: float) -> tuple[float, float]:
        return cx + slope * distance, cy + toward * distance

    (xtl, ytl), (xtr, ytr) = point(left, far), point(right, far)
    (xbl, ybl), (xbr, ybr) = point(left, near), point(right, near)
    return [xtl, ytl, xbl, ybl, xbr, ybr, xtr, ytr]


@pytest.mark.parametrize("camera", [(0.5, -70.0), (-1.0, 75.0)])
def test_camera_side_and_position_are_recovered(camera: tuple[float, float]) -> None:
    rng = np.random.default_rng(0)
    corners = np.array([_corners(camera, rng.uniform(-0.5, 0.5), rng) for _ in range(200)])
    points = camera_ground_points(corners)
    np.testing.assert_allclose(points, np.tile(camera, (200, 1)), atol=1e-8)

    estimate = estimate_camera(corners, np.repeat([1, 2], 100), match_id=9)
    assert estimate.side == ("-y" if camera[1] < 0 else "+y")
    assert (estimate.x, estimate.y) == pytest.approx(camera)
    assert estimate.share_bottom_edge == 1.0
    assert estimate.share_near_edge_shorter == 1.0
    assert estimate.share_meeting_point == 1.0
    assert set(estimate.per_period_xy) == {"1", "2"}
    assert estimate.reliable
    assert estimate.distance(camera[0] + 3.0, camera[1] + 4.0) == pytest.approx(5.0)


def test_inconsistent_cues_are_not_reliable() -> None:
    rng = np.random.default_rng(1)
    south = [_corners((0.0, -70.0), 0.0, rng) for _ in range(60)]
    north = [_corners((0.0, 70.0), 0.0, rng) for _ in range(40)]
    estimate = estimate_camera(np.array(south + north), np.ones(100), match_id=9)
    assert not estimate.reliable
    off_centre = np.array([_corners((30.0, -70.0), 0.0, rng) for _ in range(50)])
    assert not estimate_camera(off_centre, np.ones(50), match_id=9).reliable


def test_parallel_edges_give_nan() -> None:
    corners = np.array([[0.0, 10.0, 0.0, 0.0, 5.0, 0.0, 5.0, 10.0]])
    assert np.isnan(camera_ground_points(corners)).all()


def test_read_corners_and_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(2)
    lines = []
    for i in range(40):
        record = {"frame": i, "period": 1 if i < 20 else 2, "player_data": []}
        if i % 7:  # some frames have no corners
            values = _corners((0.0, -70.0), rng.uniform(-0.3, 0.3), rng)
            record["image_corners_projection"] = dict(zip(CORNER_KEYS, values, strict=True))
        lines.append(json.dumps(record))
    lines.append(json.dumps({"frame": 40, "period": None, "player_data": []}))
    path = tmp_path / "match.jsonl"
    path.write_text("\n".join(lines) + "\n")

    frames, periods, corners = read_corners(path, step=2)
    assert frames.tolist() == [i for i in range(0, 40, 2) if i % 7]
    assert corners.shape == (len(frames), 8) and set(periods) == {1, 2}

    monkeypatch.setenv("BLINDSIDE_DATA_DIR", str(tmp_path / "data"))
    first = cached_camera_estimate(path, match_id=9, step=2)
    path.unlink()
    assert cached_camera_estimate(path, match_id=9, step=2) == first
    assert first.side == "-y" and first.reliable
