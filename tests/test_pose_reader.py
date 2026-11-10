"""Tests for the streaming pose reader."""

import gzip
import json
import os
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from blindside.data.pose import iter_pose_frames, pose_to_tracking_frame, tracking_to_pose_frame


def _joint(x: float, y: float, z: float, err: float) -> dict:
    return {"xyz": [x, y, z], "p90_mae_cm": err}


FRAMES = [
    {
        "frame": 10,
        "period": 1,
        "player_data": [
            {"player_id": 1, "x": 0.0, "y": 0.0, "is_detected": True, "joints": None},
            {
                "player_id": 2,
                "x": 1.0,
                "y": 2.0,
                "is_detected": True,
                "joints": {"lHip": _joint(1.1, 2.0, 0.9, 7.5), "rHip": _joint(0.9, 2.0, 0.9, 7.4)},
            },
        ],
    },
    {"frame": 11, "period": 1, "player_data": []},
    {
        "frame": 12,
        "period": 1,
        "player_data": [
            {"player_id": 1, "x": 0.1, "y": 0.0, "is_detected": False, "joints": None},
        ],
    },
]


def _lines() -> bytes:
    return "".join(json.dumps(frame) + "\n" for frame in FRAMES).encode("utf-8")


@pytest.fixture(params=["jsonl", "gz", "zip"])
def pose_file(request: pytest.FixtureRequest, tmp_path: Path) -> Path:
    kind = request.param
    if kind == "jsonl":
        path = tmp_path / "match.jsonl"
        path.write_bytes(_lines())
    elif kind == "gz":
        path = tmp_path / "match.jsonl.gz"
        with gzip.open(path, "wb") as fh:
            fh.write(_lines())
    else:
        path = tmp_path / "match.jsonl.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("match.jsonl", _lines())
    return path


def test_reader_yields_identical_frames_for_every_container(pose_file: Path) -> None:
    assert list(iter_pose_frames(pose_file)) == FRAMES


def test_reader_is_lazy(pose_file: Path) -> None:
    frames = iter_pose_frames(pose_file)
    assert isinstance(frames, Iterator)
    first = next(frames)
    assert first["frame"] == 10
    assert first["player_data"][0]["joints"] is None
    frames.close()


def test_reader_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "match.jsonl"
    path.write_bytes(_lines() + b"\n\n")
    assert len(list(iter_pose_frames(path))) == len(FRAMES)


def test_zip_with_several_members_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "match.jsonl.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("a.jsonl", _lines())
        zf.writestr("b.jsonl", _lines())
    with pytest.raises(ValueError, match="exactly one"):
        next(iter_pose_frames(path))


def test_macos_metadata_members_are_ignored(tmp_path: Path) -> None:
    # The published archives contain "__MACOSX/._<match_id>.jsonl" next to the data.
    path = tmp_path / "match.jsonl.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("match.jsonl", _lines())
        zf.writestr("__MACOSX/._match.jsonl", b"\x00\x05\x16\x07binary")
    assert list(iter_pose_frames(path)) == FRAMES


def test_unknown_suffix_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "match.csv"
    path.write_text("x")
    with pytest.raises(ValueError, match="Unsupported"):
        next(iter_pose_frames(path))


def test_frame_conversion() -> None:
    assert tracking_to_pose_frame(2) == 5.0
    assert tracking_to_pose_frame(3) == 7.5
    assert pose_to_tracking_frame(5) == 2.0
    assert pose_to_tracking_frame(tracking_to_pose_frame(49795)) == 49795


def _opendata_sample() -> Path:
    root = Path(os.environ.get("BLINDSIDE_OPENDATA_DIR", Path(__file__).parents[2] / "opendata"))
    return root / "data" / "bodypose" / "sample_1925299_phase406.jsonl.gz"


@pytest.mark.skipif(not _opendata_sample().exists(), reason="opendata clone not available")
def test_reader_reproduces_sample_counts_from_upstream_manifest() -> None:
    n_frames = 0
    n_joints = 0
    for frame in iter_pose_frames(_opendata_sample()):
        n_frames += 1
        n_joints += sum(len(p["joints"]) for p in frame["player_data"] if p["joints"])
    assert n_frames == 309
    assert n_joints == 143_637


def test_step_decodes_every_nth_line(pose_file: Path) -> None:
    assert [frame["frame"] for frame in iter_pose_frames(pose_file, step=2)] == [10, 12]
    with pytest.raises(ValueError):
        next(iter_pose_frames(pose_file, step=0))
