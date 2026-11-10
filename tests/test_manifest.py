"""Tests for the packaged pose manifest."""

import re

import pytest

from blindside.data.manifest import JOINTS, pose_archive, pose_match_ids


def test_manifest_lists_both_pose_matches() -> None:
    assert pose_match_ids() == (1925299, 1996435)


@pytest.mark.parametrize("match_id", [1925299, 1996435])
def test_archive_entries_are_well_formed(match_id: int) -> None:
    archive = pose_archive(match_id)
    assert archive.match_id == match_id
    assert archive.size_bytes > 500_000_000
    assert re.fullmatch(r"[0-9a-f]{64}", archive.sha256)
    assert archive.url.startswith("https://huggingface.co/datasets/SkillCorner/opendata-bodypose/")
    assert archive.url.endswith(f"/raw/{match_id}.jsonl.zip")
    assert "/resolve/main/" not in archive.url, "revision must be pinned to a commit"


def test_unknown_match_raises_with_available_ids() -> None:
    with pytest.raises(ValueError, match="1925299"):
        pose_archive(42)


def test_joint_list_has_29_unique_names() -> None:
    assert len(JOINTS) == 29
    assert len(set(JOINTS)) == 29
