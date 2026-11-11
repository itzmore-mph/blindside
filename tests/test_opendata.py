"""Tests for locating and fetching the non-LFS files of the opendata repository."""

import hashlib
from pathlib import Path

import pytest

from blindside.data import opendata
from blindside.data.download import DownloadError
from blindside.data.events import load_match_metadata
from blindside.data.opendata import load_opendata_manifest, metadata_relative, opendata_file
from blindside.data.paths import DATA_DIR_ENV, OPENDATA_DIR_ENV, sample_pose_path
from blindside.data.tracking import tracking_match_ids

RELATIVE = metadata_relative(7)
PAYLOAD = b'{"pitch_length": 105, "pitch_width": 68}'


@pytest.fixture
def remote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake remote repository behind a file URL, an empty data directory and no clone."""
    source = tmp_path / "remote"
    (source / RELATIVE).parent.mkdir(parents=True)
    (source / RELATIVE).write_bytes(PAYLOAD)
    manifest = {
        "base_url": source.as_uri(),
        "files": {
            RELATIVE: {"size_bytes": len(PAYLOAD), "sha256": hashlib.sha256(PAYLOAD).hexdigest()}
        },
    }
    monkeypatch.setattr(opendata, "load_opendata_manifest", lambda: manifest)
    monkeypatch.delenv(OPENDATA_DIR_ENV, raising=False)
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path / "work" / "data"))
    (tmp_path / "work").mkdir()
    monkeypatch.chdir(tmp_path / "work")
    return source


def test_manifest_covers_events_and_metadata_of_every_match() -> None:
    files = load_opendata_manifest()["files"]
    for match_id in tracking_match_ids():
        assert opendata.events_relative(match_id) in files
        assert metadata_relative(match_id) in files
    assert "data/bodypose/sample_1925299_phase406.jsonl.gz" in files
    assert all(len(entry["sha256"]) == 64 and entry["size_bytes"] > 0 for entry in files.values())


def test_without_a_clone_the_file_is_downloaded_and_verified(remote: Path, tmp_path: Path) -> None:
    path = opendata_file(RELATIVE)
    assert path == tmp_path / "work" / "data" / "raw" / "opendata" / RELATIVE
    assert path.read_bytes() == PAYLOAD
    (remote / RELATIVE).unlink()
    assert opendata_file(RELATIVE) == path  # second call needs no remote
    assert load_match_metadata(7)["pitch_width"] == 68


def test_a_clone_next_to_the_working_directory_wins(remote: Path, tmp_path: Path) -> None:
    clone = tmp_path / "opendata" / RELATIVE
    clone.parent.mkdir(parents=True)
    clone.write_bytes(b"{}")
    assert opendata_file(RELATIVE) == clone


def test_download_that_does_not_match_the_manifest_fails(remote: Path) -> None:
    (remote / RELATIVE).write_bytes(PAYLOAD.replace(b"105", b"999"))
    with pytest.raises(DownloadError, match="does not match the manifest"):
        opendata_file(RELATIVE)


def test_explicit_clone_is_never_filled_by_a_download(
    remote: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        opendata_file(RELATIVE, root=tmp_path / "elsewhere")
    monkeypatch.setenv(OPENDATA_DIR_ENV, str(tmp_path / "elsewhere"))
    with pytest.raises(FileNotFoundError, match=OPENDATA_DIR_ENV):
        opendata_file(RELATIVE)
    assert sample_pose_path().parent == tmp_path / "elsewhere" / "data" / "bodypose"


def test_file_missing_from_the_manifest_is_reported(remote: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not in the opendata manifest"):
        opendata_file("data/matches/8/8_match.json")
