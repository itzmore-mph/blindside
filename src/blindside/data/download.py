"""Resumable, checksum-verified download of the body pose archives.

Usage: ``python -m blindside.data.download`` fetches every archive in the manifest
into ``data/raw/bodypose`` and verifies it. Files that are already verified are skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from blindside import __version__
from blindside.data.manifest import pose_archive, pose_match_ids
from blindside.data.paths import pose_raw_dir

ProgressCallback = Callable[[int, int], None]

_USER_AGENT = f"blindside/{__version__}"
_RETRYABLE = (urllib.error.URLError, http.client.HTTPException, TimeoutError, ConnectionError)


class DownloadError(RuntimeError):
    """A download failed, or the downloaded file does not match the manifest."""


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    """Hex sha256 digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _marker(dest: Path) -> Path:
    return dest.with_name(dest.name + ".sha256")


def _is_verified(dest: Path, size_bytes: int, sha256: str) -> bool:
    """True if ``dest`` is complete and matches the hash. Records a marker to skip rehashing."""
    if not dest.exists() or dest.stat().st_size != size_bytes:
        return False
    marker = _marker(dest)
    if marker.exists() and marker.read_text().strip() == sha256:
        return True
    if sha256_of(dest) == sha256:
        marker.write_text(sha256 + "\n")
        return True
    return False


def _fetch_once(
    url: str,
    part: Path,
    size_bytes: int,
    timeout: float,
    chunk_size: int,
    progress: ProgressCallback | None,
) -> None:
    """Append to ``part`` from its current size, or restart if the server ignores Range."""
    offset = part.stat().st_size if part.exists() else 0
    if offset > size_bytes:
        part.unlink()
        offset = 0
    headers = {"User-Agent": _USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        resumed = offset > 0 and response.status == 206
        done = offset if resumed else 0
        with part.open("ab" if resumed else "wb") as out:
            while chunk := response.read(chunk_size):
                out.write(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(done, size_bytes)


def download_file(
    url: str,
    dest: Path,
    *,
    size_bytes: int,
    sha256: str,
    max_retries: int = 5,
    backoff_s: float = 2.0,
    timeout: float = 60.0,
    chunk_size: int = 1 << 20,
    progress: ProgressCallback | None = None,
) -> Path:
    """Download ``url`` to ``dest`` and verify size and sha256.

    Data goes to ``<dest>.part`` first. An interrupted download resumes from the
    partial file with an HTTP Range request, also across separate calls.

    Parameters
    ----------
    url, dest
        Source and target. Parent directories are created.
    size_bytes, sha256
        Expected size and hex digest of the complete file.
    max_retries, backoff_s
        Attempts on network failure and the base of the exponential wait between them.
    progress
        Optional callback receiving ``(bytes_done, bytes_total)``.

    Raises
    ------
    DownloadError
        On an HTTP error, after the last failed attempt, or on a size or sha256 mismatch.
    """
    dest = Path(dest)
    if _is_verified(dest, size_bytes, sha256):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.unlink(missing_ok=True)
    _marker(dest).unlink(missing_ok=True)
    part = dest.with_name(dest.name + ".part")

    last_error: Exception | None = None
    for attempt in range(max_retries):
        if attempt:
            time.sleep(backoff_s * 2 ** (attempt - 1))
        try:
            _fetch_once(url, part, size_bytes, timeout, chunk_size, progress)
        except urllib.error.HTTPError as error:
            if error.code == 416:  # partial file is not a prefix of the remote file
                part.unlink(missing_ok=True)
                last_error = error
                continue
            if error.code < 500:
                raise DownloadError(f"HTTP {error.code} for {url}: {error.reason}") from error
            last_error = error
            continue
        except _RETRYABLE as error:
            last_error = error
            continue
        if part.stat().st_size >= size_bytes:
            break
        last_error = ConnectionError("connection closed before the file was complete")
    else:
        have = part.stat().st_size if part.exists() else 0
        raise DownloadError(
            f"Download of {url} failed after {max_retries} attempts ({last_error}). "
            f"{have} of {size_bytes} bytes are kept in {part.name}; run again to resume."
        ) from last_error

    actual_size = part.stat().st_size
    actual_sha256 = sha256_of(part)
    if actual_size != size_bytes or actual_sha256 != sha256:
        part.unlink()
        raise DownloadError(
            f"Downloaded file does not match the manifest for {url}: "
            f"size {actual_size} (expected {size_bytes}), "
            f"sha256 {actual_sha256} (expected {sha256}). The partial file was removed."
        )
    part.replace(dest)
    _marker(dest).write_text(sha256 + "\n")
    return dest


def download_pose_archive(
    match_id: int,
    dest_dir: Path | None = None,
    *,
    progress: ProgressCallback | None = None,
    **kwargs: float,
) -> Path:
    """Download and verify the pose archive of one match.

    Parameters
    ----------
    match_id
        A match with body pose, see :func:`blindside.data.manifest.pose_match_ids`.
    dest_dir
        Target directory, default ``data/raw/bodypose``.

    Returns
    -------
    Path
        The verified ``{match_id}.jsonl.zip``.
    """
    archive = pose_archive(match_id)
    dest = Path(dest_dir or pose_raw_dir()) / Path(archive.file).name
    return download_file(
        archive.url,
        dest,
        size_bytes=archive.size_bytes,
        sha256=archive.sha256,
        progress=progress,
        **kwargs,
    )


def _print_progress(label: str) -> ProgressCallback:
    state = {"next": 0.0}

    def report(done: int, total: int) -> None:
        share = done / total
        if share >= state["next"] or done == total:
            print(f"{label}: {done / 1e6:7.1f} / {total / 1e6:.1f} MB ({share:4.0%})", flush=True)
            state["next"] = share + 0.1

    return report


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description="Download and verify the body pose archives.")
    parser.add_argument("--match", type=int, action="append", help="match id, repeatable")
    parser.add_argument("--dest-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    for match_id in args.match or pose_match_ids():
        try:
            path = download_pose_archive(
                match_id, args.dest_dir, progress=_print_progress(str(match_id))
            )
        except (DownloadError, ValueError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        print(f"{match_id}: verified {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
