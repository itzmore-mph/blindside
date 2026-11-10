"""Tests for the resumable, checksum-verified downloader against a local HTTP server."""

import hashlib
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from blindside.data.download import DownloadError, download_file, sha256_of

PAYLOAD = bytes(range(256)) * 2000  # 512 kB, not a multiple of the chunk size used below
SHA256 = hashlib.sha256(PAYLOAD).hexdigest()


class _Server(ThreadingHTTPServer):
    """Test server whose behaviour is switched through attributes."""

    support_range = True
    truncate_first_response_at: int | None = None
    status_override: int | None = None

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.range_headers: list[str | None] = []

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/file.zip"


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    def log_message(self, *args: object) -> None:  # keep test output clean
        pass

    def do_GET(self) -> None:
        server = self.server
        range_header = self.headers.get("Range")
        server.range_headers.append(range_header)
        if server.status_override is not None:
            self.send_error(server.status_override)
            return

        start = 0
        if range_header and server.support_range:
            start = int(range_header.removeprefix("bytes=").rstrip("-"))
            if start >= len(PAYLOAD):
                self.send_error(416)
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}")
        else:
            self.send_response(200)
        body = PAYLOAD[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()

        if server.truncate_first_response_at is not None and len(server.range_headers) == 1:
            self.wfile.write(body[: server.truncate_first_response_at])
            self.wfile.flush()
            self.close_connection = True
            self.connection.close()
            return
        self.wfile.write(body)


@pytest.fixture
def server() -> Iterator[_Server]:
    srv = _Server()
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


def _download(server: _Server, dest: Path, sha256: str = SHA256) -> Path:
    return download_file(
        server.url,
        dest,
        size_bytes=len(PAYLOAD),
        sha256=sha256,
        backoff_s=0.0,
        chunk_size=7_000,
    )


def test_fresh_download_is_verified(server: _Server, tmp_path: Path) -> None:
    dest = _download(server, tmp_path / "sub" / "file.zip")
    assert dest.read_bytes() == PAYLOAD
    assert sha256_of(dest) == SHA256
    assert not dest.with_name("file.zip.part").exists()
    assert server.range_headers == [None]


def test_partial_file_is_resumed_with_range_request(server: _Server, tmp_path: Path) -> None:
    dest = tmp_path / "file.zip"
    dest.with_name("file.zip.part").write_bytes(PAYLOAD[:100_000])
    _download(server, dest)
    assert dest.read_bytes() == PAYLOAD
    assert server.range_headers == ["bytes=100000-"]


def test_server_ignoring_range_restarts_from_zero(server: _Server, tmp_path: Path) -> None:
    server.support_range = False
    dest = tmp_path / "file.zip"
    dest.with_name("file.zip.part").write_bytes(PAYLOAD[:100_000])
    _download(server, dest)
    assert dest.read_bytes() == PAYLOAD


def test_dropped_connection_is_retried_and_resumed(server: _Server, tmp_path: Path) -> None:
    server.truncate_first_response_at = 200_000
    dest = _download(server, tmp_path / "file.zip")
    assert dest.read_bytes() == PAYLOAD
    assert len(server.range_headers) >= 2
    assert server.range_headers[0] is None
    assert server.range_headers[1] is not None


def test_checksum_mismatch_raises_and_leaves_nothing_behind(
    server: _Server, tmp_path: Path
) -> None:
    dest = tmp_path / "file.zip"
    with pytest.raises(DownloadError, match="sha256"):
        _download(server, dest, sha256="0" * 64)
    assert not dest.exists()
    assert not dest.with_name("file.zip.part").exists()


def test_verified_file_is_not_fetched_again(server: _Server, tmp_path: Path) -> None:
    dest = _download(server, tmp_path / "file.zip")
    _download(server, dest)
    assert len(server.range_headers) == 1


def test_existing_file_without_marker_is_rehashed_not_refetched(
    server: _Server, tmp_path: Path
) -> None:
    dest = tmp_path / "file.zip"
    dest.write_bytes(PAYLOAD)
    _download(server, dest)
    assert server.range_headers == []


def test_corrupt_existing_file_is_replaced(server: _Server, tmp_path: Path) -> None:
    dest = tmp_path / "file.zip"
    dest.write_bytes(b"x" * len(PAYLOAD))
    _download(server, dest)
    assert dest.read_bytes() == PAYLOAD


def test_http_error_gives_clear_message(server: _Server, tmp_path: Path) -> None:
    server.status_override = 404
    with pytest.raises(DownloadError, match="404"):
        _download(server, tmp_path / "file.zip")


def test_unreachable_host_raises_download_error(tmp_path: Path) -> None:
    with pytest.raises(DownloadError, match="attempts"):
        download_file(
            "http://127.0.0.1:9/file.zip",
            tmp_path / "file.zip",
            size_bytes=10,
            sha256=SHA256,
            backoff_s=0.0,
            max_retries=2,
            timeout=2,
        )
