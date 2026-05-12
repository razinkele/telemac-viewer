"""Tests for server_import helpers."""

from __future__ import annotations

from pathlib import Path

from server_import import _DOWNLOAD_CHUNK_SIZE, _stream_file_chunks


def test_stream_file_chunks_yields_multiple_chunks_for_large_file(
    tmp_path: Path,
) -> None:
    """A file bigger than one chunk must not be read in one slurp.

    Regression: prior @render.download handlers used `yield f.read()`,
    which OOMs on multi-GB outputs from HEC-RAS imports.
    """
    path = tmp_path / "big.bin"
    payload = b"x" * (_DOWNLOAD_CHUNK_SIZE * 4 + 1234)
    path.write_bytes(payload)

    chunks = list(_stream_file_chunks(str(path)))

    assert len(chunks) >= 4, (
        f"Expected at least 4 chunks for a {len(payload)}-byte file at "
        f"chunk_size={_DOWNLOAD_CHUNK_SIZE}; got {len(chunks)}"
    )
    assert max(len(c) for c in chunks) == _DOWNLOAD_CHUNK_SIZE, (
        "No single chunk should exceed the configured chunk size — "
        "if it does, the read() loop is bypassed somehow."
    )
    assert b"".join(chunks) == payload


def test_stream_file_chunks_handles_small_file(tmp_path: Path) -> None:
    """A file smaller than one chunk yields a single non-empty chunk."""
    path = tmp_path / "small.bin"
    payload = b"hello world"
    path.write_bytes(payload)

    chunks = list(_stream_file_chunks(str(path)))

    assert chunks == [payload]


def test_stream_file_chunks_handles_empty_file(tmp_path: Path) -> None:
    """An empty file yields nothing (no zero-byte chunks)."""
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")

    assert list(_stream_file_chunks(str(path))) == []
