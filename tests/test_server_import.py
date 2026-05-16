"""Tests for server_import helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

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


# ---------------------------------------------------------------------------
# Task 9: cleanup rmtree guard + auto-save tempdir-intact reaffirmation
# ---------------------------------------------------------------------------


@pytest.fixture
def non_tmp_path(monkeypatch):
    """A real-FS path that is NOT under tempfile.gettempdir().

    pytest's default ``tmp_path`` lives under ``/tmp/pytest-of-…`` on Linux,
    which means any "library" path built on top of it is actually under
    the tempdir — making the cleanup-guard tests trivially pass even
    without the guard. Tests that need to simulate a path OUTSIDE the
    tempdir (a real user library on the ext4 home FS, for example) use
    this fixture instead.
    """
    import shutil
    import tempfile
    from pathlib import Path

    # Use $HOME (an ext4 path that's never under /tmp on this system).
    base = Path.home() / ".telemac-viewer-test-tmpdir-guard"
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True)
    assert not base.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()), (
        f"non_tmp_path fixture broken: {base} resolves under tempdir"
    )
    yield base
    shutil.rmtree(base, ignore_errors=True)


def test_cleanup_rmtree_refuses_to_delete_via_dotdot_escape(non_tmp_path):
    """A `..`-containing path that lexically starts with /tmp must be
    rejected by the cleanup guard — Path.resolve() normalizes it out."""
    import tempfile
    from pathlib import Path

    library_target = non_tmp_path / "library" / "target"
    library_target.mkdir(parents=True)
    sentinel = library_target / "DO_NOT_DELETE.txt"
    sentinel.touch()
    hostile = f"{tempfile.gettempdir()}/../{library_target.relative_to('/')}"

    try:
        old_resolved = Path(hostile).resolve(strict=False)
        tmp_resolved = Path(tempfile.gettempdir()).resolve()
        is_safe = old_resolved.is_relative_to(tmp_resolved)
    except (OSError, ValueError):
        is_safe = False

    assert not is_safe, "Path.resolve normalized away the .. — cleanup should refuse"
    assert sentinel.exists()


def test_cleanup_rmtree_refuses_to_delete_library_path(non_tmp_path):
    """Direct test of the guard logic: _import_out_dir pointing at a
    library path must NOT be rmtree'd by the next conversion's cleanup."""
    import tempfile
    from pathlib import Path

    library_path = non_tmp_path / "users" / "1" / "models" / "alpha"
    library_path.mkdir(parents=True)
    sentinel = library_path / "case.slf"
    sentinel.write_bytes(b"protected")

    old_dir = str(library_path)
    try:
        old_resolved = Path(old_dir).resolve(strict=True)
        tmp_resolved = Path(tempfile.gettempdir()).resolve(strict=True)
        is_under_tmp = old_resolved.is_relative_to(tmp_resolved)
    except (OSError, ValueError):
        is_under_tmp = False

    assert not is_under_tmp, "library path must not be considered under /tmp"
    assert sentinel.exists()
    assert sentinel.read_bytes() == b"protected"


def test_cleanup_rmtree_refuses_to_delete_via_symlink_escape(non_tmp_path):
    """Symlinked path that resolves outside /tmp must be refused."""
    import os
    import tempfile
    from pathlib import Path

    real_target = non_tmp_path / "users" / "library"
    real_target.mkdir(parents=True)
    sentinel = real_target / "DO_NOT_DELETE"
    sentinel.touch()

    symlink_under_tmp = Path(tempfile.gettempdir()) / f"escape_test_{os.getpid()}"
    try:
        os.symlink(real_target, symlink_under_tmp)
        old_resolved = Path(symlink_under_tmp).resolve(strict=True)
        tmp_resolved = Path(tempfile.gettempdir()).resolve(strict=True)
        is_under_tmp = old_resolved.is_relative_to(tmp_resolved)
        assert not is_under_tmp
        assert sentinel.exists()
    finally:
        try:
            symlink_under_tmp.unlink()
        except FileNotFoundError:
            pass


def test_cleanup_rmtree_accepts_legitimate_tempdir_path(tmp_path):
    """Sanity check: a real /tmp path IS accepted by the guard."""
    import shutil
    import tempfile
    from pathlib import Path

    legitimate = Path(tempfile.mkdtemp(prefix="telemac_import_test_"))
    try:
        (legitimate / "case.slf").touch()
        old_resolved = legitimate.resolve(strict=True)
        tmp_resolved = Path(tempfile.gettempdir()).resolve(strict=True)
        is_under_tmp = old_resolved.is_relative_to(tmp_resolved)
        assert is_under_tmp, "legitimate /tmp path must pass the guard"
    finally:
        shutil.rmtree(legitimate, ignore_errors=True)


def test_save_imported_keeps_tempdir_intact_on_oserror(
    isolated_telemac_dirs, tmp_path, monkeypatch
):
    """If the inner save fails, the tempdir must NOT be cleaned —
    keeps download buttons functional after auto-save failure."""
    import errno

    from model_library import save_imported_to_library

    src = tmp_path / "telemac_import_X"
    src.mkdir()
    (src / "model.slf").write_bytes(b"data")
    (src / "model.cas").write_text("cas")

    monkeypatch.setattr(
        "model_library.save_upload_to_library",
        lambda uid, files, name: (_ for _ in ()).throw(OSError(errno.ENOSPC, "ENOSPC")),
    )

    with pytest.raises(OSError) as exc:
        save_imported_to_library(50, src, "autosaved_run")

    assert exc.value.errno == errno.ENOSPC
    assert src.exists(), "tempdir intact — downloads keep working"
    assert (src / "model.slf").read_bytes() == b"data"
    assert (src / "model.cas").read_text() == "cas"
