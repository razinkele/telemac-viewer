"""Tests for the local model library module."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


class TestLibraryRoot:
    def test_default_when_env_unset(self, monkeypatch, tmp_path):
        from model_library import library_root, _reset_for_testing

        _reset_for_testing()
        monkeypatch.delenv("TELEMAC_VIEWER_MODELS", raising=False)
        # Redirect Path.home() so the test doesn't auto-create a real
        # ~/.telemac-viewer/models on the developer's machine.
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        result = library_root()
        assert result == (tmp_path / ".telemac-viewer" / "models").resolve()

    def test_respects_env_override(self, monkeypatch, tmp_path):
        from model_library import library_root, _reset_for_testing

        _reset_for_testing()
        target = tmp_path / "custom-lib"
        monkeypatch.setenv("TELEMAC_VIEWER_MODELS", str(target))
        result = library_root()
        assert result == target.resolve()

    def test_auto_creates_missing(self, monkeypatch, tmp_path):
        from model_library import library_root, _reset_for_testing

        _reset_for_testing()
        target = tmp_path / "fresh-lib"
        monkeypatch.setenv("TELEMAC_VIEWER_MODELS", str(target))
        assert not target.exists()
        library_root()
        assert target.is_dir()

    def test_refuses_path_inside_viewer_tree(self, monkeypatch, capsys):
        from model_library import library_root, _reset_for_testing
        import model_library

        _reset_for_testing()
        viewer_tree = Path(model_library.__file__).resolve().parent
        monkeypatch.setenv("TELEMAC_VIEWER_MODELS", str(viewer_tree / "subdir"))
        result = library_root()
        # The path-safety guard makes scan_library treat this as empty;
        # library_root itself returns the resolved path but logs a warning.
        captured = capsys.readouterr()
        assert "viewer source tree" in captured.err.lower()


class TestScanLibrary:
    def test_empty_root(self, tmp_path):
        from model_library import scan_library

        assert scan_library(tmp_path) == []

    def test_skips_non_directories(self, tmp_path):
        from model_library import scan_library

        (tmp_path / "notes.txt").write_text("ignore me")
        (tmp_path / "loose.slf").write_bytes(b"")
        assert scan_library(tmp_path) == []

    def test_skips_hidden_dirs(self, tmp_path):
        from model_library import scan_library

        hidden = tmp_path / ".staging"
        hidden.mkdir()
        (hidden / "a.slf").write_bytes(b"")
        assert scan_library(tmp_path) == []

    def test_skips_projects_without_slf(self, tmp_path):
        from model_library import scan_library

        proj = tmp_path / "no-slf-here"
        proj.mkdir()
        (proj / "readme.md").write_text("wip")
        assert scan_library(tmp_path) == []

    def test_finds_multiple_slf_per_project(self, tmp_path):
        from model_library import scan_library

        proj = tmp_path / "curonian"
        proj.mkdir()
        (proj / "results.slf").write_bytes(b"")
        (proj / "restart.slf").write_bytes(b"")
        entries = scan_library(tmp_path)
        assert len(entries) == 1
        assert entries[0].name == "curonian"
        assert len(entries[0].slf_files) == 2
        assert [p.name for p in entries[0].slf_files] == ["restart.slf", "results.slf"]

    def test_sorts_alphabetically(self, tmp_path):
        from model_library import scan_library

        for name in ("zebra", "alpha", "mango"):
            proj = tmp_path / name
            proj.mkdir()
            (proj / "r.slf").write_bytes(b"")
        names = [e.name for e in scan_library(tmp_path)]
        assert names == ["alpha", "mango", "zebra"]


class TestResolveProject:
    def _make_entry(self, root, *, slf="results.slf"):
        from model_library import ProjectEntry, scan_library

        proj = root / "proj"
        proj.mkdir()
        (proj / slf).write_bytes(b"")
        return scan_library(root)[0]

    def test_basename_match_companions(self, tmp_path):
        from model_library import resolve_project

        entry = self._make_entry(tmp_path)
        proj = entry.path
        (proj / "results.cas").write_text("X")
        (proj / "results.cli").write_text("X")
        (proj / "results.liq").write_text("X")
        # Decoy companions that should be ignored when basename match wins:
        (proj / "other.cas").write_text("X")
        files = resolve_project(entry, "results.slf")
        assert files.slf == proj / "results.slf"
        assert files.cas == proj / "results.cas"
        assert files.cli == proj / "results.cli"
        assert files.liq == proj / "results.liq"

    def test_single_companion_fallback(self, tmp_path):
        from model_library import resolve_project

        entry = self._make_entry(tmp_path)
        proj = entry.path
        (proj / "boundary.cli").write_text("X")  # no basename match
        files = resolve_project(entry, "results.slf")
        assert files.cli == proj / "boundary.cli"
        assert files.cas is None
        assert files.liq is None

    def test_no_match_returns_none(self, tmp_path):
        from model_library import resolve_project

        entry = self._make_entry(tmp_path)
        proj = entry.path
        (proj / "inflow.liq").write_text("X")
        (proj / "outflow.liq").write_text("X")  # ambiguous, no match
        files = resolve_project(entry, "results.slf")
        assert files.liq is None

    def test_raises_when_slf_missing(self, tmp_path):
        from model_library import resolve_project

        entry = self._make_entry(tmp_path)
        with pytest.raises(FileNotFoundError):
            resolve_project(entry, "ghost.slf")

    def test_unknown_extension_ignored(self, tmp_path):
        from model_library import resolve_project

        entry = self._make_entry(tmp_path)
        proj = entry.path
        (proj / "results.txt").write_text("X")  # not a TELEMAC companion
        files = resolve_project(entry, "results.slf")
        assert (files.cas, files.cli, files.liq) == (None, None, None)


class TestFindCompanion:
    def test_returns_none_when_selection_is_none(self, tmp_path):
        from model_library import find_companion

        assert find_companion(None, tmp_path, ".cas") is None

    def test_returns_path_when_basename_matches(self, tmp_path):
        from model_library import find_companion

        proj = tmp_path / "curonian"
        proj.mkdir()
        (proj / "results.slf").write_bytes(b"")
        (proj / "results.cas").write_text("X")
        result = find_companion(("curonian", "results.slf"), tmp_path, ".cas")
        assert result == proj / "results.cas"

    def test_returns_none_when_project_has_no_slf(self, tmp_path):
        """A project folder with companions but no .slf is filtered out by
        scan_library, so find_companion never reaches it.
        """
        from model_library import find_companion

        proj = tmp_path / "curonian"
        proj.mkdir()
        (proj / "results.cas").write_text("X")
        result = find_companion(("curonian", "results.slf"), tmp_path, ".cas")
        assert result is None

    def test_swallows_filenotfound_when_slf_deleted_after_scan(
        self, tmp_path, monkeypatch
    ):
        """Race: scan_library returns an entry, but the .slf is gone by
        the time resolve_project reads it. Helper silently returns None.
        """
        import model_library
        from model_library import find_companion, ProjectEntry

        proj = tmp_path / "curonian"
        proj.mkdir()
        slf = proj / "results.slf"
        slf.write_bytes(b"")
        (proj / "results.cas").write_text("X")
        # Snapshot the entry, then delete the .slf to simulate a race.
        stale_entry = ProjectEntry(name="curonian", path=proj, slf_files=(slf,))
        slf.unlink()
        monkeypatch.setattr(model_library, "scan_library", lambda r: [stale_entry])

        result = find_companion(("curonian", "results.slf"), tmp_path, ".cas")
        assert result is None

    def test_returns_none_when_project_renamed(self, tmp_path):
        from model_library import find_companion

        proj = tmp_path / "actual-name"
        proj.mkdir()
        (proj / "results.slf").write_bytes(b"")
        (proj / "results.cas").write_text("X")
        result = find_companion(("old-name", "results.slf"), tmp_path, ".cas")
        assert result is None


# --- Foundation types and validators (v3.7.0 per-user-storage) ---

_VALID_PROJECT_NAMES = ["a", "Z", "1", "A_b-1", "abc-def_ghi", "a" * 64]
_INVALID_PROJECT_NAMES = [
    "",  # empty
    "a" * 65,  # too long
    ".hidden",  # leading dot
    "foo.bar",  # dot in middle
    ".",  # current dir
    "..",  # parent dir
    "foo/bar",  # slash
    "foo bar",  # space
    "föö",  # unicode
    "a\x00b",  # NUL byte
]


@pytest.mark.parametrize("name", _VALID_PROJECT_NAMES)
def test_validate_project_name_accepts_safe_names(name):
    from model_library import _validate_project_name

    assert _validate_project_name(name) == name


@pytest.mark.parametrize("name", _INVALID_PROJECT_NAMES)
def test_validate_project_name_rejects_unsafe_names(name):
    from model_library import _validate_project_name

    with pytest.raises(ValueError):
        _validate_project_name(name)


def test_validate_companion_basename_strips_path_components():
    from model_library import _validate_companion_basename

    assert _validate_companion_basename("../../etc/passwd.slf") == "passwd.slf"
    assert _validate_companion_basename("subdir/case.cas") == "case.cas"


def test_validate_companion_basename_rejects_disallowed_extension():
    from model_library import _validate_companion_basename

    with pytest.raises(ValueError, match="disallowed extension"):
        _validate_companion_basename("case.exe")


def test_validate_companion_basename_accepts_case_insensitive_suffix():
    from model_library import _validate_companion_basename

    assert _validate_companion_basename("Case.SLF") == "Case.SLF"


def test_sanitize_for_project_name_passes_clean_input():
    from model_library import _sanitize_for_project_name

    assert _sanitize_for_project_name("alpha_run_1") == "alpha_run_1"


def test_sanitize_for_project_name_replaces_unsafe_chars():
    from model_library import _sanitize_for_project_name

    out = _sanitize_for_project_name("alpha run!1")
    assert out == "alpha_run_1"


def test_sanitize_for_project_name_falls_back_on_empty():
    from model_library import _sanitize_for_project_name

    out = _sanitize_for_project_name("???")
    assert out.startswith("hecras_import_")
    assert len(out) <= 64


def test_sanitize_for_project_name_falls_back_on_bare_prefix():
    from model_library import _sanitize_for_project_name

    out = _sanitize_for_project_name("hecras")
    assert out.startswith("hecras_import_")


def test_validate_user_id_rejects_bool_subtype():
    from model_library import _validate_user_id

    with pytest.raises(TypeError):
        _validate_user_id(True)
    with pytest.raises(TypeError):
        _validate_user_id(False)


def test_validate_user_id_rejects_non_positive():
    from model_library import _validate_user_id

    with pytest.raises(ValueError):
        _validate_user_id(0)
    with pytest.raises(ValueError):
        _validate_user_id(-1)


def test_validate_user_id_rejects_huge():
    from model_library import _validate_user_id

    with pytest.raises(ValueError):
        _validate_user_id(2**63)


def test_validate_user_id_accepts_positive_int():
    from model_library import _validate_user_id

    _validate_user_id(1)
    _validate_user_id(42)
    _validate_user_id(2**63 - 1)


def test_library_usage_size_human():
    from model_library import LibraryUsage

    assert LibraryUsage(0, 0).size_human == "0 B"
    assert LibraryUsage(0, 1023).size_human == "1023 B"
    assert LibraryUsage(0, 1024).size_human == "1.0 kB"
    assert LibraryUsage(0, 5 * (1 << 20)).size_human == "5.0 MB"
    assert LibraryUsage(0, 3 * (1 << 30)).size_human == "3.0 GB"


def test_library_source_display_label():
    from model_library import LibrarySource

    assert LibrarySource.USER.display_label == "My models"
    assert LibrarySource.SHARED.display_label == "Shared"


# --- user_library_root + _sweep_stale_partials (v3.7.0 per-user-storage) ---


def test_user_library_root_creates_0o700_under_default_umask(
    isolated_telemac_dirs, monkeypatch
):
    import os
    from model_library import user_library_root

    old = os.umask(0o022)
    try:
        root = user_library_root(7)
    finally:
        os.umask(old)

    assert root.exists() and root.is_dir()
    assert oct(root.stat().st_mode & 0o777) == "0o700"
    assert oct(root.parent.stat().st_mode & 0o777) == "0o700"


def test_user_library_root_refuses_viewer_tree_base(isolated_telemac_dirs, monkeypatch):
    import model_library

    viewer_tree = str(Path(model_library.__file__).resolve().parent)
    monkeypatch.setenv("TELEMAC_VIEWER_USERS_ROOT", viewer_tree)
    model_library._reset_for_testing()
    with pytest.raises(ValueError, match="inside the viewer source tree"):
        model_library.user_library_root(1)


def test_user_library_root_rejects_bool_subtype(isolated_telemac_dirs):
    from model_library import user_library_root

    with pytest.raises(TypeError):
        user_library_root(True)


def test_user_library_root_rejects_non_positive_uid(isolated_telemac_dirs):
    from model_library import user_library_root

    with pytest.raises(ValueError):
        user_library_root(0)
    with pytest.raises(ValueError):
        user_library_root(-1)


def test_sweep_stale_partials_does_not_remove_lock_file(isolated_telemac_dirs):
    import os
    from model_library import user_library_root, _sweep_stale_partials

    root = user_library_root(8)
    (root / ".lock").touch()
    (root / ".foo.partial-99999").mkdir()  # dead pid (unlikely to be running)
    _sweep_stale_partials(root, user_id=8)
    assert (root / ".lock").exists()
    assert not (root / ".foo.partial-99999").exists()


def test_sweep_stale_partials_bounded_at_max_per_startup(isolated_telemac_dirs):
    import model_library
    from model_library import (
        user_library_root,
        _sweep_stale_partials,
        _SWEEP_MAX_PER_STARTUP,
    )

    root = user_library_root(9)
    for i in range(_SWEEP_MAX_PER_STARTUP + 3):
        (root / f".bar{i}.partial-{99000 + i}").mkdir()
    # Reset module-level state so we can call _sweep_stale_partials directly
    # without going through user_library_root (which would also sweep and
    # consume budget, causing double-counting).
    model_library._reset_for_testing()
    removed = _sweep_stale_partials(root, user_id=9)
    # Strict equality verifies the cap IS the limit, not a coincidental
    # upper bound that would hold even if the mechanism were absent.
    assert removed == _SWEEP_MAX_PER_STARTUP
    leftover = [
        p for p in root.iterdir() if p.name.startswith(".bar") and ".partial-" in p.name
    ]
    assert len(leftover) == 3


def test_sweep_stale_partials_pid_alive_old_mtime_removed(
    isolated_telemac_dirs, monkeypatch
):
    import os
    import time
    import model_library
    from model_library import (
        user_library_root,
        _sweep_stale_partials,
        _STALE_MTIME_FALLBACK_SECONDS,
    )

    root = user_library_root(10)
    own_pid = os.getpid()
    partial = root / f".foo.partial-{own_pid}"
    partial.mkdir()
    old = time.time() - 60 * 60
    os.utime(partial, (old, old))
    # Reset the counter so this sweep starts with a fresh budget; then call
    # _sweep_stale_partials directly (avoid going through user_library_root,
    # which would also sweep and confuse the return-value contract check).
    model_library._reset_for_testing()
    removed = _sweep_stale_partials(root, user_id=10)
    assert not partial.exists()
    assert removed == 1


def test_sweep_stale_partials_permission_error_skips(
    isolated_telemac_dirs, monkeypatch
):
    import model_library
    from model_library import user_library_root, _sweep_stale_partials
    import shutil

    root = user_library_root(11)
    own_pid = os.getpid()
    partial = root / f".foo.partial-{own_pid}"
    partial.mkdir()
    monkeypatch.setattr(
        "model_library.os.kill",
        lambda pid, sig: (_ for _ in ()).throw(PermissionError("denied")),
    )
    called = {"rmtree": False}
    real_rmtree = shutil.rmtree

    def spy(*a, **kw):
        called["rmtree"] = True
        return real_rmtree(*a, **kw)

    monkeypatch.setattr("model_library.shutil.rmtree", spy)

    model_library._reset_for_testing()
    removed = _sweep_stale_partials(root, user_id=11)
    assert partial.exists()
    assert called["rmtree"] is False
    assert removed == 0


def test_project_files_iter_existing_yields_slf_and_non_none_companions(tmp_path):
    from model_library import ProjectFiles

    slf = tmp_path / "case.slf"
    cas = tmp_path / "case.cas"
    slf.touch()
    cas.touch()
    pf = ProjectFiles(slf=slf, cas=cas, cli=None, liq=None)
    files = list(pf.iter_existing())
    assert files == [slf, cas]


def test_project_entry_default_source_is_shared():
    from model_library import ProjectEntry, LibrarySource

    e = ProjectEntry(name="x", path=Path("/x"), slf_files=())
    assert e.source == LibrarySource.SHARED


def test_project_entry_can_be_constructed_with_user_source():
    from model_library import ProjectEntry, LibrarySource

    e = ProjectEntry(name="x", path=Path("/x"), slf_files=(), source=LibrarySource.USER)
    assert e.source == LibrarySource.USER


# --- Task 4: scan_library user_id merge + collision filter ---


def test_scan_library_merges_per_user_and_shared(isolated_telemac_dirs):
    from model_library import (
        user_library_root,
        library_root,
        scan_library,
        LibrarySource,
    )

    user_root = user_library_root(42)
    (user_root / "alpha").mkdir()
    (user_root / "alpha" / "case.slf").touch()
    shared = library_root()
    (shared / "beta").mkdir(parents=True)
    (shared / "beta" / "case.slf").touch()

    entries = scan_library(user_id=42)
    by_name = {e.name: e for e in entries}
    assert by_name["alpha"].source == LibrarySource.USER
    assert by_name["beta"].source == LibrarySource.SHARED


def test_scan_library_per_user_wins_collision_and_shadows_shared(
    isolated_telemac_dirs,
):
    from model_library import (
        user_library_root,
        library_root,
        scan_library,
        LibrarySource,
    )

    user_root = user_library_root(43)
    (user_root / "alpha").mkdir()
    (user_root / "alpha" / "u.slf").touch()
    shared = library_root()
    (shared / "alpha").mkdir(parents=True)
    (shared / "alpha" / "s.slf").touch()

    entries = scan_library(user_id=43)
    matching = [e for e in entries if e.name == "alpha"]
    assert len(matching) == 1, "shared duplicate must be omitted"
    assert matching[0].source == LibrarySource.USER
    assert matching[0].slf_files[0].name == "u.slf"


def test_scan_library_with_no_user_id_returns_shared_only(isolated_telemac_dirs):
    from model_library import library_root, scan_library, LibrarySource

    shared = library_root()
    (shared / "gamma").mkdir(parents=True)
    (shared / "gamma" / "case.slf").touch()

    entries = scan_library()
    assert all(e.source == LibrarySource.SHARED for e in entries)


def test_scan_library_corrupt_user_root_returns_sentinel(
    isolated_telemac_dirs, monkeypatch
):
    from model_library import LIBRARY_SENTINEL_NAME, scan_library, LibrarySource

    base = isolated_telemac_dirs / "users" / "44"
    base.mkdir(parents=True, exist_ok=True)
    (base / "models").write_text("not a dir")

    import model_library

    monkeypatch.setattr(model_library, "user_library_root", lambda uid: base / "models")

    entries = scan_library(user_id=44)
    sentinels = [e for e in entries if e.name == LIBRARY_SENTINEL_NAME]
    assert len(sentinels) == 1
    assert sentinels[0].name == LIBRARY_SENTINEL_NAME
    assert sentinels[0].source == LibrarySource.USER
    assert sentinels[0].slf_files == ()


def test_scan_library_ignores_leading_dot_dirs(isolated_telemac_dirs):
    from model_library import library_root, scan_library

    shared = library_root()
    (shared / ".hidden").mkdir(parents=True)
    (shared / ".hidden" / "case.slf").touch()
    entries = scan_library()
    assert all(not e.name.startswith(".") for e in entries)


# --- Task 5: save_upload_to_library ---


def test_save_upload_round_trip_with_companions(isolated_telemac_dirs, tmp_path):
    from model_library import (
        save_upload_to_library,
        ProjectFiles,
        user_library_root,
        LibrarySource,
    )

    src = tmp_path / "src"
    src.mkdir()
    slf = src / "case.slf"
    slf.write_bytes(b"x" * 100)
    cas = src / "case.cas"
    cas.write_text("steering")
    pf = ProjectFiles(slf=slf, cas=cas, cli=None, liq=None)

    saved = save_upload_to_library(7, pf, "alpha_run")
    assert saved.name == "alpha_run"
    assert (saved / "case.slf").read_bytes() == b"x" * 100
    assert (saved / "case.cas").read_text() == "steering"
    # Project dir 0o755, files 0o644
    assert oct(saved.stat().st_mode & 0o777) == "0o755"
    assert oct((saved / "case.slf").stat().st_mode & 0o777) == "0o644"


def test_save_upload_refuses_existing_name(isolated_telemac_dirs, tmp_path):
    from model_library import save_upload_to_library, ProjectFiles, user_library_root

    root = user_library_root(8)
    (root / "alpha_run").mkdir()
    slf = tmp_path / "case.slf"
    slf.touch()
    pf = ProjectFiles(slf=slf, cas=None, cli=None, liq=None)
    with pytest.raises(FileExistsError):
        save_upload_to_library(8, pf, "alpha_run")
    assert not list(root.glob(".alpha_run.partial-*"))


def test_save_upload_pre_stat_rejects_pre_staged_empty_dir(
    isolated_telemac_dirs, tmp_path
):
    """os.rename would silently replace an existing empty dir; the pre-stat
    refuses regardless of the target being empty or non-empty."""
    from model_library import save_upload_to_library, ProjectFiles, user_library_root

    root = user_library_root(9)
    (root / "alpha_run").mkdir()
    slf = tmp_path / "case.slf"
    slf.touch()
    pf = ProjectFiles(slf=slf, cas=None, cli=None, liq=None)
    with pytest.raises(FileExistsError):
        save_upload_to_library(9, pf, "alpha_run")


def test_save_upload_cleans_up_partial_on_copy_failure(
    isolated_telemac_dirs, tmp_path, monkeypatch
):
    import errno
    from model_library import save_upload_to_library, ProjectFiles, user_library_root

    root = user_library_root(10)
    slf = tmp_path / "case.slf"
    slf.touch()
    cas = tmp_path / "case.cas"
    cas.touch()
    pf = ProjectFiles(slf=slf, cas=cas, cli=None, liq=None)

    import shutil as _shutil_real

    real_copy = _shutil_real.copy
    call_count = {"n": 0}

    def flaky_copy(src, dst):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_copy(src, dst)

    monkeypatch.setattr("model_library.shutil.copy", flaky_copy)

    with pytest.raises(OSError) as exc_info:
        save_upload_to_library(10, pf, "alpha_run")
    assert exc_info.value.errno == errno.ENOSPC
    assert not list(root.glob(".alpha_run.partial-*"))
    assert not (root / "alpha_run").exists()


def test_save_upload_rejects_hostile_companion_basename(
    isolated_telemac_dirs, tmp_path
):
    """Defense-in-depth: _validate_companion_basename catches hostile chars
    in companion filenames before any FS write."""
    from model_library import save_upload_to_library, ProjectFiles, user_library_root

    user_library_root(11)
    slf = tmp_path / "case.slf"
    slf.touch()
    bad = tmp_path / "case\x00.cli"  # NUL byte in name
    pf = ProjectFiles(slf=slf, cas=None, cli=bad, liq=None)
    with pytest.raises(ValueError):
        save_upload_to_library(11, pf, "alpha_run")


def test_save_upload_rejects_duplicate_companion_basenames(
    isolated_telemac_dirs, tmp_path
):
    """Two companions resolving to the same basename are refused before
    any copy. ProjectFiles slots cas + cli pointed at files both named
    case.cas — by-suffix routing in _build_project_files would prevent
    this normally, but defense-in-depth at save_upload_to_library is
    tested here."""
    from model_library import save_upload_to_library, ProjectFiles, user_library_root

    user_library_root(12)
    slf = tmp_path / "case.slf"
    slf.touch()
    sub = tmp_path / "sub"
    sub.mkdir()
    cas_a = tmp_path / "case.cas"
    cas_a.touch()
    cas_b = sub / "case.cas"
    cas_b.touch()
    pf = ProjectFiles(slf=slf, cas=cas_a, cli=cas_b, liq=None)
    with pytest.raises(ValueError, match="Duplicate destination basenames"):
        save_upload_to_library(12, pf, "alpha_run")


def test_save_upload_concurrent_same_name_serializes(isolated_telemac_dirs, tmp_path):
    """flock serializes the pre-stat → rename window. One wins, the other
    sees the target existing and raises FileExistsError."""
    import threading
    from model_library import save_upload_to_library, ProjectFiles, user_library_root

    user_library_root(13)
    slf_a = tmp_path / "a.slf"
    slf_a.write_bytes(b"a")
    slf_b = tmp_path / "b.slf"
    slf_b.write_bytes(b"b")
    pf_a = ProjectFiles(slf=slf_a, cas=None, cli=None, liq=None)
    pf_b = ProjectFiles(slf=slf_b, cas=None, cli=None, liq=None)

    results = []
    barrier = threading.Barrier(2)

    def worker(pf):
        barrier.wait()
        try:
            results.append(("ok", save_upload_to_library(13, pf, "alpha")))
        except FileExistsError as e:
            results.append(("collision", e))

    t1 = threading.Thread(target=worker, args=(pf_a,))
    t2 = threading.Thread(target=worker, args=(pf_b,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    outcomes = sorted(r[0] for r in results)
    assert outcomes == ["collision", "ok"]

    # The winner's slf must survive intact — defends against a future
    # regression where both writers race past pre-stat and one clobbers
    # the other's renamed directory.
    from model_library import user_library_default_base

    final_dir = user_library_default_base() / "13" / "models" / "alpha"
    assert final_dir.is_dir()
    surviving = list(final_dir.glob("*.slf"))
    assert len(surviving) == 1
    assert surviving[0].read_bytes() in {b"a", b"b"}


def test_save_upload_lock_file_is_symlink_raises_oserror(
    isolated_telemac_dirs, tmp_path
):
    """O_NOFOLLOW on the lock open defends against pre-placed symlinks."""
    import os as _os
    from model_library import save_upload_to_library, ProjectFiles, user_library_root

    root = user_library_root(14)
    target = tmp_path / "decoy"
    target.touch()
    _os.symlink(target, root / ".lock")
    slf = tmp_path / "case.slf"
    slf.touch()
    pf = ProjectFiles(slf=slf, cas=None, cli=None, liq=None)
    with pytest.raises(OSError):
        save_upload_to_library(14, pf, "alpha_run")


def test_save_upload_partial_dir_invisible_to_scan_during_save(
    isolated_telemac_dirs, tmp_path
):
    """scan_library skips leading-dot entries, so a .name.partial-pid dir
    is never returned even mid-save."""
    from model_library import scan_library, user_library_root

    root = user_library_root(15)
    (root / ".alpha.partial-99999").mkdir()
    (root / ".alpha.partial-99999" / "case.slf").touch()
    entries = scan_library(user_id=15)
    assert all(not e.name.startswith(".") for e in entries)


# --- Task 6: save_imported_to_library + measure + delete ---


def test_save_imported_round_trip_and_cleans_tmpdir(isolated_telemac_dirs, tmp_path):
    from model_library import save_imported_to_library

    src = tmp_path / "imp"
    src.mkdir()
    (src / "model.slf").write_bytes(b"slf_data")
    (src / "model.cas").write_text("cas_data")
    (src / "model.cli").write_text("cli_data")

    saved = save_imported_to_library(20, src, "hecras_run")
    assert saved.name == "hecras_run"
    assert (saved / "model.slf").read_bytes() == b"slf_data"
    # tmpdir consumed
    assert not src.exists()


def test_save_imported_keeps_tmpdir_intact_on_failure(
    isolated_telemac_dirs, tmp_path, monkeypatch
):
    """If save_upload_to_library raises, the tmpdir is NOT rmtree'd —
    download buttons keep working. This is the load-bearing property
    that distinguishes copy-then-unlink from move."""
    from model_library import save_imported_to_library

    src = tmp_path / "imp"
    src.mkdir()
    (src / "model.slf").touch()

    monkeypatch.setattr(
        "model_library.save_upload_to_library",
        lambda uid, files, name: (_ for _ in ()).throw(OSError(28, "ENOSPC")),
    )

    with pytest.raises(OSError):
        save_imported_to_library(20, src, "hecras_run")
    # tmpdir still there, files intact
    assert src.exists()
    assert (src / "model.slf").exists()


def test_measure_user_library_empty_returns_zero(isolated_telemac_dirs):
    from model_library import measure_user_library, LibraryUsage

    assert measure_user_library(30) == LibraryUsage(0, 0)


def test_measure_user_library_walks_files_and_returns_dataclass(isolated_telemac_dirs):
    from model_library import measure_user_library, user_library_root

    root = user_library_root(31)
    proj = root / "alpha"
    proj.mkdir()
    (proj / "f1.slf").write_bytes(b"x" * 100)
    (proj / "f2.cas").write_bytes(b"y" * 50)
    usage = measure_user_library(31)
    assert usage.files == 2
    assert usage.size_bytes == 150


def test_measure_user_library_unreadable_subdir_skips_and_warns(isolated_telemac_dirs):
    import os
    from model_library import measure_user_library, user_library_root

    root = user_library_root(32)
    good = root / "alpha"
    good.mkdir()
    (good / "f.slf").write_bytes(b"x" * 10)
    bad = root / "beta"
    bad.mkdir()
    (bad / "f.slf").write_bytes(b"y" * 20)
    os.chmod(bad, 0o000)
    try:
        usage = measure_user_library(32)
    finally:
        os.chmod(bad, 0o755)
    assert usage.files >= 1
    assert usage.size_bytes >= 10


def test_delete_user_library_idempotent_on_missing_dir(isolated_telemac_dirs):
    from model_library import delete_user_library, LibraryUsage

    assert delete_user_library(99) == LibraryUsage(0, 0)


def test_delete_user_library_returns_pre_deletion_usage(isolated_telemac_dirs):
    from model_library import delete_user_library, user_library_root

    root = user_library_root(33)
    proj = root / "alpha"
    proj.mkdir()
    (proj / "f.slf").write_bytes(b"x" * 1234)
    usage = delete_user_library(33)
    assert usage.files == 1
    assert usage.size_bytes == 1234
    assert not root.parent.exists()
