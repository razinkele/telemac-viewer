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
