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
