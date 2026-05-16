"""Shared fixtures for TELEMAC Viewer tests."""

from __future__ import annotations
import pytest
from geometry import build_mesh_geometry
from tests.helpers import FakeTF, FakeSession


@pytest.fixture
def fake_tf():
    """Return a FakeTF instance."""
    return FakeTF()


@pytest.fixture
def fake_geom(fake_tf):
    """Return mesh geometry dict from FakeTF."""
    return build_mesh_geometry(fake_tf)


@pytest.fixture
def fake_session():
    """A fresh FakeSession for each test."""
    return FakeSession()


@pytest.fixture
def isolated_telemac_dirs(tmp_path, monkeypatch):
    """Redirect both library roots AND home dir to tmp_path; reset model_library
    module caches. Required for any test that touches model_library — without
    it, tests that forget to set the env vars fall back to the developer's
    ~/.telemac-viewer/ and clobber real state.
    """
    monkeypatch.setenv("TELEMAC_VIEWER_MODELS", str(tmp_path / "shared_models"))
    monkeypatch.setenv("TELEMAC_VIEWER_USERS_ROOT", str(tmp_path / "users"))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    import model_library

    model_library._reset_for_testing()
    yield tmp_path
    model_library._reset_for_testing()
