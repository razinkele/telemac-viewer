"""Shared fixtures for TELEMAC Viewer tests."""
from __future__ import annotations
import pytest
from geometry import build_mesh_geometry
from tests.helpers import FakeTF


@pytest.fixture
def fake_tf():
    """Return a FakeTF instance."""
    return FakeTF()


@pytest.fixture
def fake_geom(fake_tf):
    """Return mesh geometry dict from FakeTF."""
    return build_mesh_geometry(fake_tf)
