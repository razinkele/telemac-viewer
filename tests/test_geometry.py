"""Tests for geometry.py -- mesh geometry building."""
from __future__ import annotations
import numpy as np
import pytest
from geometry import build_mesh_geometry


class TestBuildMeshGeometry2D:
    def test_returns_required_keys(self, fake_tf):
        geom = build_mesh_geometry(fake_tf)
        for key in ("npoin", "positions", "indices", "x_off", "y_off", "extent_m", "zoom"):
            assert key in geom

    def test_npoin_matches(self, fake_tf):
        geom = build_mesh_geometry(fake_tf)
        assert geom["npoin"] == 4

    def test_positions_length(self, fake_tf):
        geom = build_mesh_geometry(fake_tf)
        assert len(geom["positions"]) == 4 * 3

    def test_indices_length(self, fake_tf):
        geom = build_mesh_geometry(fake_tf)
        assert len(geom["indices"]) == 2 * 3

    def test_center_offsets(self, fake_tf):
        geom = build_mesh_geometry(fake_tf)
        assert geom["x_off"] == pytest.approx(0.5)
        assert geom["y_off"] == pytest.approx(0.5)

    def test_positions_are_centered(self, fake_tf):
        geom = build_mesh_geometry(fake_tf)
        pos = geom["positions"]
        # Node 0: (0-0.5, 0-0.5, 0) = (-0.5, -0.5, 0)
        assert pos[0] == pytest.approx(-0.5)
        assert pos[1] == pytest.approx(-0.5)
        assert pos[2] == pytest.approx(0.0)

    def test_zoom_is_positive(self, fake_tf):
        geom = build_mesh_geometry(fake_tf)
        assert geom["zoom"] > 0

    def test_extent_m(self, fake_tf):
        geom = build_mesh_geometry(fake_tf)
        assert geom["extent_m"] == pytest.approx(1.0)


class TestBuildMeshGeometry3D:
    def test_z_values_applied(self, fake_tf):
        z = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        geom = build_mesh_geometry(fake_tf, z_values=z, z_scale=10)
        pos = geom["positions"]
        # Node 0 z = 0*10=0, Node 3 z = 3*10=30
        assert pos[2] == pytest.approx(0.0)
        assert pos[11] == pytest.approx(30.0)  # node3: index 3*3+2=11

    def test_z_none_is_flat(self, fake_tf):
        geom = build_mesh_geometry(fake_tf, z_values=None)
        pos = geom["positions"]
        z_vals = [pos[i * 3 + 2] for i in range(4)]
        assert all(z == 0.0 for z in z_vals)
