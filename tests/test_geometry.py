"""Tests for geometry.py -- mesh geometry building."""
from __future__ import annotations
import base64
import numpy as np
import pytest
from geometry import build_mesh_geometry
from crs import crs_from_epsg
from tests.helpers import FakeTF
from viewer_types import MeshGeometry


def _decode_binary(d):
    """Decode shiny-deckgl binary-encoded attribute to numpy array."""
    raw = base64.b64decode(d["value"])
    return np.frombuffer(raw, dtype=d["dtype"])


class TestBuildMeshGeometry2D:
    def test_returns_mesh_geometry(self, fake_tf):
        geom = build_mesh_geometry(fake_tf)
        assert isinstance(geom, MeshGeometry)
        for attr in ("npoin", "positions", "indices", "x_off", "y_off", "extent_m", "zoom"):
            assert hasattr(geom, attr)

    def test_npoin_matches(self, fake_tf):
        geom = build_mesh_geometry(fake_tf)
        assert geom.npoin == 4

    def test_positions_length(self, fake_tf):
        geom = build_mesh_geometry(fake_tf)
        pos = _decode_binary(geom.positions)
        assert len(pos) == 4 * 3

    def test_indices_length(self, fake_tf):
        geom = build_mesh_geometry(fake_tf)
        idx = _decode_binary(geom.indices)
        assert len(idx) == 2 * 3

    def test_center_offsets(self, fake_tf):
        geom = build_mesh_geometry(fake_tf)
        assert geom.x_off == pytest.approx(0.5)
        assert geom.y_off == pytest.approx(0.5)

    def test_positions_are_centered(self, fake_tf):
        geom = build_mesh_geometry(fake_tf)
        pos = _decode_binary(geom.positions)
        # Node 0: (0-0.5, 0-0.5, 0) = (-0.5, -0.5, 0)
        assert pos[0] == pytest.approx(-0.5)
        assert pos[1] == pytest.approx(-0.5)
        assert pos[2] == pytest.approx(0.0)

    def test_zoom_is_positive(self, fake_tf):
        geom = build_mesh_geometry(fake_tf)
        assert geom.zoom > 0

    def test_extent_m(self, fake_tf):
        geom = build_mesh_geometry(fake_tf)
        assert geom.extent_m == pytest.approx(1.0)


class TestBuildMeshGeometry3D:
    def test_z_values_applied(self, fake_tf):
        z = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        geom = build_mesh_geometry(fake_tf, z_values=z, z_scale=10)
        pos = _decode_binary(geom.positions)
        # Node 0 z = 0*10=0, Node 3 z = 3*10=30
        assert pos[2] == pytest.approx(0.0)
        assert pos[11] == pytest.approx(30.0)  # node3: index 3*3+2=11

    def test_z_none_is_flat(self, fake_tf):
        geom = build_mesh_geometry(fake_tf, z_values=None)
        pos = _decode_binary(geom.positions)
        z_vals = [pos[i * 3 + 2] for i in range(4)]
        assert all(z == 0.0 for z in z_vals)


class TestBuildMeshGeometryCRS:
    def test_no_crs_defaults_zero(self, fake_tf):
        geom = build_mesh_geometry(fake_tf)
        assert geom.lon_off == 0.0
        assert geom.lat_off == 0.0
        assert geom.crs is None

    def test_with_crs_converts_center(self):
        """LKS94 mesh center should map to real lon/lat."""
        tf = FakeTF()
        tf.meshx = np.array([490000, 510000, 490000, 510000], dtype=np.float64)
        tf.meshy = np.array([6090000, 6090000, 6110000, 6110000], dtype=np.float64)
        crs = crs_from_epsg(3346)
        geom = build_mesh_geometry(tf, crs=crs)
        assert 23.0 < geom.lon_off < 25.0
        assert 54.5 < geom.lat_off < 56.0
        assert geom.crs is crs
        # x_off/y_off still in native meters
        assert geom.x_off == pytest.approx(500000)
        assert geom.y_off == pytest.approx(6100000)

    def test_existing_tests_unchanged(self, fake_tf):
        """crs=None path produces same results as before."""
        geom = build_mesh_geometry(fake_tf, crs=None)
        assert geom.x_off == pytest.approx(0.5)
        assert geom.y_off == pytest.approx(0.5)
        assert geom.npoin == 4

    def test_iparam_offsets_applied(self):
        """SELAFIN I_ORIG/J_ORIG offsets should shift x_off/y_off."""
        tf = FakeTF()
        tf.iparam = [0, 0, 1000, 2000, 0, 0, 0, 0, 0, 0]
        geom = build_mesh_geometry(tf)
        assert geom.x_off == pytest.approx(0.5 + 1000)
        assert geom.y_off == pytest.approx(0.5 + 2000)
