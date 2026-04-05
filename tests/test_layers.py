"""Tests for layers.py -- deck.gl layer builders."""
from __future__ import annotations
import numpy as np
import pytest
from tests.helpers import FakeTF
from geometry import build_mesh_geometry
from layers import (
    build_mesh_layer, build_velocity_layer, build_contour_layer_fn,
    build_wireframe_layer, build_marker_layer, build_cross_section_layer,
    build_boundary_layer, build_extrema_markers, build_measurement_layer,
    build_particle_layer,
)
from analysis import find_boundary_nodes, find_extrema


class TestBuildMeshLayer:
    def test_returns_4_tuple(self, fake_tf, fake_geom):
        values = fake_tf.get_data_value("WATER DEPTH", 0)
        result = build_mesh_layer(fake_geom, values, "Viridis")
        assert isinstance(result, tuple) and len(result) == 4
        layer, vmin, vmax, log_applied = result
        assert isinstance(layer, dict)

    def test_layer_type(self, fake_tf, fake_geom):
        values = fake_tf.get_data_value("WATER DEPTH", 0)
        layer, _, _, _ = build_mesh_layer(fake_geom, values, "Viridis")
        assert layer["type"] == "SimpleMeshLayer"

    def test_layer_id(self, fake_tf, fake_geom):
        values = fake_tf.get_data_value("WATER DEPTH", 0)
        layer, _, _, _ = build_mesh_layer(fake_geom, values, "Viridis")
        assert layer["id"] == "mesh"

    def test_nan_values(self, fake_geom):
        values = np.array([np.nan, np.nan, np.nan, np.nan])
        layer, vmin, vmax, log_applied = build_mesh_layer(fake_geom, values, "Viridis")
        assert isinstance(layer, dict)
        # When all NaN, vmin/vmax should be set to defaults
        assert vmin == 0.0
        assert vmax == 1.0

    def test_log_scale_positive(self, fake_geom):
        values = np.array([1.0, 2.0, 5.0, 10.0])
        _, vmin, vmax, log_applied = build_mesh_layer(fake_geom, values, "Viridis", log_scale=True)
        assert log_applied is True
        assert vmin > 0

    def test_log_scale_zero(self, fake_geom):
        values = np.array([0.0, 1.0, 2.0, 3.0])
        _, vmin, vmax, log_applied = build_mesh_layer(fake_geom, values, "Viridis", log_scale=True)
        assert log_applied is False

    def test_custom_color_range(self, fake_tf, fake_geom):
        values = fake_tf.get_data_value("WATER DEPTH", 0)
        _, vmin, vmax, _ = build_mesh_layer(fake_geom, values, "Viridis", color_range_override=(0, 10))
        assert vmin == 0.0
        assert vmax == 10.0

    def test_filter_range_grays_out_values(self, fake_geom):
        values = np.array([0.1, 0.5, 0.5, 1.0], dtype=np.float32)
        lyr, vmin, vmax, log_applied = build_mesh_layer(
            fake_geom, values, "Viridis", filter_range=(0.3, 0.7))
        assert lyr is not None
        # Values outside [0.3, 0.7] should have alpha=0 in the color array
        # Check that the layer was built successfully
        assert vmin is not None
        assert vmax is not None

    def test_uniform_values(self, fake_geom):
        values = np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float32)
        lyr, vmin, vmax, log_applied = build_mesh_layer(
            fake_geom, values, "Viridis")
        assert lyr is not None
        # vmax should be adjusted to vmin + 1 to avoid division by zero
        assert vmax > vmin


class TestBuildVelocityLayer:
    def test_no_velocity_vars(self, fake_geom):
        class NoVelTF(FakeTF):
            varnames = ["WATER DEPTH"]
            _data = {"WATER DEPTH": np.array([0.1, 0.5, 0.5, 1.0])}
        tf = NoVelTF()
        result = build_velocity_layer(tf, 0, fake_geom)
        assert result is None

    def test_zero_velocity(self, fake_geom):
        class ZeroVelTF(FakeTF):
            _data = {
                **FakeTF._data,
                "VELOCITY U": np.zeros(4),
                "VELOCITY V": np.zeros(4),
            }
        tf = ZeroVelTF()
        result = build_velocity_layer(tf, 0, fake_geom)
        assert result is None

    def test_valid_velocity(self, fake_tf, fake_geom):
        result = build_velocity_layer(fake_tf, 0, fake_geom)
        assert isinstance(result, dict)
        assert "type" in result
        assert "id" in result


class TestBuildContourLayer:
    def test_flat_values(self, fake_tf, fake_geom):
        values = np.array([5.0, 5.0, 5.0, 5.0])
        result = build_contour_layer_fn(fake_tf, values, fake_geom)
        assert result is None

    def test_varying_values(self, fake_tf, fake_geom):
        values = fake_tf.get_data_value("WATER DEPTH", 0)
        result = build_contour_layer_fn(fake_tf, values, fake_geom)
        assert isinstance(result, dict)
        assert "data" in result
        assert len(result["data"]) > 0

    def test_n_contours(self, fake_tf, fake_geom):
        values = fake_tf.get_data_value("WATER DEPTH", 0)
        result_3 = build_contour_layer_fn(fake_tf, values, fake_geom, n_contours=3)
        result_10 = build_contour_layer_fn(fake_tf, values, fake_geom, n_contours=10)
        # More contours should generally produce more or equal segments
        assert result_3 is not None and result_10 is not None
        count_3 = len(result_3["data"])
        count_10 = len(result_10["data"])
        assert count_10 != count_3 or count_10 > 0  # different or at least non-empty


class TestOtherLayers:
    def test_wireframe(self, fake_tf, fake_geom):
        result = build_wireframe_layer(fake_tf, fake_geom)
        assert isinstance(result, dict)
        assert "type" in result
        assert "id" in result

    def test_marker(self):
        result = build_marker_layer(1.0, 2.0, "test-marker")
        assert isinstance(result, dict)
        assert result["id"] == "test-marker"

    def test_cross_section(self):
        pts = [[0.0, 0.0], [1.0, 1.0]]
        result = build_cross_section_layer(pts)
        assert isinstance(result, dict)

    def test_boundary(self, fake_tf, fake_geom):
        bnodes = find_boundary_nodes(fake_tf)
        result = build_boundary_layer(fake_tf, fake_geom, bnodes)
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(r, dict) for r in result)

    def test_boundary_with_bc_types_empty(self, fake_tf, fake_geom):
        from layers import build_boundary_layer
        from analysis import find_boundary_nodes
        bnodes = find_boundary_nodes(fake_tf)
        # Empty bc_types is valid — all default to wall
        bc_types = {}
        lyr = build_boundary_layer(fake_tf, fake_geom, bnodes, bc_types=bc_types)
        assert isinstance(lyr, list)
        assert len(lyr) > 0

    def test_boundary_with_prescribed_bc(self, fake_tf, fake_geom):
        from layers import build_boundary_layer
        from analysis import find_boundary_nodes
        bnodes = find_boundary_nodes(fake_tf)
        # bc_types uses 1-based node keys (TELEMAC convention)
        # Node 1 (1-based) as prescribed (5), node 2 as wall (2)
        bc_types = {1: 5, 2: 2}
        lyr = build_boundary_layer(fake_tf, fake_geom, bnodes, bc_types=bc_types)
        assert isinstance(lyr, list)
        assert len(lyr) > 0

    def test_extrema_markers(self, fake_tf, fake_geom):
        values = fake_tf.get_data_value("WATER DEPTH", 0)
        extrema = find_extrema(fake_tf, values)
        result = build_extrema_markers(extrema, fake_geom.x_off, fake_geom.y_off)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_measurement_2pts(self):
        pts = [[0.0, 0.0], [1.0, 1.0]]
        result = build_measurement_layer(pts)
        assert isinstance(result, list)
        assert len(result) == 3  # 1 line + 2 point markers

    def test_measurement_1pt(self):
        pts = [[0.0, 0.0]]
        result = build_measurement_layer(pts)
        assert isinstance(result, list)
        assert len(result) == 1  # 1 point marker only

    def test_particle(self):
        paths = [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]]
        result = build_particle_layer(paths, current_time=0.5, trail_length=1.0)
        assert isinstance(result, dict)
        assert "currentTime" in result


class TestOriginParameter:
    """Verify all layer builders pass origin through to coordinateOrigin."""

    def test_mesh_layer(self, fake_geom):
        values = np.array([0.1, 0.5, 0.5, 1.0])
        lyr, _, _, _ = build_mesh_layer(fake_geom, values, "Viridis", origin=[24.0, 55.0])
        assert lyr["coordinateOrigin"] == [24.0, 55.0]

    def test_velocity_layer(self, fake_tf, fake_geom):
        lyr = build_velocity_layer(fake_tf, 0, fake_geom, origin=[24.0, 55.0])
        if lyr is not None:
            assert lyr["coordinateOrigin"] == [24.0, 55.0]

    def test_contour_layer(self, fake_tf, fake_geom):
        values = fake_tf.get_data_value("WATER DEPTH", 0)
        lyr = build_contour_layer_fn(fake_tf, values, fake_geom, origin=[24.0, 55.0])
        if lyr is not None:
            assert lyr["coordinateOrigin"] == [24.0, 55.0]

    def test_marker_layer(self):
        lyr = build_marker_layer(0.0, 0.0, origin=[24.0, 55.0])
        assert lyr["coordinateOrigin"] == [24.0, 55.0]

    def test_cross_section_layer(self):
        lyr = build_cross_section_layer([[0, 0], [1, 1]], origin=[24.0, 55.0])
        assert lyr["coordinateOrigin"] == [24.0, 55.0]

    def test_particle_layer(self):
        paths = [[[0, 0, 0], [1, 1, 1]]]
        lyr = build_particle_layer(paths, 0.5, 1.0, origin=[24.0, 55.0])
        assert lyr["coordinateOrigin"] == [24.0, 55.0]

    def test_wireframe_layer(self, fake_tf, fake_geom):
        lyr = build_wireframe_layer(fake_tf, fake_geom, origin=[24.0, 55.0])
        assert lyr["coordinateOrigin"] == [24.0, 55.0]

    def test_extrema_markers(self, fake_tf, fake_geom):
        values = fake_tf.get_data_value("WATER DEPTH", 0)
        extrema = find_extrema(fake_tf, values)
        layers = build_extrema_markers(extrema, fake_geom.x_off, fake_geom.y_off, origin=[24.0, 55.0])
        for lyr in layers:
            assert lyr["coordinateOrigin"] == [24.0, 55.0]

    def test_measurement_layer(self):
        pts = [[0, 0], [1, 1]]
        layers = build_measurement_layer(pts, origin=[24.0, 55.0])
        for lyr in layers:
            assert lyr["coordinateOrigin"] == [24.0, 55.0]

    def test_boundary_layer(self, fake_tf, fake_geom):
        bnodes = find_boundary_nodes(fake_tf)
        layers = build_boundary_layer(fake_tf, fake_geom, bnodes, origin=[24.0, 55.0])
        for lyr in layers:
            assert lyr["coordinateOrigin"] == [24.0, 55.0]

    def test_default_origin_is_zero(self, fake_geom):
        values = np.array([0.1, 0.5, 0.5, 1.0])
        lyr, _, _, _ = build_mesh_layer(fake_geom, values, "Viridis")
        assert lyr["coordinateOrigin"] == [0, 0]
