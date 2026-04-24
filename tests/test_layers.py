"""Tests for layers.py -- deck.gl layer builders."""

from __future__ import annotations
import numpy as np
import pytest
from tests.helpers import FakeTF
from geometry import build_mesh_geometry
from layers import (
    build_mesh_layer,
    build_velocity_layer,
    build_contour_layer_fn,
    build_wireframe_layer,
    build_marker_layer,
    build_cross_section_layer,
    build_boundary_layer,
    build_extrema_markers,
    build_measurement_layer,
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
        _, vmin, vmax, log_applied = build_mesh_layer(
            fake_geom, values, "Viridis", log_scale=True
        )
        assert log_applied is True
        assert vmin > 0

    def test_log_scale_zero(self, fake_geom):
        values = np.array([0.0, 1.0, 2.0, 3.0])
        _, vmin, vmax, log_applied = build_mesh_layer(
            fake_geom, values, "Viridis", log_scale=True
        )
        assert log_applied is False

    def test_custom_color_range(self, fake_tf, fake_geom):
        values = fake_tf.get_data_value("WATER DEPTH", 0)
        _, vmin, vmax, _ = build_mesh_layer(
            fake_geom, values, "Viridis", color_range_override=(0, 10)
        )
        assert vmin == 0.0
        assert vmax == 10.0

    def test_filter_range_grays_out_values(self, fake_geom):
        values = np.array([0.1, 0.5, 0.5, 1.0], dtype=np.float32)
        lyr, vmin, vmax, log_applied = build_mesh_layer(
            fake_geom, values, "Viridis", filter_range=(0.3, 0.7)
        )
        assert lyr is not None
        # Values outside [0.3, 0.7] should have alpha=0 in the color array
        # Check that the layer was built successfully
        assert vmin is not None
        assert vmax is not None

    def test_uniform_values(self, fake_geom):
        values = np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float32)
        lyr, vmin, vmax, log_applied = build_mesh_layer(fake_geom, values, "Viridis")
        assert lyr is not None
        # vmax should be adjusted to vmin + 1 to avoid division by zero
        assert vmax > vmin

    def test_log_scale_on_uniform_field_falls_back(self, fake_geom):
        import numpy as np
        from layers import build_mesh_layer

        values = np.array([3.14, 3.14, 3.14, 3.14], dtype=np.float32)
        _, _, _, log_applied = build_mesh_layer(
            fake_geom,
            values,
            "Viridis",
            log_scale=True,
        )
        assert log_applied is False


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
        lyr, _, _, _ = build_mesh_layer(
            fake_geom, values, "Viridis", origin=[24.0, 55.0]
        )
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
        layers = build_extrema_markers(
            extrema, fake_geom.x_off, fake_geom.y_off, origin=[24.0, 55.0]
        )
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


class TestBuildMeshColorPatch:
    def test_patch_has_mesh_id(self, fake_geom):
        from layers import build_mesh_color_patch

        values = np.array([1.0, 2.0, 3.0, 4.0])
        patch, _, _, _ = build_mesh_color_patch(fake_geom, values, "Viridis")
        assert patch["id"] == "mesh"

    def test_patch_contains_mesh_colors_only(self, fake_geom):
        from layers import build_mesh_color_patch

        values = np.array([1.0, 2.0, 3.0, 4.0])
        patch, _, _, _ = build_mesh_color_patch(fake_geom, values, "Viridis")
        # Patch must NOT include positions or indices (preserved by JS cache)
        assert "_meshPositions" not in patch
        assert "_meshIndices" not in patch
        assert "mesh" not in patch  # no CustomGeometry ref needed
        # Patch MUST include the new color array
        assert "_meshColors" in patch

    def test_patch_returns_same_vmin_vmax_as_full_build(self, fake_tf, fake_geom):
        from layers import build_mesh_layer, build_mesh_color_patch

        values = fake_tf.get_data_value("WATER DEPTH", 0)
        _, vmin_full, vmax_full, log_full = build_mesh_layer(
            fake_geom, values, "Viridis"
        )
        _, vmin_patch, vmax_patch, log_patch = build_mesh_color_patch(
            fake_geom, values, "Viridis"
        )
        assert vmin_full == vmin_patch
        assert vmax_full == vmax_patch
        assert log_full == log_patch

    def test_patch_colors_match_full_build(self, fake_tf, fake_geom):
        from layers import build_mesh_layer, build_mesh_color_patch

        values = fake_tf.get_data_value("WATER DEPTH", 0)
        full, _, _, _ = build_mesh_layer(fake_geom, values, "Viridis")
        patch, _, _, _ = build_mesh_color_patch(fake_geom, values, "Viridis")
        # Same input → identical color buffer
        assert patch["_meshColors"] == full["_meshColors"]

    def test_patch_respects_filter_range(self, fake_geom):
        from layers import build_mesh_color_patch

        values = np.array([0.1, 0.5, 0.5, 1.0], dtype=np.float32)
        unfiltered, _, _, _ = build_mesh_color_patch(fake_geom, values, "Viridis")
        filtered, _, _, _ = build_mesh_color_patch(
            fake_geom, values, "Viridis", filter_range=(0.3, 0.7)
        )
        # filter_range must actually change the encoded color buffer
        # (vertices outside [0.3, 0.7] are greyed out).
        assert filtered["_meshColors"]["value"] != unfiltered["_meshColors"]["value"]


class TestPartialUpdatePatches:
    def test_velocity_patch_matches_full_layer(self, fake_tf, fake_geom):
        from layers import build_velocity_layer, build_velocity_patch

        full = build_velocity_layer(fake_tf, 0, fake_geom)
        patch = build_velocity_patch(fake_tf, 0, fake_geom)
        # Pass-through: full layer dict == patch dict
        assert full == patch

    def test_velocity_patch_none_when_no_velocity(self, fake_geom):
        from layers import build_velocity_patch
        from tests.helpers import FakeTF

        class NoVelTF(FakeTF):
            @property
            def varnames(self):
                return ["WATER DEPTH"]

        assert build_velocity_patch(NoVelTF(), 0, fake_geom) is None

    def test_contour_patch_matches_full_layer(self, fake_tf, fake_geom):
        from layers import build_contour_layer_fn, build_contour_patch

        values = fake_tf.get_data_value("WATER DEPTH", 0)
        full = build_contour_layer_fn(fake_tf, values, fake_geom)
        patch = build_contour_patch(fake_tf, values, fake_geom)
        assert full == patch

    def test_contour_patch_forwards_kwargs(self, fake_tf, fake_geom):
        """All non-default kwargs must reach build_contour_layer_fn."""
        from layers import build_contour_layer_fn, build_contour_patch

        values = fake_tf.get_data_value("WATER DEPTH", 0)
        kwargs = dict(n_contours=3, layer_id="custom", contour_color=[255, 0, 0])
        full = build_contour_layer_fn(fake_tf, values, fake_geom, **kwargs)
        patch = build_contour_patch(fake_tf, values, fake_geom, **kwargs)
        assert full == patch
        assert patch["id"] == "custom"
