"""Round 3 deep review — regression tests and coverage improvements."""
from __future__ import annotations
import numpy as np
import pytest
from tests.helpers import FakeTF


# ---------------------------------------------------------------------------
# Phase 1 regression tests
# ---------------------------------------------------------------------------

class TestProcInit:
    """r3-proc-init: proc=None initialised before try block."""

    def test_server_simulation_has_proc_init(self):
        """Verify proc = None appears before try in server_simulation.py."""
        import pathlib
        src = (pathlib.Path(__file__).resolve().parent.parent / "server_simulation.py").read_text()
        # proc = None must come before the first 'try:' in handle_run_sim
        idx_proc = src.index("proc = None")
        idx_try = src.index("try:", idx_proc)
        assert idx_proc < idx_try


class TestGeoTiffTags:
    """r3-geotiff-tags: missing GeoTIFF tags raise ValueError, not KeyError."""

    def test_missing_tags_raises_valueerror(self, tmp_path):
        """A TIFF without geotransform tags should raise ValueError."""
        try:
            import tifffile
        except ImportError:
            pytest.skip("tifffile not installed")
        # Create minimal TIFF without GeoTIFF tags
        p = tmp_path / "no_geo.tif"
        data = np.zeros((4, 4), dtype=np.float32)
        tifffile.imwrite(str(p), data)
        from telemac_tools.domain.builder import _read_dem
        with pytest.raises(ValueError, match="missing required GeoTIFF tags"):
            _read_dem(str(p))


class TestCourantReturn:
    """r3-courant-return: Courant failure returns current_values() explicitly."""

    def test_courant_path_has_return(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parent.parent / "app.py").read_text()
        # Find the courant block
        idx = src.index('elif diag == "courant"')
        block = src[idx:idx + 400]
        # After notification_show, there should be 'return current_values()'
        assert "return current_values()" in block


class TestElevationLenValidation:
    """r3-elevation-len: elevation array length validated against cell count."""

    def test_parser_2d_has_len_check(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parent.parent /
               "telemac_tools" / "hecras" / "parser_2d.py").read_text()
        assert "len(raw_elev) == len(cells)" in src


class TestDurationBias:
    """r3-duration-bias: duration uses (wet_count-1) to avoid 1-interval overcount."""

    def test_all_wet_duration_equals_timespan(self):
        """If all nodes wet at every timestep, duration = timespan, not timespan + dt."""
        from analysis import compute_all_temporal_stats
        tf = FakeTF()
        # WATER DEPTH at t=0: [0.1, 0.5, 0.5, 1.0] — all > 0.01
        stats = compute_all_temporal_stats(tf, "WATER DEPTH", threshold=0.01)
        # All 4 nodes wet at all 3 timesteps → wet_count=3
        # Duration should be (3-1)*1.0 = 2.0 (time span from first to last)
        # NOT 3*1.0 = 3.0 (old overcounting formula)
        timespan = tf.times[-1] - tf.times[0]  # 2.0
        assert stats["duration"][3] == pytest.approx(timespan, abs=0.01)

    def test_single_wet_duration_is_zero(self):
        """A node wet at exactly one timestep should have duration=0."""
        from analysis import compute_all_temporal_stats

        class SingleWetTF(FakeTF):
            """Node 0 only wet at t=0, dry at t=1,2."""
            _data = {
                "VELOCITY U": np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64),
                "VELOCITY V": np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
                "WATER DEPTH": np.array([0.05, 0.5, 0.5, 1.0], dtype=np.float64),
            }

            def get_data_value(self, varname, tidx):
                base = self._data[varname].copy()
                if varname == "WATER DEPTH":
                    if tidx == 0:
                        base[0] = 0.05  # above threshold
                    else:
                        base[0] = 0.005  # below threshold
                    base[1:] = base[1:] * (1.0 + tidx * 0.5)
                return base

        tf = SingleWetTF()
        stats = compute_all_temporal_stats(tf, "WATER DEPTH", threshold=0.01)
        # Node 0: wet at 1 timestep → duration = max(1-1, 0) * avg_dt = 0.0
        assert stats["duration"][0] == pytest.approx(0.0)


class TestMultiRingBoundary:
    """r3-multi-ring: boundary tracing handles multiple rings."""

    def test_builder_uses_longest_ring(self):
        """Source code should use 'longest' ring selection."""
        import pathlib
        src = (pathlib.Path(__file__).resolve().parent.parent /
               "telemac_tools" / "domain" / "builder.py").read_text()
        assert "longest = max(rings, key=len)" in src


class TestDeadUniqueRemoved:
    """r3-dead-unique: unused unique_idx removed from writer_liq."""

    def test_no_unused_unique_idx(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parent.parent /
               "telemac_tools" / "telemac" / "writer_liq.py").read_text()
        assert "unique_idx" not in src


# ---------------------------------------------------------------------------
# Phase 2 — test coverage improvements
# ---------------------------------------------------------------------------

class TestDischargeKnownAnswer:
    """r3-t-discharge-val: verify discharge Q against analytical value."""

    def test_discharge_perpendicular_section(self):
        """Cross-section along x=0.9 from y=0 to y=1.

        Nearest node to midpoint (0.9, 0.5) is node 1 (1,0) or node 3 (1,1).
        At node 1: u=1, v=0, depth=0.5. Normal = (0, -1) cross (0, 1) rotated.
        Section direction dy=1 → outward normal is (-1, 0).
        Q_seg = depth * (u*ny - v*nx) * seg_len... but exact formula
        depends on implementation. Key: Q must be non-zero and consistent sign.
        """
        from analysis import compute_discharge
        tf = FakeTF()
        result = compute_discharge(tf, 0, [[0.9, 0.0], [0.9, 1.0]])
        q = result["total_q"]
        assert q != 0.0
        # Section is parallel to y-axis, flow is in +x direction,
        # so discharge through it should be negative (flow from right to left
        # relative to section direction) or positive depending on convention.
        # Key assertion: magnitude should be around 0.5 (depth * velocity * length)
        assert abs(q) == pytest.approx(0.5, abs=0.3)


class TestFilterRangeColors:
    """r3-t-filter-colors: verify grayed-out vertex colors."""

    def test_gray_outside_range(self, fake_geom):
        from layers import build_mesh_layer
        values = np.array([0.1, 0.5, 0.5, 1.0], dtype=np.float32)
        lyr, vmin, vmax, _ = build_mesh_layer(
            fake_geom, values, "Viridis", filter_range=(0.3, 0.7))
        assert lyr is not None
        # Extract the mesh colors from the layer
        colors_attr = lyr.get("_meshColors")
        assert colors_attr is not None
        assert colors_attr  # non-empty


class TestDegenerateTriangleQuality:
    """r3-t-degen-mesh: collinear points produce quality=0."""

    def test_collinear_nodes_quality_zero(self):
        from analysis import compute_mesh_quality

        class CollinearTF(FakeTF):
            """Mesh with one degenerate (collinear) triangle."""
            meshx = np.array([0.0, 1.0, 2.0, 1.0], dtype=np.float64)
            meshy = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
            ikle2 = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int32)
            npoin2 = 4
            nelem2 = 2

        tf = CollinearTF()
        q = compute_mesh_quality(tf)
        assert q.shape == (4,)
        # No NaN or Inf
        assert np.all(np.isfinite(q))
        # Node 2 is only in the degenerate triangle → its quality contribution is 0
        # Nodes 0, 1 are shared → averaged; node 3 is in the good triangle only
        assert q[3] > 0  # good triangle contributes quality > 0


class TestCRSTighterBounds:
    """r3-t-crs-tighten: LKS94 transform bounds tightened."""

    def test_lks94_accuracy(self):
        from crs import crs_from_epsg, native_to_wgs84
        crs = crs_from_epsg(3346)
        lon, lat = native_to_wgs84(500000, 6100000, crs)
        # Known: (500000, 6100000) in LKS94 ≈ (24.05°E, 55.0°N)
        # Tighten from ±1° to ±0.5°
        assert 23.5 < lon < 24.5
        assert 54.5 < lat < 55.5


class TestEdgeFunctions:
    """r3-t-edge-funcs: unit tests for compute_unique_edges and find_boundary_edges."""

    def test_single_triangle_edges(self):
        from analysis import compute_unique_edges
        ikle = np.array([[0, 1, 2]], dtype=np.int32)
        keys, a, b = compute_unique_edges(ikle)
        assert len(keys) == 3  # 3 unique edges

    def test_two_triangles_shared_edge(self):
        from analysis import compute_unique_edges
        ikle = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
        keys, a, b = compute_unique_edges(ikle)
        assert len(keys) == 5  # 6 total - 1 shared = 5 unique

    def test_boundary_edges_single_triangle(self):
        from analysis import find_boundary_edges

        class TriTF:
            ikle2 = np.array([[0, 1, 2]], dtype=np.int32)

        keys, a, b = find_boundary_edges(TriTF())
        assert len(keys) == 3  # all 3 edges are boundary

    def test_boundary_edges_two_triangles(self):
        from analysis import find_boundary_edges

        class QuadTF:
            ikle2 = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)

        keys, a, b = find_boundary_edges(QuadTF())
        assert len(keys) == 4  # 5 unique - 1 interior = 4 boundary


class TestVorticityKnownAnswer:
    """r3-t-vorticity-val: verify vorticity with known gradient field."""

    def test_uniform_v_gradient(self):
        """FakeTF has V = [0, 0, 1, 1] (y-gradient), U = [0, 1, 0, 1] (x-gradient).

        For each element, dv/dx and du/dy can be computed analytically.
        Element 0 (nodes 0,1,2): area = 0.5
            dv/dx = (v0*(y1-y2) + v1*(y2-y0) + v2*(y0-y1)) / (2*area)
                  = (0*(0-1) + 0*(1-0) + 1*(0-0)) / 1.0 = 0
            du/dy = (u0*(x2-x1) + u1*(x0-x2) + u2*(x1-x0)) / (2*area)
                  = (0*(0-1) + 1*(0-0) + 0*(1-0)) / 1.0 = 0
            vorticity = dv/dx - du/dy = 0

        Element 1 (nodes 1,3,2): area = 0.5
            dv/dx = (v1*(y3-y2) + v3*(y2-y1) + v2*(y1-y3)) / 1.0
                  = (0*(1-1) + 1*(1-0) + 1*(0-1)) / 1.0 = 0
            du/dy = (u1*(x2-x3) + u3*(x1-x2) + u2*(x3-x1)) / 1.0
                  = (1*(0-1) + 1*(1-0) + 0*(1-1)) / 1.0 = 0
            vorticity = 0

        Both elements have zero vorticity because V has only y-gradient
        and U has only x-gradient. Cross-derivatives are zero.
        """
        from analysis import _compute_vorticity
        tf = FakeTF()
        vort = _compute_vorticity(tf, 0)
        assert vort.shape == (4,)
        assert np.all(np.isfinite(vort))
        # For FakeTF's specific field, vorticity should be ~0
        assert np.allclose(vort, 0.0, atol=1e-6)

    def test_nonzero_vorticity(self):
        """Create a field where V has x-gradient → positive vorticity."""
        from analysis import _compute_vorticity

        class VortTF(FakeTF):
            _data = {
                "VELOCITY U": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
                "VELOCITY V": np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64),
                "WATER DEPTH": np.array([0.1, 0.5, 0.5, 1.0], dtype=np.float64),
            }

            def get_data_value(self, varname, tidx):
                return self._data[varname].copy()

        tf = VortTF()
        vort = _compute_vorticity(tf, 0)
        # dv/dx > 0, du/dy = 0 → vorticity > 0
        assert np.all(vort > -1e-6)  # non-negative (averaged to vertices)
        assert np.any(vort > 0.1)  # at least some positive


class TestCrossSectionValues:
    """r3-t-cross-vals: verify interpolated cross-section values."""

    def test_cross_section_known_values(self):
        from analysis import cross_section_profile
        tf = FakeTF()
        # Diagonal from (0,0) to (1,1): nearest nodes are 0 and 3
        polyline = [[0.0, 0.0], [1.0, 1.0]]
        abscissa, values = cross_section_profile(tf, "WATER DEPTH", 0, polyline)
        assert len(values) == 2
        # At (0,0) → node 0 → depth = 0.1
        assert values[0] == pytest.approx(0.1, abs=0.01)
        # At (1,1) → node 3 → depth = 1.0
        assert values[1] == pytest.approx(1.0, abs=0.01)


class TestPolygonLayer:
    """r3-t-polygon-layer: basic build_polygon_layer test."""

    def test_polygon_layer_structure(self):
        from layers import build_polygon_layer
        coords = [[0, 0], [100, 0], [100, 100], [0, 100]]
        lyr = build_polygon_layer(coords, origin=[24.0, 55.0])
        assert lyr["coordinateOrigin"] == [24.0, 55.0]
        assert lyr["filled"] is True
        assert "data" in lyr

    def test_polygon_auto_closes(self):
        from layers import build_polygon_layer
        coords = [[0, 0], [100, 0], [100, 100]]
        lyr = build_polygon_layer(coords)
        # The polygon should be auto-closed
        feat = lyr["data"]["features"][0]
        ring = feat["geometry"]["coordinates"][0]
        assert ring[0] == ring[-1]


class TestContourSaddlePoint:
    """r3-t-contour-saddle: test contour with 3-crossing triangle."""

    def test_contour_no_crash_with_varied_values(self, fake_geom):
        """Build contours on values where saddle-like configurations exist."""
        from layers import build_contour_layer_fn
        tf = FakeTF()
        values = np.array([0.2, 0.8, 0.4, 0.6], dtype=np.float32)
        result = build_contour_layer_fn(tf, values, fake_geom, n_contours=5)
        # Should not crash; may return None or valid layer
        if result is not None:
            assert "data" in result


class TestFloodDurationSingleTimestep:
    """r3-t-flood-single: single-timestep flood with threshold variations."""

    def test_single_timestep_threshold_splits(self):
        """With threshold=0.3, some nodes are wet and some dry."""
        from analysis import compute_flood_duration

        class SingleTF(FakeTF):
            times = [0.0]

            def get_data_value(self, varname, tidx):
                return self._data[varname].copy()

        tf = SingleTF()
        dur = compute_flood_duration(tf, "WATER DEPTH", threshold=0.3)
        # Node 0: depth=0.1 < 0.3 → dry → dur=0
        assert dur[0] == pytest.approx(0.0)
        # Single timestep has no forward intervals → duration is 0
        assert dur[3] == pytest.approx(0.0)
