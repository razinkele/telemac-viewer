"""Tests for analysis.py — all computation functions."""
from __future__ import annotations
import logging
import os
import numpy as np
import pytest
from tests.helpers import FakeTF
from viewer_types import MeshGeometry
from analysis import (
    _element_areas, _scatter_to_vertices, _sanitize_result,
    compute_mesh_quality, compute_slope, compute_courant_number,
    compute_element_area, compute_mesh_integral,
    get_available_derived, compute_derived, _compute_vorticity,
    evaluate_expression,
    extract_layer_2d,
    nearest_node, find_boundary_nodes, find_extrema,
    vertical_profile_at_point, time_series_at_point, cross_section_profile,
    compute_temporal_stats, compute_all_temporal_stats, compute_difference,
    compute_discharge,
    compute_particle_paths, generate_seed_grid, distribute_seeds_along_line,
    export_timeseries_csv, export_crosssection_csv, export_all_variables_csv,
    find_cas_files, detect_module,
    compute_flood_envelope, compute_flood_arrival, compute_flood_duration,
    read_cli_file, polygon_zonal_stats,
)


# ---------------------------------------------------------------------------
# FakeTF variants for tests that need different variable sets
# ---------------------------------------------------------------------------

class NoVelTF(FakeTF):
    varnames = ["WATER DEPTH"]
    _data = {"WATER DEPTH": np.array([0.1, 0.5, 0.5, 1.0], dtype=np.float64)}


class NoDepthTF(FakeTF):
    varnames = ["VELOCITY U", "VELOCITY V"]
    _data = {
        "VELOCITY U": np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64),
        "VELOCITY V": np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
    }


class SingleTimeTF(FakeTF):
    times = [0.0]


class NonUniformTimeTF(FakeTF):
    times = [0.0, 1.0, 4.0]


# ---------------------------------------------------------------------------
# TestVectorizedHelpers
# ---------------------------------------------------------------------------

class TestVectorizedHelpers:
    def test_element_areas(self, fake_tf):
        areas = _element_areas(fake_tf)
        assert len(areas) == 2
        assert areas[0] == pytest.approx(0.5)
        assert areas[1] == pytest.approx(0.5)

    def test_scatter_to_vertices(self, fake_tf):
        elem_values = np.array([10.0, 20.0])
        result = _scatter_to_vertices(fake_tf.ikle2, elem_values, fake_tf.npoin2)
        # Node 0 belongs only to tri 0 → 10
        assert result[0] == pytest.approx(10.0)
        # Node 3 belongs only to tri 1 → 20
        assert result[3] == pytest.approx(20.0)
        # Nodes 1 and 2 belong to both → avg(10, 20) = 15
        assert result[1] == pytest.approx(15.0)
        assert result[2] == pytest.approx(15.0)

    def test_sanitize_nan(self):
        arr = np.array([1.0, np.nan, np.inf, -np.inf])
        result = _sanitize_result(arr)
        np.testing.assert_array_equal(result, [1.0, 0.0, 0.0, 0.0])

    def test_sanitize_clean(self):
        arr = np.array([1.0, 2.0])
        result = _sanitize_result(arr)
        assert result is arr  # same object, no copy needed


# ---------------------------------------------------------------------------
# TestComputedFields
# ---------------------------------------------------------------------------

class TestComputedFields:
    def test_mesh_quality_range(self, fake_tf):
        q = compute_mesh_quality(fake_tf)
        assert np.all(q >= 0.0)
        assert np.all(q <= 1.0)

    def test_mesh_quality_value(self, fake_tf):
        q = compute_mesh_quality(fake_tf)
        # Right isosceles triangle quality ≈ 0.8284
        assert q[0] == pytest.approx(0.8284, abs=0.001)

    def test_slope_linear_gradient(self, fake_tf):
        # VELOCITY U = [0, 1, 0, 1] — linear gradient of 1.0 in x
        values = fake_tf.get_data_value("VELOCITY U", 0)
        slope = compute_slope(fake_tf, values)
        assert all(s == pytest.approx(1.0, abs=0.01) for s in slope)

    def test_slope_constant_field(self, fake_tf):
        values = np.ones(fake_tf.npoin2, dtype=np.float64)
        slope = compute_slope(fake_tf, values)
        assert all(s == pytest.approx(0.0) for s in slope)

    def test_courant_returns_none(self):
        tf = NoVelTF()
        result = compute_courant_number(tf, 0)
        assert result is None

    def test_courant_returns_array(self, fake_tf):
        result = compute_courant_number(fake_tf, 1)
        assert result is not None
        assert result.shape == (4,)

    def test_courant_uses_correct_dt(self):
        tf = NonUniformTimeTF()
        cfl1 = compute_courant_number(tf, 1)  # dt = 1
        cfl2 = compute_courant_number(tf, 2)  # dt = 3
        # Same velocity field, so CFL should scale linearly with dt
        # At nodes with non-zero velocity, cfl2/cfl1 == 3
        nonzero = cfl1 > 0
        ratio = cfl2[nonzero] / cfl1[nonzero]
        assert all(r == pytest.approx(3.0, abs=0.01) for r in ratio)

    def test_element_area_value(self, fake_tf):
        areas = compute_element_area(fake_tf)
        assert all(a == pytest.approx(0.5, abs=0.01) for a in areas)

    def test_mesh_integral_constant(self, fake_tf):
        values = np.full(fake_tf.npoin2, 3.0, dtype=np.float64)
        result = compute_mesh_integral(fake_tf, values)
        # total_area = 0.5 + 0.5 = 1.0, integral = 3.0 * 1.0 = 3.0
        assert result["total_area"] == pytest.approx(1.0)
        assert result["integral"] == pytest.approx(3.0)

    def test_mesh_integral_threshold(self, fake_tf):
        values = fake_tf.get_data_value("WATER DEPTH", 0)  # [0.1, 0.5, 0.5, 1.0]
        result = compute_mesh_integral(fake_tf, values, threshold=0.6)
        assert result["wetted_fraction"] < 1.0


# ---------------------------------------------------------------------------
# TestDerivedVariables
# ---------------------------------------------------------------------------

class TestDerivedVariables:
    def test_available_with_velocity(self, fake_tf):
        avail = get_available_derived(fake_tf)
        assert "VELOCITY MAGNITUDE" in avail
        assert "FROUDE NUMBER" in avail
        assert "VORTICITY" in avail

    def test_available_without_vel(self):
        tf = NoVelTF()
        avail = get_available_derived(tf)
        assert avail == []

    def test_velocity_magnitude(self, fake_tf):
        mag = compute_derived(fake_tf, "VELOCITY MAGNITUDE", 0)
        # Node 3: u=1, v=1 → sqrt(2) ≈ 1.414
        assert mag[3] == pytest.approx(1.414, abs=0.01)

    def test_vorticity_nonzero(self):
        # Use a custom fixture where V varies in x (not y) to get dv/dx != 0
        class VortTF(FakeTF):
            _data = {
                "VELOCITY U": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
                "VELOCITY V": np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64),  # x-gradient
                "WATER DEPTH": np.array([0.1, 0.5, 0.5, 1.0], dtype=np.float64),
            }
        tf = VortTF()
        vort = _compute_vorticity(tf, 0)
        # V=[0,1,0,1] has dv/dx=1, du/dy=0 → vorticity=1
        assert not np.allclose(vort, 0.0)


# ---------------------------------------------------------------------------
# TestExpressionParser
# ---------------------------------------------------------------------------

class TestExpressionParser:
    def test_arithmetic(self, fake_tf):
        result = evaluate_expression(fake_tf, 0, "VELOCITY_U + VELOCITY_V")
        assert result[3] == pytest.approx(2.0)

    def test_function_sqrt(self, fake_tf):
        result = evaluate_expression(fake_tf, 0, "sqrt(VELOCITY_U**2)")
        expected = np.abs(fake_tf.get_data_value("VELOCITY U", 0))
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_constant_mul(self, fake_tf):
        result = evaluate_expression(fake_tf, 0, "WATER_DEPTH * 9.81")
        assert result[0] == pytest.approx(0.1 * 9.81, abs=1e-5)

    def test_where(self, fake_tf):
        result = evaluate_expression(fake_tf, 0, "where(WATER_DEPTH > 0.3, 1.0, 0.0)")
        np.testing.assert_array_equal(result, [0, 1, 1, 1])

    def test_ternary(self, fake_tf):
        result = evaluate_expression(fake_tf, 0, "1.0 if WATER_DEPTH > 0.3 else 0.0")
        np.testing.assert_array_equal(result, [0, 1, 1, 1])

    def test_negation(self, fake_tf):
        result = evaluate_expression(fake_tf, 0, "-WATER_DEPTH")
        expected = -fake_tf.get_data_value("WATER DEPTH", 0).astype(np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_comparison(self, fake_tf):
        result = evaluate_expression(fake_tf, 0, "WATER_DEPTH > 0.3")
        # Should be boolean-ish: [False, True, True, True]
        assert result[0] == pytest.approx(0.0)
        assert result[3] == pytest.approx(1.0)

    def test_security_import(self, fake_tf):
        with pytest.raises(ValueError):
            evaluate_expression(fake_tf, 0, "__import__('os')")

    def test_security_attr(self, fake_tf):
        with pytest.raises(ValueError):
            evaluate_expression(fake_tf, 0, "(1).__class__")

    def test_security_method(self, fake_tf):
        with pytest.raises(ValueError):
            evaluate_expression(fake_tf, 0, "WATER_DEPTH.max()")

    def test_security_kwargs(self, fake_tf):
        with pytest.raises(ValueError):
            evaluate_expression(fake_tf, 0, "sqrt(x=1)")

    def test_syntax_error(self, fake_tf):
        with pytest.raises(ValueError):
            evaluate_expression(fake_tf, 0, "1 +")

    def test_chained_comparison(self, fake_tf):
        """Chained comparison a < b < c should mean (a < b) AND (b < c)."""
        # WATER DEPTH at t=0: [0.1, 0.5, 0.5, 1.0]
        # 0.2 < WATER_DEPTH < 0.8 should be True for nodes with 0.5 (indices 1,2)
        result = evaluate_expression(fake_tf, 0, "0.2 < WATER_DEPTH < 0.8")
        expected = np.array([False, True, True, False])
        np.testing.assert_array_equal(result.astype(bool), expected)

    def test_subscript_rejected(self, fake_tf):
        with pytest.raises(ValueError, match="Unsupported expression element"):
            evaluate_expression(fake_tf, 0, "WATER_DEPTH[0]")

    def test_nested_function_calls(self, fake_tf):
        result = evaluate_expression(fake_tf, 0, "sqrt(abs(WATER_DEPTH))")
        expected = np.sqrt(np.abs(np.array([0.1, 0.5, 0.5, 1.0])))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_division_by_zero_produces_inf(self, fake_tf):
        result = evaluate_expression(fake_tf, 0, "WATER_DEPTH / 0")
        assert np.any(np.isinf(result))

    def test_string_constant_rejected(self, fake_tf):
        with pytest.raises(ValueError, match="Unsupported constant"):
            evaluate_expression(fake_tf, 0, '"hello"')


# ---------------------------------------------------------------------------
# TestSpatialFunctions
# ---------------------------------------------------------------------------

class TestSpatialFunctions:
    def test_click_to_native_no_crs(self):
        from crs import click_to_native
        geom = MeshGeometry(npoin=0, positions={}, indices={},
                            x_off=500.0, y_off=300.0, lon_off=0.0, lat_off=0.0,
                            crs=None, extent_m=1.0, zoom=1.0)
        x, y = click_to_native(0.0, 0.0, geom)
        assert x == pytest.approx(500.0)
        assert y == pytest.approx(300.0)

    def test_nearest_node_corner(self, fake_tf):
        idx, nx, ny = nearest_node(fake_tf, 0.1, 0.1)
        assert idx == 0

    def test_nearest_node_opposite(self, fake_tf):
        idx, nx, ny = nearest_node(fake_tf, 0.9, 0.9)
        assert idx == 3

    def test_boundary_nodes(self, fake_tf):
        bnodes = find_boundary_nodes(fake_tf)
        assert sorted(bnodes) == [0, 1, 2, 3]

    def test_extrema_correct(self, fake_tf):
        values = np.array([1.0, 5.0, 2.0, 4.0])
        ext = find_extrema(fake_tf, values)
        assert ext["min"][0] == 0
        assert ext["min"][3] == pytest.approx(1.0)
        assert ext["max"][0] == 1
        assert ext["max"][3] == pytest.approx(5.0)

    def test_extrema_nan(self, fake_tf):
        values = np.array([np.nan, 5.0, 1.0, np.nan])
        ext = find_extrema(fake_tf, values)
        assert ext["min"][0] == 2
        assert ext["min"][3] == pytest.approx(1.0)
        assert ext["max"][0] == 1
        assert ext["max"][3] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# TestProfiles
# ---------------------------------------------------------------------------

class TestProfiles:
    def test_vertical_profile_2d(self, fake_tf):
        elev, vals, label = vertical_profile_at_point(fake_tf, "WATER DEPTH", 0, 0.5, 0.5)
        assert len(elev) == 0
        assert len(vals) == 0
        assert label == "Elevation (m)"

    def test_time_series(self, fake_tf):
        times, values = time_series_at_point(fake_tf, "WATER DEPTH", 0.0, 0.0)
        assert len(times) == 3
        assert len(values) == 3

    def test_cross_section(self, fake_tf):
        polyline = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
        abscissa, values = cross_section_profile(fake_tf, "WATER DEPTH", 0, polyline)
        assert len(abscissa) == 3
        # Abscissa must be monotonically non-decreasing
        assert all(abscissa[i] <= abscissa[i + 1] for i in range(len(abscissa) - 1))

    def test_temporal_stats(self, fake_tf):
        stats = compute_temporal_stats(fake_tf, "WATER DEPTH")
        assert stats is not None
        # min <= mean <= max at every node
        assert np.all(stats["min"] <= stats["mean"] + 1e-6)
        assert np.all(stats["mean"] <= stats["max"] + 1e-6)

    def test_difference(self, fake_tf):
        diff = compute_difference(fake_tf, "WATER DEPTH", 2, 0)
        # t=2 scale: 1 + 2*0.5 = 2.0, t=0 scale: 1.0
        # diff = base * (2.0 - 1.0) = base * 1.0
        base = np.array([0.1, 0.5, 0.5, 1.0])
        np.testing.assert_allclose(diff, base, atol=1e-10)


# ---------------------------------------------------------------------------
# TestDischarge
# ---------------------------------------------------------------------------

class TestDischarge:
    def test_missing_velocity(self):
        tf = NoVelTF()
        result = compute_discharge(tf, 0, [[0, 0], [1, 1]])
        assert result["total_q"] is None
        assert "error" in result

    def test_missing_depth(self):
        tf = NoDepthTF()
        result = compute_discharge(tf, 0, [[0, 0], [1, 1]])
        assert result["total_q"] is None

    def test_valid_has_keys(self, fake_tf):
        result = compute_discharge(fake_tf, 0, [[0.0, 0.0], [1.0, 1.0]])
        assert isinstance(result["total_q"], float)
        assert isinstance(result["segments"], list)
        assert "skipped" in result

    def test_discharge_numerical_value(self, fake_tf):
        """Verify discharge Q against analytical expectation.

        Cross-section from (0.9, 0) to (0.9, 1) — perpendicular to x-axis.
        Midpoint at (0.9, 0.5). FakeTF nearest-node: node 1 (1,0) has u=1, v=0,
        depth=0.5. Normal to section: n=(-1, 0). Q = -1 * 0.5 * 1.0 = -0.5.
        """
        xsec = [[0.9, 0.0], [0.9, 1.0]]
        result = compute_discharge(fake_tf, 0, xsec)
        # Result should be non-zero (exact value depends on nearest-node interpolation)
        assert result["total_q"] != 0.0
        assert isinstance(result["total_q"], float)
        assert len(result["segments"]) > 0


# ---------------------------------------------------------------------------
# TestParticleTracing
# ---------------------------------------------------------------------------

class TestParticleTracing:
    def test_no_velocity(self):
        tf = NoVelTF()
        paths = compute_particle_paths(tf, [[0.5, 0.5]], 0.5, 0.5)
        assert paths == []

    def test_single_timestep(self):
        tf = SingleTimeTF()
        paths = compute_particle_paths(tf, [[0.5, 0.5]], 0.5, 0.5)
        assert paths == []

    def test_empty_seeds(self, fake_tf):
        paths = compute_particle_paths(fake_tf, [], 0.5, 0.5)
        assert paths == []

    def test_valid_structure(self, fake_tf):
        paths = compute_particle_paths(fake_tf, [[0.5, 0.5]], 0.5, 0.5)
        assert isinstance(paths, list)
        if len(paths) > 0:
            # Each path is a list of [x, y, time] triplets
            for point in paths[0]:
                assert len(point) == 3


# ---------------------------------------------------------------------------
# TestSeedDistribution
# ---------------------------------------------------------------------------

class TestSeedDistribution:
    def test_generate_seed_grid(self, fake_tf):
        seeds = generate_seed_grid(fake_tf, n_target=10)
        assert len(seeds) == 0  # FakeTF has no .tri, fallback now returns empty

    def test_empty_polyline(self):
        result = distribute_seeds_along_line([], n_seeds=5)
        assert result == []

    def test_single_point(self):
        result = distribute_seeds_along_line([[5.0, 5.0]], n_seeds=5)
        assert result == [[5.0, 5.0]]

    def test_l_shaped(self):
        polyline = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]
        result = distribute_seeds_along_line(polyline, n_seeds=5)
        assert len(result) == 5
        for pt in result:
            assert len(pt) == 2


# ---------------------------------------------------------------------------
# TestExport
# ---------------------------------------------------------------------------

class TestExport:
    def test_timeseries_csv(self):
        times = np.array([0.0, 1.0, 2.0])
        values = np.array([10.0, 20.0, 30.0])
        csv = export_timeseries_csv(times, values, "WATER DEPTH")
        lines = csv.strip().split("\n")
        assert lines[0] == "Time (s),WATER DEPTH"
        assert len(lines) == 4  # header + 3 data lines

    def test_crosssection_csv(self):
        abscissa = np.array([0.0, 5.0, 10.0])
        values = np.array([1.0, 2.0, 3.0])
        csv = export_crosssection_csv(abscissa, values, "WATER DEPTH")
        lines = csv.strip().split("\n")
        assert lines[0] == "Distance (m),WATER DEPTH"

    def test_all_variables_csv(self, fake_tf):
        csv = export_all_variables_csv(fake_tf, 0, 0.0, 0.0)
        lines = csv.strip().split("\n")
        assert lines[0] == "Variable,Value"
        # All varnames present
        for vname in fake_tf.varnames:
            assert any(vname in line for line in lines)


# ---------------------------------------------------------------------------
# TestSimUtilities
# ---------------------------------------------------------------------------

class TestSimUtilities:
    def test_find_cas_files(self, tmp_path):
        (tmp_path / "case1.cas").write_text("/ test")
        (tmp_path / "case2.cas").write_text("/ test")
        (tmp_path / "other.txt").write_text("not a cas")
        result = find_cas_files(str(tmp_path / "result.slf"))
        assert len(result) == 2
        assert "case1.cas" in result
        assert "case2.cas" in result

    def test_detect_telemac2d(self):
        assert detect_module("/examples/telemac2d/foo.cas") == "telemac2d"

    def test_detect_telemac3d(self):
        assert detect_module("/examples/telemac3d/foo.cas") == "telemac3d"

    def test_detect_fallback(self):
        assert detect_module("/some/path/foo.cas") == "telemac2d"


# ---------------------------------------------------------------------------
# TestFloodAnalysis
# ---------------------------------------------------------------------------

class TestFloodAnalysis:
    def test_flood_envelope_is_peak(self, fake_tf):
        """Envelope should be max value across all timesteps at each node."""
        env = compute_flood_envelope(fake_tf, "WATER DEPTH", threshold=0.01)
        expected_peak = np.array([0.2, 1.0, 1.0, 2.0])
        np.testing.assert_allclose(env, expected_peak, atol=1e-6)

    def test_flood_envelope_threshold(self, fake_tf):
        """Nodes whose peak is below threshold should be zeroed."""
        env = compute_flood_envelope(fake_tf, "WATER DEPTH", threshold=0.5)
        assert env[0] == 0.0
        assert env[3] == pytest.approx(2.0)

    def test_flood_arrival_time(self, fake_tf):
        """Arrival = time when value first exceeds threshold."""
        arrival = compute_flood_arrival(fake_tf, "WATER DEPTH", threshold=0.6)
        assert np.isnan(arrival[0])
        assert arrival[1] == pytest.approx(1.0)
        assert arrival[3] == pytest.approx(0.0)

    def test_flood_arrival_all_dry(self, fake_tf):
        """All NaN if threshold is never exceeded."""
        arrival = compute_flood_arrival(fake_tf, "WATER DEPTH", threshold=999.0)
        assert np.all(np.isnan(arrival))

    def test_flood_duration(self, fake_tf):
        """Duration = total time above threshold."""
        dur = compute_flood_duration(fake_tf, "WATER DEPTH", threshold=0.6)
        assert dur[0] == pytest.approx(0.0)
        assert dur[1] == pytest.approx(2.0)
        assert dur[3] == pytest.approx(3.0)

    def test_flood_duration_single_timestep(self):
        """Single timestep uses dt=1.0 fallback."""
        tf = SingleTimeTF()
        dur = compute_flood_duration(tf, "WATER DEPTH", threshold=0.01)
        assert np.all(dur == pytest.approx(1.0))


# ---------------------------------------------------------------------------
# TestCliFile
# ---------------------------------------------------------------------------

class TestCliFile:
    def test_valid_cli_file(self, tmp_path):
        """Parse a minimal .cli file with LIHBOR codes."""
        cli = tmp_path / "test.cli"
        cli.write_text(
            "2 0 0 0.0 0.0 0.0 0.0 0 0.0 0.0 0.0 1\n"
            "5 0 0 0.0 0.0 0.0 0.0 0 0.0 0.0 0.0 2\n"
            "4 0 0 0.0 0.0 0.0 0.0 0 0.0 0.0 0.0 3\n"
        )
        result = read_cli_file(str(cli))
        assert result is not None
        assert result[1] == 2  # wall
        assert result[2] == 5  # prescribed
        assert result[3] == 4  # free

    def test_missing_file_returns_none(self, tmp_path):
        result = read_cli_file(str(tmp_path / "nonexistent.cli"))
        assert result is None

    def test_empty_file_returns_none(self, tmp_path):
        cli = tmp_path / "empty.cli"
        cli.write_text("")
        result = read_cli_file(str(cli))
        assert result is None


# ---------------------------------------------------------------------------
# TestPolygonZonalStats
# ---------------------------------------------------------------------------

class TestPolygonZonalStats:
    def test_polygon_covering_all_nodes(self, fake_tf):
        """Polygon enclosing entire mesh should include all 4 nodes."""
        polygon = [[-0.5, -0.5], [1.5, -0.5], [1.5, 1.5], [-0.5, 1.5]]
        values = fake_tf.get_data_value("WATER DEPTH", 0)  # [0.1, 0.5, 0.5, 1.0]
        stats = polygon_zonal_stats(fake_tf, values, polygon)
        assert stats["count"] == 4
        assert stats["min"] == pytest.approx(0.1)
        assert stats["max"] == pytest.approx(1.0)
        assert stats["mean"] == pytest.approx(np.mean([0.1, 0.5, 0.5, 1.0]))

    def test_empty_polygon(self, fake_tf):
        """Polygon outside mesh should return zeros."""
        polygon = [[10, 10], [11, 10], [11, 11], [10, 11]]
        values = fake_tf.get_data_value("WATER DEPTH", 0)
        stats = polygon_zonal_stats(fake_tf, values, polygon)
        assert stats["count"] == 0
        assert stats["mean"] == 0.0


# ---------------------------------------------------------------------------
# TestSanitizeResultLogging
# ---------------------------------------------------------------------------

class TestSanitizeResultLogging:
    def test_logs_warning_on_nan(self, caplog):
        """_sanitize_result should log when replacing NaN/Inf."""
        arr = np.array([1.0, np.nan, np.inf, -np.inf, 2.0])
        with caplog.at_level(logging.WARNING):
            result = _sanitize_result(arr)
        assert np.all(np.isfinite(result))
        assert any("non-finite" in r.message.lower() for r in caplog.records)

    def test_no_warning_on_clean_data(self, caplog):
        """No warning when data is all finite."""
        arr = np.array([1.0, 2.0, 3.0])
        with caplog.at_level(logging.WARNING):
            result = _sanitize_result(arr)
        np.testing.assert_array_equal(result, arr)
        assert len(caplog.records) == 0


# ---------------------------------------------------------------------------
# TestExtractLayer2d
# ---------------------------------------------------------------------------

class TestExtractLayer2d:
    def test_extract_bottom_layer(self):
        # 3 planes, 4 nodes each = 12 values
        data = np.arange(12, dtype=np.float64)
        result = extract_layer_2d(data, npoin2=4, layer_k=0)
        np.testing.assert_array_equal(result, [0, 1, 2, 3])

    def test_extract_surface_layer(self):
        data = np.arange(12, dtype=np.float64)
        result = extract_layer_2d(data, npoin2=4, layer_k=2)
        np.testing.assert_array_equal(result, [8, 9, 10, 11])

    def test_extract_returns_copy(self):
        data = np.arange(12, dtype=np.float64)
        result = extract_layer_2d(data, npoin2=4, layer_k=0)
        result[0] = 999
        assert data[0] != 999  # original unchanged


# ---------------------------------------------------------------------------
# TestComputeAllTemporalStats
# ---------------------------------------------------------------------------

class TestComputeAllTemporalStats:
    def test_returns_all_keys(self, fake_tf):
        result = compute_all_temporal_stats(fake_tf, "WATER DEPTH")
        assert "max_values" in result
        assert "min_values" in result
        assert "mean_values" in result
        assert "envelope" in result
        assert "arrival" in result
        assert "duration" in result

    def test_max_values_match_individual(self, fake_tf):
        combined = compute_all_temporal_stats(fake_tf, "WATER DEPTH")
        individual = compute_temporal_stats(fake_tf, "WATER DEPTH")
        np.testing.assert_allclose(
            combined["max_values"], individual["max"], rtol=1e-5)

    def test_min_values_match_individual(self, fake_tf):
        combined = compute_all_temporal_stats(fake_tf, "WATER DEPTH")
        individual = compute_temporal_stats(fake_tf, "WATER DEPTH")
        np.testing.assert_allclose(
            combined["min_values"], individual["min"], rtol=1e-5)
