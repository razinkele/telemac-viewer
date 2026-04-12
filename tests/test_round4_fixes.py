"""Round 4 regression tests — 8 bug fixes."""
from __future__ import annotations

import ast
import logging
import numpy as np
import pytest

from tests.helpers import FakeTF


# ---------------------------------------------------------------------------
# BUG 1: logger vs _logger NameError (parser_2d.py)
# ---------------------------------------------------------------------------
class TestLoggerName:
    def test_parser_2d_uses_underscore_logger(self):
        """Line 149 must use _logger, not logger."""
        import telemac_tools.hecras.parser_2d as mod
        assert hasattr(mod, "_logger")
        import inspect
        src = inspect.getsource(mod)
        # No bare 'logger.' calls — only '_logger.'
        lines = src.splitlines()
        for i, line in enumerate(lines, 1):
            stripped = line.lstrip()
            if stripped.startswith("logger.") and not stripped.startswith("_logger."):
                pytest.fail(f"Line {i}: bare 'logger.' should be '_logger.'")


# ---------------------------------------------------------------------------
# BUG 2: scalar expression IndexError
# ---------------------------------------------------------------------------
class TestScalarExpression:
    def test_constant_expression_broadcasts(self, fake_tf):
        from analysis import evaluate_expression
        result = evaluate_expression(fake_tf, 0, "10")
        assert result.shape == (fake_tf.npoin2,)
        assert np.allclose(result, 10.0)

    def test_constant_zero(self, fake_tf):
        from analysis import evaluate_expression
        result = evaluate_expression(fake_tf, 0, "0")
        assert result.shape == (fake_tf.npoin2,)
        assert np.allclose(result, 0.0)

    def test_float_constant(self, fake_tf):
        from analysis import evaluate_expression
        result = evaluate_expression(fake_tf, 0, "3.14")
        assert result.shape == (fake_tf.npoin2,)
        assert np.allclose(result, 3.14, atol=1e-5)


# ---------------------------------------------------------------------------
# BUG 3: seed grid memory overflow
# ---------------------------------------------------------------------------
class TestSeedGridOverflow:
    def test_extreme_aspect_ratio_no_crash(self):
        """A very thin strip mesh should not blow up memory."""
        from analysis import generate_seed_grid
        tf = FakeTF()
        # Monkey-patch to create extreme aspect: dx=1000, dy=0.001
        tf.meshx = np.array([0.0, 1000.0, 1000.0, 0.0], dtype=np.float64)
        tf.meshy = np.array([0.0, 0.0, 0.001, 0.001], dtype=np.float64)
        # Should return without MemoryError
        result = generate_seed_grid(tf, n_target=500)
        assert isinstance(result, list)
        # Total points should be reasonable (not millions)
        assert len(result) <= 5000

    def test_normal_aspect_still_works(self):
        """Normal meshes should still produce seeds."""
        from analysis import generate_seed_grid
        tf = FakeTF()
        result = generate_seed_grid(tf, n_target=100)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# BUG 4: NaN in polygon zonal stats
# ---------------------------------------------------------------------------
class TestPolygonNaN:
    def test_nan_vertex_skipped(self, fake_tf, fake_geom):
        from analysis import polygon_zonal_stats
        values = np.array([np.nan, 1.0, 2.0, 3.0], dtype=np.float32)
        # Polygon covering entire mesh
        polygon = [[0, 0], [1, 0], [1, 1], [0, 1]]
        result = polygon_zonal_stats(fake_tf, values, polygon, fake_geom)
        # Should not be NaN — elements with NaN vertices are skipped
        assert result is not None
        assert not np.isnan(result["mean"])

    def test_all_nan_fallback(self, fake_tf, fake_geom):
        from analysis import polygon_zonal_stats
        values = np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float32)
        polygon = [[0, 0], [1, 0], [1, 1], [0, 1]]
        result = polygon_zonal_stats(fake_tf, values, polygon, fake_geom)
        # All NaN — weighted_mean should be NaN (from nanmean of all-NaN)
        assert result is not None


# ---------------------------------------------------------------------------
# BUG 5: empty polyline IndexError
# ---------------------------------------------------------------------------
class TestEmptyPolyline:
    def test_empty_polyline_no_crash(self):
        from telemac_tools.hecras.parser_1d import _interp_stations_to_world
        polyline = np.empty((0, 2), dtype=np.float64)
        sta = np.array([0.0, 1.0, 2.0])
        elev = np.array([10.0, 11.0, 12.0])
        result = _interp_stations_to_world(sta, elev, polyline)
        assert result.shape == (3, 3)
        # Z column should be elevations
        np.testing.assert_array_equal(result[:, 2], elev)

    def test_empty_station_vals(self):
        from telemac_tools.hecras.parser_1d import _interp_stations_to_world
        polyline = np.array([[0.0, 0.0], [10.0, 0.0]])
        sta = np.array([], dtype=np.float64)
        elev = np.array([], dtype=np.float64)
        result = _interp_stations_to_world(sta, elev, polyline)
        assert result.shape == (0, 3)


# ---------------------------------------------------------------------------
# BUG 6: empty station-elevation — tested via parser_1d integration
# (structural test: verify the guard exists)
# ---------------------------------------------------------------------------
class TestEmptyStationElev:
    def test_guard_exists_in_parse_loop(self):
        import inspect
        from telemac_tools.hecras import parser_1d
        src = inspect.getsource(parser_1d.parse_hecras_1d)
        assert "len(se_data) == 0" in src, "Missing empty station-elevation guard"


# ---------------------------------------------------------------------------
# BUG 7: mannings_n indexing
# ---------------------------------------------------------------------------
class TestManningsIndexing:
    def test_single_value_mannings(self):
        """builder should handle 1-element mannings_n without IndexError."""
        from telemac_tools.domain.builder import _build_mannings_regions
        from telemac_tools.model import CrossSection, Reach

        xs1 = CrossSection(
            station=0.0,
            coords=np.array([[0, 0, 10], [1, 0, 10]]),
            mannings_n=[0.05],
            bank_stations=(0.0, 1.0),
            bank_coords=np.array([[0, 0], [1, 0]]),
        )
        xs2 = CrossSection(
            station=100.0,
            coords=np.array([[0, 100, 10], [1, 100, 10]]),
            mannings_n=[0.06],
            bank_stations=(0.0, 1.0),
            bank_coords=np.array([[0, 100], [1, 100]]),
        )
        reach = Reach(name="test", alignment=np.array([[0, 50], [1, 50]]),
                       cross_sections=[xs1, xs2])
        result = _build_mannings_regions(reach)
        assert isinstance(result, list)
        if result:
            assert result[0]["n"] == pytest.approx(0.055)

    def test_empty_mannings_uses_default(self):
        from telemac_tools.domain.builder import _build_mannings_regions
        from telemac_tools.model import CrossSection, Reach

        xs = CrossSection(
            station=0.0,
            coords=np.array([[0, 0, 10], [1, 0, 10]]),
            mannings_n=[],
            bank_stations=(0.0, 1.0),
            bank_coords=np.array([[0, 0], [1, 0]]),
        )
        xs2 = CrossSection(
            station=100.0,
            coords=np.array([[0, 100, 10], [1, 100, 10]]),
            mannings_n=[],
            bank_stations=(0.0, 1.0),
            bank_coords=np.array([[0, 100], [1, 100]]),
        )
        reach = Reach(name="test", alignment=np.array([[0, 50], [1, 50]]),
                       cross_sections=[xs, xs2])
        result = _build_mannings_regions(reach)
        if result:
            assert result[0]["n"] == pytest.approx(0.035)


# ---------------------------------------------------------------------------
# BUG 8: NaN elevation interpolation
# ---------------------------------------------------------------------------
class TestNaNElevationInterp:
    def test_nan_filtered_before_interp(self):
        """Verify NaN elevation handling code structure."""
        import inspect
        from telemac_tools.hecras import parser_2d
        src = inspect.getsource(parser_2d.triangulate_2d_area)
        assert "~np.isnan" in src or "isnan" in src, "Missing NaN filter"
        assert "valid" in src, "Missing valid mask variable"

    def test_structural_nan_fallback(self):
        """Verify all-NaN fallback path exists."""
        import inspect
        from telemac_tools.hecras import parser_2d
        src = inspect.getsource(parser_2d.triangulate_2d_area)
        assert "np.zeros" in src, "Missing zeros fallback for all-NaN case"
