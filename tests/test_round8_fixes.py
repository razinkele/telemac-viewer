"""Round 8 regression tests — empty mesh, all-NaN extrema, empty timeseries,
empty polygon guard."""
from __future__ import annotations

import numpy as np
import pytest
from tests.helpers import FakeTF


# ── Fix 1: nearest_node on empty mesh ──


class TestNearestNodeEmpty:
    """nearest_node raises ValueError on empty mesh."""

    def test_empty_mesh_raises(self):
        from analysis import nearest_node

        class EmptyTF:
            meshx = np.array([], dtype=np.float64)
            meshy = np.array([], dtype=np.float64)
            npoin2 = 0

        with pytest.raises(ValueError, match="no 2D points"):
            nearest_node(EmptyTF(), 0.5, 0.5)

    def test_normal_mesh_works(self):
        from analysis import nearest_node

        tf = FakeTF()
        idx, x, y = nearest_node(tf, 0.1, 0.1)
        assert idx == 0
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(0.0)


# ── Fix 2: find_extrema on all-NaN ──


class TestFindExtremaAllNaN:
    """find_extrema returns sentinel values for all-NaN input."""

    def test_all_nan(self):
        from analysis import find_extrema

        tf = FakeTF()
        values = np.full(tf.npoin2, np.nan)
        result = find_extrema(tf, values)
        assert result["min"][0] == -1
        assert result["max"][0] == -1
        assert np.isnan(result["min"][3])
        assert np.isnan(result["max"][3])

    def test_normal_values(self):
        from analysis import find_extrema

        tf = FakeTF()
        values = np.array([1.0, 3.0, 2.0, 0.5])
        result = find_extrema(tf, values)
        assert result["min"][0] == 3  # node 3 has value 0.5
        assert result["max"][0] == 1  # node 1 has value 3.0
        assert result["min"][3] == pytest.approx(0.5)
        assert result["max"][3] == pytest.approx(3.0)

    def test_mixed_nan(self):
        from analysis import find_extrema

        tf = FakeTF()
        values = np.array([np.nan, 2.0, np.nan, 1.0])
        result = find_extrema(tf, values)
        assert result["min"][0] == 3  # node 3 has value 1.0
        assert result["max"][0] == 1  # node 1 has value 2.0


# ── Fix 3: time_series_at_point empty result ──


class TestTimeSeriesEmptyResult:
    """time_series_at_point returns NaN array if no data available."""

    def test_empty_timeseries_returns_nan(self):
        from analysis import time_series_at_point

        class EmptyTsTF(FakeTF):
            def get_timeseries_on_points(self, varname, points):
                return []  # simulate empty result

        tf = EmptyTsTF()
        times, vals = time_series_at_point(tf, "WATER DEPTH", 0.5, 0.5)
        assert len(times) == len(tf.times)
        assert len(vals) == len(tf.times)
        assert all(np.isnan(v) for v in vals)

    def test_normal_timeseries(self):
        from analysis import time_series_at_point

        tf = FakeTF()
        times, vals = time_series_at_point(tf, "WATER DEPTH", 0.0, 0.0)
        assert len(times) == 3
        assert len(vals) == 3
        assert vals[0] == pytest.approx(0.1)


# ── Fix 4: polygon_zonal_stats empty polygon ──


class TestPolygonZonalStatsEmpty:
    """polygon_zonal_stats handles empty or degenerate polygons."""

    def test_empty_polygon(self):
        from analysis import polygon_zonal_stats

        tf = FakeTF()
        values = tf.get_data_value("WATER DEPTH", 0)
        result = polygon_zonal_stats(tf, values, [])
        assert result["count"] == 0
        assert result["area"] == 0.0

    def test_single_point_polygon(self):
        from analysis import polygon_zonal_stats

        tf = FakeTF()
        values = tf.get_data_value("WATER DEPTH", 0)
        result = polygon_zonal_stats(tf, values, [[0.5, 0.5]])
        assert result["count"] == 0

    def test_two_point_polygon(self):
        from analysis import polygon_zonal_stats

        tf = FakeTF()
        values = tf.get_data_value("WATER DEPTH", 0)
        result = polygon_zonal_stats(tf, values, [[0.0, 0.0], [1.0, 1.0]])
        assert result["count"] == 0

    def test_valid_polygon_still_works(self):
        from analysis import polygon_zonal_stats

        tf = FakeTF()
        values = tf.get_data_value("WATER DEPTH", 0)
        polygon = [(-.5, -.5), (1.5, -.5), (1.5, 1.5), (-.5, 1.5)]
        result = polygon_zonal_stats(tf, values, polygon)
        assert result["count"] == 4
        assert result["area"] > 0
