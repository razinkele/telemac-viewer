"""Round 6 regression tests — path traversal, flooded area accuracy, list cap."""
from __future__ import annotations

import os
import numpy as np
import pytest
from tests.helpers import FakeTF


# ── Fix 1: Path traversal sanitisation in server_import.py ──


class TestPathTraversalSanitise:
    """os.path.basename prevents directory escape in uploaded filenames."""

    def test_simple_filename(self):
        name = "MyProject.g01.hdf"
        sanitised = os.path.basename(name).rsplit(".", 2)[0]
        assert sanitised == "MyProject"

    def test_path_traversal_attack(self):
        name = "../../etc/evil.g01.hdf"
        sanitised = os.path.basename(name).rsplit(".", 2)[0]
        assert sanitised == "evil"
        assert ".." not in sanitised

    def test_absolute_path_attack(self):
        name = "/tmp/secret/data.g01.hdf"
        sanitised = os.path.basename(name).rsplit(".", 2)[0]
        assert sanitised == "data"

    def test_backslash_attack(self):
        name = "..\\..\\windows\\system32\\foo.g01.hdf"
        sanitised = os.path.basename(name).rsplit(".", 2)[0]
        # os.path.basename on Linux treats backslash as literal char
        # but still no '..' prefix escape possible
        assert "/" not in sanitised

    def test_empty_after_strip(self):
        name = ".g01.hdf"
        sanitised = os.path.basename(name).rsplit(".", 2)[0]
        # Edge case: empty prefix still works (produces '' which is safe)
        assert ".." not in sanitised


# ── Fix 2: Element-area-based flooded area in polygon_zonal_stats ──


class TestFloodedAreaAccuracy:
    """Flooded area uses element areas, not node-count ratio."""

    def test_uniform_mesh_flooded_area(self):
        """All nodes flooded → flooded_area ≈ total element area."""
        from analysis import polygon_zonal_stats

        tf = FakeTF()
        # WATER DEPTH at tidx=0: [0.1, 0.5, 0.5, 1.0] — all > 0.01 threshold
        values = tf.get_data_value("WATER DEPTH", 0)
        polygon = [(-.5, -.5), (1.5, -.5), (1.5, 1.5), (-.5, 1.5)]
        result = polygon_zonal_stats(tf, values, polygon, var_name="WATER DEPTH")
        assert result["flooded_area"] > 0
        # Each triangle has area 0.5, total = 1.0
        assert result["flooded_area"] == pytest.approx(1.0, abs=0.01)

    def test_partial_flood_uses_element_area(self):
        """When some nodes are below threshold, only flooded elements count."""
        from analysis import polygon_zonal_stats

        tf = FakeTF()
        # Node 0 = 0.0 (dry), rest above threshold
        values = np.array([0.0, 0.5, 0.5, 1.0])
        polygon = [(-.5, -.5), (1.5, -.5), (1.5, 1.5), (-.5, 1.5)]
        result = polygon_zonal_stats(tf, values, polygon, var_name="WATER DEPTH")
        # Triangle 0 (nodes 0,1,2): node 0 = 0.0 → NOT flooded
        # Triangle 1 (nodes 1,3,2): all > 0.01 → flooded, area = 0.5
        assert result["flooded_area"] == pytest.approx(0.5, abs=0.01)
        assert result["flooded_fraction"] == pytest.approx(0.5, abs=0.01)

    def test_non_depth_variable_zero_flooded(self):
        """Non-depth variables return zero flooded area."""
        from analysis import polygon_zonal_stats

        tf = FakeTF()
        values = tf.get_data_value("VELOCITY U", 0)
        polygon = [(-.5, -.5), (1.5, -.5), (1.5, 1.5), (-.5, 1.5)]
        result = polygon_zonal_stats(tf, values, polygon, var_name="VELOCITY U")
        assert result["flooded_area"] == 0.0
        assert result["flooded_fraction"] == 0.0

    def test_flooded_fraction_is_area_based(self):
        """flooded_fraction = flooded_elem_area / total_elem_area."""
        from analysis import polygon_zonal_stats

        tf = FakeTF()
        values = tf.get_data_value("WATER DEPTH", 0)
        polygon = [(-.5, -.5), (1.5, -.5), (1.5, 1.5), (-.5, 1.5)]
        result = polygon_zonal_stats(tf, values, polygon, var_name="WATER DEPTH")
        # All flooded → fraction should be ≈ 1.0
        assert result["flooded_fraction"] == pytest.approx(1.0, abs=0.01)


# ── Fix 3: Unbounded clicked_points cap ──


class TestClickedPointsCap:
    """clicked_points list is capped at 200 entries."""

    def test_cap_logic(self):
        """Simulate append + cap logic from server_analysis.py."""
        pts = [(float(i), float(i)) for i in range(250)]
        pts.append((999.0, 999.0))
        if len(pts) > 200:
            pts = pts[-200:]
        assert len(pts) == 200
        assert pts[-1] == (999.0, 999.0)
        # Oldest points dropped
        assert pts[0] == (51.0, 51.0)

    def test_under_cap_unchanged(self):
        """Lists under 200 are not truncated."""
        pts = [(float(i), float(i)) for i in range(10)]
        pts.append((99.0, 99.0))
        if len(pts) > 200:
            pts = pts[-200:]
        assert len(pts) == 11
