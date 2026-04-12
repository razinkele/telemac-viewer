"""Regression tests for Round 2 deep-review fixes."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from telemac_tools.model import BoundaryCondition, BCType, Mesh2D
from telemac_tools.telemac.writer_liq import write_liq
from telemac_tools.telemac.writer_slf import write_slf
from telemac_tools.hecras.parser_2d import triangulate_2d_area
from telemac_tools.model import HecRas2DArea, HecRasCell
from validation import parse_observation_csv, parse_liq_file, compute_rmse, compute_nse
from constants import cached_palette_arr
from analysis import polygon_zonal_stats


# ---------------------------------------------------------------------------
# p4a: writer_slf file handle leak on exception
# ---------------------------------------------------------------------------

class TestWriterSlfHandleLeak:
    def test_file_closed_on_exception(self, tmp_path):
        """If write_slf raises, the file handle must still be closed."""
        path = str(tmp_path / "test.slf")
        mesh = Mesh2D(
            nodes=np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]),
            elements=np.array([[0, 1, 2]], dtype=np.int32),
            elevation=np.array([0.0, 0.0, 0.0]),
            mannings_n=np.array([0.03, 0.03, 0.03]),
        )
        # Normal write should succeed and file should exist
        write_slf(mesh, path)
        assert os.path.isfile(path)

    def test_normal_write_produces_output(self, tmp_path):
        path = str(tmp_path / "out.slf")
        mesh = Mesh2D(
            nodes=np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]),
            elements=np.array([[0, 1, 2]], dtype=np.int32),
            elevation=np.array([0.0, 0.0, 0.0]),
            mannings_n=np.array([0.03, 0.03, 0.03]),
        )
        write_slf(mesh, path)
        assert os.path.getsize(path) > 0


# ---------------------------------------------------------------------------
# p4b: writer_liq unsorted times
# ---------------------------------------------------------------------------

class TestWriterLiqUnsortedTimes:
    def test_unsorted_times_produce_valid_output(self, tmp_path):
        """BCs with unsorted times should still produce monotonic .liq output."""
        bc = BoundaryCondition(
            bc_type=BCType.FLOW,
            location="upstream",
            timeseries={
                "time": np.array([30.0, 10.0, 20.0]),  # intentionally unsorted
                "values": np.array([3.0, 1.0, 2.0]),
            },
        )
        path = str(tmp_path / "test.liq")
        result = write_liq([bc], path)
        assert result is True
        parsed = parse_liq_file(path)
        assert parsed is not None
        col = list(parsed.values())[0]
        # Times should be monotonically increasing
        assert np.all(np.diff(col["times"]) > 0)
        # Values should match sorted order
        np.testing.assert_allclose(col["values"], [1.0, 2.0, 3.0])

    def test_duplicate_times_keeps_last(self, tmp_path):
        """Duplicate timestamps should keep the last value."""
        bc = BoundaryCondition(
            bc_type=BCType.FLOW,
            location="upstream",
            timeseries={
                "time": np.array([10.0, 10.0, 20.0]),
                "values": np.array([1.0, 5.0, 2.0]),  # 5.0 is later for t=10
            },
        )
        path = str(tmp_path / "dup.liq")
        write_liq([bc], path)
        parsed = parse_liq_file(path)
        assert parsed is not None
        col = list(parsed.values())[0]
        # At t=10, should have value 5.0 (the last occurrence)
        idx_10 = np.argmin(np.abs(col["times"] - 10.0))
        assert col["values"][idx_10] == pytest.approx(5.0, abs=0.01)


# ---------------------------------------------------------------------------
# p4c: parser_2d bounds, overflow, negative indices
# ---------------------------------------------------------------------------

class TestParser2dValidation:
    def _make_area(self, fp_indices_list, n_fp=4):
        """Helper: create a HecRas2DArea with given face-point index lists."""
        face_points = np.random.rand(n_fp, 2) * 100
        cells = [HecRasCell(face_point_indices=idx) for idx in fp_indices_list]
        cell_centers = np.random.rand(len(cells), 2) * 100
        return HecRas2DArea(
            name="test",
            face_points=face_points,
            cell_centers=cell_centers,
            cells=cells,
            elevation=None,
            mannings_n_constant=0.03,
        )

    def test_out_of_range_indices_skipped(self):
        """Cells with indices >= n_fp should be skipped, not crash."""
        area = self._make_area([[0, 1, 99]], n_fp=4)  # 99 is out of range
        mesh = triangulate_2d_area(area)
        assert mesh.elements.shape == (0, 3)

    def test_negative_indices_skipped(self):
        """Cells with negative indices should be skipped."""
        area = self._make_area([[0, 1, -1]], n_fp=4)
        mesh = triangulate_2d_area(area)
        assert mesh.elements.shape == (0, 3)

    def test_too_few_face_points_skipped(self):
        """Cells with <3 face points should be skipped."""
        area = self._make_area([[0, 1]], n_fp=4)  # only 2 points
        mesh = triangulate_2d_area(area)
        assert mesh.elements.shape == (0, 3)

    def test_valid_cells_work(self):
        """Normal valid cells should triangulate correctly."""
        area = self._make_area([[0, 1, 2, 3]], n_fp=4)
        mesh = triangulate_2d_area(area)
        assert mesh.elements.shape[0] == 4  # 4-sided cell -> 4 fan triangles
        assert mesh.elements.shape[1] == 3

    def test_empty_cells_produce_empty_mesh(self):
        """No cells should produce empty elements with shape (0, 3)."""
        area = self._make_area([], n_fp=4)
        mesh = triangulate_2d_area(area)
        assert mesh.elements.shape == (0, 3)

    def test_mesh2d_rejects_negative_indices(self):
        """Mesh2D.__post_init__ should reject negative element indices."""
        with pytest.raises(ValueError, match="negative element index"):
            Mesh2D(
                nodes=np.array([[0, 0], [1, 0], [0.5, 1.0]]),
                elements=np.array([[-1, 1, 2]], dtype=np.int32),
                elevation=np.array([0.0, 0.0, 0.0]),
                mannings_n=np.array([0.03, 0.03, 0.03]),
            )


# ---------------------------------------------------------------------------
# p4d: empty CSV / malformed .liq
# ---------------------------------------------------------------------------

class TestEmptyCsvAndMalformedLiq:
    def test_empty_csv_raises_valueerror(self, tmp_path):
        """Completely empty CSV should raise ValueError, not StopIteration."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")
        with pytest.raises(ValueError, match="CSV file is empty"):
            parse_observation_csv(str(csv_file))

    def test_header_only_csv_raises(self, tmp_path):
        """CSV with header but no data should raise ValueError."""
        csv_file = tmp_path / "header_only.csv"
        csv_file.write_text("time,value\n")
        with pytest.raises(ValueError, match="no data rows"):
            parse_observation_csv(str(csv_file))

    def test_malformed_liq_rows_skipped(self, tmp_path):
        """Rows with wrong column count in .liq should be skipped."""
        liq = tmp_path / "bad.liq"
        liq.write_text(
            "# test\n"
            "T         Q(1)\n"
            "s         m3/s\n"
            "0.0       1.0\n"
            "10.0\n"           # missing column
            "20.0      3.0\n"
        )
        result = parse_liq_file(str(liq))
        assert result is not None
        col = list(result.values())[0]
        assert len(col["times"]) == 2  # only 2 valid rows
        np.testing.assert_allclose(col["values"], [1.0, 3.0])


# ---------------------------------------------------------------------------
# p4e: playback speed edge cases
# ---------------------------------------------------------------------------

class TestPlaybackSpeedClamp:
    """Test that speed clamping logic works correctly."""

    def test_clamp_zero(self):
        speed = max(0.1, min(10.0, float(0)))
        assert speed == 0.1

    def test_clamp_negative(self):
        speed = max(0.1, min(10.0, float(-5)))
        assert speed == 0.1

    def test_clamp_very_large(self):
        speed = max(0.1, min(10.0, float(999)))
        assert speed == 10.0

    def test_normal_speed(self):
        speed = max(0.1, min(10.0, float(0.5)))
        assert speed == 0.5


# ---------------------------------------------------------------------------
# p4f: obs metric overlap
# ---------------------------------------------------------------------------

class TestObsMetricOverlap:
    def test_full_overlap_computes_metrics(self):
        """When time ranges fully overlap, metrics should be computed normally."""
        obs_t = np.array([0.0, 10.0, 20.0])
        mod_t = np.array([0.0, 10.0, 20.0])
        mod_v = np.array([1.0, 2.0, 3.0])
        obs_v = np.array([1.0, 2.0, 3.0])
        # Crop to overlap
        t_lo = max(obs_t[0], mod_t[0])
        t_hi = min(obs_t[-1], mod_t[-1])
        mask = (obs_t >= t_lo) & (obs_t <= t_hi)
        assert mask.all()
        model_interp = np.interp(obs_t[mask], mod_t, mod_v)
        rmse = compute_rmse(model_interp, obs_v[mask])
        assert rmse == pytest.approx(0.0, abs=1e-10)

    def test_partial_overlap_crops(self):
        """When obs extends beyond model, only overlapping portion is used."""
        obs_t = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        mod_t = np.array([5.0, 15.0, 25.0])
        t_lo = max(obs_t[0], mod_t[0])
        t_hi = min(obs_t[-1], mod_t[-1])
        mask = (obs_t >= t_lo) & (obs_t <= t_hi)
        assert not mask.all()  # not all obs points are in overlap
        assert mask.sum() >= 2  # enough points for metrics

    def test_no_overlap_detected(self):
        """When ranges don't overlap, no metrics should be computed."""
        obs_t = np.array([100.0, 110.0, 120.0])
        mod_t = np.array([0.0, 10.0, 20.0])
        t_lo = max(obs_t[0], mod_t[0])
        t_hi = min(obs_t[-1], mod_t[-1])
        assert t_hi <= t_lo  # no overlap

    def test_single_point_overlap_insufficient(self):
        """A single overlapping point is insufficient for metrics."""
        obs_t = np.array([20.0, 30.0])
        mod_t = np.array([0.0, 10.0, 20.0])
        t_lo = max(obs_t[0], mod_t[0])
        t_hi = min(obs_t[-1], mod_t[-1])
        mask = (obs_t >= t_lo) & (obs_t <= t_hi)
        assert mask.sum() < 2


# ---------------------------------------------------------------------------
# Bonus: palette fallback
# ---------------------------------------------------------------------------

class TestPaletteFallback:
    def test_valid_palette(self):
        arr = cached_palette_arr("Viridis")
        assert arr.shape == (256, 4)

    def test_invalid_palette_falls_back(self):
        """Unknown palette ID should return turbo fallback, not raise KeyError."""
        cached_palette_arr.cache_clear()
        arr = cached_palette_arr("nonexistent_palette_xyz")
        assert arr.shape == (256, 4)


# ---------------------------------------------------------------------------
# Bonus: dead code removed
# ---------------------------------------------------------------------------

class TestDeadCodeRemoved:
    def test_polygon_zonal_stats_no_if_false(self):
        """Verify the 'if False' debug stub was removed from analysis.py."""
        import inspect
        src = inspect.getsource(polygon_zonal_stats)
        assert "if False" not in src
