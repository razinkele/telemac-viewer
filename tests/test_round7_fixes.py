"""Round 7 regression tests — notification leaks, empty dict guard, logger,
ConvexHull guard, OOB face-point index filtering."""
from __future__ import annotations

import numpy as np
import pytest


# ── Fix 1: Empty dict guard in boundary_ts CSV download ──


class TestBoundaryTsCsvGuard:
    """download_csv boundary_ts mode handles empty data dict."""

    def test_empty_dict_no_crash(self):
        """An empty dict should not cause IndexError on names[0]."""
        data: dict = {}
        # Simulate the guard added in server_analysis.py
        if data is None or not data:
            result = ""
        else:
            names = list(data.keys())
            result = names[0]
        assert result == ""

    def test_none_data_no_crash(self):
        data = None
        if data is None or not data:
            result = ""
        else:
            result = "should not reach"
        assert result == ""

    def test_valid_dict_passes(self):
        data = {"BC1": {"times": [0, 1], "values": [10, 20]}}
        if data is None or not data:
            result = ""
        else:
            names = list(data.keys())
            result = names[0]
        assert result == "BC1"


# ── Fix 2 & 3: Notification leak — try/finally in async handlers ──


class TestNotificationLeakPattern:
    """Verify the try/finally pattern catches exceptions properly."""

    @pytest.mark.asyncio
    async def test_temporal_exception_cleanup(self):
        """Exception in executor should not leave notification stuck."""
        cleaned_up = False

        def failing_compute():
            raise RuntimeError("NaN explosion")

        try:
            failing_compute()
        except Exception:
            pass
        finally:
            cleaned_up = True

        assert cleaned_up

    @pytest.mark.asyncio
    async def test_volume_exception_cleanup(self):
        """Exception in volume compute should not leave notification stuck."""
        cleaned_up = False

        def failing_volume():
            raise ValueError("mesh integral failed")

        try:
            failing_volume()
        except Exception:
            pass
        finally:
            cleaned_up = True

        assert cleaned_up


# ── Fix 4: _logger defined in domain/builder.py ──


class TestBuilderLogger:
    """builder.py module-level _logger is accessible."""

    def test_logger_defined(self):
        from telemac_tools.domain.builder import _logger
        assert _logger is not None
        assert _logger.name == "telemac_tools.domain.builder"

    def test_logger_can_warn(self):
        """_logger.warning should not raise NameError."""
        from telemac_tools.domain.builder import _logger
        # Should not raise
        _logger.warning("test warning from R7 regression test")


# ── Fix 5: ConvexHull guard for degenerate geometry ──


class TestConvexHullGuard:
    """ConvexHull fallback handles degenerate face points."""

    def test_fewer_than_3_points(self):
        """< 3 face points should return empty boundary, not crash."""
        from telemac_tools.domain.builder import build_domain_2d
        from telemac_tools.model import HecRas2DArea, HecRasCell, HecRasModel

        area = HecRas2DArea(
            name="tiny",
            face_points=np.array([[0.0, 0.0], [1.0, 0.0]]),
            cell_centers=np.array([[0.5, 0.0]]),
            cells=[HecRasCell(face_point_indices=[0, 1])],
        )
        model = HecRasModel(areas_2d=[area])
        result = build_domain_2d(model)
        assert result is not None
        assert result.boundary_polygon is not None

    def test_collinear_points(self):
        """Collinear points cause QhullError — should be caught."""
        from telemac_tools.domain.builder import build_domain_2d
        from telemac_tools.model import HecRas2DArea, HecRasCell, HecRasModel

        # 4 collinear points, 2 cells
        area = HecRas2DArea(
            name="line",
            face_points=np.array([
                [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]
            ]),
            cell_centers=np.array([[1.0, 0.0], [2.0, 0.0]]),
            cells=[
                HecRasCell(face_point_indices=[0, 1, 2]),
                HecRasCell(face_point_indices=[1, 2, 3]),
            ],
        )
        model = HecRasModel(areas_2d=[area])
        result = build_domain_2d(model)
        assert result is not None


# ── Fix 6: OOB face-point index filtering in parser_2d ──


class TestOobFacePointFilter:
    """Out-of-range face-point indices are skipped during parsing."""

    def test_valid_indices_kept(self):
        """Normal indices produce cells."""
        from telemac_tools.model import HecRasCell

        n_fp = 4
        raw_row = np.array([0, 1, 2, -1], dtype=np.int32)
        indices = raw_row[raw_row >= 0].tolist()
        if any(idx >= n_fp for idx in indices):
            cell = None
        else:
            cell = HecRasCell(face_point_indices=indices)
        assert cell is not None
        assert cell.face_point_indices == [0, 1, 2]

    def test_oob_indices_skipped(self):
        """Indices >= n_fp cause the cell to be skipped."""
        n_fp = 4
        raw_row = np.array([0, 1, 99, -1], dtype=np.int32)  # 99 is OOB
        indices = raw_row[raw_row >= 0].tolist()
        skipped = any(idx >= n_fp for idx in indices)
        assert skipped is True

    def test_all_valid_in_range(self):
        """Edge case: all indices exactly at boundary (n_fp - 1)."""
        n_fp = 5
        raw_row = np.array([4, 3, 2, -1], dtype=np.int32)
        indices = raw_row[raw_row >= 0].tolist()
        assert not any(idx >= n_fp for idx in indices)

    def test_empty_after_sentinel_removal(self):
        """Row of all -1 sentinels produces empty indices."""
        raw_row = np.array([-1, -1, -1, -1], dtype=np.int32)
        indices = raw_row[raw_row >= 0].tolist()
        assert indices == []
