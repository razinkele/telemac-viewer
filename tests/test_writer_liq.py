"""Tests for the .liq writer and round-trip with the parser."""
from __future__ import annotations
import os
import tempfile
import numpy as np
import pytest

from telemac_tools.model import BoundaryCondition, BCType
from telemac_tools.telemac.writer_liq import write_liq, _liq_column_name
from validation import parse_liq_file


# ---------------------------------------------------------------------------
# Column naming
# ---------------------------------------------------------------------------

class TestLiqColumnName:
    def test_flow_bc(self):
        bc = BoundaryCondition(bc_type=BCType.FLOW, location="upstream")
        header, unit = _liq_column_name(bc, 0)
        assert header == "Q(1)"
        assert unit == "m3/s"

    def test_stage_bc(self):
        bc = BoundaryCondition(bc_type=BCType.STAGE, location="downstream")
        header, unit = _liq_column_name(bc, 1)
        assert header == "SL(2)"
        assert unit == "m"

    def test_unknown_defaults_to_flow(self):
        bc = BoundaryCondition(bc_type=BCType.UNKNOWN, location="x")
        header, unit = _liq_column_name(bc, 2)
        assert header == "Q(3)"
        assert unit == "m3/s"


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class TestWriteLiq:
    def test_no_timeseries_returns_false(self, tmp_path):
        bcs = [BoundaryCondition(bc_type=BCType.FLOW, location="up")]
        path = str(tmp_path / "test.liq")
        assert write_liq(bcs, path) is False
        assert not os.path.exists(path)

    def test_single_bc(self, tmp_path):
        bc = BoundaryCondition(
            bc_type=BCType.FLOW,
            location="upstream",
            timeseries={
                "time": np.array([0.0, 3600.0, 7200.0]),
                "values": np.array([100.0, 200.0, 150.0]),
                "unit": "m3/s",
            },
        )
        path = str(tmp_path / "test.liq")
        assert write_liq([bc], path) is True
        assert os.path.exists(path)

        content = open(path).read()
        lines = [l for l in content.splitlines() if not l.startswith("#")]
        assert "T" in lines[0]
        assert "Q(1)" in lines[0]
        assert "s" in lines[1]
        assert "m3/s" in lines[1]
        assert len(lines) == 5  # header + unit + 3 data rows

    def test_two_bcs(self, tmp_path):
        bcs = [
            BoundaryCondition(
                bc_type=BCType.FLOW, location="upstream",
                timeseries={"time": np.array([0.0, 100.0]), "values": np.array([50.0, 60.0]), "unit": "m3/s"},
            ),
            BoundaryCondition(
                bc_type=BCType.STAGE, location="downstream",
                timeseries={"time": np.array([0.0, 100.0]), "values": np.array([1.5, 1.4]), "unit": "m"},
            ),
        ]
        path = str(tmp_path / "test.liq")
        assert write_liq(bcs, path) is True
        content = open(path).read()
        assert "Q(1)" in content
        assert "SL(2)" in content

    def test_mixed_bc_with_and_without_timeseries(self, tmp_path):
        bcs = [
            BoundaryCondition(bc_type=BCType.FLOW, location="up"),  # no TS
            BoundaryCondition(
                bc_type=BCType.STAGE, location="down",
                timeseries={"time": np.array([0.0, 60.0]), "values": np.array([2.0, 2.5]), "unit": "m"},
            ),
        ]
        path = str(tmp_path / "test.liq")
        assert write_liq(bcs, path) is True
        content = open(path).read()
        # Only the one with timeseries should appear (indexed as first ts BC)
        assert "SL(1)" in content


# ---------------------------------------------------------------------------
# Round-trip: write then parse
# ---------------------------------------------------------------------------

class TestWriteParseRoundTrip:
    def test_roundtrip_single(self, tmp_path):
        times = np.array([0.0, 1800.0, 3600.0])
        values = np.array([500.0, 450.0, 400.0])
        bc = BoundaryCondition(
            bc_type=BCType.FLOW, location="upstream",
            timeseries={"time": times, "values": values, "unit": "m3/s"},
        )
        path = str(tmp_path / "roundtrip.liq")
        write_liq([bc], path)

        parsed = parse_liq_file(path)
        assert parsed is not None
        assert "Q(1)" in parsed
        np.testing.assert_allclose(parsed["Q(1)"]["times"], times, atol=0.1)
        np.testing.assert_allclose(parsed["Q(1)"]["values"], values, atol=0.001)
        assert parsed["Q(1)"]["unit"] == "m3/s"

    def test_roundtrip_two_bcs(self, tmp_path):
        bcs = [
            BoundaryCondition(
                bc_type=BCType.FLOW, location="upstream",
                timeseries={"time": np.array([0.0, 60.0]), "values": np.array([100.0, 120.0]), "unit": "m3/s"},
            ),
            BoundaryCondition(
                bc_type=BCType.STAGE, location="downstream",
                timeseries={"time": np.array([0.0, 60.0]), "values": np.array([5.0, 4.8]), "unit": "m"},
            ),
        ]
        path = str(tmp_path / "roundtrip2.liq")
        write_liq(bcs, path)

        parsed = parse_liq_file(path)
        assert parsed is not None
        assert "Q(1)" in parsed
        assert "SL(2)" in parsed
        np.testing.assert_allclose(parsed["Q(1)"]["values"], [100.0, 120.0], atol=0.001)
        np.testing.assert_allclose(parsed["SL(2)"]["values"], [5.0, 4.8], atol=0.001)

    def test_roundtrip_interpolated_times(self, tmp_path):
        """When BCs have different time arrays, writer interpolates onto union."""
        bcs = [
            BoundaryCondition(
                bc_type=BCType.FLOW, location="upstream",
                timeseries={"time": np.array([0.0, 100.0]), "values": np.array([10.0, 20.0]), "unit": "m3/s"},
            ),
            BoundaryCondition(
                bc_type=BCType.STAGE, location="downstream",
                timeseries={"time": np.array([0.0, 50.0, 100.0]), "values": np.array([1.0, 1.5, 2.0]), "unit": "m"},
            ),
        ]
        path = str(tmp_path / "interp.liq")
        write_liq(bcs, path)

        parsed = parse_liq_file(path)
        assert parsed is not None
        # Common times should be union: 0, 50, 100
        assert len(parsed["Q(1)"]["times"]) == 3
        # Q at t=50 should be interpolated: 15.0
        np.testing.assert_allclose(parsed["Q(1)"]["values"][1], 15.0, atol=0.1)
