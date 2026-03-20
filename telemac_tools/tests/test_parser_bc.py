"""Tests for BC time series parsing."""
from __future__ import annotations
import numpy as np
import pytest
from telemac_tools.hecras.parser_bc import parse_bc_timeseries


class TestParseBcTimeseries:
    def test_returns_list(self, hdf_unsteady):
        bcs = parse_bc_timeseries(hdf_unsteady)
        assert isinstance(bcs, list)
        assert len(bcs) == 2

    def test_flow_hydrograph(self, hdf_unsteady):
        bcs = parse_bc_timeseries(hdf_unsteady)
        upstream = [bc for bc in bcs if bc.location == "upstream"][0]
        assert upstream.timeseries is not None
        assert "time" in upstream.timeseries
        assert "values" in upstream.timeseries
        assert len(upstream.timeseries["time"]) == 5
        assert upstream.bc_type == "flow"

    def test_stage_hydrograph(self, hdf_unsteady):
        bcs = parse_bc_timeseries(hdf_unsteady)
        downstream = [bc for bc in bcs if bc.location == "downstream"][0]
        assert downstream.timeseries is not None
        assert downstream.bc_type == "stage"

    def test_no_unsteady_returns_empty(self, tmp_path):
        import h5py
        path = tmp_path / "no_bc.hdf"
        with h5py.File(path, "w") as f:
            f.create_group("Geometry")
        bcs = parse_bc_timeseries(str(path))
        assert bcs == []
