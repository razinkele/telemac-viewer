"""Tests for 1D HEC-RAS geometry parsing."""
from __future__ import annotations
import numpy as np
import pytest
from telemac_tools.hecras.parser_1d import parse_hecras_1d
from telemac_tools.model import BCType, HecRasModel, HecRasParseError


class TestParseHecras1d:
    def test_returns_model(self, hdf_1d):
        model = parse_hecras_1d(hdf_1d)
        assert isinstance(model, HecRasModel)

    def test_one_reach(self, hdf_1d):
        model = parse_hecras_1d(hdf_1d)
        assert len(model.rivers) == 1
        assert model.rivers[0].name == "River1/Reach1"

    def test_three_cross_sections(self, hdf_1d):
        model = parse_hecras_1d(hdf_1d)
        xs_list = model.rivers[0].cross_sections
        assert len(xs_list) == 3

    def test_cross_section_stations(self, hdf_1d):
        model = parse_hecras_1d(hdf_1d)
        stations = [xs.station for xs in model.rivers[0].cross_sections]
        assert stations == [0.0, 100.0, 200.0]

    def test_cross_section_coords_shape(self, hdf_1d):
        model = parse_hecras_1d(hdf_1d)
        xs = model.rivers[0].cross_sections[0]
        assert xs.coords.shape == (5, 3)

    def test_mannings_n(self, hdf_1d):
        model = parse_hecras_1d(hdf_1d)
        xs = model.rivers[0].cross_sections[0]
        assert xs.mannings_n == [0.06, 0.035, 0.06]

    def test_bank_stations(self, hdf_1d):
        model = parse_hecras_1d(hdf_1d)
        xs = model.rivers[0].cross_sections[0]
        assert xs.bank_stations == (25.0, 75.0)

    def test_alignment_shape(self, hdf_1d):
        model = parse_hecras_1d(hdf_1d)
        assert model.rivers[0].alignment.shape == (4, 2)

    def test_boundary_conditions(self, hdf_1d):
        model = parse_hecras_1d(hdf_1d)
        assert len(model.boundaries) == 2
        assert model.boundaries[0].bc_type == BCType.FLOW
        assert model.boundaries[1].bc_type == BCType.NORMAL_DEPTH

    def test_bc_line_coords(self, hdf_1d):
        model = parse_hecras_1d(hdf_1d)
        assert model.boundaries[0].line_coords.shape == (2, 2)

    def test_invalid_file_raises(self, tmp_path):
        import h5py
        bad = tmp_path / "empty.hdf"
        with h5py.File(bad, "w") as f:
            f.create_group("NoGeometry")
        with pytest.raises(HecRasParseError):
            parse_hecras_1d(str(bad))
