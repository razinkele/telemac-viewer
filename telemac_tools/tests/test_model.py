"""Tests for telemac_tools.model dataclasses."""
from __future__ import annotations
import numpy as np
import pytest
from telemac_tools.model import (
    CrossSection, Reach, BoundaryCondition, HecRasCell, HecRas2DArea,
    Mesh2D, HecRasModel, BCSegment, TelemacDomain, HecRasParseError,
)


class TestDataclasses:
    def test_cross_section(self):
        xs = CrossSection(
            station=100.0,
            coords=np.zeros((5, 3)),
            mannings_n=[0.06, 0.035, 0.06],
            bank_stations=(20.0, 80.0),
            bank_coords=np.zeros((2, 2)),
        )
        assert xs.station == 100.0
        assert len(xs.mannings_n) == 3

    def test_reach(self):
        r = Reach(name="River1", alignment=np.zeros((10, 2)))
        assert r.name == "River1"
        assert len(r.cross_sections) == 0

    def test_mesh2d(self):
        m = Mesh2D(
            nodes=np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64),
            elements=np.array([[0, 1, 2]], dtype=np.int32),
            elevation=np.array([0.0, 1.0, 0.5]),
            mannings_n=np.array([0.035, 0.035, 0.035]),
        )
        assert m.nodes.shape == (3, 2)
        assert m.elements.shape == (1, 3)

    def test_hecras_model_defaults(self):
        m = HecRasModel()
        assert m.rivers == []
        assert m.boundaries == []
        assert m.areas_2d == []

    def test_bc_segment(self):
        bc = BCSegment(node_indices=[1, 2, 3], lihbor=5, prescribed_h=2.0)
        assert bc.lihbor == 5

    def test_parse_error(self):
        with pytest.raises(HecRasParseError):
            raise HecRasParseError("Missing group")

    def test_mesh2d_rejects_mismatched_elevation(self):
        nodes = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
        elems = np.array([[0, 1, 2]], dtype=np.int32)
        with pytest.raises(ValueError, match="elevation length"):
            Mesh2D(nodes=nodes, elements=elems,
                   elevation=np.zeros(5),
                   mannings_n=np.full(3, 0.035))

    def test_mesh2d_rejects_out_of_bounds_element(self):
        nodes = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
        elems = np.array([[0, 1, 99]], dtype=np.int32)
        with pytest.raises(ValueError, match="element index"):
            Mesh2D(nodes=nodes, elements=elems,
                   elevation=np.zeros(3),
                   mannings_n=np.full(3, 0.035))
