"""Tests for Gmsh mesh generation backend."""
from __future__ import annotations
import numpy as np
import pytest
from telemac_tools.meshing import generate_mesh
from telemac_tools.model import TelemacDomain, Mesh2D, BCSegment


def _simple_domain():
    poly = np.array([[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]], dtype=np.float64)
    return TelemacDomain(boundary_polygon=poly, bc_segments=[BCSegment(node_indices=[], lihbor=2)])


class TestGmshBackend:
    def test_returns_mesh2d(self):
        mesh = generate_mesh(_simple_domain(), backend="gmsh")
        assert isinstance(mesh, Mesh2D)

    def test_has_triangles(self):
        mesh = generate_mesh(_simple_domain(), backend="gmsh")
        assert mesh.elements.shape[1] == 3
        assert mesh.elements.shape[0] > 0

    def test_nodes_inside_domain(self):
        mesh = generate_mesh(_simple_domain(), backend="gmsh")
        assert mesh.nodes[:, 0].min() >= -1
        assert mesh.nodes[:, 0].max() <= 101

    def test_elevation_assigned(self):
        mesh = generate_mesh(_simple_domain(), backend="gmsh")
        assert len(mesh.elevation) == mesh.nodes.shape[0]

    def test_mannings_assigned(self):
        mesh = generate_mesh(_simple_domain(), backend="gmsh")
        assert len(mesh.mannings_n) == mesh.nodes.shape[0]
        assert np.all(mesh.mannings_n > 0)

    def test_max_area_affects_density(self):
        coarse = generate_mesh(_simple_domain(), backend="gmsh", max_area=500)
        fine = generate_mesh(_simple_domain(), backend="gmsh", max_area=50)
        assert fine.elements.shape[0] > coarse.elements.shape[0]

    def test_with_channel_constraints(self):
        domain = _simple_domain()
        domain.channel_points = np.array([[50, 0, -1], [50, 50, -2], [50, 100, -3]], dtype=np.float64)
        domain.channel_segments = np.array([[0, 1], [1, 2]], dtype=np.int32)
        mesh = generate_mesh(domain, backend="gmsh")
        assert mesh.elements.shape[0] > 0
        assert mesh.elevation.min() < 0
