"""Tests for 2D HEC-RAS geometry parsing."""
from __future__ import annotations
import numpy as np
import pytest
from telemac_tools.hecras.parser_2d import parse_hecras_2d
from telemac_tools.model import HecRasModel, Mesh2D, HecRasParseError


class TestParseHecras2d:
    def test_returns_model(self, hdf_2d):
        model = parse_hecras_2d(hdf_2d)
        assert isinstance(model, HecRasModel)

    def test_one_2d_area(self, hdf_2d):
        model = parse_hecras_2d(hdf_2d)
        assert len(model.areas_2d) == 1
        assert model.areas_2d[0].name == "TestArea"

    def test_face_points(self, hdf_2d):
        model = parse_hecras_2d(hdf_2d)
        assert model.areas_2d[0].face_points.shape == (9, 2)

    def test_cell_centers(self, hdf_2d):
        model = parse_hecras_2d(hdf_2d)
        assert model.areas_2d[0].cell_centers.shape == (4, 2)

    def test_four_cells(self, hdf_2d):
        model = parse_hecras_2d(hdf_2d)
        assert len(model.areas_2d[0].cells) == 4

    def test_cell_face_point_count(self, hdf_2d):
        model = parse_hecras_2d(hdf_2d)
        for cell in model.areas_2d[0].cells:
            assert len(cell.face_point_indices) == 4

    def test_elevation(self, hdf_2d):
        model = parse_hecras_2d(hdf_2d)
        area = model.areas_2d[0]
        assert area.elevation is not None
        assert len(area.elevation) == 4

    def test_invalid_file_raises(self, tmp_path):
        import h5py
        bad = tmp_path / "empty.hdf"
        with h5py.File(bad, "w") as f:
            f.create_group("Geometry")
        with pytest.raises(HecRasParseError):
            parse_hecras_2d(str(bad))


class TestVoronoiToTriangles:
    def test_triangulate_returns_mesh(self, hdf_2d):
        from telemac_tools.hecras.parser_2d import triangulate_2d_area
        model = parse_hecras_2d(hdf_2d)
        mesh = triangulate_2d_area(model.areas_2d[0])
        assert isinstance(mesh, Mesh2D)
        assert mesh.nodes.shape[1] == 2
        assert mesh.elements.shape[1] == 3

    def test_triangles_cover_cells(self, hdf_2d):
        from telemac_tools.hecras.parser_2d import triangulate_2d_area
        model = parse_hecras_2d(hdf_2d)
        mesh = triangulate_2d_area(model.areas_2d[0])
        # 4 square cells -> fan triangulation: 4 triangles per cell = 16
        assert mesh.elements.shape[0] >= 4

    def test_elevation_interpolated(self, hdf_2d):
        from telemac_tools.hecras.parser_2d import triangulate_2d_area
        model = parse_hecras_2d(hdf_2d)
        mesh = triangulate_2d_area(model.areas_2d[0])
        assert len(mesh.elevation) == mesh.nodes.shape[0]
        assert not np.any(np.isnan(mesh.elevation))
