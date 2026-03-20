"""Tests for domain builder (geometry + DEM -> TelemacDomain)."""
from __future__ import annotations
import numpy as np
import pytest
from telemac_tools.domain.builder import build_domain_1d, build_domain_2d, sample_dem
from telemac_tools.model import TelemacDomain


class TestBuildDomain1d:
    def test_returns_domain(self, hdf_1d, fake_dem):
        from telemac_tools.hecras.parser_1d import parse_hecras_1d
        model = parse_hecras_1d(hdf_1d)
        domain = build_domain_1d(model, dem_path=fake_dem, floodplain_width=80.0)
        assert isinstance(domain, TelemacDomain)

    def test_boundary_polygon_closed(self, hdf_1d, fake_dem):
        from telemac_tools.hecras.parser_1d import parse_hecras_1d
        model = parse_hecras_1d(hdf_1d)
        domain = build_domain_1d(model, dem_path=fake_dem, floodplain_width=80.0)
        poly = domain.boundary_polygon
        assert poly.shape[1] == 2
        assert np.allclose(poly[0], poly[-1])

    def test_has_channel_points(self, hdf_1d, fake_dem):
        from telemac_tools.hecras.parser_1d import parse_hecras_1d
        model = parse_hecras_1d(hdf_1d)
        domain = build_domain_1d(model, dem_path=fake_dem, floodplain_width=80.0)
        assert domain.channel_points is not None
        assert domain.channel_points.shape[1] == 3

    def test_has_bc_segments(self, hdf_1d, fake_dem):
        from telemac_tools.hecras.parser_1d import parse_hecras_1d
        model = parse_hecras_1d(hdf_1d)
        domain = build_domain_1d(model, dem_path=fake_dem, floodplain_width=80.0)
        assert len(domain.bc_segments) >= 1

    def test_dem_loaded(self, hdf_1d, fake_dem):
        from telemac_tools.hecras.parser_1d import parse_hecras_1d
        model = parse_hecras_1d(hdf_1d)
        domain = build_domain_1d(model, dem_path=fake_dem, floodplain_width=80.0)
        assert domain._dem_data is not None
        assert domain._dem_transform is not None

    def test_sample_dem(self, hdf_1d, fake_dem):
        from telemac_tools.hecras.parser_1d import parse_hecras_1d
        model = parse_hecras_1d(hdf_1d)
        domain = build_domain_1d(model, dem_path=fake_dem, floodplain_width=80.0)
        z = sample_dem(np.array([50.0]), np.array([0.0]),
                       domain._dem_data, domain._dem_transform)
        assert z[0] == pytest.approx(5.0, abs=0.5)


class TestBuildDomain2d:
    def test_returns_domain(self, hdf_2d):
        from telemac_tools.hecras.parser_2d import parse_hecras_2d
        model = parse_hecras_2d(hdf_2d)
        domain = build_domain_2d(model)
        assert isinstance(domain, TelemacDomain)

    def test_boundary_from_mesh(self, hdf_2d):
        from telemac_tools.hecras.parser_2d import parse_hecras_2d
        model = parse_hecras_2d(hdf_2d)
        domain = build_domain_2d(model)
        assert domain.boundary_polygon.shape[1] == 2
