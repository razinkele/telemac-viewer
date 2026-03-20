# tests/test_crs.py
"""Tests for crs.py — coordinate reference system transforms."""
from __future__ import annotations
import numpy as np
import pytest
from crs import CRS, crs_from_epsg, native_to_wgs84, wgs84_to_native


class TestCrsFromEpsg:
    def test_lks94(self):
        crs = crs_from_epsg(3346)
        assert crs.epsg == 3346
        assert "LKS" in crs.name or "Lithuania" in crs.name

    def test_utm_north(self):
        crs = crs_from_epsg(32634)
        assert crs.epsg == 32634

    def test_wgs84(self):
        crs = crs_from_epsg(4326)
        assert crs.epsg == 4326

    def test_invalid_raises(self):
        with pytest.raises(Exception):
            crs_from_epsg(99999)


class TestTransforms:
    def test_lks94_to_wgs84(self):
        crs = crs_from_epsg(3346)
        lon, lat = native_to_wgs84(500000, 6100000, crs)
        assert 23.0 < lon < 25.0
        assert 54.5 < lat < 56.0

    def test_roundtrip(self):
        crs = crs_from_epsg(3346)
        x_orig, y_orig = 500000.0, 6100000.0
        lon, lat = native_to_wgs84(x_orig, y_orig, crs)
        x_back, y_back = wgs84_to_native(lon, lat, crs)
        assert x_back == pytest.approx(x_orig, abs=0.01)
        assert y_back == pytest.approx(y_orig, abs=0.01)

    def test_wgs84_identity(self):
        """WGS84 CRS should be near-identity transform."""
        crs = crs_from_epsg(4326)
        lon, lat = native_to_wgs84(24.0, 55.0, crs)
        assert lon == pytest.approx(24.0, abs=0.001)
        assert lat == pytest.approx(55.0, abs=0.001)
