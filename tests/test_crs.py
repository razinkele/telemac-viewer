# tests/test_crs.py
"""Tests for crs.py — coordinate reference system transforms."""
from __future__ import annotations
import numpy as np
import pytest
from crs import (
    CRS, crs_from_epsg, native_to_wgs84, wgs84_to_native,
    detect_crs_from_cas, guess_crs_from_coords,
    click_to_native, meters_to_wgs84,
)
from viewer_types import MeshGeometry


def _make_geom(**kwargs) -> MeshGeometry:
    """Build a minimal MeshGeometry for testing crs helpers."""
    defaults = dict(
        npoin=0, positions={}, indices={},
        x_off=0.0, y_off=0.0,
        lon_off=0.0, lat_off=0.0,
        crs=None, extent_m=1.0, zoom=1.0,
    )
    defaults.update(kwargs)
    return MeshGeometry(**defaults)


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


# --- Task 2: .cas detection and coordinate heuristic ---

class TestDetectCrsFromCas:
    def test_lambert_zone1(self, tmp_path):
        cas = tmp_path / "test.cas"
        cas.write_text(
            "TITLE = 'TEST'\n"
            "GEOGRAPHIC SYSTEM : 4\n"
            "ZONE NUMBER IN GEOGRAPHIC SYSTEM : 1\n"
        )
        crs = detect_crs_from_cas(str(cas))
        assert crs is not None
        assert crs.epsg == 27561

    def test_utm_north_zone34(self, tmp_path):
        cas = tmp_path / "test.cas"
        cas.write_text(
            "GEOGRAPHIC SYSTEM = 2\n"
            "ZONE NUMBER IN GEOGRAPHIC SYSTEM = 34\n"
        )
        crs = detect_crs_from_cas(str(cas))
        assert crs is not None
        assert crs.epsg == 32634

    def test_no_keywords_returns_none(self, tmp_path):
        cas = tmp_path / "test.cas"
        cas.write_text("TITLE = 'NO CRS'\nTIME STEP = 10\n")
        crs = detect_crs_from_cas(str(cas))
        assert crs is None

    def test_comments_stripped(self, tmp_path):
        cas = tmp_path / "test.cas"
        cas.write_text(
            "/ This is a comment\n"
            "GEOGRAPHIC SYSTEM = 1 / WGS84\n"
        )
        crs = detect_crs_from_cas(str(cas))
        assert crs is not None
        assert crs.epsg == 4326

    def test_lambert_zone93(self, tmp_path):
        """Zone 93 is RGF93 Lambert (different datum from zones 1-4)."""
        cas = tmp_path / "test.cas"
        cas.write_text(
            "GEOGRAPHIC SYSTEM = 4\n"
            "ZONE NUMBER IN GEOGRAPHIC SYSTEM = 93\n"
        )
        crs = detect_crs_from_cas(str(cas))
        assert crs is not None
        assert crs.epsg == 2154

    def test_french_keyword(self, tmp_path):
        cas = tmp_path / "test.cas"
        cas.write_text("SYSTEME GEOGRAPHIQUE : 5\n")
        crs = detect_crs_from_cas(str(cas))
        assert crs is not None
        assert crs.epsg == 3395

    def test_french_keyword_with_zone(self, tmp_path):
        cas = tmp_path / "test.cas"
        cas.write_text(
            "SYSTEME GEOGRAPHIQUE : 2\n"
            "NUMERO DE ZONE DU SYSTEME GEOGRAPHIQUE : 34\n"
        )
        crs = detect_crs_from_cas(str(cas))
        assert crs is not None
        assert crs.epsg == 32634

    def test_geosyst_minus1_returns_none(self, tmp_path):
        cas = tmp_path / "test.cas"
        cas.write_text("GEOGRAPHIC SYSTEM = -1\n")
        crs = detect_crs_from_cas(str(cas))
        assert crs is None

    def test_geosyst_zero_returns_none(self, tmp_path):
        cas = tmp_path / "test.cas"
        cas.write_text("GEOGRAPHIC SYSTEM = 0\n")
        crs = detect_crs_from_cas(str(cas))
        assert crs is None

    def test_real_tide_cas(self):
        """Test with actual TELEMAC tide example if available."""
        import os
        cas_path = os.path.join(
            os.environ.get("HOMETEL", ""),
            "examples/telemac2d/tide/t2d_tide_local-jmj_type_gen.cas",
        )
        if not os.path.isfile(cas_path):
            pytest.skip("Tide example not available")
        crs = detect_crs_from_cas(cas_path)
        assert crs is not None
        assert crs.epsg == 27561  # Lambert zone I


class TestGuessCrsFromCoords:
    def test_lks94_range(self):
        x = np.array([400000, 500000, 600000], dtype=np.float64)
        y = np.array([6000000, 6100000, 6200000], dtype=np.float64)
        crs = guess_crs_from_coords(x, y)
        assert crs is not None
        assert crs.epsg == 3346

    def test_wgs84_range_returns_none(self):
        """WGS84 cannot be detected from coordinates alone — too ambiguous."""
        x = np.array([20.0, 25.0, 30.0], dtype=np.float64)
        y = np.array([50.0, 55.0, 60.0], dtype=np.float64)
        crs = guess_crs_from_coords(x, y)
        assert crs is None

    def test_small_local_returns_none(self):
        """Small lab/flume model (e.g. Gouttedo 0-1m) must NOT trigger WGS84."""
        x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        y = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        crs = guess_crs_from_coords(x, y)
        assert crs is None

    def test_gouttedo_scale_returns_none(self):
        """Gouttedo-scale mesh (~100m) must not be mistaken for WGS84."""
        x = np.array([0.0, 50.0, 100.0], dtype=np.float64)
        y = np.array([0.0, 50.0, 100.0], dtype=np.float64)
        crs = guess_crs_from_coords(x, y)
        assert crs is None

    def test_gouttedo_real_scale_returns_none(self):
        """Actual Gouttedo mesh (0-20m) must not trigger WGS84."""
        x = np.array([0.0, 10.0, 20.1], dtype=np.float64)
        y = np.array([0.0, 10.0, 20.1], dtype=np.float64)
        crs = guess_crs_from_coords(x, y)
        assert crs is None


# --- Task 3: click conversion and display helpers ---

class TestClickToNative:
    def test_with_crs_roundtrip(self):
        """Round-trip: native → wgs84 → click_to_native should recover original."""
        crs = crs_from_epsg(3346)
        x_orig, y_orig = 500000.0, 6100000.0
        lon, lat = native_to_wgs84(x_orig, y_orig, crs)
        geom = _make_geom(x_off=x_orig, y_off=y_orig, crs=crs)
        x, y = click_to_native(lon, lat, geom)
        assert x == pytest.approx(x_orig, abs=1.0)
        assert y == pytest.approx(y_orig, abs=1.0)

    def test_without_crs_fallback(self):
        """Without CRS, should match old coord_to_meters behavior."""
        from constants import _M2D
        geom = _make_geom(x_off=100.0, y_off=200.0, crs=None)
        lon, lat = 0.001, 0.002
        x, y = click_to_native(lon, lat, geom)
        assert x == pytest.approx(lon * _M2D + 100.0)
        assert y == pytest.approx(lat * _M2D + 200.0)


class TestMetersToWgs84:
    def test_with_crs(self):
        crs = crs_from_epsg(3346)
        geom = _make_geom(crs=crs)
        result = meters_to_wgs84(500000, 6100000, geom)
        assert result is not None
        lon, lat = result
        assert 23.0 < lon < 25.0
        assert 54.5 < lat < 56.0

    def test_without_crs(self):
        geom = _make_geom(crs=None)
        result = meters_to_wgs84(100, 200, geom)
        assert result is None


class TestOriginOffset:
    def test_offset_shifts_lonlat(self):
        from geometry import build_mesh_geometry
        from tests.helpers import FakeTF
        tf = FakeTF()
        crs = crs_from_epsg(3346)
        geom = build_mesh_geometry(tf, crs=crs, origin_offset=(309424, 6132619))
        # Mesh center (0.5, 0.5) + offset -> (309424.5, 6132619.5) in LKS94
        # Should map to approximately 20.9E, 55.3N (Curonian Lagoon)
        assert 20.0 < geom.lon_off < 22.0
        assert 54.5 < geom.lat_off < 56.0

    def test_zero_offset_unchanged(self):
        from geometry import build_mesh_geometry
        from tests.helpers import FakeTF
        geom_default = build_mesh_geometry(FakeTF())
        geom_zero = build_mesh_geometry(FakeTF(), origin_offset=(0, 0))
        assert geom_default.x_off == geom_zero.x_off
        assert geom_default.y_off == geom_zero.y_off
