# crs.py
"""Coordinate Reference System transforms for TELEMAC Viewer.

Single source of truth for all CRS operations: EPSG lookup,
native↔WGS84 transforms, .cas auto-detection, click conversion.
"""
from __future__ import annotations
from dataclasses import dataclass
from pyproj import Transformer, CRS as ProjCRS
from constants import _M2D


@dataclass
class CRS:
    epsg: int
    name: str
    transformer: Transformer      # native → WGS84
    inv_transformer: Transformer   # WGS84 → native


def crs_from_epsg(code: int) -> CRS:
    """Create CRS from EPSG code. Raises pyproj.exceptions.CRSError for invalid codes."""
    proj_crs = ProjCRS.from_epsg(code)
    fwd = Transformer.from_crs(code, 4326, always_xy=True)
    inv = Transformer.from_crs(4326, code, always_xy=True)
    return CRS(epsg=code, name=proj_crs.name, transformer=fwd, inv_transformer=inv)


def native_to_wgs84(x: float, y: float, crs: CRS) -> tuple[float, float]:
    """Transform native CRS coordinates to WGS84 (lon, lat)."""
    return crs.transformer.transform(x, y)


def wgs84_to_native(lon: float, lat: float, crs: CRS) -> tuple[float, float]:
    """Transform WGS84 (lon, lat) to native CRS coordinates."""
    return crs.inv_transformer.transform(lon, lat)
