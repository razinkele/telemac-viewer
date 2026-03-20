# crs.py
"""Coordinate Reference System transforms for TELEMAC Viewer.

Single source of truth for all CRS operations: EPSG lookup,
native↔WGS84 transforms, .cas auto-detection, click conversion.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from numpy import ndarray
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


# --- .cas file detection ---

# TELEMAC GEOGRAPHIC SYSTEM → EPSG mapping
_LAMBERT_ZONES = {1: 27561, 2: 27562, 3: 27563, 4: 27564, 93: 2154}

_GEOSYST_KEYWORDS = re.compile(
    r'(?:GEOGRAPHIC SYSTEM|SYSTEME GEOGRAPHIQUE)\s*[:=]\s*(-?\d+)',
    re.IGNORECASE,
)
_NUMZONE_KEYWORDS = re.compile(
    r'(?:ZONE NUMBER IN GEOGRAPHIC SYSTEM|NUMERO DE ZONE DU SYSTEME GEOGRAPHIQUE)\s*[:=]\s*(\d+)',
    re.IGNORECASE,
)


def _geosyst_to_epsg(geosyst: int, numzone: int) -> int | None:
    """Map TELEMAC GEOGRAPHIC SYSTEM code to EPSG."""
    if geosyst == 1:
        return 4326
    if geosyst == 2:
        return 32600 + numzone  # UTM North
    if geosyst == 3:
        return 32700 + numzone  # UTM South
    if geosyst == 4:
        return _LAMBERT_ZONES.get(numzone)
    if geosyst == 5:
        return 3395
    return None


def detect_crs_from_cas(cas_path: str) -> CRS | None:
    """Parse TELEMAC .cas file for GEOGRAPHIC SYSTEM and ZONE NUMBER.

    Returns CRS if detected, None otherwise.
    """
    try:
        with open(cas_path) as f:
            text = f.read()
    except (OSError, IOError):
        return None

    # Strip comments (everything after / on each line)
    lines = []
    for line in text.split('\n'):
        idx = line.find('/')
        if idx >= 0:
            line = line[:idx]
        lines.append(line)
    clean = '\n'.join(lines)

    geo_match = _GEOSYST_KEYWORDS.search(clean)
    if not geo_match:
        return None

    geosyst = int(geo_match.group(1))
    zone_match = _NUMZONE_KEYWORDS.search(clean)
    numzone = int(zone_match.group(1)) if zone_match else 0

    epsg = _geosyst_to_epsg(geosyst, numzone)
    if epsg is None:
        return None

    try:
        return crs_from_epsg(epsg)
    except Exception:
        return None


# --- Coordinate heuristic ---

def guess_crs_from_coords(x: ndarray, y: ndarray) -> CRS | None:
    """Heuristic CRS guess from coordinate ranges.

    Returns a CRS suggestion or None if indeterminate.
    """
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    # WGS84 cannot be reliably detected from coordinates alone — small lab
    # models (0-100m) are indistinguishable from geographic degrees.
    # WGS84 is only set via .cas file detection or manual EPSG entry.

    # LKS94 (Lithuania) — check before general UTM
    if 300_000 <= x_min and x_max <= 700_000 and 5_950_000 <= y_min and y_max <= 6_250_000:
        return crs_from_epsg(3346)

    # General UTM range — too ambiguous without more info
    return None


# --- Click conversion and display helpers ---

def click_to_native(lon: float, lat: float, geom: dict) -> tuple[float, float]:
    """Convert deck.gl click coordinates to native CRS meters.

    When CRS is set, deck.gl reports real lon/lat — use pyproj inverse.
    When CRS is None, fall back to _M2D approximation (old behavior).
    """
    crs = geom.get("crs")
    if crs is not None:
        return wgs84_to_native(lon, lat, crs)
    return float(lon) * _M2D + geom["x_off"], float(lat) * _M2D + geom["y_off"]


def meters_to_wgs84(x_m: float, y_m: float, geom: dict) -> tuple[float, float] | None:
    """Convert native CRS meters to WGS84 for display. Returns None if no CRS."""
    crs = geom.get("crs")
    if crs is None:
        return None
    return native_to_wgs84(x_m, y_m, crs)
