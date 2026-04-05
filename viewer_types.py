"""Shared types for the TELEMAC Viewer."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MeshGeometry:
    """Mesh geometry for deck.gl rendering.

    Produced by build_mesh_geometry(), consumed by all layer builders
    and analysis functions.
    """
    npoin: int
    positions: dict[str, Any]   # binary-encoded for deck.gl
    indices: dict[str, Any]     # binary-encoded for deck.gl
    x_off: float
    y_off: float
    lon_off: float
    lat_off: float
    crs: Any                    # CRS | None
    extent_m: float
    zoom: float
