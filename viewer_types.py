"""Shared types for the TELEMAC Viewer."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol, Sequence
import numpy as np


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


class TelemacFileProtocol(Protocol):
    """Structural type for TelemacFile-like objects."""
    meshx: np.ndarray
    meshy: np.ndarray
    npoin2: int
    nelem2: int
    ikle2: np.ndarray
    varnames: list[str]
    times: Sequence[float]
    nplan: int

    def get_data_value(self, varname: str, tidx: int) -> np.ndarray: ...
    def get_z_name(self) -> str: ...
    def get_timeseries_on_points(self, varname: str, points: list) -> list: ...
    def get_data_on_points(self, varname: str, tidx: int, points: list) -> list: ...
    def get_data_on_polyline(self, varname: str, tidx: int, polyline: list) -> tuple: ...
    def close(self) -> None: ...
