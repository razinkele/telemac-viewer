"""Intermediate data model for HEC-RAS → TELEMAC pipeline."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, IntEnum
import numpy as np


class BCType(str, Enum):
    FLOW = "flow"
    STAGE = "stage"
    NORMAL_DEPTH = "normal_depth"
    RATING_CURVE = "rating_curve"
    UNKNOWN = "unknown"


class LIHBOR(IntEnum):
    WALL = 2
    FREE = 4
    PRESCRIBED = 5


@dataclass
class CrossSection:
    station: float
    coords: np.ndarray              # (N, 3) x, y, z in world coordinates
    mannings_n: list[float]          # [left_overbank, channel, right_overbank]
    bank_stations: tuple[float, float]
    bank_coords: np.ndarray          # (2, 2) left/right bank x, y world


@dataclass
class Reach:
    name: str
    alignment: np.ndarray            # (M, 2) river centerline x, y
    cross_sections: list[CrossSection] = field(default_factory=list)


@dataclass
class BoundaryCondition:
    bc_type: BCType                  # flow, stage, normal_depth, rating_curve, unknown
    location: str                    # "upstream" / "downstream" / reach name
    line_coords: np.ndarray | None = None
    timeseries: dict | None = None  # {"time": array, "values": array, "unit": str}


@dataclass
class HecRasCell:
    face_point_indices: list[int]


@dataclass
class HecRas2DArea:
    name: str
    face_points: np.ndarray          # (F, 2)
    cell_centers: np.ndarray         # (C, 2)
    cells: list[HecRasCell] = field(default_factory=list)
    elevation: np.ndarray | None = None
    mannings_n_constant: float = 0.035
    mannings_n_raster: str | None = None


@dataclass
class Mesh2D:
    nodes: np.ndarray                # (N, 2) x, y
    elements: np.ndarray             # (E, 3) triangles (0-based)
    elevation: np.ndarray            # (N,) bed elevation
    mannings_n: np.ndarray           # (N,) Manning's n

    def __post_init__(self):
        n = len(self.nodes)
        if len(self.elevation) != n:
            raise ValueError(f"elevation length {len(self.elevation)} != node count {n}")
        if len(self.mannings_n) != n:
            raise ValueError(f"mannings_n length {len(self.mannings_n)} != node count {n}")
        if self.elements.size > 0 and int(self.elements.max()) >= n:
            raise ValueError(f"element index {int(self.elements.max())} >= node count {n}")
        if self.elements.size > 0 and int(self.elements.min()) < 0:
            raise ValueError(f"negative element index {int(self.elements.min())}")


@dataclass
class HecRasModel:
    rivers: list[Reach] = field(default_factory=list)
    boundaries: list[BoundaryCondition] = field(default_factory=list)
    areas_2d: list[HecRas2DArea] = field(default_factory=list)
    crs: str | None = None


@dataclass
class BCSegment:
    node_indices: list[int]
    lihbor: LIHBOR                   # WALL=2, FREE=4, PRESCRIBED=5
    prescribed_h: float | None = None
    prescribed_q: float | None = None
    _line_coords: np.ndarray | None = None


@dataclass
class TelemacDomain:
    boundary_polygon: np.ndarray     # (P, 2)
    refinement_zones: list[dict] = field(default_factory=list)
    channel_points: np.ndarray | None = None
    channel_segments: np.ndarray | None = None
    mannings_regions: list[dict] = field(default_factory=list)
    bc_segments: list[BCSegment] = field(default_factory=list)
    _dem_data: np.ndarray | None = None
    _dem_transform: dict | None = None


class HecRasParseError(Exception):
    """Raised when HDF5 structure is invalid or missing expected groups."""
