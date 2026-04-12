"""Parse 2D HEC-RAS geometry from HDF5 (.g01.hdf) files."""
from __future__ import annotations

import logging

import h5py
import numpy as np
from scipy.interpolate import NearestNDInterpolator

from telemac_tools.model import (
    HecRas2DArea,
    HecRasCell,
    HecRasModel,
    HecRasParseError,
    Mesh2D,
)

_logger = logging.getLogger(__name__)


def parse_hecras_2d(path: str) -> HecRasModel:
    """Parse 2D geometry from a HEC-RAS HDF5 file.

    Parameters
    ----------
    path : str
        Path to the .g01.hdf file.

    Returns
    -------
    HecRasModel with areas_2d populated.

    Raises
    ------
    HecRasParseError
        If the file lacks a "Geometry/2D Flow Areas" group.
    """
    with h5py.File(path, "r") as f:
        if "Geometry" not in f or "2D Flow Areas" not in f["Geometry"]:
            raise HecRasParseError(
                f"No 'Geometry/2D Flow Areas' group found in {path}"
            )
        areas_grp = f["Geometry/2D Flow Areas"]

        # Read global attributes for Manning's n default
        default_mannings = float(areas_grp.attrs.get("Manning's n", 0.035))

        areas: list[HecRas2DArea] = []
        for name in areas_grp:
            item = areas_grp[name]
            if not isinstance(item, h5py.Group):
                continue

            # Face points (F, 2)
            face_points = item["FacePoints Coordinate"][:].astype(np.float64)

            # Cell centers: try known name variants
            if "Cells Center Coordinate" in item:
                cell_centers = item["Cells Center Coordinate"][:].astype(np.float64)
            elif "Cell Points" in item:
                cell_centers = item["Cell Points"][:].astype(np.float64)
            else:
                # Synthesize cell centers from face geometry if missing
                cell_centers = None

            # Cell face point indices: may be a padded 2D array (N, max_faces)
            # with -1 sentinels, or a flat 1D array read via offset/count.
            if "Cells FacePoint Indexes" in item:
                raw_idx = item["Cells FacePoint Indexes"][:]
                # Validate uint64 won't overflow int32
                if raw_idx.dtype == np.uint64 and raw_idx.size > 0:
                    max_val = int(raw_idx.max())
                    if max_val >= 2**31:
                        _logger.warning(
                            "Area %s: face-point index %d overflows int32, skipping",
                            name, max_val)
                        continue
                raw_cfpi = raw_idx.astype(int)
            elif "Faces FacePoint Indexes" in item:
                raw_cfpi = None  # will use face-based reconstruction below
            else:
                raw_cfpi = None

            # Cell face counts from orientation info
            if "Cells Face and Orientation Info" in item:
                face_info = item["Cells Face and Orientation Info"][:].astype(int)
            else:
                face_info = None

            # Reconstruct cells
            cells: list[HecRasCell] = []
            n_fp = len(face_points)

            if raw_cfpi is not None and raw_cfpi.ndim == 2:
                # Padded 2D array: each row is face-point indices with -1 fill
                for row in raw_cfpi:
                    indices = row[row >= 0].tolist()
                    if any(idx >= n_fp for idx in indices):
                        continue  # skip cells with out-of-range indices
                    cells.append(HecRasCell(face_point_indices=indices))
            elif raw_cfpi is not None and raw_cfpi.ndim == 1 and face_info is not None:
                # Flat 1D array with offset/count from face info
                for off, cnt in face_info:
                    indices = raw_cfpi[off : off + cnt].tolist()
                    if any(idx >= n_fp for idx in indices):
                        continue  # skip cells with out-of-range indices
                    cells.append(HecRasCell(face_point_indices=indices))
            elif face_info is not None:
                # No cell-level face-point indices; build from face-level data
                # Use Faces FacePoint Indexes to reconstruct per-cell polygons
                face_orient_vals = None
                if "Cells Face and Orientation Values" in item:
                    face_orient_vals = item["Cells Face and Orientation Values"][:].astype(int)
                faces_fp = None
                if "Faces FacePoint Indexes" in item:
                    faces_fp = item["Faces FacePoint Indexes"][:].astype(int)

                if face_orient_vals is not None and faces_fp is not None:
                    for off, cnt in face_info:
                        fp_set: list[int] = []
                        for j in range(off, off + cnt):
                            face_idx = abs(face_orient_vals[j, 0])
                            if face_idx < len(faces_fp):
                                for fp in faces_fp[face_idx]:
                                    if fp >= 0 and fp < n_fp and fp not in fp_set:
                                        fp_set.append(int(fp))
                        cells.append(HecRasCell(face_point_indices=fp_set))
                else:
                    # Last resort: create dummy cells
                    n_cells = len(face_info)
                    for _ in range(n_cells):
                        cells.append(HecRasCell(face_point_indices=[]))

            if cell_centers is None and cells:
                # Synthesize cell centers as centroid of face points
                centers = []
                for cell in cells:
                    if cell.face_point_indices:
                        pts = face_points[cell.face_point_indices]
                        centers.append(pts.mean(axis=0))
                    else:
                        centers.append(np.array([0.0, 0.0]))
                cell_centers = np.array(centers)
            elif cell_centers is None:
                cell_centers = np.empty((0, 2))

            # Elevation per cell (optional)
            elevation = None
            if "Cells Minimum Elevation" in item:
                raw_elev = item["Cells Minimum Elevation"][:].astype(np.float64)
                if len(raw_elev) == len(cells):
                    elevation = raw_elev
                else:
                    _logger.warning(
                        "Elevation array length (%d) != cell count (%d), skipping",
                        len(raw_elev), len(cells))

            areas.append(
                HecRas2DArea(
                    name=name,
                    face_points=face_points,
                    cell_centers=cell_centers,
                    cells=cells,
                    elevation=elevation,
                    mannings_n_constant=default_mannings,
                )
            )

    return HecRasModel(areas_2d=areas)


def triangulate_2d_area(area: HecRas2DArea) -> Mesh2D:
    """Convert a HEC-RAS 2D area (Voronoi cells) to a triangle mesh.

    Uses fan triangulation: for each cell, creates triangles from the cell
    center to consecutive face point pairs.

    Parameters
    ----------
    area : HecRas2DArea
        Parsed 2D area with face points, cell centers, and cells.

    Returns
    -------
    Mesh2D with triangulated nodes, elements, elevation, and Manning's n.
    """
    n_fp = area.face_points.shape[0]
    n_cc = area.cell_centers.shape[0]

    # Nodes: face_points first, then cell_centers
    nodes = np.vstack([area.face_points, area.cell_centers])

    # Fan-triangulate each cell
    triangles: list[list[int]] = []
    for cell_idx, cell in enumerate(area.cells):
        if cell_idx >= n_cc:
            break  # more cells than cell_centers
        center_node = n_fp + cell_idx  # index into nodes array
        fp_indices = cell.face_point_indices
        n = len(fp_indices)
        if n < 3:
            continue
        # Validate all face-point indices are in range
        if any(idx < 0 or idx >= n_fp for idx in fp_indices):
            continue
        for j in range(n):
            j_next = (j + 1) % n
            triangles.append([center_node, fp_indices[j], fp_indices[j_next]])

    elements = np.array(triangles, dtype=np.int32) if triangles else np.empty((0, 3), dtype=np.int32)

    # Interpolate elevation from cell centers to all nodes
    if area.elevation is not None:
        valid = ~np.isnan(area.elevation)
        if valid.any():
            interp = NearestNDInterpolator(
                area.cell_centers[valid], area.elevation[valid])
            elevation = interp(nodes)
        else:
            _logger.warning("All cell elevations are NaN, using zeros")
            elevation = np.zeros(nodes.shape[0])
    else:
        elevation = np.zeros(nodes.shape[0])

    # Manning's n: constant across all nodes
    mannings_n = np.full(nodes.shape[0], area.mannings_n_constant)

    return Mesh2D(
        nodes=nodes,
        elements=elements,
        elevation=elevation,
        mannings_n=mannings_n,
    )
