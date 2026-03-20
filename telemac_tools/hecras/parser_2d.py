"""Parse 2D HEC-RAS geometry from HDF5 (.g01.hdf) files."""
from __future__ import annotations

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

            # Cell centers (C, 2)
            cell_centers = item["Cell Points"][:].astype(np.float64)

            # Cell face counts and face point indices
            face_counts = item["Cells Face and Orientation Info"][:].astype(int)
            face_values = item["Faces FacePoint Indexes"][:].astype(int)

            # Reconstruct cells
            cells: list[HecRasCell] = []
            offset = 0
            for count in face_counts:
                indices = face_values[offset : offset + count].tolist()
                cells.append(HecRasCell(face_point_indices=indices))
                offset += count

            # Elevation per cell (optional)
            elevation = None
            if "Cells Minimum Elevation" in item:
                elevation = item["Cells Minimum Elevation"][:].astype(np.float64)

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
        center_node = n_fp + cell_idx  # index into nodes array
        fp_indices = cell.face_point_indices
        n = len(fp_indices)
        for j in range(n):
            j_next = (j + 1) % n
            triangles.append([center_node, fp_indices[j], fp_indices[j_next]])

    elements = np.array(triangles, dtype=np.int32)

    # Interpolate elevation from cell centers to all nodes
    if area.elevation is not None:
        interp = NearestNDInterpolator(area.cell_centers, area.elevation)
        elevation = interp(nodes)
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
