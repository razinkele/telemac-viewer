"""Write a TELEMAC Selafin (.slf) geometry/initial-conditions file."""
from __future__ import annotations

import numpy as np
from data_manip.formats.selafin import Selafin

from telemac_tools.model import Mesh2D


def write_slf(
    mesh: Mesh2D,
    path: str,
    *,
    init_depth: float = 0.1,
    boundary_nodes: list[int] | None = None,
) -> None:
    """Create a .slf file with BOTTOM, FRICTION, WATER DEPTH, FREE SURFACE.

    Parameters
    ----------
    mesh : Mesh2D
        Triangle mesh with elevation and Manning's n.
    path : str
        Output file path.
    init_depth : float
        Initial water depth applied everywhere (default 0.1 m).
    boundary_nodes : list[int] or None
        Sorted list of node indices on the mesh boundary.  When provided,
        the ``ipob2`` / ``ipob3`` arrays are populated so that TELEMAC can
        link .cli entries to mesh nodes.
    """
    slf = Selafin("")

    slf.title = "TELEMAC GEOMETRY".ljust(80)[:80]

    # Variables
    slf.nbv1 = 4
    slf.nbv2 = 0
    slf.nvar = 4
    slf.varindex = list(range(4))
    slf.varnames = [
        "BOTTOM".ljust(16)[:16],
        "FRICTION COEFFIC".ljust(16)[:16],
        "WATER DEPTH".ljust(16)[:16],
        "FREE SURFACE".ljust(16)[:16],
    ]
    slf.varunits = [
        "M".ljust(16)[:16],
        "".ljust(16)[:16],
        "M".ljust(16)[:16],
        "M".ljust(16)[:16],
    ]
    slf.cldnames = []
    slf.cldunits = []

    # Mesh dimensions
    npoin = mesh.nodes.shape[0]
    nelem = mesh.elements.shape[0]
    slf.nelem3 = nelem
    slf.npoin3 = npoin
    slf.ndp3 = 3
    slf.nplan = 1
    slf.nelem2 = nelem
    slf.npoin2 = npoin
    slf.ndp2 = 3

    # Connectivity (0-based internally; append_header_slf adds 1)
    slf.ikle3 = mesh.elements.astype(np.int32)
    slf.ikle2 = slf.ikle3

    # Boundary pointer (0 = interior, >0 = boundary node number)
    ipob2 = np.zeros(npoin, dtype=np.int32)
    if boundary_nodes:
        for k, node in enumerate(boundary_nodes):
            ipob2[node] = k + 1  # 1-based boundary pointer
    slf.ipob2 = ipob2
    slf.ipob3 = ipob2

    # Coordinates
    slf.meshx = mesh.nodes[:, 0].astype(np.float64)
    slf.meshy = mesh.nodes[:, 1].astype(np.float64)

    # iparam: 10 integers, all zero (no date record)
    slf.iparam = np.zeros(10, dtype=np.int32)

    # Time tags
    slf.tags = {"cores": [], "times": [0.0]}

    # Open output file
    slf.fole = {"name": path, "endian": ">", "float": ("f", 4)}
    slf.fole["hook"] = open(path, "wb")

    # Write header
    slf.append_header_slf()

    # Build variable arrays for timestep 0
    bottom = mesh.elevation.astype(np.float32)
    friction = mesh.mannings_n.astype(np.float32)
    depth = np.full(npoin, init_depth, dtype=np.float32)
    surface = bottom + depth

    slf.append_core_time_slf(0.0)
    slf.append_core_vars_slf([bottom, friction, depth, surface])

    slf.fole["hook"].close()
