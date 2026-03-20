"""Write a TELEMAC boundary-conditions (.cli) file."""
from __future__ import annotations

import numpy as np

from telemac_tools.model import Mesh2D, TelemacDomain


def _find_boundary_nodes(elements: np.ndarray, nnodes: int) -> list[int]:
    """Return sorted list of node indices that lie on the mesh boundary.

    A boundary edge is one shared by exactly one triangle.
    """
    edge_count: dict[tuple[int, int], int] = {}
    for tri in elements:
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i + 1) % 3])
            edge = (min(a, b), max(a, b))
            edge_count[edge] = edge_count.get(edge, 0) + 1

    boundary_nodes: set[int] = set()
    for (a, b), count in edge_count.items():
        if count == 1:
            boundary_nodes.add(a)
            boundary_nodes.add(b)

    return sorted(boundary_nodes)


def write_cli(
    mesh: Mesh2D,
    domain: TelemacDomain,
    path: str,
) -> None:
    """Write a .cli boundary-condition file.

    Parameters
    ----------
    mesh : Mesh2D
        The triangle mesh.
    domain : TelemacDomain
        Domain with BC segments specifying node indices and codes.
    path : str
        Output file path.
    """
    boundary_nodes = _find_boundary_nodes(mesh.elements, mesh.nodes.shape[0])

    # Build lookup: node_index -> (lihbor, prescribed_h)
    lihbor_map: dict[int, tuple[int, float | None]] = {}
    for seg in domain.bc_segments:
        for ni in seg.node_indices:
            lihbor_map[ni] = (seg.lihbor, seg.prescribed_h)

    lines: list[str] = []
    for idx, node in enumerate(boundary_nodes):
        lihbor, prescribed_h = lihbor_map.get(node, (2, None))

        if lihbor == 5:
            # Prescribed depth / level
            hbor = prescribed_h if prescribed_h is not None else 0.0
            line = (
                f"5 5 5 {hbor:.1f} 0.0 0.0 0.0 "
                f"5 0.0 0.0 0.0 {node + 1} {idx + 1}"
            )
        elif lihbor == 4:
            # Free / outflow
            line = (
                f"4 4 4 0.0 0.0 0.0 0.0 "
                f"4 0.0 0.0 0.0 {node + 1} {idx + 1}"
            )
        else:
            # Wall (default)
            line = (
                f"2 0 0 0.0 0.0 0.0 0.0 "
                f"0 0.0 0.0 0.0 {node + 1} {idx + 1}"
            )

        lines.append(line)

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
