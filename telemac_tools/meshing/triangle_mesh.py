"""Triangle-based mesh generation backend."""
from __future__ import annotations

import numpy as np
import triangle as tr

from telemac_tools.domain.builder import sample_dem
from telemac_tools.meshing.base import MeshBackend
from telemac_tools.model import Mesh2D, TelemacDomain


class TriangleBackend(MeshBackend):
    """Mesh generator using Jonathan Shewchuk's Triangle library."""

    def generate(
        self,
        domain: TelemacDomain,
        min_angle: float = 30.0,
        max_area: float | None = None,
    ) -> Mesh2D:
        """Generate a quality triangular mesh for *domain*.

        Parameters
        ----------
        domain : TelemacDomain
            Must have a closed ``boundary_polygon`` (first == last vertex).
        min_angle : float
            Minimum angle constraint (degrees).
        max_area : float | None
            Maximum triangle area.  Computed from domain extent if *None*.

        Returns
        -------
        Mesh2D
        """
        # ------------------------------------------------------------------
        # 1. Prepare boundary vertices & segments
        # ------------------------------------------------------------------
        poly = np.asarray(domain.boundary_polygon, dtype=np.float64)

        # Remove closing duplicate if present
        if np.allclose(poly[0], poly[-1]):
            poly = poly[:-1]

        n_bnd = len(poly)
        segments = np.array(
            [[i, (i + 1) % n_bnd] for i in range(n_bnd)], dtype=np.int32
        )

        vertices = poly.copy()

        # ------------------------------------------------------------------
        # 2. Optionally add channel constraint vertices/segments
        # ------------------------------------------------------------------
        if domain.channel_points is not None and len(domain.channel_points) > 0:
            ch_pts = np.asarray(domain.channel_points, dtype=np.float64)
            offset = len(vertices)
            vertices = np.vstack([vertices, ch_pts])

            if domain.channel_segments is not None and len(domain.channel_segments) > 0:
                ch_segs = np.asarray(domain.channel_segments, dtype=np.int32) + offset
                segments = np.vstack([segments, ch_segs])

        # ------------------------------------------------------------------
        # 3. Build PSLG and triangulate
        # ------------------------------------------------------------------
        pslg: dict = {"vertices": vertices, "segments": segments}

        # Compute default max_area from bounding box if not given
        if max_area is None:
            dx = vertices[:, 0].max() - vertices[:, 0].min()
            dy = vertices[:, 1].max() - vertices[:, 1].min()
            max_area = (dx * dy) / 200.0  # ~200 triangles as a default

        opts = f"pq{min_angle:.1f}a{max_area:.6f}"
        result = tr.triangulate(pslg, opts)

        nodes = result["vertices"]
        elements = result["triangles"]

        # ------------------------------------------------------------------
        # 4. Assign elevations
        # ------------------------------------------------------------------
        n_nodes = nodes.shape[0]
        elevation = np.zeros(n_nodes, dtype=np.float64)

        if domain._dem_data is not None and domain._dem_transform is not None:
            elevation = sample_dem(
                nodes[:, 0], nodes[:, 1], domain._dem_data, domain._dem_transform
            )
            # Replace any NaN from out-of-bounds sampling with 0
            elevation = np.nan_to_num(elevation, nan=0.0)

        # Override channel-node elevations when channel points carry Z
        if domain.channel_points is not None and len(domain.channel_points) > 0:
            ch_pts = np.asarray(domain.channel_points, dtype=np.float64)
            if ch_pts.ndim == 2 and ch_pts.shape[1] >= 3:
                # For every mesh node within 1 m of a channel point, use
                # the channel point's elevation.
                for cp in ch_pts:
                    dists = np.hypot(nodes[:, 0] - cp[0], nodes[:, 1] - cp[1])
                    mask = dists < 1.0
                    elevation[mask] = cp[2]

        # ------------------------------------------------------------------
        # 5. Assign Manning's n
        # ------------------------------------------------------------------
        mannings_n = np.full(n_nodes, 0.035, dtype=np.float64)

        # Apply per-region overrides if available
        for region in domain.mannings_regions:
            poly_r = np.asarray(region["polygon"], dtype=np.float64)
            value = float(region["n"])
            inside = _points_in_polygon(nodes, poly_r)
            mannings_n[inside] = value

        return Mesh2D(
            nodes=nodes,
            elements=elements,
            elevation=elevation,
            mannings_n=mannings_n,
        )


def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Ray-casting point-in-polygon test.

    Parameters
    ----------
    points : (N, 2) array
    polygon : (M, 2) array – closed or open polygon vertices.

    Returns
    -------
    mask : (N,) bool array
    """
    n = len(polygon)
    inside = np.zeros(len(points), dtype=bool)
    px, py = points[:, 0], points[:, 1]

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        cond = ((yi > py) != (yj > py)) & (
            px < (xj - xi) * (py - yi) / (yj - yi + 1e-30) + xi
        )
        inside ^= cond
        j = i
    return inside
