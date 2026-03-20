"""Gmsh-based mesh generation backend."""
from __future__ import annotations

import math

import gmsh
import numpy as np

from telemac_tools.domain.builder import sample_dem
from telemac_tools.meshing.base import MeshBackend
from telemac_tools.model import Mesh2D, TelemacDomain


class GmshBackend(MeshBackend):
    """Mesh generator using the Gmsh library."""

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
        # 1. Initialize Gmsh
        # ------------------------------------------------------------------
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)  # suppress output
        gmsh.model.add("telemac_domain")

        try:
            return self._generate_impl(domain, min_angle, max_area)
        finally:
            gmsh.finalize()

    def _generate_impl(
        self,
        domain: TelemacDomain,
        min_angle: float,
        max_area: float | None,
    ) -> Mesh2D:
        poly = np.asarray(domain.boundary_polygon, dtype=np.float64)

        # Remove closing duplicate if present
        if np.allclose(poly[0], poly[-1]):
            poly = poly[:-1]

        n_bnd = len(poly)

        # ------------------------------------------------------------------
        # 2. Add boundary points and lines
        # ------------------------------------------------------------------
        point_tags = []
        for i in range(n_bnd):
            tag = gmsh.model.occ.addPoint(poly[i, 0], poly[i, 1], 0.0)
            point_tags.append(tag)

        line_tags = []
        for i in range(n_bnd):
            j = (i + 1) % n_bnd
            tag = gmsh.model.occ.addLine(point_tags[i], point_tags[j])
            line_tags.append(tag)

        # ------------------------------------------------------------------
        # 3. Create curve loop and plane surface
        # ------------------------------------------------------------------
        loop_tag = gmsh.model.occ.addCurveLoop(line_tags)
        surface_tag = gmsh.model.occ.addPlaneSurface([loop_tag])

        # ------------------------------------------------------------------
        # 4. Embed channel constraints if present
        # ------------------------------------------------------------------
        has_channels = (
            domain.channel_points is not None
            and len(domain.channel_points) > 0
            and domain.channel_segments is not None
            and len(domain.channel_segments) > 0
        )

        if has_channels:
            ch_pts = np.asarray(domain.channel_points, dtype=np.float64)
            ch_segs = np.asarray(domain.channel_segments, dtype=np.int32)

            # Create channel lines as separate OCC entities
            ch_point_tags = []
            for i in range(len(ch_pts)):
                tag = gmsh.model.occ.addPoint(ch_pts[i, 0], ch_pts[i, 1], 0.0)
                ch_point_tags.append(tag)

            ch_line_tags = []
            for seg in ch_segs:
                tag = gmsh.model.occ.addLine(ch_point_tags[seg[0]], ch_point_tags[seg[1]])
                ch_line_tags.append(tag)

            # Use fragment to properly merge channel lines into the surface.
            # This handles points on boundary edges correctly.
            out_dimtags, out_map = gmsh.model.occ.fragment(
                [(2, surface_tag)],
                [(1, t) for t in ch_line_tags],
            )
            # After fragment, find the resulting surface tag(s)
            surface_tag = None
            for dim, tag in out_dimtags:
                if dim == 2:
                    surface_tag = tag
                    break

        # ------------------------------------------------------------------
        # 5. Synchronize
        # ------------------------------------------------------------------
        gmsh.model.occ.synchronize()

        # ------------------------------------------------------------------
        # 6. Set mesh size
        # ------------------------------------------------------------------
        dx = poly[:, 0].max() - poly[:, 0].min()
        dy = poly[:, 1].max() - poly[:, 1].min()

        if max_area is not None:
            # Convert max triangle area to characteristic element size
            # For equilateral triangle: area = sqrt(3)/4 * h^2 => h = sqrt(4*area/sqrt(3))
            elem_size = math.sqrt(4.0 * max_area / math.sqrt(3.0))
        else:
            extent = max(dx, dy)
            elem_size = extent / 20.0

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", elem_size * 0.5)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", elem_size)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

        # ------------------------------------------------------------------
        # 7. Generate 2D mesh
        # ------------------------------------------------------------------
        gmsh.model.mesh.generate(2)

        # ------------------------------------------------------------------
        # 8. Extract nodes and triangles
        # ------------------------------------------------------------------
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        # node_coords is flat [x1,y1,z1, x2,y2,z2, ...]
        node_coords = np.array(node_coords).reshape(-1, 3)
        nodes_xy = node_coords[:, :2]

        # Build tag-to-index mapping (gmsh tags are 1-based, may have gaps)
        tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

        # Get 2D triangle elements (type 2 = 3-node triangle)
        elem_types, elem_tags_list, elem_node_tags_list = gmsh.model.mesh.getElements(2)

        triangles = []
        for etype, etags, enodes in zip(elem_types, elem_tags_list, elem_node_tags_list):
            if etype == 2:  # 3-node triangle
                enodes = np.array(enodes, dtype=np.int64)
                n_tri = len(enodes) // 3
                enodes = enodes.reshape(n_tri, 3)
                for tri in enodes:
                    triangles.append([tag_to_idx[int(tri[0])],
                                      tag_to_idx[int(tri[1])],
                                      tag_to_idx[int(tri[2])]])

        elements = np.array(triangles, dtype=np.int32)

        # ------------------------------------------------------------------
        # 9. Assign elevations
        # ------------------------------------------------------------------
        n_nodes = nodes_xy.shape[0]
        elevation = np.zeros(n_nodes, dtype=np.float64)

        if domain._dem_data is not None and domain._dem_transform is not None:
            elevation = sample_dem(
                nodes_xy[:, 0], nodes_xy[:, 1],
                domain._dem_data, domain._dem_transform,
            )
            elevation = np.nan_to_num(elevation, nan=0.0)

        # Override channel-node elevations when channel points carry Z
        if domain.channel_points is not None and len(domain.channel_points) > 0:
            ch_pts = np.asarray(domain.channel_points, dtype=np.float64)
            if ch_pts.ndim == 2 and ch_pts.shape[1] >= 3:
                for cp in ch_pts:
                    dists = np.hypot(nodes_xy[:, 0] - cp[0], nodes_xy[:, 1] - cp[1])
                    mask = dists < 1.0
                    elevation[mask] = cp[2]

        # ------------------------------------------------------------------
        # 10. Assign Manning's n
        # ------------------------------------------------------------------
        mannings_n = np.full(n_nodes, 0.035, dtype=np.float64)

        return Mesh2D(
            nodes=nodes_xy,
            elements=elements,
            elevation=elevation,
            mannings_n=mannings_n,
        )
