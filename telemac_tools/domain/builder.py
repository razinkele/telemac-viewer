"""Domain builder: HEC-RAS model + DEM -> TelemacDomain."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from telemac_tools.domain.channel_carve import build_channel_points
from telemac_tools.model import (
    BCSegment,
    BoundaryCondition,
    HecRasModel,
    LIHBOR,
    TelemacDomain,
)


# ---------------------------------------------------------------------------
# DEM reading
# ---------------------------------------------------------------------------

def _read_dem(dem_path: str) -> tuple[np.ndarray, dict]:
    """Read a GeoTIFF DEM using tifffile.

    Returns
    -------
    data : 2-D array (rows, cols) of elevation values.
    transform : dict with keys origin_x, origin_y, pixel_w, pixel_h.
        origin_x/y is the top-left corner; pixel_h is positive (pixel step
        in the y-down direction of the raster).
    """
    import tifffile

    with tifffile.TiffFile(dem_path) as tif:
        page = tif.pages[0]
        data = page.asarray()

        # Extract geotransform from GeoTIFF tags
        tiepoint = page.tags[33922].value   # ModelTiepointTag
        scale = page.tags[33550].value      # ModelPixelScaleTag

        origin_x = tiepoint[3]
        origin_y = tiepoint[4]
        pixel_w = scale[0]
        pixel_h = scale[1]  # positive value; y decreases down rows

    transform = {
        "origin_x": origin_x,
        "origin_y": origin_y,
        "pixel_w": pixel_w,
        "pixel_h": pixel_h,
    }
    return data, transform


def sample_dem(
    x: np.ndarray,
    y: np.ndarray,
    dem_data: np.ndarray,
    transform: dict,
) -> np.ndarray:
    """Sample DEM elevations at (x, y) locations using bilinear interpolation.

    Parameters
    ----------
    x, y : 1-D arrays of world coordinates.
    dem_data : 2-D raster (rows, cols).
    transform : geotransform dict from ``_read_dem``.

    Returns
    -------
    z : 1-D array of interpolated elevations.
    """
    nrows, ncols = dem_data.shape
    ox = transform["origin_x"]
    oy = transform["origin_y"]
    pw = transform["pixel_w"]
    ph = transform["pixel_h"]

    # Build coordinate arrays for each axis.
    # Row 0 corresponds to origin_y (top); row i -> origin_y - i*pixel_h
    # We need ascending y for RegularGridInterpolator, so flip.
    y_coords = oy - np.arange(nrows) * ph            # descending
    x_coords = ox + np.arange(ncols) * pw             # ascending

    # Flip y_coords and data so y is ascending
    y_asc = y_coords[::-1]
    data_asc = dem_data[::-1, :]

    interp = RegularGridInterpolator(
        (y_asc, x_coords), data_asc,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    pts = np.column_stack([y, x])
    return interp(pts)


# ---------------------------------------------------------------------------
# Boundary polygon from 1D alignment
# ---------------------------------------------------------------------------

def _buffer_alignment(alignment: np.ndarray, distance: float) -> np.ndarray:
    """Offset an alignment polyline left/right to create a closed polygon.

    Parameters
    ----------
    alignment : (M, 2) river centreline x, y.
    distance : perpendicular offset (half-width of floodplain).

    Returns
    -------
    polygon : (2*M + 1, 2) closed polygon (left side forward, right side
              reversed, first point repeated).
    """
    n = len(alignment)
    left = np.empty((n, 2))
    right = np.empty((n, 2))

    for i in range(n):
        # Tangent vector (forward difference / central / backward)
        if i == 0:
            tang = alignment[1] - alignment[0]
        elif i == n - 1:
            tang = alignment[-1] - alignment[-2]
        else:
            tang = alignment[i + 1] - alignment[i - 1]

        length = np.linalg.norm(tang)
        if length == 0:
            perp = np.array([0.0, 1.0])
        else:
            tang = tang / length
            # Perpendicular (90 deg CCW)
            perp = np.array([-tang[1], tang[0]])

        left[i] = alignment[i] + perp * distance
        right[i] = alignment[i] - perp * distance

    # Polygon: left forward, right reversed, close
    poly = np.vstack([left, right[::-1], left[0:1]])
    return poly


# ---------------------------------------------------------------------------
# BC segment helpers
# ---------------------------------------------------------------------------

def _bc_type_to_lihbor(bc: BoundaryCondition) -> tuple[LIHBOR, float | None, float | None]:
    """Map HEC-RAS BC type to TELEMAC LIHBOR code.

    For v1 we do NOT import flow/stage time series, so:
    - upstream flow BCs get LIHBOR=5 with a nominal HBOR=0.1 (user must update)
    - stage BCs get LIHBOR=5 with a nominal HBOR=0.1 (user must update)
    - downstream free BCs get LIHBOR=4
    """
    bt = bc.bc_type.lower()
    loc = (bc.location or "").lower()
    if bt in ("flow", "hydrograph"):
        if loc == "downstream":
            return LIHBOR.FREE, None, None          # free outflow
        return LIHBOR.PRESCRIBED, 0.1, None         # prescribed — nominal wet depth
    elif bt in ("stage", "known_ws"):
        return LIHBOR.PRESCRIBED, 0.1, None         # prescribed — nominal wet depth
    elif bt in ("normal_depth", "rating_curve"):
        return LIHBOR.FREE, None, None              # free / Neumann
    import warnings
    warnings.warn(f"Unknown BC type '{bc.bc_type}' — defaulting to wall (LIHBOR=2)")
    return LIHBOR.WALL, None, None                  # wall


# ---------------------------------------------------------------------------
# Manning's regions
# ---------------------------------------------------------------------------

def _build_mannings_regions(reach) -> list[dict]:
    """Build Manning's n spatial regions from cross-section bank positions."""
    if not reach.cross_sections:
        return []

    # Channel zone polygon: connect left banks forward, right banks backward
    left_banks = []
    right_banks = []
    for xs in sorted(reach.cross_sections, key=lambda x: x.station):
        left_banks.append(xs.bank_coords[0])
        right_banks.append(xs.bank_coords[1])

    if len(left_banks) < 2:
        return []

    # Average channel Manning's n
    avg_n = np.mean([xs.mannings_n[1] for xs in reach.cross_sections])

    # Build closed polygon: left forward + right reversed + close
    channel_poly = np.vstack(left_banks + right_banks[::-1] + [left_banks[0]])

    return [{"polygon": channel_poly, "n": float(avg_n)}]


# ---------------------------------------------------------------------------
# 1D domain builder
# ---------------------------------------------------------------------------

def build_domain_1d(
    model: HecRasModel,
    dem_path: str,
    floodplain_width: float = 500.0,
    channel_spacing: float = 10.0,
) -> TelemacDomain:
    """Build a TelemacDomain from a 1D HEC-RAS model and a DEM.

    Parameters
    ----------
    model : HecRasModel with rivers and boundaries populated.
    dem_path : path to GeoTIFF DEM.
    floodplain_width : total width of the floodplain buffer (half on each side).
    channel_spacing : spacing for channel constraint points.

    Returns
    -------
    TelemacDomain ready for meshing.
    """
    dem_data, dem_transform = _read_dem(dem_path)

    # --- Boundary polygon (buffer the first reach's alignment) ---
    reach = model.rivers[0]
    boundary_polygon = _buffer_alignment(reach.alignment, floodplain_width / 2.0)

    # --- Channel points ---
    channel_pts, channel_segs = build_channel_points(reach, spacing=channel_spacing)

    # --- Refinement zones (channel corridor) ---
    refinement_zones: list[dict] = []
    if len(reach.cross_sections) >= 2:
        bank_left = []
        bank_right = []
        for xs in sorted(reach.cross_sections, key=lambda x: x.station):
            bank_left.append(xs.bank_coords[0])
            bank_right.append(xs.bank_coords[1])
        bank_left_arr = np.array(bank_left)
        bank_right_arr = np.array(bank_right)
        channel_poly = np.vstack([
            bank_left_arr,
            bank_right_arr[::-1],
            bank_left_arr[0:1],
        ])
        refinement_zones.append({
            "polygon": channel_poly,
            "max_area": (channel_spacing ** 2) / 2.0,
            "label": "channel",
        })

    # --- Manning's regions ---
    mannings_regions = _build_mannings_regions(reach)

    # --- BC segments ---
    bc_segments: list[BCSegment] = []
    for bc in model.boundaries:
        lihbor, h, q = _bc_type_to_lihbor(bc)
        seg = BCSegment(
            node_indices=[],  # filled after meshing
            lihbor=lihbor,
            prescribed_h=h,
            prescribed_q=q,
            _line_coords=bc.line_coords,
        )
        bc_segments.append(seg)

    return TelemacDomain(
        boundary_polygon=boundary_polygon,
        refinement_zones=refinement_zones,
        channel_points=channel_pts,
        channel_segments=channel_segs,
        mannings_regions=mannings_regions,
        bc_segments=bc_segments,
        _dem_data=dem_data,
        _dem_transform=dem_transform,
    )


# ---------------------------------------------------------------------------
# 2D domain builder
# ---------------------------------------------------------------------------

def build_domain_2d(model: HecRasModel) -> TelemacDomain:
    """Build a TelemacDomain from a 2D HEC-RAS model.

    Extracts the boundary polygon from the mesh topology by finding
    boundary edges (edges shared by exactly one cell).

    Parameters
    ----------
    model : HecRasModel with areas_2d populated.

    Returns
    -------
    TelemacDomain with boundary_polygon set.
    """
    area = model.areas_2d[0]
    face_points = area.face_points

    # Count edge occurrences across all cells
    edge_count: dict[tuple[int, int], int] = defaultdict(int)
    for cell in area.cells:
        fpi = cell.face_point_indices
        n = len(fpi)
        for j in range(n):
            a, b = fpi[j], fpi[(j + 1) % n]
            edge = (min(a, b), max(a, b))
            edge_count[edge] += 1

    # Boundary edges = shared by exactly one cell
    boundary_edges = [e for e, c in edge_count.items() if c == 1]

    if not boundary_edges:
        # Fallback: convex hull
        from scipy.spatial import ConvexHull
        hull = ConvexHull(face_points)
        verts = hull.vertices
        poly = np.vstack([face_points[verts], face_points[verts[0:1]]])
        return TelemacDomain(boundary_polygon=poly)

    # Build adjacency list for boundary edges
    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in boundary_edges:
        adj[a].append(b)
        adj[b].append(a)

    # Walk the boundary ring
    start = boundary_edges[0][0]
    ring = [start]
    visited = {start}
    current = start
    while True:
        found_next = False
        for nb in adj[current]:
            if nb not in visited:
                ring.append(nb)
                visited.add(nb)
                current = nb
                found_next = True
                break
        if not found_next:
            break

    # Close the ring
    ring.append(ring[0])

    poly = face_points[ring]
    return TelemacDomain(boundary_polygon=poly)


# ---------------------------------------------------------------------------
# Post-mesh BC node assignment
# ---------------------------------------------------------------------------

def assign_bc_nodes(
    mesh_boundary_nodes: np.ndarray,
    mesh_boundary_coords: np.ndarray,
    domain: TelemacDomain,
    tolerance: float = 50.0,
) -> None:
    """Match mesh boundary nodes to BC segments by proximity.

    Parameters
    ----------
    mesh_boundary_nodes : (B,) node indices on the mesh boundary.
    mesh_boundary_coords : (B, 2) coordinates of those nodes.
    domain : TelemacDomain whose bc_segments will be updated in-place.
    tolerance : max distance from a BC line to claim a node.
    """
    for seg in domain.bc_segments:
        if seg._line_coords is None or len(seg._line_coords) < 2:
            continue
        line = seg._line_coords
        matched = []
        for i, coord in enumerate(mesh_boundary_coords):
            # Distance from point to polyline (segment-by-segment)
            min_d = _point_to_polyline_dist(coord, line)
            if min_d <= tolerance:
                matched.append(int(mesh_boundary_nodes[i]))
        seg.node_indices = matched


def _point_to_polyline_dist(pt: np.ndarray, line: np.ndarray) -> float:
    """Minimum distance from a point to a polyline."""
    min_d = np.inf
    for i in range(len(line) - 1):
        d = _point_to_segment_dist(pt, line[i], line[i + 1])
        if d < min_d:
            min_d = d
    return min_d


def _point_to_segment_dist(pt: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Distance from point to line segment a-b."""
    ab = b - a
    ab_sq = np.dot(ab, ab)
    if ab_sq == 0:
        return float(np.linalg.norm(pt - a))
    t = np.clip(np.dot(pt - a, ab) / ab_sq, 0.0, 1.0)
    proj = a + t * ab
    return float(np.linalg.norm(pt - proj))
