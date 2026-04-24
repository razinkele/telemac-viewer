# layers.py
from __future__ import annotations
import warnings
import numpy as np
from viewer_types import MeshGeometry, TelemacFileProtocol
from shiny_deckgl import (
    layer,
    simple_mesh_layer,
    line_layer,
    scatterplot_layer,
    path_layer,
    trips_layer,
    encode_binary_attribute,
)

# deck.gl COORDINATE_SYSTEM.METER_OFFSETS: layer x/y are meters relative to
# the lon/lat origin set via coordinateOrigin on each layer.
_COORD_METER_OFFSETS = 2
from constants import cached_palette_arr
from telemac_defaults import find_velocity_pair

_GRAY = np.array([0.85, 0.85, 0.85], dtype=np.float32)
_WIRE_COLOR = [100, 100, 100, 80]


def _compute_mesh_colors(
    geom: MeshGeometry,
    values: np.ndarray,
    palette_id: str,
    filter_range: tuple[float, float] | None = None,
    color_range_override: tuple[float, float] | None = None,
    log_scale: bool = False,
    reverse_palette: bool = False,
) -> tuple[np.ndarray, float, float, bool]:
    """Compute per-vertex colors (float32, flattened RGB) for the mesh.

    Returns (colors_flat, vmin, vmax, log_applied). Shared between
    build_mesh_layer (full-build) and build_mesh_color_patch (partial
    update), so there is a single source of truth for the value→color
    mapping.
    """
    npoin = geom.npoin

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        vmin, vmax = (
            float(np.nanmin(values[:npoin])),
            float(np.nanmax(values[:npoin])),
        )
    if np.isnan(vmin):
        vmin, vmax = 0.0, 1.0
    if color_range_override is not None:
        cmin, cmax = color_range_override
        if cmin is not None and cmax is not None and cmax > cmin:
            vmin, vmax = float(cmin), float(cmax)
    uniform_field = vmax == vmin
    if uniform_field:
        vmax = vmin + 1.0

    palette_arr = cached_palette_arr(palette_id, reverse=reverse_palette)
    log_applied = False
    # Log scale needs strictly positive min AND a non-degenerate range.
    # Uniform fields would normalize to ~0 everywhere → silently black mesh.
    if log_scale and vmin > 0 and not uniform_field:
        log_vals = np.log10(np.maximum(values[:npoin], vmin))
        log_min, log_max = np.log10(vmin), np.log10(vmax)
        normalized = np.clip((log_vals - log_min) / (log_max - log_min), 0, 1)
        log_applied = True
    else:
        normalized = np.clip((values[:npoin] - vmin) / (vmax - vmin), 0, 1)
    idx = np.nan_to_num(normalized * 255, nan=0.0).astype(int)
    np.clip(idx, 0, 255, out=idx)
    vertex_colors_u8 = palette_arr[idx]
    colors_f32 = vertex_colors_u8[:, :3].astype(np.float32) / 255.0

    if filter_range is not None:
        lo, hi = filter_range
        mask = (values[:npoin] < lo) | (values[:npoin] > hi)
        colors_f32[mask] = _GRAY

    return colors_f32.flatten(), vmin, vmax, log_applied


def build_mesh_layer(
    geom: MeshGeometry,
    values: np.ndarray,
    palette_id: str,
    filter_range: tuple[float, float] | None = None,
    color_range_override: tuple[float, float] | None = None,
    log_scale: bool = False,
    reverse_palette: bool = False,
    origin: list[float] | None = None,
) -> tuple[dict, float, float, bool]:
    """Build SimpleMeshLayer with per-vertex coloring and optional value filter."""
    colors_flat, vmin, vmax, log_applied = _compute_mesh_colors(
        geom,
        values,
        palette_id,
        filter_range=filter_range,
        color_range_override=color_range_override,
        log_scale=log_scale,
        reverse_palette=reverse_palette,
    )

    lyr = simple_mesh_layer(
        "mesh",
        data=[{"position": [0, 0, 0]}],
        mesh="@@CustomGeometry",
        _meshPositions=geom.positions,
        _meshNormals=[],
        _meshColors=encode_binary_attribute(colors_flat),
        _meshIndices=geom.indices,
        coordinateSystem=_COORD_METER_OFFSETS,
        coordinateOrigin=origin or [0, 0],
        sizeScale=1,
        getPosition="@@=d.position",
        getColor=[255, 255, 255, 255],
        pickable=True,
    )
    return lyr, vmin, vmax, log_applied


def build_mesh_color_patch(
    geom: MeshGeometry,
    values: np.ndarray,
    palette_id: str,
    filter_range: tuple[float, float] | None = None,
    color_range_override: tuple[float, float] | None = None,
    log_scale: bool = False,
    reverse_palette: bool = False,
) -> tuple[dict, float, float, bool]:
    """Build a partial-update patch for the mesh layer (colors only).

    Returns a dict suitable for ``map_widget.partial_update([patch])``.
    The JS-side layer cache preserves positions, indices, origin, and
    coordinate system — only the ``_meshColors`` binary attribute is
    resent. For a 100k-node float32 mesh this cuts roughly ~60-70% of
    the per-tick payload (positions + indices) when the user scrubs
    through time or switches palette.
    """
    colors_flat, vmin, vmax, log_applied = _compute_mesh_colors(
        geom,
        values,
        palette_id,
        filter_range=filter_range,
        color_range_override=color_range_override,
        log_scale=log_scale,
        reverse_palette=reverse_palette,
    )
    # Intentionally only `id` + `_meshColors`: partial_update merges by
    # key, so every other layer prop (positions, indices, coordinate
    # system, origin, pickable, sizeScale, ...) is preserved from the
    # JS-side cache. Adding any of them here defeats the optimization.
    patch = {
        "id": "mesh",
        "_meshColors": encode_binary_attribute(colors_flat),
    }
    return patch, vmin, vmax, log_applied


def build_velocity_patch(
    tf: TelemacFileProtocol,
    tidx: int,
    geom: MeshGeometry,
    origin: list[float] | None = None,
) -> dict | None:
    """Partial-update patch for the velocity arrow layer.

    Today this is a pass-through to ``build_velocity_layer`` — velocity
    arrows are cheap relative to mesh colors, so we resend the full layer
    dict. The wrapper exists so the fast path in ``update_map`` can call
    ``build_*_patch`` uniformly; if we later introduce sparse attribute
    updates for arrows, only this function changes.
    """
    return build_velocity_layer(tf, tidx, geom, origin=origin)


def build_contour_patch(
    tf: TelemacFileProtocol,
    values: np.ndarray,
    geom: MeshGeometry,
    n_contours: int = 6,
    layer_id: str = "contours",
    contour_color: list[int] | None = None,
    origin: list[float] | None = None,
) -> dict | None:
    """Partial-update patch for a contour layer.

    Today this is a pass-through to ``build_contour_layer_fn`` — contours
    are recomputed from scratch when the field changes because the marching
    triangles pass produces entirely new crossing points per tick. A true
    sparse update would require segment-level diffing that isn't worth
    the complexity yet; the seam exists so the map dispatcher can call
    ``build_*_patch`` uniformly.
    """
    return build_contour_layer_fn(
        tf,
        values,
        geom,
        n_contours=n_contours,
        layer_id=layer_id,
        contour_color=contour_color,
        origin=origin,
    )


def build_velocity_layer(
    tf: TelemacFileProtocol,
    tidx: int,
    geom: MeshGeometry,
    origin: list[float] | None = None,
) -> dict | None:
    """Build velocity arrow layer from U/V components."""
    pair = find_velocity_pair(tf.varnames)
    if pair is None:
        return None

    u = tf.get_data_value(pair[0], tidx)
    v = tf.get_data_value(pair[1], tidx)
    x, y = tf.meshx, tf.meshy
    npoin = tf.npoin2
    x_off, y_off = geom.x_off, geom.y_off

    mag = np.sqrt(u[:npoin] ** 2 + v[:npoin] ** 2)
    max_mag = float(mag.max())
    if max_mag < 1e-10:
        return None

    step = max(1, npoin // 1500)
    idx = np.arange(0, npoin, step)

    extent = max(
        float(x[:npoin].max() - x[:npoin].min()),
        float(y[:npoin].max() - y[:npoin].min()),
        1.0,
    )
    # Scale arrows so the longest is ~1/30th of the mesh extent
    arrow_scale = extent / (max_mag * 30)

    # Vectorised filter + positioning; then emit in deck.gl binary-attribute
    # mode so the per-tick wire payload is ~5x smaller than a list of dicts.
    mask = mag[idx] >= max_mag * 0.01
    kept = idx[mask]
    if kept.size == 0:
        return None

    sx = (x[kept] - x_off).astype(np.float32)
    sy = (y[kept] - y_off).astype(np.float32)
    tx = sx + (u[kept] * arrow_scale).astype(np.float32)
    ty = sy + (v[kept] * arrow_scale).astype(np.float32)
    src_xy = np.column_stack([sx, sy])
    tgt_xy = np.column_stack([tx, ty])

    return line_layer(
        "velocity",
        data={"length": int(kept.size)},
        getSourcePosition=encode_binary_attribute(src_xy),
        getTargetPosition=encode_binary_attribute(tgt_xy),
        getColor=[20, 20, 20, 160],
        getWidth=2,
        widthMinPixels=1,
        widthMaxPixels=3,
        pickable=False,
        coordinateSystem=_COORD_METER_OFFSETS,
        coordinateOrigin=origin or [0, 0],
    )


def build_contour_layer_fn(
    tf: TelemacFileProtocol,
    values: np.ndarray,
    geom: MeshGeometry,
    n_contours: int = 6,
    layer_id: str = "contours",
    contour_color: list[int] | None = None,
    origin: list[float] | None = None,
) -> dict | None:
    """Build FEM-exact contour lines using marching triangles.

    Walks each triangle to find edges where the field value crosses
    contour thresholds, interpolates the exact crossing point, and
    builds line segments. Much more accurate than grid-based KDE
    contouring for unstructured meshes.
    """
    npoin = tf.npoin2
    ikle = tf.ikle2
    x, y = tf.meshx, tf.meshy
    x_off, y_off = geom.x_off, geom.y_off
    vmin, vmax = float(np.nanmin(values[:npoin])), float(np.nanmax(values[:npoin]))
    if vmax == vmin:
        return None

    c_color = contour_color or [0, 0, 0]
    step = (vmax - vmin) / (n_contours + 1)
    # Slight perturbation avoids gaps when threshold lands exactly on a node value
    eps = (vmax - vmin) * 1e-10
    thresholds = [vmin + step * (i + 1) + eps for i in range(n_contours)]

    # Pre-compute centered coordinates
    cx = (x[:npoin] - x_off).astype(np.float64)
    cy = (y[:npoin] - y_off).astype(np.float64)
    v = values[:npoin].astype(np.float64)
    i0, i1, i2 = ikle[:, 0], ikle[:, 1], ikle[:, 2]

    # Helper: find edge crossings for all elements (vectorized)
    def _edge_crossings(da, db, ia, ib):
        cross = (da * db) < 0  # different signs = crossing
        denom = da - db
        valid = cross & (np.abs(denom) > 1e-12)
        t = np.where(
            valid,
            da / np.where(valid, denom, 1.0),
            0.0,
        )
        px = np.where(cross, cx[ia] + t * (cx[ib] - cx[ia]), 0.0)
        py = np.where(cross, cy[ia] + t * (cy[ib] - cy[ia]), 0.0)
        return cross, px, py

    all_src = []  # source positions for line segments
    all_tgt = []  # target positions for line segments

    for threshold in thresholds:
        d0 = v[i0] - threshold
        d1 = v[i1] - threshold
        d2 = v[i2] - threshold

        c01, px01, py01 = _edge_crossings(d0, d1, i0, i1)
        c12, px12, py12 = _edge_crossings(d1, d2, i1, i2)
        c20, px20, py20 = _edge_crossings(d2, d0, i2, i0)

        # Each triangle with exactly 2 crossings produces one line segment
        n_cross = c01.astype(int) + c12.astype(int) + c20.astype(int)
        has_contour = n_cross == 2
        if not has_contour.any():
            continue

        # Vectorized extraction of the two crossing points per triangle
        # For triangles with 2 crossings, exactly 2 of (c01, c12, c20) are True
        # Use conditional stacking to get the two points
        idx = np.where(has_contour)[0]
        # Collect crossing coordinates for each edge at contour triangles
        pts_x = np.column_stack([px01[idx], px12[idx], px20[idx]])  # (n, 3)
        pts_y = np.column_stack([py01[idx], py12[idx], py20[idx]])  # (n, 3)
        mask = np.column_stack([c01[idx], c12[idx], c20[idx]])  # (n, 3) bool

        # For each row, pick the 2 True columns.
        # Since exactly 2 are True (normal case), argmax on cumsum gives the first,
        # and flip+argmax gives the second. In rare saddle-point cases (all 3 True),
        # this picks columns 0 and 2, drawing a valid but simplified contour segment.
        first = np.argmax(mask, axis=1)
        flipped = mask[:, ::-1]
        last = 2 - np.argmax(flipped, axis=1)

        n = len(idx)
        rows = np.arange(n)
        sx = pts_x[rows, first]
        sy = pts_y[rows, first]
        tx = pts_x[rows, last]
        ty = pts_y[rows, last]

        all_src.extend(zip(sx.tolist(), sy.tolist()))
        all_tgt.extend(zip(tx.tolist(), ty.tolist()))

    if not all_src:
        return None

    # Binary-attribute mode: deck.gl reads positions from a base64-encoded
    # Float32Array at index i, dramatically smaller on the wire than the
    # equivalent list of {sourcePosition, targetPosition} dicts.
    src_xy = np.array(all_src, dtype=np.float32).reshape(-1, 2)
    tgt_xy = np.array(all_tgt, dtype=np.float32).reshape(-1, 2)

    return line_layer(
        layer_id,
        data={"length": int(src_xy.shape[0])},
        getSourcePosition=encode_binary_attribute(src_xy),
        getTargetPosition=encode_binary_attribute(tgt_xy),
        getColor=c_color + [180] if len(c_color) == 3 else c_color,
        getWidth=1,
        widthMinPixels=1,
        widthMaxPixels=2,
        pickable=False,
        coordinateSystem=_COORD_METER_OFFSETS,
        coordinateOrigin=origin or [0, 0],
    )


def build_marker_layer(
    x_m: float, y_m: float, layer_id: str = "marker", origin: list[float] | None = None
) -> dict:
    """Build a single-point scatterplot layer to mark clicked location."""
    return scatterplot_layer(
        layer_id,
        [{"position": [x_m, y_m]}],
        getPosition="@@=d.position",
        getColor=[255, 0, 0, 220],
        getRadius=8,
        radiusMinPixels=6,
        radiusMaxPixels=12,
        pickable=False,
        coordinateSystem=_COORD_METER_OFFSETS,
        coordinateOrigin=origin or [0, 0],
    )


def build_cross_section_layer(
    points_m: list[list[float]], origin: list[float] | None = None
) -> dict:
    """Build a path layer showing the cross-section polyline on the map.

    points_m: list of [x, y] in meters relative to mesh center.
    """
    return path_layer(
        "cross-section",
        [{"path": points_m}],
        getPath="@@=d.path",
        getColor=[255, 50, 50, 200],
        getWidth=3,
        widthMinPixels=2,
        widthMaxPixels=5,
        pickable=False,
        coordinateSystem=_COORD_METER_OFFSETS,
        coordinateOrigin=origin or [0, 0],
    )


def build_particle_layer(
    paths: list[list[list[float]]],
    current_time: float,
    trail_length: float,
    origin: list[float] | None = None,
) -> dict:
    """Build a TripsLayer for particle trace animation.

    paths: list of particle trajectories, each is list of [x, y, timestamp].
    Coordinates in meters relative to mesh center.
    """
    return trips_layer(
        "particles",
        [{"path": p} for p in paths],
        getPath="@@=d.path",
        getColor=[255, 80, 20, 200],
        currentTime=current_time,
        trailLength=trail_length,
        widthMinPixels=2,
        coordinateSystem=_COORD_METER_OFFSETS,
        coordinateOrigin=origin or [0, 0],
    )


def build_wireframe_layer(
    tf: TelemacFileProtocol, geom: MeshGeometry, origin: list[float] | None = None
) -> dict:
    """Build mesh wireframe as line segments (triangle edges)."""
    from analysis import compute_unique_edges

    x, y = tf.meshx, tf.meshy
    ikle = tf.ikle2
    x_off, y_off = geom.x_off, geom.y_off

    # Extract unique edges using shared helper
    unique_keys, a_idx_raw, b_idx_raw = compute_unique_edges(ikle)

    # Subsample if too many edges (>50k)
    step = max(1, len(unique_keys) // 50000)
    a_idx = a_idx_raw[::step]
    b_idx = b_idx_raw[::step]

    src_x = (x[a_idx] - x_off).tolist()
    src_y = (y[a_idx] - y_off).tolist()
    tgt_x = (x[b_idx] - x_off).tolist()
    tgt_y = (y[b_idx] - y_off).tolist()
    lines = [
        {"sourcePosition": [sx, sy], "targetPosition": [tx, ty]}
        for sx, sy, tx, ty in zip(src_x, src_y, tgt_x, tgt_y)
    ]

    return line_layer(
        "wireframe",
        lines,
        getColor=_WIRE_COLOR,
        getWidth=1,
        widthMinPixels=1,
        widthMaxPixels=1,
        pickable=False,
        coordinateSystem=_COORD_METER_OFFSETS,
        coordinateOrigin=origin or [0, 0],
    )


def build_extrema_markers(
    extrema: dict[str, tuple],
    x_off: float,
    y_off: float,
    origin: list[float] | None = None,
) -> list[dict]:
    """Build scatterplot markers for min/max value locations."""
    _colors = {"min": [0, 100, 255, 220], "max": [255, 50, 0, 220]}
    layers = []
    for key in ("min", "max"):
        _, x_m, y_m, val = extrema[key]
        color = _colors[key]
        layers.append(
            scatterplot_layer(
                f"extrema-{key}",
                [{"position": [float(x_m - x_off), float(y_m - y_off)]}],
                getPosition="@@=d.position",
                getColor=color,
                getRadius=12,
                radiusMinPixels=8,
                radiusMaxPixels=16,
                stroked=True,
                lineWidthMinPixels=2,
                getFillColor=[255, 255, 255, 180],
                getLineColor=color,
                pickable=False,
                coordinateSystem=_COORD_METER_OFFSETS,
                coordinateOrigin=origin or [0, 0],
            )
        )
    return layers


def build_measurement_layer(
    points_m: list[list[float]], origin: list[float] | None = None
) -> list[dict]:
    """Build a line + endpoint markers for distance measurement."""
    layers = []
    if len(points_m) >= 2:
        layers.append(
            line_layer(
                "measurement-line",
                [{"sourcePosition": points_m[0], "targetPosition": points_m[1]}],
                getColor=[255, 165, 0, 220],
                getWidth=3,
                widthMinPixels=2,
                widthMaxPixels=4,
                pickable=False,
                coordinateSystem=_COORD_METER_OFFSETS,
                coordinateOrigin=origin or [0, 0],
            )
        )
    for i, pt in enumerate(points_m):
        layers.append(
            scatterplot_layer(
                f"measure-pt-{i}",
                [{"position": pt}],
                getPosition="@@=d.position",
                getColor=[255, 165, 0, 220],
                getRadius=6,
                radiusMinPixels=5,
                radiusMaxPixels=10,
                pickable=False,
                coordinateSystem=_COORD_METER_OFFSETS,
                coordinateOrigin=origin or [0, 0],
            )
        )
    return layers


def build_boundary_layer(
    tf: TelemacFileProtocol,
    geom: MeshGeometry,
    boundary_nodes: list[int],
    bc_types: dict[int, int] | None = None,
    boundary_edges: tuple | None = None,
    origin: list[float] | None = None,
) -> list[dict]:
    """Build boundary edge lines color-coded by hydrodynamic type.

    Returns a list of layers: one line layer per boundary type plus
    a scatterplot layer for prescribed-value nodes.

    LIHBOR codes: 2=wall (gray), 4=free/Neumann (green), 5=prescribed (blue).
    """
    from analysis import find_boundary_edges as _find_edges

    x, y = tf.meshx, tf.meshy
    x_off, y_off = geom.x_off, geom.y_off

    # Get boundary edges
    if boundary_edges is not None:
        _, nodes_a, nodes_b = boundary_edges
    else:
        _, nodes_a, nodes_b = _find_edges(tf)

    _BC_COLORS = {
        2: [160, 160, 170, 220],  # wall — light gray
        4: [0, 200, 80, 220],  # free/Neumann — green
        5: [40, 120, 255, 240],  # prescribed H or Q — blue
    }
    _BC_LABELS = {2: "Wall", 4: "Free", 5: "Prescribed"}
    default_color = [255, 100, 255, 180]  # magenta fallback

    # Group edges by boundary type
    edges_by_type: dict[int, list[dict]] = {}
    prescribed_nodes: list[dict] = []

    for na, nb in zip(nodes_a.tolist(), nodes_b.tolist()):
        # Determine edge type from both endpoint nodes
        type_a = bc_types.get(na + 1, 0) if bc_types else 0
        type_b = bc_types.get(nb + 1, 0) if bc_types else 0
        # Use the more specific (higher) code
        bc_type = max(type_a, type_b) if (type_a and type_b) else (type_a or type_b)
        if bc_type == 0:
            bc_type = 2  # default to wall if no .cli data

        edges_by_type.setdefault(bc_type, []).append(
            {
                "sourcePosition": [float(x[na] - x_off), float(y[na] - y_off)],
                "targetPosition": [float(x[nb] - x_off), float(y[nb] - y_off)],
            }
        )

        # Collect prescribed nodes for marker display
        if bc_type == 5:
            for n in (na, nb):
                prescribed_nodes.append(
                    {
                        "position": [float(x[n] - x_off), float(y[n] - y_off)],
                    }
                )

    layers = []
    for bc_type, edges in edges_by_type.items():
        color = _BC_COLORS.get(bc_type, default_color)
        label = _BC_LABELS.get(bc_type, f"BC {bc_type}")
        layers.append(
            line_layer(
                f"boundary-{label.lower()}",
                edges,
                getColor=color,
                getWidth=3,
                widthMinPixels=2,
                widthMaxPixels=5,
                pickable=False,
                coordinateSystem=_COORD_METER_OFFSETS,
                coordinateOrigin=origin or [0, 0],
            )
        )

    # Add diamond markers at prescribed boundary nodes (inlets/outlets)
    if prescribed_nodes:
        # Deduplicate by position
        seen = set()
        unique = []
        for p in prescribed_nodes:
            key = (p["position"][0], p["position"][1])
            if key not in seen:
                seen.add(key)
                unique.append(p)
        layers.append(
            scatterplot_layer(
                "boundary-prescribed-markers",
                unique,
                getPosition="@@=d.position",
                getFillColor=[40, 120, 255, 180],
                getLineColor=[255, 255, 255, 220],
                getRadius=6,
                radiusMinPixels=4,
                radiusMaxPixels=10,
                stroked=True,
                lineWidthMinPixels=1,
                pickable=False,
                coordinateSystem=_COORD_METER_OFFSETS,
                coordinateOrigin=origin or [0, 0],
            )
        )

    return layers


def build_polygon_layer(
    polygon_coords: list[list[float]], origin: list[float] | None = None
) -> dict:
    """Build a GeoJsonLayer outlining the user-drawn polygon.

    polygon_coords: list of [x_m, y_m] points in mesh-offset coordinates.
    """
    coords = list(polygon_coords)
    if coords and coords[0] != coords[-1]:
        coords.append(coords[0])

    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords],
                },
                "properties": {},
            }
        ],
    }

    return layer(
        "GeoJsonLayer",
        id="polygon-overlay",
        data=geojson,
        filled=True,
        getFillColor=[255, 200, 0, 40],
        getLineColor=[255, 160, 0, 200],
        getLineWidth=2,
        lineWidthMinPixels=2,
        pickable=False,
        coordinateSystem=_COORD_METER_OFFSETS,
        coordinateOrigin=origin or [0, 0],
    )
