# layers.py
import numpy as np
from shiny_deckgl import (
    layer,
    line_layer,
    contour_layer,
    scatterplot_layer,
    path_layer,
    trips_layer,
)

# deck.gl _COORD_METER_OFFSETS = 2
_COORD_METER_OFFSETS = 2
from constants import cached_palette_arr

_GRAY = np.array([0.85, 0.85, 0.85], dtype=np.float32)
_WIRE_COLOR = [100, 100, 100, 80]


def build_mesh_layer(geom, values, palette_id, filter_range=None,
                     color_range_override=None, log_scale=False, reverse_palette=False):
    """Build SimpleMeshLayer with per-vertex coloring and optional value filter."""
    npoin = geom["npoin"]

    vmin, vmax = float(np.nanmin(values[:npoin])), float(np.nanmax(values[:npoin]))
    if np.isnan(vmin):
        vmin, vmax = 0.0, 1.0
    if color_range_override is not None:
        cmin, cmax = color_range_override
        if cmin is not None and cmax is not None and cmax > cmin:
            vmin, vmax = float(cmin), float(cmax)
    if vmax == vmin:
        vmax = vmin + 1.0

    palette_arr = cached_palette_arr(palette_id, reverse=reverse_palette)
    log_applied = False
    if log_scale and vmin > 0:
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

    # Apply value filter: gray out vertices outside range
    if filter_range is not None:
        lo, hi = filter_range
        mask = (values[:npoin] < lo) | (values[:npoin] > hi)
        colors_f32[mask] = _GRAY

    lyr = layer("SimpleMeshLayer",
        "mesh",
        data=[{"position": [0, 0, 0]}],
        mesh="@@CustomGeometry",
        _meshPositions=geom["positions"],
        _meshNormals=[],
        _meshColors=colors_f32.flatten().tolist(),
        _meshIndices=geom["indices"],
        coordinateSystem=_COORD_METER_OFFSETS,
        coordinateOrigin=[0, 0],
        sizeScale=1,
        getPosition="@@=d.position",
        getColor=[255, 255, 255, 255],
        pickable=True,
    )
    return lyr, vmin, vmax, log_applied


def build_velocity_layer(tf, time_idx, geom):
    """Build velocity arrow layer from U/V components."""
    varnames = [v.strip() for v in tf.varnames]
    if "VELOCITY U" not in varnames or "VELOCITY V" not in varnames:
        return None

    u = tf.get_data_value("VELOCITY U", time_idx)
    v = tf.get_data_value("VELOCITY V", time_idx)
    x, y = tf.meshx, tf.meshy
    npoin = tf.npoin2
    x_off, y_off = geom["x_off"], geom["y_off"]

    mag = np.sqrt(u[:npoin]**2 + v[:npoin]**2)
    max_mag = float(mag.max())
    if max_mag < 1e-10:
        return None

    step = max(1, npoin // 1500)
    idx = np.arange(0, npoin, step)

    extent = max(float(x[:npoin].max() - x[:npoin].min()),
                 float(y[:npoin].max() - y[:npoin].min()), 1.0)
    arrow_scale = extent / (max_mag * 30)

    arrows = []
    for i in idx:
        if mag[i] < max_mag * 0.01:
            continue
        sx = float(x[i] - x_off)
        sy = float(y[i] - y_off)
        arrows.append({
            "sourcePosition": [sx, sy],
            "targetPosition": [sx + float(u[i] * arrow_scale),
                               sy + float(v[i] * arrow_scale)],
        })

    if not arrows:
        return None

    return line_layer(
        "velocity",
        arrows,
        getColor=[20, 20, 20, 160],
        getWidth=2,
        widthMinPixels=1,
        widthMaxPixels=3,
        pickable=False,
        coordinateSystem=_COORD_METER_OFFSETS,
        coordinateOrigin=[0, 0],
    )


def build_contour_layer_fn(tf, values, geom, n_contours=6, layer_id="contours",
                           contour_color=None):
    """Build a ContourLayer from mesh node positions and values."""
    x, y = tf.meshx, tf.meshy
    npoin = tf.npoin2
    x_off, y_off = geom["x_off"], geom["y_off"]
    vmin, vmax = float(np.nanmin(values[:npoin])), float(np.nanmax(values[:npoin]))
    if vmax == vmin:
        return None

    # Subsample if too many nodes (ContourLayer uses KDE, doesn't need all points)
    step = max(1, npoin // 5000)
    indices = np.arange(0, npoin, step)
    points = [{"position": [float(x[i] - x_off), float(y[i] - y_off)],
               "weight": float(values[i])}
              for i in indices]

    c_color = contour_color or [0, 0, 0]
    step = (vmax - vmin) / (n_contours + 1)
    contours = [
        {"threshold": vmin + step * (i + 1), "color": c_color, "strokeWidth": 1}
        for i in range(n_contours)
    ]

    dx = float(x[:npoin].max() - x[:npoin].min())
    dy = float(y[:npoin].max() - y[:npoin].min())
    cell_size = max(dx, dy) / 100

    return contour_layer(
        layer_id,
        points,
        contours=contours,
        cellSize=cell_size,
        getPosition="@@=d.position",
        getWeight="@@d.weight",
        pickable=False,
        coordinateSystem=_COORD_METER_OFFSETS,
        coordinateOrigin=[0, 0],
    )


def build_marker_layer(x_m, y_m, layer_id="marker"):
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
        coordinateOrigin=[0, 0],
    )


def build_cross_section_layer(points_m):
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
        coordinateOrigin=[0, 0],
    )


def build_particle_layer(paths, current_time, trail_length):
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
        coordinateOrigin=[0, 0],
    )


def build_wireframe_layer(tf, geom):
    """Build mesh wireframe as line segments (triangle edges)."""
    x, y = tf.meshx, tf.meshy
    ikle = tf.ikle2
    x_off, y_off = geom["x_off"], geom["y_off"]

    # Extract unique edges using integer key encoding (faster than np.unique with axis=0)
    e0 = np.column_stack([ikle[:, 0], ikle[:, 1]])
    e1 = np.column_stack([ikle[:, 1], ikle[:, 2]])
    e2 = np.column_stack([ikle[:, 2], ikle[:, 0]])
    all_edges = np.vstack([e0, e1, e2])
    all_edges.sort(axis=1)
    max_node = int(all_edges.max()) + 1
    edge_keys = all_edges[:, 0].astype(np.int64) * max_node + all_edges[:, 1].astype(np.int64)
    unique_keys = np.unique(edge_keys)

    # Subsample if too many edges (>50k)
    step = max(1, len(unique_keys) // 50000)
    sampled_keys = unique_keys[::step]

    # Decode back to node pairs
    a_idx = (sampled_keys // max_node).astype(np.int32)
    b_idx = (sampled_keys % max_node).astype(np.int32)
    src_x = (x[a_idx] - x_off).tolist()
    src_y = (y[a_idx] - y_off).tolist()
    tgt_x = (x[b_idx] - x_off).tolist()
    tgt_y = (y[b_idx] - y_off).tolist()
    lines = [{"sourcePosition": [sx, sy], "targetPosition": [tx, ty]}
             for sx, sy, tx, ty in zip(src_x, src_y, tgt_x, tgt_y)]

    return line_layer(
        "wireframe",
        lines,
        getColor=_WIRE_COLOR,
        getWidth=1,
        widthMinPixels=1,
        widthMaxPixels=1,
        pickable=False,
        coordinateSystem=_COORD_METER_OFFSETS,
        coordinateOrigin=[0, 0],
    )


def build_extrema_markers(extrema, x_off, y_off):
    """Build scatterplot markers for min/max value locations."""
    _colors = {"min": [0, 100, 255, 220], "max": [255, 50, 0, 220]}
    layers = []
    for key in ("min", "max"):
        _, x_m, y_m, val = extrema[key]
        color = _colors[key]
        layers.append(scatterplot_layer(
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
            coordinateOrigin=[0, 0],
        ))
    return layers


def build_measurement_layer(points_m):
    """Build a line + endpoint markers for distance measurement."""
    layers = []
    if len(points_m) >= 2:
        layers.append(line_layer(
            "measurement-line",
            [{"sourcePosition": points_m[0], "targetPosition": points_m[1]}],
            getColor=[255, 165, 0, 220],
            getWidth=3,
            widthMinPixels=2,
            widthMaxPixels=4,
            pickable=False,
            coordinateSystem=_COORD_METER_OFFSETS,
            coordinateOrigin=[0, 0],
        ))
    for i, pt in enumerate(points_m):
        layers.append(scatterplot_layer(
            f"measure-pt-{i}",
            [{"position": pt}],
            getPosition="@@=d.position",
            getColor=[255, 165, 0, 220],
            getRadius=6,
            radiusMinPixels=5,
            radiusMaxPixels=10,
            pickable=False,
            coordinateSystem=_COORD_METER_OFFSETS,
            coordinateOrigin=[0, 0],
        ))
    return layers


def build_boundary_layer(tf, geom, boundary_nodes):
    """Build scatterplot layer highlighting boundary nodes."""
    x, y = tf.meshx, tf.meshy
    x_off, y_off = geom["x_off"], geom["y_off"]
    data = [{"position": [float(x[n] - x_off), float(y[n] - y_off)]}
            for n in boundary_nodes]
    return scatterplot_layer(
        "boundary",
        data,
        getPosition="@@=d.position",
        getColor=[255, 0, 255, 160],
        getRadius=4,
        radiusMinPixels=2,
        radiusMaxPixels=6,
        pickable=False,
        coordinateSystem=_COORD_METER_OFFSETS,
        coordinateOrigin=[0, 0],
    )
