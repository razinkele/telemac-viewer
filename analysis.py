# analysis.py
import io
import os
import numpy as np
from constants import _M2D

# Derived variable definitions: name -> (required_vars, compute_fn)
DERIVED_VARIABLES = {
    "VELOCITY MAGNITUDE": {
        "requires": ["VELOCITY U", "VELOCITY V"],
        "compute": lambda tf, tidx: np.sqrt(
            tf.get_data_value("VELOCITY U", tidx)**2 +
            tf.get_data_value("VELOCITY V", tidx)**2
        ),
    },
    "FROUDE NUMBER": {
        "requires": ["VELOCITY U", "VELOCITY V", "WATER DEPTH"],
        "compute": lambda tf, tidx: (
            lambda u, v, h: np.where(h > 0.001, np.sqrt(u**2 + v**2) / np.sqrt(9.81 * np.maximum(h, 0.001)), 0.0)
        )(tf.get_data_value("VELOCITY U", tidx), tf.get_data_value("VELOCITY V", tidx), tf.get_data_value("WATER DEPTH", tidx)),
    },
    "VORTICITY": {
        "requires": ["VELOCITY U", "VELOCITY V"],
        "compute": None,  # needs mesh info, computed separately
    },
}


def get_available_derived(tf):
    """Return list of derived variable names available for this file."""
    varnames = [v.strip() for v in tf.varnames]
    available = []
    for name, spec in DERIVED_VARIABLES.items():
        if name == "VORTICITY":
            continue  # skip for now — needs gradient computation
        if all(r in varnames for r in spec["requires"]):
            available.append(name)
    return available


def compute_derived(tf, varname, tidx):
    """Compute a derived variable. Returns numpy array."""
    spec = DERIVED_VARIABLES[varname]
    return spec["compute"](tf, tidx)


def export_timeseries_csv(times, values, varname):
    """Format time series data as CSV string."""
    buf = io.StringIO()
    buf.write(f"Time (s),{varname}\n")
    for t, v in zip(times, values):
        buf.write(f"{t},{v}\n")
    return buf.getvalue()


def export_crosssection_csv(abscissa, values, varname):
    """Format cross-section data as CSV string."""
    buf = io.StringIO()
    buf.write(f"Distance (m),{varname}\n")
    for d, v in zip(abscissa, values):
        buf.write(f"{d},{v}\n")
    return buf.getvalue()


def compute_mesh_integral(tf, values, threshold=None):
    """Compute area-weighted integral and statistics over the mesh.

    If threshold is set, only elements where all vertices exceed it are included.
    Returns dict with total_area, integral, mean, wetted_area, wetted_fraction.
    """
    x, y = tf.meshx, tf.meshy
    npoin = tf.npoin2
    ikle = tf.ikle2
    nelem = tf.nelem2

    total_area = 0.0
    integral = 0.0
    wetted_area = 0.0

    for e in range(nelem):
        i0, i1, i2 = ikle[e]
        x0, y0 = x[i0], y[i0]
        x1, y1 = x[i1], y[i1]
        x2, y2 = x[i2], y[i2]
        area = abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)) / 2.0

        total_area += area
        v_avg = (float(values[i0]) + float(values[i1]) + float(values[i2])) / 3.0
        integral += v_avg * area

        if threshold is not None:
            if values[i0] > threshold and values[i1] > threshold and values[i2] > threshold:
                wetted_area += area
        else:
            wetted_area += area

    mean = integral / total_area if total_area > 0 else 0.0
    return {
        "total_area": total_area,
        "integral": integral,
        "mean": mean,
        "wetted_area": wetted_area,
        "wetted_fraction": wetted_area / total_area if total_area > 0 else 0.0,
    }


def evaluate_expression(tf, tidx, expression):
    """Evaluate a user-defined expression using variable names.

    Variables are referenced by name (e.g., "VELOCITY_U**2 + VELOCITY_V**2").
    Spaces in variable names are replaced with underscores.
    Returns numpy array or raises ValueError.
    This is a local desktop app — eval with restricted builtins is acceptable.
    """
    npoin = tf.npoin2
    # Build namespace with all variables (spaces -> underscores)
    namespace = {"np": np}
    for vname in tf.varnames:
        safe_name = vname.strip().replace(" ", "_")
        namespace[safe_name] = tf.get_data_value(vname, tidx)[:npoin].astype(np.float64)

    # Provide common math functions
    namespace["sqrt"] = np.sqrt
    namespace["abs"] = np.abs
    namespace["log"] = np.log
    namespace["log10"] = np.log10
    namespace["exp"] = np.exp
    namespace["sin"] = np.sin
    namespace["cos"] = np.cos

    # Replace variable names with safe names in expression
    safe_expr = expression
    for vname in sorted(tf.varnames, key=len, reverse=True):
        safe_name = vname.strip().replace(" ", "_")
        safe_expr = safe_expr.replace(vname.strip(), safe_name)

    # eval with restricted builtins — local desktop app only  # noqa: S307
    result = eval(safe_expr, {"__builtins__": {}}, namespace)  # noqa: S307
    return np.asarray(result, dtype=np.float32)[:npoin]


def compute_slope(tf, values):
    """Compute slope magnitude (gradient) of a scalar field on the mesh.

    Uses per-element gradient averaged to vertices. Returns per-vertex slope array.
    """
    x, y = tf.meshx, tf.meshy
    npoin = tf.npoin2
    ikle = tf.ikle2
    nelem = tf.nelem2

    vertex_slope = np.zeros(npoin, dtype=np.float64)
    vertex_count = np.zeros(npoin, dtype=np.int32)

    for e in range(nelem):
        i0, i1, i2 = ikle[e]
        # Triangle vertex coordinates
        x0, y0 = x[i0], y[i0]
        x1, y1 = x[i1], y[i1]
        x2, y2 = x[i2], y[i2]
        v0, v1, v2 = float(values[i0]), float(values[i1]), float(values[i2])

        # Area * 2
        area2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area2) < 1e-30:
            continue

        # Gradient components
        dv_dx = (v0 * (y1 - y2) + v1 * (y2 - y0) + v2 * (y0 - y1)) / area2
        dv_dy = (v0 * (x2 - x1) + v1 * (x0 - x2) + v2 * (x1 - x0)) / area2
        slope = np.sqrt(dv_dx**2 + dv_dy**2)

        for node in (i0, i1, i2):
            vertex_slope[node] += slope
            vertex_count[node] += 1

    mask = vertex_count > 0
    vertex_slope[mask] /= vertex_count[mask]
    return vertex_slope.astype(np.float32)


def export_all_variables_csv(tf, tidx, x_m, y_m):
    """Export all variable values at a point for a given timestep as CSV string."""
    npoin = tf.npoin2
    x, y = tf.meshx, tf.meshy
    dists = (x[:npoin] - x_m)**2 + (y[:npoin] - y_m)**2
    nearest = int(np.argmin(dists))

    buf = io.StringIO()
    buf.write(f"Variable,Value\n")
    for vname in tf.varnames:
        val = float(tf.get_data_value(vname, tidx)[nearest])
        buf.write(f"{vname},{val}\n")
    return buf.getvalue()


def find_boundary_nodes(tf):
    """Find boundary nodes of the 2D mesh.

    Boundary edges appear in only one triangle. Returns list of node indices.
    """
    npoin = tf.npoin2
    ikle = tf.ikle2
    nelem = tf.nelem2

    # Count how many triangles share each edge
    edge_count = {}
    for e in range(nelem):
        i0, i1, i2 = ikle[e]
        for a, b in [(i0, i1), (i1, i2), (i2, i0)]:
            edge = (min(a, b), max(a, b))
            edge_count[edge] = edge_count.get(edge, 0) + 1

    # Boundary edges are shared by exactly 1 triangle
    boundary_nodes = set()
    for (a, b), count in edge_count.items():
        if count == 1:
            boundary_nodes.add(a)
            boundary_nodes.add(b)

    return sorted(boundary_nodes)


def coord_to_meters(lon, lat, x_off, y_off):
    """Convert pseudo-degree coordinate back to mesh meters."""
    return float(lon) * _M2D + x_off, float(lat) * _M2D + y_off


def meters_to_coord(x_m, y_m, x_off, y_off):
    """Convert mesh meters to centered coordinates (for layers)."""
    return float(x_m - x_off), float(y_m - y_off)


def time_series_at_point(tf, varname, x_m, y_m):
    """Extract time series for a variable at a mesh point.

    Returns (times, values) arrays.
    """
    ts = tf.get_timeseries_on_points(varname, [[x_m, y_m]])
    return np.array(tf.times), ts[0]


def cross_section_profile(tf, varname, record, polyline_m):
    """Extract variable values along a polyline.

    polyline_m: list of [x, y] in mesh meters.
    Returns (curvilinear_abscissa, values).
    """
    points, abscissa, values = tf.get_data_on_polyline(
        varname, record, polyline_m
    )
    return np.array(abscissa), np.array(values)


def compute_particle_paths(tf, seed_points, x_off, y_off):
    """Compute Lagrangian particle trajectories from velocity field.

    seed_points: list of [x_m, y_m] in mesh meters.
    Returns list of paths, each path is [[x_centered, y_centered, time], ...].
    """
    varnames = [v.strip() for v in tf.varnames]
    if "VELOCITY U" not in varnames or "VELOCITY V" not in varnames:
        return []

    ntimes = len(tf.times)
    if ntimes < 2:
        return []

    paths = []
    for sx, sy in seed_points:
        path = [[float(sx - x_off), float(sy - y_off), float(tf.times[0])]]
        x, y = float(sx), float(sy)
        for t in range(ntimes - 1):
            dt = float(tf.times[t + 1] - tf.times[t])
            if dt <= 0:
                continue
            try:
                u_val = tf.get_data_on_points("VELOCITY U", t, [[x, y]])
                v_val = tf.get_data_on_points("VELOCITY V", t, [[x, y]])
            except Exception:
                break
            u, v = float(u_val[0]), float(v_val[0])
            if np.isnan(u) or np.isnan(v):
                break
            x += u * dt
            y += v * dt
            path.append([float(x - x_off), float(y - y_off), float(tf.times[t + 1])])
        if len(path) > 1:
            paths.append(path)
    return paths


def generate_seed_grid(tf, n_target=500):
    """Generate a regular grid of seed points within the mesh bounding box.

    Returns list of [x_m, y_m] in mesh meters. Filters to points inside mesh.
    """
    x, y = tf.meshx, tf.meshy
    npoin = tf.npoin2
    xmin, xmax = float(x[:npoin].min()), float(x[:npoin].max())
    ymin, ymax = float(y[:npoin].min()), float(y[:npoin].max())

    dx = xmax - xmin
    dy = ymax - ymin
    aspect = dx / dy if dy > 0 else 1.0
    ny = max(2, int(np.sqrt(n_target / aspect)))
    nx = max(2, int(ny * aspect))

    gx = np.linspace(xmin + dx * 0.05, xmax - dx * 0.05, nx)
    gy = np.linspace(ymin + dy * 0.05, ymax - dy * 0.05, ny)
    grid_x, grid_y = np.meshgrid(gx, gy)
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # Filter to points inside mesh using matplotlib triangulation
    tri = tf.tri
    finder = tri.get_trifinder()
    inside = finder(points[:, 0], points[:, 1]) >= 0
    return points[inside].tolist()


def compute_mesh_quality(tf):
    """Compute per-vertex mesh quality (0=worst, 1=best).

    Uses element aspect ratio: ratio of inscribed to circumscribed circle
    radii, normalized so equilateral triangle = 1.0. Averaged to vertices.
    """
    x, y = tf.meshx, tf.meshy
    npoin = tf.npoin2
    ikle = tf.ikle2
    nelem = tf.nelem2

    # Compute per-element quality
    elem_quality = np.zeros(nelem, dtype=np.float64)
    for e in range(nelem):
        i0, i1, i2 = ikle[e]
        ax, ay = x[i0], y[i0]
        bx, by = x[i1], y[i1]
        cx, cy = x[i2], y[i2]
        a = np.sqrt((bx-cx)**2 + (by-cy)**2)
        b = np.sqrt((ax-cx)**2 + (ay-cy)**2)
        c = np.sqrt((ax-bx)**2 + (ay-by)**2)
        s = (a + b + c) / 2.0
        area = np.sqrt(max(s * (s-a) * (s-b) * (s-c), 0.0))
        if area > 0 and (a*b*c) > 0:
            # 2 * inradius / circumradius, normalized so equilateral = 1
            inradius = area / s
            circumradius = (a * b * c) / (4.0 * area)
            elem_quality[e] = min(2.0 * inradius / circumradius, 1.0)

    # Average to vertices
    vertex_quality = np.zeros(npoin, dtype=np.float64)
    vertex_count = np.zeros(npoin, dtype=np.int32)
    for e in range(nelem):
        for node in ikle[e]:
            vertex_quality[node] += elem_quality[e]
            vertex_count[node] += 1
    mask = vertex_count > 0
    vertex_quality[mask] /= vertex_count[mask]
    return vertex_quality


def vertical_profile_at_point(tf, varname, tidx, x_m, y_m):
    """Extract vertical profile at a point for 3D files.

    Returns (elevations, values) arrays for each vertical plane.
    """
    nplan = getattr(tf, 'nplan', 0)
    if nplan <= 1:
        return np.array([]), np.array([])

    npoin2 = tf.npoin2
    x, y = tf.meshx, tf.meshy
    dists = (x[:npoin2] - x_m)**2 + (y[:npoin2] - y_m)**2
    nearest_2d = int(np.argmin(dists))

    # Get variable values at all vertical planes for this 2D node
    vals_3d = tf.get_data_value(varname, tidx)
    values = np.array([float(vals_3d[nearest_2d + k * npoin2]) for k in range(nplan)])

    # Get elevations from Z coordinate
    try:
        z_name = tf.get_z_name()
        z_3d = tf.get_data_value(z_name, tidx)
        elevations = np.array([float(z_3d[nearest_2d + k * npoin2]) for k in range(nplan)])
    except Exception:
        elevations = np.arange(nplan, dtype=np.float64)

    return elevations, values


def find_extrema(tf, values):
    """Find locations of min and max values on the 2D mesh.

    Returns dict with 'min'/'max' keys, each containing (node_idx, x_m, y_m, value).
    """
    npoin = tf.npoin2
    v = values[:npoin]
    imin = int(np.argmin(v))
    imax = int(np.argmax(v))
    x, y = tf.meshx, tf.meshy
    return {
        "min": (imin, float(x[imin]), float(y[imin]), float(v[imin])),
        "max": (imax, float(x[imax]), float(y[imax]), float(v[imax])),
    }


def compute_temporal_stats(tf, varname):
    """Compute min, max, mean across all timesteps for a variable.

    Returns dict with 'min', 'max', 'mean' arrays (per-node).
    """
    ntimes = len(tf.times)
    npoin = tf.npoin2
    first = tf.get_data_value(varname, 0)[:npoin]
    running_min = first.copy()
    running_max = first.copy()
    running_sum = first.copy().astype(np.float64)

    for t in range(1, ntimes):
        vals = tf.get_data_value(varname, t)[:npoin]
        running_min = np.minimum(running_min, vals)
        running_max = np.maximum(running_max, vals)
        running_sum += vals.astype(np.float64)

    return {
        "min": running_min,
        "max": running_max,
        "mean": (running_sum / ntimes).astype(np.float32),
    }


def compute_difference(tf, varname, tidx, ref_tidx):
    """Compute difference between current and reference timestep."""
    current = tf.get_data_value(varname, tidx)
    reference = tf.get_data_value(varname, ref_tidx)
    return current - reference


def find_cas_files(example_path):
    """Find .cas steering files in the same directory as an example .slf file."""
    import glob
    directory = os.path.dirname(example_path)
    cas_files = sorted(glob.glob(os.path.join(directory, "*.cas")))
    return {os.path.basename(f): f for f in cas_files}


def detect_module(cas_path):
    """Detect which TELEMAC module a .cas file belongs to based on directory."""
    parts = cas_path.replace("\\", "/").split("/")
    for i, p in enumerate(parts):
        if p == "examples" and i + 1 < len(parts):
            module = parts[i + 1]
            module_map = {
                "telemac2d": "telemac2d",
                "telemac3d": "telemac3d",
                "artemis": "artemis",
                "tomawac": "tomawac",
                "gaia": "gaia",
                "khione": "khione",
                "waqtel": "waqtel",
                "mascaret": "mascaret",
            }
            return module_map.get(module, "telemac2d")
    # Fallback: guess from filename prefix
    basename = os.path.basename(cas_path).lower()
    if basename.startswith("t3d"):
        return "telemac3d"
    if basename.startswith("art"):
        return "artemis"
    if basename.startswith("tom") or basename.startswith("fom"):
        return "tomawac"
    return "telemac2d"
