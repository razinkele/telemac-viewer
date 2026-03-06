# analysis.py
import io
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
