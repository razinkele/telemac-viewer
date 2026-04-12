# analysis.py
from __future__ import annotations
import io
import os
import ast
import glob
import operator
import re
import numpy as np
from typing import Any
from constants import _M2D
from viewer_types import TelemacFileProtocol
from telemac_defaults import find_velocity_pair
import logging

_logger = logging.getLogger(__name__)

# Derived variable definitions: name -> (required_vars, compute_fn)
DERIVED_VARIABLES = {
    "VELOCITY MAGNITUDE": {
        "requires": ["VELOCITY U", "VELOCITY V"],
        "compute": lambda tf, tidx: np.sqrt(
            tf.get_data_value("VELOCITY U", tidx) ** 2
            + tf.get_data_value("VELOCITY V", tidx) ** 2
        ),
    },
    "FROUDE NUMBER": {
        "requires": ["VELOCITY U", "VELOCITY V", "WATER DEPTH"],
        "compute": lambda tf, tidx: (
            lambda u, v, h: np.where(
                h > 0.001,
                np.sqrt(u**2 + v**2) / np.sqrt(9.81 * np.maximum(h, 0.001)),
                0.0,
            )
        )(
            tf.get_data_value("VELOCITY U", tidx),
            tf.get_data_value("VELOCITY V", tidx),
            tf.get_data_value("WATER DEPTH", tidx),
        ),
    },
    "VORTICITY": {
        "requires": ["VELOCITY U", "VELOCITY V"],
        "compute": "vorticity",  # special: uses mesh gradient, handled in compute_derived
    },
}


def get_available_derived(tf: TelemacFileProtocol) -> list[str]:
    """Return list of derived variable names available for this file."""
    varnames = [v.strip() for v in tf.varnames]
    available = []
    for name, spec in DERIVED_VARIABLES.items():
        if all(r in varnames for r in spec["requires"]):
            available.append(name)
    return available


def compute_derived(tf: TelemacFileProtocol, varname: str, tidx: int) -> np.ndarray:
    """Compute a derived variable. Returns numpy array."""
    spec = DERIVED_VARIABLES[varname]
    if spec["compute"] == "vorticity":
        return _compute_vorticity(tf, tidx)
    return spec["compute"](tf, tidx)


def get_var_values(tf: TelemacFileProtocol, varname: str, tidx: int) -> np.ndarray:
    """Get variable values, handling both file and derived variables.

    Use this instead of tf.get_data_value() when the variable name may be
    a derived quantity (e.g. VELOCITY MAGNITUDE, FROUDE NUMBER, VORTICITY).
    """
    if varname in DERIVED_VARIABLES and varname in get_available_derived(tf):
        return compute_derived(tf, varname, tidx)
    return tf.get_data_value(varname, tidx)


def _compute_vorticity(tf, tidx):
    """Compute vorticity (dv/dx - du/dy) on the mesh using per-element gradients."""
    x, y = tf.meshx, tf.meshy
    npoin = tf.npoin2
    ikle = tf.ikle2
    i0, i1, i2 = ikle[:, 0], ikle[:, 1], ikle[:, 2]

    u = tf.get_data_value("VELOCITY U", tidx)
    v = tf.get_data_value("VELOCITY V", tidx)

    # Area * 2 for each element
    area2 = (x[i1] - x[i0]) * (y[i2] - y[i0]) - (x[i2] - x[i0]) * (y[i1] - y[i0])
    safe_area2 = np.where(np.abs(area2) < 1e-30, 1.0, area2)

    # dv/dx
    dv_dx = (
        v[i0] * (y[i1] - y[i2]) + v[i1] * (y[i2] - y[i0]) + v[i2] * (y[i0] - y[i1])
    ) / safe_area2
    # du/dy
    du_dy = (
        u[i0] * (x[i2] - x[i1]) + u[i1] * (x[i0] - x[i2]) + u[i2] * (x[i1] - x[i0])
    ) / safe_area2

    elem_vort = dv_dx - du_dy
    elem_vort[np.abs(area2) < 1e-30] = 0.0

    return _sanitize_result(
        _scatter_to_vertices(ikle, elem_vort, npoin).astype(np.float32)
    )


def export_timeseries_csv(times: np.ndarray, values: np.ndarray, varname: str) -> str:
    """Format time series data as CSV string."""
    buf = io.StringIO()
    buf.write(f"Time (s),{varname}\n")
    for t, v in zip(times, values):
        buf.write(f"{t},{v}\n")
    return buf.getvalue()


def export_crosssection_csv(
    abscissa: np.ndarray, values: np.ndarray, varname: str
) -> str:
    """Format cross-section data as CSV string."""
    buf = io.StringIO()
    buf.write(f"Distance (m),{varname}\n")
    for d, v in zip(abscissa, values):
        buf.write(f"{d},{v}\n")
    return buf.getvalue()


def compute_discharge(
    tf: TelemacFileProtocol, tidx: int, polyline_m: list[list[float]]
) -> dict[str, Any]:
    """Compute discharge (Q) through a cross-section polyline.

    Q = integral of (velocity · normal) * depth along the section.
    Returns total discharge in m³/s and per-segment breakdown.
    """
    varnames = [v.strip() for v in tf.varnames]
    pair = find_velocity_pair(tf.varnames)
    if pair is None:
        return {
            "total_q": None,
            "segments": [],
            "error": "Missing VELOCITY U/V variables",
        }
    if "WATER DEPTH" not in varnames:
        return {
            "total_q": None,
            "segments": [],
            "error": "Missing WATER DEPTH variable",
        }

    total_q = 0.0
    segments = []
    skipped = 0

    for i in range(len(polyline_m) - 1):
        x1, y1 = polyline_m[i]
        x2, y2 = polyline_m[i + 1]
        # Midpoint
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        seg_len = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if seg_len < 1e-10:
            continue

        # Normal vector (perpendicular to segment, pointing left)
        nx, ny = -(y2 - y1) / seg_len, (x2 - x1) / seg_len

        try:
            u = float(tf.get_data_on_points(pair[0], tidx, [[mx, my]])[0])
            v = float(tf.get_data_on_points(pair[1], tidx, [[mx, my]])[0])
            h = float(tf.get_data_on_points("WATER DEPTH", tidx, [[mx, my]])[0])
        except (ValueError, IndexError, KeyError) as exc:
            if skipped == 0:
                _logger.warning("Discharge segment skipped: %s", exc)
            skipped += 1
            continue

        # Q = (V · n) * h * length
        vn = u * nx + v * ny
        q = vn * max(h, 0.0) * seg_len
        total_q += q
        segments.append({"length": seg_len, "q": q, "depth": h, "velocity_n": vn})

    return {"total_q": total_q, "segments": segments, "skipped": skipped}


def _element_areas(tf: TelemacFileProtocol) -> np.ndarray:
    """Vectorized element area computation. Returns (nelem,) float64 array."""
    x, y = tf.meshx, tf.meshy
    ikle = tf.ikle2
    i0, i1, i2 = ikle[:, 0], ikle[:, 1], ikle[:, 2]
    return (
        np.abs((x[i1] - x[i0]) * (y[i2] - y[i0]) - (x[i2] - x[i0]) * (y[i1] - y[i0]))
        / 2.0
    )


def _scatter_to_vertices(
    ikle: np.ndarray, elem_values: np.ndarray, npoin: int
) -> np.ndarray:
    """Scatter per-element values to vertices (average). Returns (npoin,) float64."""
    nodes = ikle.ravel()
    weights = np.repeat(elem_values, 3)
    vertex_sum = np.bincount(nodes, weights=weights, minlength=npoin).astype(np.float64)
    vertex_count = np.bincount(nodes, minlength=npoin).astype(np.float64)
    mask = vertex_count > 0
    vertex_sum[mask] /= vertex_count[mask]
    return vertex_sum


def _sanitize_result(arr: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with 0 in computed field results."""
    if not np.all(np.isfinite(arr)):
        bad_count = int(np.count_nonzero(~np.isfinite(arr)))
        _logger.warning(
            "%d non-finite values replaced with 0 in computed result", bad_count
        )
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def compute_courant_number(tf: TelemacFileProtocol, tidx: int) -> np.ndarray | None:
    """Compute Courant number (CFL) per vertex: CFL = V * dt / dx.

    dx is estimated as sqrt(average element area around each vertex).

    Note: dt is the output file interval between saved timesteps, not the
    solver's internal computational timestep. The true CFL governing
    numerical stability may be much smaller. This metric is useful for
    comparing relative flow intensity across the domain.
    """
    pair = find_velocity_pair(tf.varnames)
    if pair is None:
        return None

    npoin = tf.npoin2

    # Use interval around current timestep for variable output frequency
    if tidx > 0:
        dt = float(tf.times[tidx] - tf.times[tidx - 1])
    elif len(tf.times) > 1:
        dt = float(tf.times[1] - tf.times[0])
    else:
        dt = 1.0

    u = tf.get_data_value(pair[0], tidx)[:npoin]
    v = tf.get_data_value(pair[1], tidx)[:npoin]
    speed = np.sqrt(u**2 + v**2)

    areas = _element_areas(tf)
    avg_area = _scatter_to_vertices(tf.ikle2, areas, npoin)
    dx = np.sqrt(np.maximum(avg_area, 1e-20))
    cfl = speed * dt / dx
    return _sanitize_result(cfl.astype(np.float32))


def compute_element_area(tf: TelemacFileProtocol) -> np.ndarray:
    """Compute per-vertex element area (average of adjacent element areas)."""
    areas = _element_areas(tf)
    return _scatter_to_vertices(tf.ikle2, areas, tf.npoin2).astype(np.float32)


def compute_mesh_integral(
    tf: TelemacFileProtocol, values: np.ndarray, threshold: float | None = None
) -> dict[str, float]:
    """Compute area-weighted integral and statistics over the mesh.

    If threshold is set, only elements where all vertices exceed it are included.
    Returns dict with total_area, integral, mean, wetted_area, wetted_fraction.
    """
    ikle = tf.ikle2
    i0, i1, i2 = ikle[:, 0], ikle[:, 1], ikle[:, 2]
    areas = _element_areas(tf)
    tri_avg = (values[i0] + values[i1] + values[i2]) / 3.0

    total_area = float(areas.sum())
    integral = float((tri_avg * areas).sum())

    if threshold is not None:
        wetted_mask = (
            (values[i0] > threshold)
            & (values[i1] > threshold)
            & (values[i2] > threshold)
        )
        wetted_area = float(areas[wetted_mask].sum())
    else:
        wetted_area = total_area

    mean = integral / total_area if total_area > 0 else 0.0
    return {
        "total_area": total_area,
        "integral": integral,
        "mean": mean,
        "wetted_area": wetted_area,
        "wetted_fraction": wetted_area / total_area if total_area > 0 else 0.0,
    }


_SAFE_MATH = {
    "sqrt": np.sqrt,
    "abs": np.abs,
    "log": np.log,
    "log10": np.log10,
    "exp": np.exp,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "pi": np.pi,
    "maximum": np.maximum,
    "minimum": np.minimum,
    "where": np.where,
}

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

_SAFE_COMPARE = {
    ast.Gt: operator.gt,
    ast.Lt: operator.lt,
    ast.GtE: operator.ge,
    ast.LtE: operator.le,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}


def evaluate_expression(
    tf: TelemacFileProtocol, tidx: int, expression: str
) -> np.ndarray:
    """Evaluate a user-defined math expression safely using AST parsing.

    Variables are referenced by name with spaces replaced by underscores
    (e.g., "VELOCITY_U**2 + VELOCITY_V**2").
    Available functions: sqrt, abs, log, log10, exp, sin, cos, tan,
    maximum, minimum, where, pi.
    Returns numpy array or raises ValueError.
    """
    npoin = tf.npoin2
    namespace = {}
    for vname in tf.varnames:
        safe_name = vname.strip().replace(" ", "_")
        namespace[safe_name] = tf.get_data_value(vname, tidx)[:npoin].astype(np.float64)
    namespace.update(_SAFE_MATH)

    # Replace variable names in expression using word-boundary regex
    safe_expr = expression
    for vname in sorted(tf.varnames, key=len, reverse=True):
        safe_name = vname.strip().replace(" ", "_")
        safe_expr = re.sub(
            r"\b" + re.escape(vname.strip()) + r"\b", safe_name, safe_expr
        )

    try:
        tree = ast.parse(safe_expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Syntax error in expression: {e}") from e

    result = _ast_eval(tree.body, namespace)
    result_arr = np.asarray(result, dtype=np.float32)
    if result_arr.ndim == 0:
        return np.full(npoin, result_arr.item(), dtype=np.float32)
    return result_arr[:npoin]


def _ast_eval(node, ns):
    """Recursively evaluate an AST node against a safe namespace."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

    if isinstance(node, ast.Name):
        if node.id in ns:
            return ns[node.id]
        raise ValueError(f"Unknown variable or function: '{node.id}'")

    if isinstance(node, ast.BinOp):
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op(_ast_eval(node.left, ns), _ast_eval(node.right, ns))

    if isinstance(node, ast.UnaryOp):
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op(_ast_eval(node.operand, ns))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError(
                "Only simple function calls allowed (no methods or chaining)"
            )
        if node.keywords:
            raise ValueError("Keyword arguments are not supported in expressions")
        fname = node.func.id
        if fname not in ns or not callable(ns[fname]):
            raise ValueError(f"Unknown function: '{fname}'")
        args = [_ast_eval(a, ns) for a in node.args]
        return ns[fname](*args)

    if isinstance(node, ast.Compare):
        left = _ast_eval(node.left, ns)
        result = None
        for op_node, comparator in zip(node.ops, node.comparators):
            op = _SAFE_COMPARE.get(type(op_node))
            if op is None:
                raise ValueError(f"Unsupported comparison: {type(op_node).__name__}")
            right = _ast_eval(comparator, ns)
            cmp = op(left, right)
            result = cmp if result is None else np.logical_and(result, cmp)
            left = right
        return result

    if isinstance(node, ast.IfExp):
        test = _ast_eval(node.test, ns)
        body = _ast_eval(node.body, ns)
        orelse = _ast_eval(node.orelse, ns)
        return np.where(test, body, orelse)

    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def compute_slope(tf: TelemacFileProtocol, values: np.ndarray) -> np.ndarray:
    """Compute slope magnitude (gradient) of a scalar field on the mesh.

    Uses per-element gradient averaged to vertices. Returns per-vertex slope array.
    """
    x, y = tf.meshx, tf.meshy
    npoin = tf.npoin2
    ikle = tf.ikle2
    i0, i1, i2 = ikle[:, 0], ikle[:, 1], ikle[:, 2]

    # Vectorized area * 2
    area2 = (x[i1] - x[i0]) * (y[i2] - y[i0]) - (x[i2] - x[i0]) * (y[i1] - y[i0])
    # Avoid division by zero for degenerate triangles
    safe_area2 = np.where(np.abs(area2) < 1e-30, 1.0, area2)

    v0, v1, v2 = values[i0], values[i1], values[i2]
    dv_dx = (
        v0 * (y[i1] - y[i2]) + v1 * (y[i2] - y[i0]) + v2 * (y[i0] - y[i1])
    ) / safe_area2
    dv_dy = (
        v0 * (x[i2] - x[i1]) + v1 * (x[i0] - x[i2]) + v2 * (x[i1] - x[i0])
    ) / safe_area2
    elem_slope = np.sqrt(dv_dx**2 + dv_dy**2)
    # Zero out degenerate elements
    elem_slope[np.abs(area2) < 1e-30] = 0.0

    return _sanitize_result(
        _scatter_to_vertices(ikle, elem_slope, npoin).astype(np.float32)
    )


def export_all_variables_csv(
    tf: TelemacFileProtocol, tidx: int, x_m: float, y_m: float
) -> str:
    """Export all variable values at a point for a given timestep as CSV string."""
    nearest, _, _ = nearest_node(tf, x_m, y_m)

    buf = io.StringIO()
    buf.write(f"Variable,Value\n")
    for vname in tf.varnames:
        val = float(tf.get_data_value(vname, tidx)[nearest])
        buf.write(f"{vname},{val}\n")
    return buf.getvalue()


def find_boundary_nodes(tf: TelemacFileProtocol) -> list[int]:
    """Find boundary nodes of the 2D mesh.

    Boundary edges appear in only one triangle. Returns list of node indices.
    """
    _, nodes_a, nodes_b = find_boundary_edges(tf)
    return sorted(set(nodes_a.tolist() + nodes_b.tolist()))


def compute_unique_edges(ikle: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract unique edges from triangle connectivity.

    Returns (edge_keys, nodes_a, nodes_b) where nodes_a[i]-nodes_b[i]
    form the i-th unique edge.
    """
    e0 = np.column_stack([ikle[:, 0], ikle[:, 1]])
    e1 = np.column_stack([ikle[:, 1], ikle[:, 2]])
    e2 = np.column_stack([ikle[:, 2], ikle[:, 0]])
    all_edges = np.vstack([e0, e1, e2])
    all_edges.sort(axis=1)
    max_node = int(all_edges.max()) + 1
    edge_keys = all_edges[:, 0].astype(np.int64) * max_node + all_edges[:, 1].astype(
        np.int64
    )
    unique_keys, unique_idx = np.unique(edge_keys, return_index=True)
    nodes_a = (unique_keys // max_node).astype(np.int32)
    nodes_b = (unique_keys % max_node).astype(np.int32)
    return unique_keys, nodes_a, nodes_b


def find_boundary_edges(
    tf: TelemacFileProtocol,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find boundary edges of the 2D mesh.

    Returns (boundary_keys, nodes_a, nodes_b) where nodes_a[i]-nodes_b[i]
    form the i-th boundary edge.
    """
    ikle = tf.ikle2
    e0 = np.column_stack([ikle[:, 0], ikle[:, 1]])
    e1 = np.column_stack([ikle[:, 1], ikle[:, 2]])
    e2 = np.column_stack([ikle[:, 2], ikle[:, 0]])
    all_edges = np.vstack([e0, e1, e2])
    all_edges.sort(axis=1)

    max_node = int(all_edges.max()) + 1
    edge_keys = all_edges[:, 0].astype(np.int64) * max_node + all_edges[:, 1].astype(
        np.int64
    )
    unique_keys, counts = np.unique(edge_keys, return_counts=True)
    boundary_keys = unique_keys[counts == 1]

    nodes_a = (boundary_keys // max_node).astype(np.int32)
    nodes_b = (boundary_keys % max_node).astype(np.int32)
    return boundary_keys, nodes_a, nodes_b


def read_cli_file(cli_path: str) -> dict[int, int] | None:
    """Read TELEMAC .cli boundary condition file.

    Returns dict mapping node_number (1-based) to LIHBOR code:
      2 = solid wall, 4 = free (Neumann), 5 = prescribed H or Q.
    Returns None if file cannot be read.
    """
    try:
        with open(cli_path) as f:
            raw_lines = f.readlines()
    except OSError:
        return None
    try:
        bc_types: dict[int, int] = {}
        for line in raw_lines:
            parts = line.split()
            if len(parts) >= 12:
                lihbor = int(parts[0])
                node_num = int(parts[11])
                bc_types[node_num] = lihbor
        return bc_types if bc_types else None
    except (ValueError, IndexError) as exc:
        _logger.warning("Failed to parse .cli file '%s': %s", cli_path, exc)
        return None


def extract_layer_2d(values_3d: np.ndarray, npoin2: int, layer_k: int) -> np.ndarray:
    """Extract a single horizontal layer from 3D variable data.

    For TELEMAC-3D files with nplan vertical planes, variable arrays have
    length npoin2 * nplan. This extracts plane k as a 2D array of length npoin2.

    layer_k=0 is the bottom, layer_k=nplan-1 is the surface.
    """
    start = layer_k * npoin2
    return values_3d[start : start + npoin2].copy()


def polygon_zonal_stats(
    tf: TelemacFileProtocol, values: np.ndarray, polygon_m: list[list[float]],
    flood_threshold: float = 0.01,
    var_name: str = "",
) -> dict[str, float]:
    """Compute statistics of a variable within a drawn polygon.

    polygon_m: list of [x_m, y_m] vertices in mesh meters.
    flood_threshold: depth threshold for flooded fraction (default 0.01 m).
    var_name: variable name; flooded metrics only computed for depth-like vars.
    Returns dict with area, mean, min, max, count, flooded_area, flooded_fraction.
    """
    from matplotlib.path import Path

    if not polygon_m or len(polygon_m) < 3:
        return {
            "area": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0,
            "count": 0, "flooded_area": 0.0, "flooded_fraction": 0.0,
        }

    x, y = tf.meshx, tf.meshy
    npoin = tf.npoin2

    # Check which nodes are inside the polygon
    poly_path = Path(polygon_m)
    points = np.column_stack([x[:npoin], y[:npoin]])
    inside = poly_path.contains_points(points)

    n_inside = int(inside.sum())
    if n_inside == 0:
        return {
            "area": 0.0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "count": 0,
            "flooded_area": 0.0,
            "flooded_fraction": 0.0,
        }

    vals_inside = values[:npoin][inside]

    # Element-weighted mean: use areas of elements with all vertices inside
    ikle = tf.ikle2
    elem_areas = _element_areas(tf)
    inside_idx = np.where(inside)[0]
    inside_set = set(inside_idx.tolist())
    weighted_sum = 0.0
    total_area_elem = 0.0
    for ei in range(len(ikle)):
        n0, n1, n2 = ikle[ei]
        if n0 in inside_set and n1 in inside_set and n2 in inside_set:
            v0, v1, v2 = values[n0], values[n1], values[n2]
            if np.isnan(v0) or np.isnan(v1) or np.isnan(v2):
                continue
            a = float(elem_areas[ei])
            v_avg = float(v0 + v1 + v2) / 3.0
            weighted_sum += v_avg * a
            total_area_elem += a
    weighted_mean = weighted_sum / total_area_elem if total_area_elem > 0 else float(np.nanmean(vals_inside))

    # Estimate area using Shoelace formula on the polygon
    poly = np.array(polygon_m)
    n = len(poly)
    area = 0.5 * abs(
        sum(
            poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1]
            for i in range(n)
        )
    )

    # Flooded fraction — only meaningful for depth-like variables
    _DEPTH_KEYWORDS = {"DEPTH", "HAUTEUR", "WATER DEPTH", "FREE SURFACE"}
    is_depth = any(kw in var_name.upper() for kw in _DEPTH_KEYWORDS)
    flooded_area_elem = 0.0
    if is_depth:
        flooded = int((vals_inside > flood_threshold).sum())
        # Element-area-based flooded area (handles non-uniform meshes correctly)
        for ei in range(len(ikle)):
            n0, n1, n2 = ikle[ei]
            if n0 in inside_set and n1 in inside_set and n2 in inside_set:
                v0, v1, v2 = values[n0], values[n1], values[n2]
                if np.isnan(v0) or np.isnan(v1) or np.isnan(v2):
                    continue
                if (v0 > flood_threshold and v1 > flood_threshold
                        and v2 > flood_threshold):
                    flooded_area_elem += float(elem_areas[ei])
    else:
        flooded = 0

    return {
        "area": float(area),
        "mean": float(weighted_mean),
        "min": float(np.nanmin(vals_inside)),
        "max": float(np.nanmax(vals_inside)),
        "count": n_inside,
        "flooded_area": flooded_area_elem if is_depth else 0.0,
        "flooded_fraction": float(flooded_area_elem / total_area_elem) if total_area_elem > 0 and is_depth else 0.0,
    }


def nearest_node(
    tf: TelemacFileProtocol, x_m: float, y_m: float
) -> tuple[int, float, float]:
    """Find the nearest 2D mesh node to a point in mesh meters.

    Returns (node_index, node_x, node_y).
    """
    x, y = tf.meshx, tf.meshy
    npoin = tf.npoin2
    if npoin == 0:
        raise ValueError("Mesh has no 2D points (npoin2=0)")
    dists = (x[:npoin] - x_m) ** 2 + (y[:npoin] - y_m) ** 2
    idx = int(np.argmin(dists))
    return idx, float(x[idx]), float(y[idx])


def time_series_at_point(
    tf: TelemacFileProtocol, varname: str, x_m: float, y_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Extract time series for a variable at a mesh point.

    Returns (times, values) arrays.  Supports derived variables.
    """
    if varname in DERIVED_VARIABLES and varname in get_available_derived(tf):
        idx, _, _ = nearest_node(tf, x_m, y_m)
        vals = np.array([compute_derived(tf, varname, t)[idx]
                         for t in range(len(tf.times))])
        return np.array(tf.times), vals
    ts = tf.get_timeseries_on_points(varname, [[x_m, y_m]])
    if not ts:
        return np.array(tf.times), np.full(len(tf.times), np.nan)
    return np.array(tf.times), ts[0]


def cross_section_profile(
    tf: TelemacFileProtocol, varname: str, record: int, polyline_m: list[list[float]]
) -> tuple[np.ndarray, np.ndarray]:
    """Extract variable values along a polyline.

    polyline_m: list of [x, y] in mesh meters.
    Returns (curvilinear_abscissa, values).  Supports derived variables.
    """
    if varname in DERIVED_VARIABLES and varname in get_available_derived(tf):
        vals_full = compute_derived(tf, varname, record)
        points, abscissa, values = tf.get_data_on_polyline(
            tf.varnames[0], record, polyline_m)
        if not points or len(points) < 3:
            return np.array(abscissa), np.array(values)
        # Re-interpolate derived values at the polyline nodes
        from scipy.interpolate import LinearNDInterpolator
        mesh_pts = np.column_stack((tf.meshx[:tf.npoin2], tf.meshy[:tf.npoin2]))
        interp = LinearNDInterpolator(mesh_pts, vals_full[:tf.npoin2])
        pts_arr = np.array(points)
        derived_vals = interp(pts_arr[:, 0], pts_arr[:, 1])
        return np.array(abscissa), derived_vals
    points, abscissa, values = tf.get_data_on_polyline(varname, record, polyline_m)
    return np.array(abscissa), np.array(values)


def compute_particle_paths(
    tf: TelemacFileProtocol, seed_points: list[list[float]], x_off: float, y_off: float
) -> list[list[list[float]]]:
    """Compute Lagrangian particle trajectories from velocity field.

    Uses batched interpolation: fetches velocity field once per timestep,
    interpolates all active particles simultaneously.
    seed_points: list of [x_m, y_m] in mesh meters.
    Returns list of paths, each path is [[x_centered, y_centered, time], ...].
    """
    pair = find_velocity_pair(tf.varnames)
    if pair is None:
        return []

    ntimes = len(tf.times)
    if ntimes < 2:
        return []

    n_seeds = len(seed_points)
    if n_seeds == 0:
        return []

    # Initialize particle positions (mesh meters)
    pos = np.array(seed_points, dtype=np.float64)  # (n_seeds, 2)
    active = np.ones(n_seeds, dtype=bool)

    # Build triangulation for interpolation
    x_mesh, y_mesh = tf.meshx, tf.meshy
    npoin = tf.npoin2
    try:
        tri = tf.tri
    except AttributeError:
        _logger.info("TelemacFile has no .tri attribute; building from ikle2")
        from matplotlib.tri import Triangulation

        tri = Triangulation(x_mesh[:npoin], y_mesh[:npoin], tf.ikle2)
    finder = tri.get_trifinder()

    # Store paths as list of lists
    all_paths = [
        [[float(pos[i, 0] - x_off), float(pos[i, 1] - y_off), float(tf.times[0])]]
        for i in range(n_seeds)
    ]

    for t in range(ntimes - 1):
        if not active.any():
            break
        dt = float(tf.times[t + 1] - tf.times[t])
        if dt <= 0:
            continue

        # Fetch velocity field once for this timestep
        u_field = tf.get_data_value(pair[0], t)[:npoin]
        v_field = tf.get_data_value(pair[1], t)[:npoin]

        # Find which triangle each active particle is in
        active_idx = np.where(active)[0]
        tri_ids = finder(pos[active_idx, 0], pos[active_idx, 1])
        outside = tri_ids < 0
        # Deactivate particles outside the mesh
        active[active_idx[outside]] = False
        active_idx = active_idx[~outside]
        tri_ids = tri_ids[~outside]

        if len(active_idx) == 0:
            continue

        # Barycentric interpolation within each triangle
        ikle = tf.ikle2
        i0 = ikle[tri_ids, 0]
        i1 = ikle[tri_ids, 1]
        i2 = ikle[tri_ids, 2]

        px = pos[active_idx, 0]
        py = pos[active_idx, 1]

        # Barycentric coordinates
        x0, y0 = x_mesh[i0], y_mesh[i0]
        x1, y1 = x_mesh[i1], y_mesh[i1]
        x2, y2 = x_mesh[i2], y_mesh[i2]
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        denom = np.where(np.abs(denom) < 1e-30, 1.0, denom)
        w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
        w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
        w2 = 1.0 - w0 - w1

        # Interpolate velocity
        u_interp = w0 * u_field[i0] + w1 * u_field[i1] + w2 * u_field[i2]
        v_interp = w0 * v_field[i0] + w1 * v_field[i1] + w2 * v_field[i2]

        # Check for NaN
        nan_mask = np.isnan(u_interp) | np.isnan(v_interp)
        active[active_idx[nan_mask]] = False
        valid = ~nan_mask
        valid_idx = active_idx[valid]

        # Advance positions
        pos[valid_idx, 0] += u_interp[valid] * dt
        pos[valid_idx, 1] += v_interp[valid] * dt

        # Record path points
        time_val = float(tf.times[t + 1])
        for j, idx in enumerate(valid_idx):
            all_paths[idx].append(
                [
                    float(pos[idx, 0] - x_off),
                    float(pos[idx, 1] - y_off),
                    time_val,
                ]
            )

    # Filter to paths with >1 point
    return [p for p in all_paths if len(p) > 1]


def generate_seed_grid(
    tf: TelemacFileProtocol, n_target: int = 500
) -> list[list[float]]:
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
    aspect = max(0.01, min(aspect, 10000.0))  # clamp extreme ratios
    ny = max(2, int(np.sqrt(n_target / aspect)))
    nx = max(2, int(ny * aspect))
    # Cap total grid points to prevent memory explosion on degenerate meshes
    max_total = n_target * 4
    if nx * ny > max_total:
        scale = np.sqrt(max_total / (nx * ny))
        nx = max(2, int(nx * scale))
        ny = max(2, int(ny * scale))

    gx = np.linspace(xmin + dx * 0.05, xmax - dx * 0.05, nx)
    gy = np.linspace(ymin + dy * 0.05, ymax - dy * 0.05, ny)
    grid_x, grid_y = np.meshgrid(gx, gy)
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # Filter to points inside mesh using matplotlib triangulation
    try:
        tri = tf.tri
        finder = tri.get_trifinder()
        inside = finder(points[:, 0], points[:, 1]) >= 0
        return points[inside].tolist()
    except (AttributeError, RuntimeError, ValueError):
        _logger.warning(
            "Triangulation unavailable for seed filtering; returning empty seed list"
        )
        return []


def distribute_seeds_along_line(
    polyline_m: list[list[float]], n_seeds: int = 100
) -> list[list[float]]:
    """Distribute seed points evenly along a polyline.

    polyline_m: list of [x, y] in mesh meters.
    Returns list of [x, y] seed points.
    """
    if len(polyline_m) < 2:
        return polyline_m
    segments = np.array(polyline_m)
    diffs = np.diff(segments, axis=0)
    seg_lengths = np.sqrt((diffs**2).sum(axis=1))
    total_len = seg_lengths.sum()
    if total_len <= 0:
        return [polyline_m[0]]

    # Cumulative distances at each polyline vertex
    cum_dist = np.concatenate([[0], np.cumsum(seg_lengths)])
    targets = np.linspace(0, total_len, n_seeds, endpoint=False)

    # Find which segment each target falls in
    seg_idx = np.searchsorted(cum_dist[1:], targets, side="right")
    seg_idx = np.clip(seg_idx, 0, len(seg_lengths) - 1)

    # Interpolate within each segment
    t = np.where(
        seg_lengths[seg_idx] > 0,
        (targets - cum_dist[seg_idx]) / seg_lengths[seg_idx],
        0.0,
    )
    seeds_x = segments[seg_idx, 0] + t * diffs[seg_idx, 0]
    seeds_y = segments[seg_idx, 1] + t * diffs[seg_idx, 1]
    return np.column_stack([seeds_x, seeds_y]).tolist()


def compute_mesh_quality(tf: TelemacFileProtocol) -> np.ndarray:
    """Compute per-vertex mesh quality (0=worst, 1=best).

    Uses element aspect ratio: ratio of inscribed to circumscribed circle
    radii, normalized so equilateral triangle = 1.0. Averaged to vertices.
    """
    x, y = tf.meshx, tf.meshy
    npoin = tf.npoin2
    ikle = tf.ikle2
    i0, i1, i2 = ikle[:, 0], ikle[:, 1], ikle[:, 2]

    # Edge lengths
    a = np.sqrt((x[i1] - x[i2]) ** 2 + (y[i1] - y[i2]) ** 2)
    b = np.sqrt((x[i0] - x[i2]) ** 2 + (y[i0] - y[i2]) ** 2)
    c = np.sqrt((x[i0] - x[i1]) ** 2 + (y[i0] - y[i1]) ** 2)

    s = (a + b + c) / 2.0
    # Heron's formula (clamp to zero for degenerate triangles)
    area = np.sqrt(np.maximum(s * (s - a) * (s - b) * (s - c), 0.0))

    # Quality = 2 * inradius / circumradius, equilateral = 1.0
    valid = (area > 0) & (a * b * c > 0)
    elem_quality = np.zeros(len(ikle), dtype=np.float64)
    inradius = np.where(valid, area / s, 0.0)
    circumradius = np.where(valid, (a * b * c) / (4.0 * area + 1e-30), 1.0)
    elem_quality[valid] = np.minimum(2.0 * inradius[valid] / circumradius[valid], 1.0)

    return _sanitize_result(_scatter_to_vertices(ikle, elem_quality, npoin))


def vertical_profile_at_point(
    tf: TelemacFileProtocol, varname: str, tidx: int, x_m: float, y_m: float
) -> tuple[np.ndarray, np.ndarray, str]:
    """Extract vertical profile at a point for 3D files.

    Returns (elevations, values) arrays for each vertical plane.
    """
    nplan = getattr(tf, "nplan", 0)
    if nplan <= 1:
        return np.array([]), np.array([]), "Elevation (m)"

    npoin2 = tf.npoin2
    x, y = tf.meshx, tf.meshy
    dists = (x[:npoin2] - x_m) ** 2 + (y[:npoin2] - y_m) ** 2
    nearest_2d = int(np.argmin(dists))

    # Get variable values at all vertical planes for this 2D node
    vals_3d = tf.get_data_value(varname, tidx)
    values = np.array([float(vals_3d[nearest_2d + k * npoin2]) for k in range(nplan)])

    # Get elevations from Z coordinate
    elevation_label = "Elevation (m)"
    try:
        z_name = tf.get_z_name()
        z_3d = tf.get_data_value(z_name, tidx)
        elevations = np.array(
            [float(z_3d[nearest_2d + k * npoin2]) for k in range(nplan)]
        )
    except (KeyError, IndexError, AttributeError) as exc:
        _logger.warning("Elevation data unavailable for vertical profile: %s", exc)
        elevations = np.arange(nplan, dtype=np.float64)
        elevation_label = "Layer index (elevation unavailable)"

    return elevations, values, elevation_label


def find_extrema(tf: TelemacFileProtocol, values: np.ndarray) -> dict[str, tuple]:
    """Find locations of min and max values on the 2D mesh.

    Returns dict with 'min'/'max' keys, each containing (node_idx, x_m, y_m, value).
    """
    npoin = tf.npoin2
    v = values[:npoin]
    if npoin == 0 or not np.isfinite(v).any():
        return {
            "min": (-1, 0.0, 0.0, float("nan")),
            "max": (-1, 0.0, 0.0, float("nan")),
        }
    imin = int(np.nanargmin(v))
    imax = int(np.nanargmax(v))
    x, y = tf.meshx, tf.meshy
    return {
        "min": (imin, float(x[imin]), float(y[imin]), float(v[imin])),
        "max": (imax, float(x[imax]), float(y[imax]), float(v[imax])),
    }


def compute_all_temporal_stats(
    tf: TelemacFileProtocol, varname: str, threshold: float = 0.01
) -> dict[str, np.ndarray]:
    """Single-pass computation of all temporal statistics.

    Returns dict with keys: max_values, min_values, mean_values,
    envelope, arrival, duration.
    """
    ntimes = len(tf.times)
    npoin = tf.npoin2
    if ntimes == 0:
        z = np.zeros(npoin, dtype=np.float32)
        return {
            "max_values": z,
            "min_values": z,
            "mean_values": z,
            "envelope": z,
            "arrival": np.full(npoin, np.nan, dtype=np.float32),
            "duration": z,
        }

    first = tf.get_data_value(varname, 0)[:npoin]
    running_min = first.copy()
    running_max = first.copy()
    running_sum = first.copy().astype(np.float64)
    arrival = np.full(npoin, np.nan, dtype=np.float32)
    wet_count = np.zeros(npoin, dtype=np.int32)

    wet = first > threshold
    arrival[wet] = float(tf.times[0])
    wet_count[wet] += 1

    for t in range(1, ntimes):
        vals = tf.get_data_value(varname, t)[:npoin]
        running_min = np.minimum(running_min, vals)
        running_max = np.maximum(running_max, vals)
        running_sum += vals.astype(np.float64)
        wet = vals > threshold
        first_wet = wet & np.isnan(arrival)
        arrival[first_wet] = float(tf.times[t])
        wet_count[wet] += 1

    # Duration: use average dt
    if ntimes > 1:
        avg_dt = (tf.times[-1] - tf.times[0]) / (ntimes - 1)
    else:
        avg_dt = 1.0

    peak = running_max.copy()
    peak[peak < threshold] = 0.0

    return {
        "max_values": running_max,
        "min_values": running_min,
        "mean_values": (running_sum / ntimes).astype(np.float32),
        "envelope": peak.astype(np.float32),
        "arrival": arrival,
        "duration": (np.maximum(wet_count - 1, 0).astype(np.float64) * avg_dt).astype(np.float32),
    }


def compute_temporal_stats(
    tf: TelemacFileProtocol, varname: str
) -> dict[str, np.ndarray] | None:
    """Compute min, max, mean across all timesteps for a variable.

    Returns dict with 'min', 'max', 'mean' arrays (per-node).
    """
    ntimes = len(tf.times)
    if ntimes == 0:
        return None
    npoin = tf.npoin2
    first = get_var_values(tf, varname, 0)[:npoin]
    running_min = first.copy()
    running_max = first.copy()
    running_sum = first.copy().astype(np.float64)

    for t in range(1, ntimes):
        vals = get_var_values(tf, varname, t)[:npoin]
        running_min = np.minimum(running_min, vals)
        running_max = np.maximum(running_max, vals)
        running_sum += vals.astype(np.float64)

    return {
        "min": running_min,
        "max": running_max,
        "mean": (running_sum / ntimes).astype(np.float32),
    }


def compute_flood_envelope(tf, varname, threshold=0.01):
    """Compute maximum value over all timesteps per node.

    Returns per-node array of the peak value across the entire simulation.
    Nodes where the peak is below threshold are set to 0.
    """
    npoin = tf.npoin2
    ntimes = len(tf.times)
    if ntimes == 0:
        return np.zeros(npoin, dtype=np.float32)
    peak = get_var_values(tf, varname, 0)[:npoin].copy()
    for t in range(1, ntimes):
        vals = get_var_values(tf, varname, t)[:npoin]
        peak = np.maximum(peak, vals)
    peak[peak < threshold] = 0.0
    return peak.astype(np.float32)


def compute_flood_arrival(tf, varname="WATER DEPTH", threshold=0.01):
    """Compute flood arrival time per node.

    Returns per-node array of the simulation time (seconds) when the value
    first exceeds the threshold. Nodes never exceeding threshold get NaN.
    """
    npoin = tf.npoin2
    ntimes = len(tf.times)
    arrival = np.full(npoin, np.nan, dtype=np.float32)
    for t in range(ntimes):
        vals = get_var_values(tf, varname, t)[:npoin]
        newly_wet = np.isnan(arrival) & (vals > threshold)
        arrival[newly_wet] = float(tf.times[t])
    return arrival


def compute_flood_duration(tf, varname="WATER DEPTH", threshold=0.01):
    """Compute total flood duration per node.

    Returns per-node array of total time (seconds) spent above threshold.
    Accumulates forward intervals for each wet timestep (except the last).
    """
    npoin = tf.npoin2
    ntimes = len(tf.times)
    duration = np.zeros(npoin, dtype=np.float32)
    for t in range(ntimes - 1):
        vals = get_var_values(tf, varname, t)[:npoin]
        wet = vals > threshold
        dt = float(tf.times[t + 1] - tf.times[t])
        duration[wet] += dt
    return duration


def compute_difference(
    tf: TelemacFileProtocol, varname: str, tidx: int, ref_tidx: int
) -> np.ndarray:
    """Compute difference between current and reference timestep."""
    current = get_var_values(tf, varname, tidx)
    reference = get_var_values(tf, varname, ref_tidx)
    return current - reference


def find_cas_files(example_path: str) -> dict[str, str]:
    """Find .cas steering files in the same directory as an example .slf file."""
    directory = os.path.dirname(example_path)
    cas_files = sorted(glob.glob(os.path.join(directory, "*.cas")))
    return {os.path.basename(f): f for f in cas_files}


def detect_module_from_path(cas_path: str) -> str:
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
