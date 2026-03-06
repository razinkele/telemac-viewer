# geometry.py
import math
import numpy as np
from constants import _M2D


def build_mesh_geometry(tf):
    """Build static mesh geometry arrays for SimpleMeshLayer."""
    x, y = tf.meshx, tf.meshy
    npoin = tf.npoin2
    ikle = tf.ikle2

    x_off = float((x.min() + x.max()) / 2)
    y_off = float((y.min() + y.max()) / 2)

    positions = np.zeros((npoin, 3), dtype=np.float32)
    positions[:, 0] = (x[:npoin] - x_off).astype(np.float32)
    positions[:, 1] = (y[:npoin] - y_off).astype(np.float32)

    indices = ikle.flatten().astype(np.int32)

    extent_m = max(float(x.max() - x.min()), float(y.max() - y.min()), 1.0)
    extent_deg = extent_m / _M2D
    zoom = math.log2(600 * 360 / (256 * extent_deg)) if extent_deg > 0 else 0

    return {
        "npoin": npoin,
        "positions": positions.flatten().tolist(),
        "indices": indices.tolist(),
        "x_off": x_off, "y_off": y_off,
        "extent_m": extent_m,
        "zoom": zoom,
    }


def build_mesh_geometry_3d(tf, z_values, z_scale):
    """Build 3D extruded mesh geometry (top surface with elevation).

    Same as 2D but positions[:,2] = z_values * z_scale.
    """
    x, y = tf.meshx, tf.meshy
    npoin = tf.npoin2
    ikle = tf.ikle2

    x_off = float((x.min() + x.max()) / 2)
    y_off = float((y.min() + y.max()) / 2)

    positions = np.zeros((npoin, 3), dtype=np.float32)
    positions[:, 0] = (x[:npoin] - x_off).astype(np.float32)
    positions[:, 1] = (y[:npoin] - y_off).astype(np.float32)
    positions[:, 2] = (z_values[:npoin] * z_scale).astype(np.float32)

    indices = ikle.flatten().astype(np.int32)

    extent_m = max(float(x.max() - x.min()), float(y.max() - y.min()), 1.0)
    extent_deg = extent_m / _M2D
    zoom = math.log2(600 * 360 / (256 * extent_deg)) if extent_deg > 0 else 0

    return {
        "npoin": npoin,
        "positions": positions.flatten().tolist(),
        "indices": indices.tolist(),
        "x_off": x_off, "y_off": y_off,
        "extent_m": extent_m,
        "zoom": zoom,
    }
