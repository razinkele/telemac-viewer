# geometry.py
from __future__ import annotations
import math
import numpy as np
from typing import Any
from shiny_deckgl import encode_binary_attribute
from constants import _M2D


def build_mesh_geometry(tf: Any, z_values: np.ndarray | None = None, z_scale: float = 1) -> dict[str, Any]:
    """Build mesh geometry for SimpleMeshLayer.

    Transforms TELEMAC mesh coordinates (arbitrary metric CRS) into
    centered-meter coordinates by subtracting the bounding box center
    (x_off, y_off). All layer rendering uses these centered coordinates
    via deck.gl's METER_OFFSETS coordinate system.

    If z_values and z_scale are provided, positions[:,2] is set to
    z_values * z_scale for 3D visualization.

    Returns dict with positions and indices as binary-encoded dicts
    (base64 transport for efficient WebSocket transfer), plus
    x_off/y_off (mesh center for round-trip), extent_m, zoom.
    """
    x, y = tf.meshx, tf.meshy
    npoin = tf.npoin2
    ikle = tf.ikle2

    x_off = float((x.min() + x.max()) / 2)
    y_off = float((y.min() + y.max()) / 2)

    positions = np.zeros((npoin, 3), dtype=np.float32)
    positions[:, 0] = (x[:npoin] - x_off).astype(np.float32)
    positions[:, 1] = (y[:npoin] - y_off).astype(np.float32)
    if z_values is not None:
        positions[:, 2] = (z_values[:npoin] * z_scale).astype(np.float32)

    indices = ikle.flatten().astype(np.int32)

    extent_m = max(float(x.max() - x.min()), float(y.max() - y.min()), 1.0)
    extent_deg = extent_m / _M2D
    zoom = math.log2(600 * 360 / (256 * extent_deg)) if extent_deg > 0 else 0

    return {
        "npoin": npoin,
        "positions": encode_binary_attribute(positions.flatten()),
        "indices": encode_binary_attribute(indices),
        "x_off": x_off, "y_off": y_off,
        "extent_m": extent_m,
        "zoom": zoom,
    }
