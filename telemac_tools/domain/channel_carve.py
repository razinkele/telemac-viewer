"""Station→world coordinate transforms and thalweg interpolation."""
from __future__ import annotations
import numpy as np
from telemac_tools.model import CrossSection, Reach


def interpolate_thalweg(reach: Reach, spacing: float = 10.0) -> np.ndarray:
    """Interpolate thalweg (deepest channel point) between cross-sections.

    Returns (N, 3) array of x, y, z points along the thalweg.
    """
    xs_list = sorted(reach.cross_sections, key=lambda xs: xs.station)
    if len(xs_list) < 2:
        if xs_list:
            xs = xs_list[0]
            min_idx = int(np.argmin(xs.coords[:, 2]))
            return xs.coords[min_idx:min_idx + 1]
        return np.zeros((0, 3))

    thalweg_stations = []
    thalweg_xyz = []
    for xs in xs_list:
        min_idx = int(np.argmin(xs.coords[:, 2]))
        thalweg_xyz.append(xs.coords[min_idx])
        thalweg_stations.append(xs.station)

    thalweg_xyz = np.array(thalweg_xyz)
    thalweg_stations = np.array(thalweg_stations)

    total = thalweg_stations[-1] - thalweg_stations[0]
    n_pts = max(int(total / spacing) + 1, 2)
    interp_stations = np.linspace(thalweg_stations[0], thalweg_stations[-1], n_pts)

    x = np.interp(interp_stations, thalweg_stations, thalweg_xyz[:, 0])
    y = np.interp(interp_stations, thalweg_stations, thalweg_xyz[:, 1])
    z = np.interp(interp_stations, thalweg_stations, thalweg_xyz[:, 2])

    return np.column_stack([x, y, z])


def build_channel_points(reach: Reach, spacing: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Build channel constraint points and segments for PSLG meshing.

    Returns:
        points: (N, 3) channel bed points.
        segments: (N-1, 2) segment index pairs.
    """
    points = interpolate_thalweg(reach, spacing)
    n = len(points)
    if n < 2:
        return points, np.zeros((0, 2), dtype=np.int32)
    segments = np.column_stack([np.arange(n - 1), np.arange(1, n)]).astype(np.int32)
    return points, segments
