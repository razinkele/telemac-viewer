"""Shared test helpers — FakeTF and variants."""
from __future__ import annotations
import numpy as np


class FakeTF:
    """Minimal TelemacFile-like object for unit testing.

    Mesh: 4 nodes forming a unit square, 2 right-isosceles triangles.

        Node 2 (0,1) --- Node 3 (1,1)
           |  \\            |
           |    \\  tri 1   |
           |      \\        |
           | tri 0  \\      |
           |          \\    |
        Node 0 (0,0) --- Node 1 (1,0)

    Variables (3 timesteps at t=0, 1, 2):
        VELOCITY U: x-gradient [0, 1, 0, 1] -- constant across timesteps
        VELOCITY V: y-gradient [0, 0, 1, 1] -- constant across timesteps
        WATER DEPTH: [0.1, 0.5, 0.5, 1.0] scaled by (1 + tidx * 0.5)
    """
    meshx = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    meshy = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    ikle2 = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    npoin2 = 4
    nelem2 = 2
    varnames = ["VELOCITY U", "VELOCITY V", "WATER DEPTH"]
    times = [0.0, 1.0, 2.0]
    nplan = 0

    _data = {
        "VELOCITY U": np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64),
        "VELOCITY V": np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
        "WATER DEPTH": np.array([0.1, 0.5, 0.5, 1.0], dtype=np.float64),
    }

    def get_data_value(self, varname, tidx):
        base = self._data[varname].copy()
        if varname == "WATER DEPTH":
            base = base * (1.0 + tidx * 0.5)
        return base

    def get_data_on_points(self, varname, tidx, points):
        """Nearest-node interpolation for each point."""
        vals = self.get_data_value(varname, tidx)
        result = []
        for px, py in points:
            dists = (self.meshx - px) ** 2 + (self.meshy - py) ** 2
            result.append(float(vals[np.argmin(dists)]))
        return result

    def get_timeseries_on_points(self, varname, points):
        """Return time series at nearest node for each point."""
        result = []
        for px, py in points:
            dists = (self.meshx - px) ** 2 + (self.meshy - py) ** 2
            nearest = np.argmin(dists)
            ts = [
                float(self.get_data_value(varname, t)[nearest])
                for t in range(len(self.times))
            ]
            result.append(np.array(ts))
        return result

    def get_data_on_polyline(self, varname, record, polyline):
        """Sample along polyline using nearest-node interpolation."""
        vals = self.get_data_value(varname, record)
        points_out, abscissa, values_out = [], [], []
        cum_dist = 0.0
        for i, (px, py) in enumerate(polyline):
            if i > 0:
                dx = px - polyline[i - 1][0]
                dy = py - polyline[i - 1][1]
                cum_dist += np.sqrt(dx ** 2 + dy ** 2)
            dists = (self.meshx - px) ** 2 + (self.meshy - py) ** 2
            nearest = np.argmin(dists)
            points_out.append([px, py])
            abscissa.append(cum_dist)
            values_out.append(float(vals[nearest]))
        return points_out, abscissa, values_out

    def get_z_name(self):
        raise KeyError("No Z variable in FakeTF")

    def close(self):
        pass


# Verify FakeTF satisfies the protocol (checked at import time by type checkers)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from viewer_types import TelemacFileProtocol
    _: TelemacFileProtocol = FakeTF()  # type: ignore[assignment]
