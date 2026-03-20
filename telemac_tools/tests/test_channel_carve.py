"""Tests for channel carving and station→world coordinate transforms."""
from __future__ import annotations
import numpy as np
import pytest
from telemac_tools.domain.channel_carve import interpolate_thalweg, build_channel_points
from telemac_tools.model import CrossSection, Reach


def _make_reach():
    """Simple reach: 200m along x-axis, 3 XS perpendicular."""
    alignment = np.array([[0, 0], [100, 0], [200, 0]], dtype=np.float64)
    xs0 = CrossSection(
        station=0.0,
        coords=np.array([[0, -50, 5], [0, -25, 2], [0, 0, 0], [0, 25, 2], [0, 50, 5]]),
        mannings_n=[0.06, 0.035, 0.06],
        bank_stations=(25.0, 75.0),
        bank_coords=np.array([[0, -25], [0, 25]]),
    )
    xs1 = CrossSection(
        station=100.0,
        coords=np.array([[100, -50, 5], [100, -25, 2], [100, 0, -1], [100, 25, 2], [100, 50, 5]]),
        mannings_n=[0.06, 0.035, 0.06],
        bank_stations=(25.0, 75.0),
        bank_coords=np.array([[100, -25], [100, 25]]),
    )
    xs2 = CrossSection(
        station=200.0,
        coords=np.array([[200, -50, 5], [200, -25, 2], [200, 0, -2], [200, 25, 2], [200, 50, 5]]),
        mannings_n=[0.06, 0.035, 0.06],
        bank_stations=(25.0, 75.0),
        bank_coords=np.array([[200, -25], [200, 25]]),
    )
    return Reach(name="test", alignment=alignment, cross_sections=[xs0, xs1, xs2])


class TestInterpolateThalweg:
    def test_returns_xyz(self):
        pts = interpolate_thalweg(_make_reach(), spacing=50.0)
        assert pts.shape[1] == 3

    def test_spacing(self):
        pts = interpolate_thalweg(_make_reach(), spacing=50.0)
        assert pts.shape[0] >= 4  # 200m / 50m = ~5 points

    def test_elevation_interpolated(self):
        pts = interpolate_thalweg(_make_reach(), spacing=50.0)
        # Thalweg z: 0 at station 0, -1 at 100, -2 at 200 → monotonically decreasing
        assert pts[0, 2] > pts[-1, 2]


class TestBuildChannelPoints:
    def test_returns_points_and_segments(self):
        points, segments = build_channel_points(_make_reach(), spacing=50.0)
        assert points.shape[1] == 3
        assert segments.shape[1] == 2
        assert segments.shape[0] >= 1
