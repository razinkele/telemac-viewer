"""Tests for chart-builder helpers extracted from server_analysis."""

from __future__ import annotations

import numpy as np
import pytest
import plotly.graph_objects as go


class TestBuildTimeseriesChart:
    def test_empty_points_returns_empty_figure(self, fake_tf):
        from server_analysis import build_timeseries_chart

        fig = build_timeseries_chart(
            fake_tf,
            "WATER DEPTH",
            0,
            points=[],
            obs=None,
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_single_point_produces_one_trace(self, fake_tf):
        from server_analysis import build_timeseries_chart

        fig = build_timeseries_chart(
            fake_tf,
            "WATER DEPTH",
            0,
            points=[(0.0, 0.0)],
            obs=None,
        )
        # 1 model trace — vlines don't count in fig.data (they're in fig.layout.shapes)
        assert len(fig.data) == 1
        assert fig.data[0].mode == "lines"

    def test_three_points_produce_three_traces(self, fake_tf):
        from server_analysis import build_timeseries_chart

        pts = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        fig = build_timeseries_chart(
            fake_tf,
            "WATER DEPTH",
            0,
            points=pts,
            obs=None,
        )
        assert len(fig.data) == 3

    def test_observation_overlay_adds_obs_trace_and_rmse_annotation(self, fake_tf):
        from server_analysis import build_timeseries_chart

        obs_times = np.array([0.0, 1.0, 2.0, 3.0])
        obs_values = np.array([0.5, 0.6, 0.7, 0.6])
        obs = (obs_times, obs_values, "observed")
        fig = build_timeseries_chart(
            fake_tf,
            "WATER DEPTH",
            0,
            points=[(0.0, 0.0)],
            obs=obs,
        )
        # 1 model + 1 obs trace
        assert len(fig.data) == 2
        # RMSE/NSE annotation (or "Too few points" / "No time overlap") must be present
        assert len(fig.layout.annotations) >= 1


class TestBuildCrosssectionChart:
    def test_no_path_returns_empty_figure(self, fake_tf):
        from server_analysis import build_crosssection_chart

        fig = build_crosssection_chart(
            fake_tf,
            "WATER DEPTH",
            0,
            path_points=None,
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_path_produces_profile_trace(self, fake_tf):
        from server_analysis import build_crosssection_chart

        path = [(0.0, 0.0), (1.0, 1.0)]
        fig = build_crosssection_chart(
            fake_tf,
            "WATER DEPTH",
            0,
            path_points=path,
        )
        assert len(fig.data) >= 1
