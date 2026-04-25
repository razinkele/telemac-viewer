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


class TestBuildVertprofileChart:
    def test_empty_points_returns_empty_figure(self, fake_tf):
        from server_analysis import build_vertprofile_chart

        fig = build_vertprofile_chart(fake_tf, "WATER DEPTH", 0, points=[])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_point_produces_profile_or_empty(self, fake_tf):
        """FakeTF is 2D — vertical_profile_at_point may return empty arrays.
        Either no traces (no z-data) or at least one trace is acceptable;
        the contract is that no exception escapes."""
        from server_analysis import build_vertprofile_chart

        fig = build_vertprofile_chart(
            fake_tf, "WATER DEPTH", 0, points=[(0.0, 0.0)]
        )
        assert isinstance(fig, go.Figure)


class TestBuildHistogramChart:
    def test_returns_one_histogram_trace(self, fake_tf):
        from server_analysis import build_histogram_chart

        values = fake_tf.get_data_value("WATER DEPTH", 0)
        fig = build_histogram_chart(fake_tf, "WATER DEPTH", values=values)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        # Plotly stores type as a string for bar/histogram traces
        assert fig.data[0].type == "histogram"

    def test_uses_sliced_values(self, fake_tf):
        """Histogram should respect tf.npoin2 slicing."""
        from server_analysis import build_histogram_chart

        values = fake_tf.get_data_value("WATER DEPTH", 0)
        fig = build_histogram_chart(fake_tf, "WATER DEPTH", values=values)
        # Histogram x array length matches npoin2 (or the underlying values
        # if shorter). FakeTF has npoin2 == 4.
        n = len(fig.data[0].x)
        assert n == fake_tf.npoin2


class TestBuildMultivarChart:
    def test_empty_points_returns_empty_figure(self, fake_tf):
        from server_analysis import build_multivar_chart

        fig = build_multivar_chart(fake_tf, 0, points=[])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_one_trace_per_variable(self, fake_tf):
        from server_analysis import build_multivar_chart

        fig = build_multivar_chart(fake_tf, 0, points=[(0.0, 0.0)])
        # FakeTF has multiple varnames; expect one trace per
        assert len(fig.data) == len(fake_tf.varnames)


class TestBuildRatingChart:
    def test_no_path_returns_empty_with_zero_skipped(self, fake_tf):
        from server_analysis import build_rating_chart

        fig, skipped, ntimes = build_rating_chart(fake_tf, path_points=None)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
        assert skipped == 0
        assert ntimes == len(fake_tf.times)

    def test_path_returns_three_tuple(self, fake_tf):
        """Rating chart returns (fig, skipped, ntimes); shim consumes
        skipped/ntimes for the >50% notification gate."""
        from server_analysis import build_rating_chart

        path = [(0.0, 0.0), (1.0, 1.0)]
        result = build_rating_chart(fake_tf, path_points=path)
        assert isinstance(result, tuple) and len(result) == 3
        fig, skipped, ntimes = result
        assert isinstance(fig, go.Figure)
        assert isinstance(skipped, int) and skipped >= 0
        assert ntimes == len(fake_tf.times)


class TestBuildVolumeChart:
    def test_none_cache_returns_empty(self):
        from server_analysis import build_volume_chart

        fig = build_volume_chart(cache=None)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_cache_dict_produces_one_trace(self):
        from server_analysis import build_volume_chart

        cache = {
            "times": np.array([0.0, 1.0, 2.0]),
            "volumes": np.array([100.0, 99.5, 99.0]),
        }
        fig = build_volume_chart(cache=cache)
        assert len(fig.data) == 1
        assert fig.data[0].mode == "lines"


class TestBuildBoundaryTsChart:
    def test_none_liq_returns_empty(self):
        from server_analysis import build_boundary_ts_chart

        fig = build_boundary_ts_chart(liq=None)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_liq_dict_produces_one_trace_per_column(self):
        from server_analysis import build_boundary_ts_chart

        liq = {
            "Q(1)": {
                "times": np.array([0.0, 1.0, 2.0]),
                "values": np.array([10.0, 12.0, 15.0]),
                "unit": "m3/s",
            },
            "SL(2)": {
                "times": np.array([0.0, 1.0, 2.0]),
                "values": np.array([5.0, 5.1, 5.2]),
                "unit": "m",
            },
        }
        fig = build_boundary_ts_chart(liq=liq)
        assert len(fig.data) == 2
