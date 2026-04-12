"""Smoke tests verifying server modules import and critical wiring."""
from __future__ import annotations
import importlib
import pytest
import numpy as np
from tests.helpers import FakeTF


class TestModuleImports:
    """Verify all server modules import without error."""

    @pytest.mark.parametrize("module", [
        "analysis",
        "validation",
        "geometry",
        "layers",
        "crs",
        "constants",
        "server_core",
        "server_analysis",
        "server_simulation",
        "server_import",
        "server_playback",
    ])
    def test_import(self, module):
        importlib.import_module(module)


class TestGetVarValues:
    """Verify the centralized get_var_values helper."""

    def test_file_variable(self, fake_tf):
        from analysis import get_var_values
        vals = get_var_values(fake_tf, "WATER DEPTH", 0)
        np.testing.assert_allclose(vals, [0.1, 0.5, 0.5, 1.0])

    def test_derived_variable(self, fake_tf):
        from analysis import get_var_values
        vals = get_var_values(fake_tf, "VELOCITY MAGNITUDE", 0)
        expected = np.sqrt(
            np.array([0.0, 1.0, 0.0, 1.0]) ** 2
            + np.array([0.0, 0.0, 1.0, 1.0]) ** 2
        )
        np.testing.assert_allclose(vals, expected, atol=1e-10)

    def test_unknown_derived_falls_back(self, fake_tf):
        """A name that looks derived but isn't available raises ValueError."""
        from analysis import get_var_values
        with pytest.raises((ValueError, KeyError)):
            get_var_values(fake_tf, "NONEXISTENT VAR", 0)


class TestDerivedTimeSeries:
    """Verify time_series_at_point works for derived variables."""

    def test_velocity_magnitude_timeseries(self, fake_tf):
        from analysis import time_series_at_point
        times, vals = time_series_at_point(fake_tf, "VELOCITY MAGNITUDE", 1.0, 1.0)
        assert len(times) == 3
        # Node 3 (1,1): U=1, V=1 at all timesteps -> magnitude = sqrt(2)
        np.testing.assert_allclose(vals, np.sqrt(2.0), atol=1e-10)

    def test_file_var_timeseries(self, fake_tf):
        from analysis import time_series_at_point
        times, vals = time_series_at_point(fake_tf, "WATER DEPTH", 0.0, 0.0)
        assert len(times) == 3
        # Node 0: depth=0.1 * (1 + tidx*0.5)
        np.testing.assert_allclose(vals, [0.1, 0.15, 0.2], atol=1e-10)


class TestDerivedDifference:
    """Verify compute_difference works for derived variables."""

    def test_file_var_difference(self, fake_tf):
        from analysis import compute_difference
        diff = compute_difference(fake_tf, "WATER DEPTH", 2, 0)
        # tidx=2: depth * 2.0, tidx=0: depth * 1.0 -> diff = depth
        np.testing.assert_allclose(diff, [0.1, 0.5, 0.5, 1.0])

    def test_derived_var_difference(self, fake_tf):
        from analysis import compute_difference
        # Velocity magnitude is constant across timesteps for FakeTF
        diff = compute_difference(fake_tf, "VELOCITY MAGNITUDE", 1, 0)
        np.testing.assert_allclose(diff, 0.0, atol=1e-10)


class TestDerivedTemporalStats:
    """Verify compute_temporal_stats works for derived variables."""

    def test_derived_temporal_stats(self, fake_tf):
        from analysis import compute_temporal_stats
        stats = compute_temporal_stats(fake_tf, "VELOCITY MAGNITUDE")
        assert stats is not None
        expected = np.sqrt(
            np.array([0.0, 1.0, 0.0, 1.0]) ** 2
            + np.array([0.0, 0.0, 1.0, 1.0]) ** 2
        )[:4]
        # Velocity magnitude is constant -> min==max==mean
        np.testing.assert_allclose(stats["min"], expected, atol=1e-6)
        np.testing.assert_allclose(stats["max"], expected, atol=1e-6)
