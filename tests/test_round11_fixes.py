"""Round 11 regression tests — flood duration fix, volume variable fallback."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from helpers import FakeTF


# ---------------------------------------------------------------------------
# flood_duration: forward-interval only (no double-counting)
# ---------------------------------------------------------------------------

class AllWetTF(FakeTF):
    """All nodes above any threshold at all 4 timesteps."""
    times = [0.0, 10.0, 20.0, 30.0]

    def get_data_value(self, varname, tidx):
        return np.array([1.0, 1.0, 1.0, 1.0])


def test_flood_duration_no_double_count():
    """All-wet for 4 timesteps spanning 30s → duration must be 30, not 40."""
    from analysis import compute_flood_duration
    tf = AllWetTF()
    dur = compute_flood_duration(tf, "WATER DEPTH", threshold=0.01)
    assert dur[0] == pytest.approx(30.0)
    assert dur[3] == pytest.approx(30.0)


def test_flood_duration_uneven_intervals():
    """Uneven timesteps: intervals [0→1]=1s, [1→5]=4s, [5→6]=1s → total 6s."""
    from analysis import compute_flood_duration

    class UnevenTF(FakeTF):
        times = [0.0, 1.0, 5.0, 6.0]

        def get_data_value(self, varname, tidx):
            return np.array([1.0, 1.0, 1.0, 1.0])

    dur = compute_flood_duration(UnevenTF(), "WATER DEPTH", threshold=0.01)
    assert dur[0] == pytest.approx(6.0)


def test_flood_duration_partial_wet():
    """Node wet at t=0,1 only → duration = interval [0,1] = 10s."""
    from analysis import compute_flood_duration

    class PartialTF(FakeTF):
        times = [0.0, 10.0, 20.0]

        def get_data_value(self, varname, tidx):
            if tidx < 2:
                return np.array([1.0, 1.0, 1.0, 1.0])
            return np.array([0.0, 0.0, 0.0, 0.0])

    dur = compute_flood_duration(PartialTF(), "WATER DEPTH", threshold=0.5)
    # wet at t=0 (interval [0,10]=10) and t=1 (interval [10,20]=10), dry at t=2
    # But t=1 is the last wet step; its forward interval [10,20] IS counted
    # because we iterate t=0 and t=1 (both < ntimes-1=2)
    assert dur[0] == pytest.approx(20.0)


def test_flood_duration_two_timesteps():
    """Two timesteps spanning 5s → one interval of 5s."""
    from analysis import compute_flood_duration

    class TwoTF(FakeTF):
        times = [0.0, 5.0]

        def get_data_value(self, varname, tidx):
            return np.array([1.0, 1.0, 1.0, 1.0])

    dur = compute_flood_duration(TwoTF(), "WATER DEPTH", threshold=0.01)
    assert dur[0] == pytest.approx(5.0)


def test_flood_duration_consistent_with_temporal_stats():
    """flood_duration should match compute_all_temporal_stats duration."""
    from analysis import compute_flood_duration, compute_all_temporal_stats
    tf = AllWetTF()
    dur_standalone = compute_flood_duration(tf, "WATER DEPTH", threshold=0.01)
    stats = compute_all_temporal_stats(tf, "WATER DEPTH", threshold=0.01)
    np.testing.assert_allclose(dur_standalone, stats["duration"], rtol=1e-5)


# ---------------------------------------------------------------------------
# volume timeseries: French depth variable fallback
# ---------------------------------------------------------------------------

def test_volume_uses_french_depth():
    """Volume computation should find HAUTEUR D'EAU for French TELEMAC files."""
    from validation import compute_volume_timeseries

    class FrenchTF(FakeTF):
        varnames = ["VITESSE U", "VITESSE V", "HAUTEUR D'EAU"]
        _data = {
            "VITESSE U": np.array([0.0, 1.0, 0.0, 1.0]),
            "VITESSE V": np.array([0.0, 0.0, 1.0, 1.0]),
            "HAUTEUR D'EAU": np.array([0.1, 0.5, 0.5, 1.0]),
        }
        times = [0.0, 1.0]

        def get_data_value(self, varname, tidx):
            return self._data[varname].copy()

    def fake_integral(tf, vals, threshold=0.001):
        return {"integral": float(np.sum(vals[vals > threshold]))}

    times, volumes = compute_volume_timeseries(FrenchTF(), fake_integral)
    assert len(volumes) == 2
    # Should use HAUTEUR D'EAU, not VITESSE U
    assert volumes[0] == pytest.approx(0.1 + 0.5 + 0.5 + 1.0)  # sum of all depth > 0.001


def test_volume_english_depth_preferred():
    """WATER DEPTH should be preferred when present."""
    from validation import compute_volume_timeseries

    class EnglishTF(FakeTF):
        times = [0.0]

        def get_data_value(self, varname, tidx):
            if varname == "WATER DEPTH":
                return np.array([2.0, 2.0, 2.0, 2.0])
            return np.array([99.0, 99.0, 99.0, 99.0])

    def fake_integral(tf, vals, threshold=0.001):
        return {"integral": float(np.sum(vals))}

    _, volumes = compute_volume_timeseries(EnglishTF(), fake_integral)
    assert volumes[0] == pytest.approx(8.0)  # 4 * 2.0, not 4 * 99.0
