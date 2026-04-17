"""Tests for fixes from docs/codebase-review-findings.md (Rounds 1 + 2)."""

import numpy as np
import pytest

from tests.helpers import FakeTF


def test_mesh_identity_hash_matches_identical_mesh():
    from analysis import mesh_identity_hash

    assert mesh_identity_hash(FakeTF()) == mesh_identity_hash(FakeTF())


def test_mesh_identity_hash_rejects_shifted_nodes():
    from analysis import mesh_identity_hash

    class Shifted(FakeTF):
        meshx = FakeTF.meshx + 1e-6  # sub-millimetre geometry noise

    assert mesh_identity_hash(FakeTF()) != mesh_identity_hash(Shifted())


def test_mesh_identity_hash_rejects_different_connectivity():
    from analysis import mesh_identity_hash

    class Reconnected(FakeTF):
        ikle2 = np.array([[0, 2, 1], [1, 2, 3]], dtype=np.int32)

    assert mesh_identity_hash(FakeTF()) != mesh_identity_hash(Reconnected())


def test_volume_fallback_to_free_surface_minus_bottom():
    """When no depth variable is present but FREE SURFACE and BOTTOM are,
    compute_volume_timeseries should compute depth = FS - B per timestep."""
    from validation import compute_volume_timeseries

    class NoDepthTF(FakeTF):
        varnames = ["VELOCITY U", "VELOCITY V", "FREE SURFACE", "BOTTOM"]

        def get_data_value(self, varname, tidx):
            if varname == "FREE SURFACE":
                return np.array([2.5, 2.5, 2.5, 2.5])  # water elevation
            if varname == "BOTTOM":
                return np.array([0.5, 0.5, 0.5, 0.5])  # bed elevation
            return np.array([99.0, 99.0, 99.0, 99.0])  # should NOT be used

    def fake_integral(tf, vals, threshold=0.001):
        # Sum depths > threshold — use this to check the right var went in.
        return {"integral": float(np.sum(vals[vals > threshold]))}

    _, volumes = compute_volume_timeseries(NoDepthTF(), fake_integral)
    # Depth = 2.5 - 0.5 = 2.0 at each of 4 nodes
    assert volumes[0] == pytest.approx(8.0)


def test_volume_does_not_treat_free_surface_as_depth():
    """FREE SURFACE alone must not appear in the depth-alias search."""
    from validation import compute_volume_timeseries

    class FsOnlyTF(FakeTF):
        varnames = ["VELOCITY U", "FREE SURFACE"]  # no BOTTOM, no DEPTH

        def get_data_value(self, varname, tidx):
            if varname == "FREE SURFACE":
                return np.array([5.0, 5.0, 5.0, 5.0])
            return np.array([1.0, 1.0, 1.0, 1.0])

    def fake_integral(tf, vals, threshold=0.001):
        return {"integral": float(np.sum(vals))}

    # Last-resort legacy path picks first var (VELOCITY U) — NOT FREE SURFACE.
    _, volumes = compute_volume_timeseries(FsOnlyTF(), fake_integral)
    assert volumes[0] == pytest.approx(4.0)  # = 4 * 1.0 from VELOCITY U


# ---------------------------------------------------------------------------
# R2 #3: barycentric interpolation for derived point sampling
# ---------------------------------------------------------------------------


def test_derived_point_sampling_uses_barycentric_interp():
    """VELOCITY MAGNITUDE at (0.25, 0.25) should interpolate the derived
    field evaluated at mesh nodes — matching what the map layer shows —
    not take the nearest node."""
    from analysis import time_series_at_point

    # Triangle 0 = nodes [(0,0), (1,0), (0,1)].
    # Node |V| values: sqrt(U²+V²) = [0, 1, 1].
    # At (0.25, 0.25): bary weights (0.5, 0.25, 0.25) →
    # interp = 0.5·0 + 0.25·1 + 0.25·1 = 0.5
    _, vals = time_series_at_point(FakeTF(), "VELOCITY MAGNITUDE", 0.25, 0.25)
    assert vals[0] == pytest.approx(0.5, rel=1e-6)
    # Nearest-node would give node 0 → |V|=0. Confirming we are NOT nearest-node.
    assert vals[0] > 0.3


def test_derived_point_sampling_outside_mesh_falls_back():
    """A point outside the mesh should fall back to nearest-node without error."""
    from analysis import time_series_at_point

    _, vals = time_series_at_point(FakeTF(), "VELOCITY MAGNITUDE", 99.0, 99.0)
    # Nearest node to (99, 99) is (1, 1) → U=1, V=1 → mag = sqrt(2)
    assert vals[0] == pytest.approx(np.sqrt(2.0), rel=1e-6)


def test_derived_sampling_matches_mesh_node_exactly():
    """At a mesh node, barycentric interp should equal the nodal value."""
    from analysis import time_series_at_point

    # Node 3 is at (1, 1) with U=1, V=1, so |V| = sqrt(2)
    _, vals = time_series_at_point(FakeTF(), "VELOCITY MAGNITUDE", 1.0, 1.0)
    assert vals[0] == pytest.approx(np.sqrt(2.0), rel=1e-6)
