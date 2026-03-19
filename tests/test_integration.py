"""Integration tests using real SELAFIN files.

Skipped automatically when TELEMAC is not installed.
"""
from __future__ import annotations
import os
import numpy as np
import pytest

HOMETEL = os.environ.get("HOMETEL", "/home/razinka/telemac/telemac-v8p5r1")
GOUTTEDO = os.path.join(HOMETEL, "examples/telemac2d/gouttedo/r2d_gouttedo_v1p0.slf")
TELEMAC_AVAILABLE = os.path.isfile(GOUTTEDO)

pytestmark = pytest.mark.skipif(
    not TELEMAC_AVAILABLE,
    reason="TELEMAC not installed or Gouttedo file not found",
)


@pytest.fixture(scope="module")
def tf():
    from data_manip.extraction.telemac_file import TelemacFile
    return TelemacFile(GOUTTEDO)


@pytest.fixture(scope="module")
def geom(tf):
    from geometry import build_mesh_geometry
    return build_mesh_geometry(tf)


def test_load_file(tf):
    assert tf.npoin2 > 0
    assert tf.nelem2 > 0
    assert len(tf.times) > 0


def test_geometry(tf, geom):
    assert geom["npoin"] == tf.npoin2
    assert len(geom["positions"]) == tf.npoin2 * 3
    assert geom["zoom"] > 0


def test_mesh_layer(tf, geom):
    from layers import build_mesh_layer
    vals = tf.get_data_value("WATER DEPTH", len(tf.times) - 1)
    lyr, vmin, vmax, log_applied = build_mesh_layer(geom, vals, "Viridis")
    assert isinstance(lyr, dict)
    assert vmin < vmax


def test_velocity_layer(tf, geom):
    from layers import build_velocity_layer
    lyr = build_velocity_layer(tf, len(tf.times) - 1, geom)
    assert lyr is not None
    assert isinstance(lyr, dict)


def test_fem_contours(tf, geom):
    from layers import build_contour_layer_fn
    vals = tf.get_data_value("WATER DEPTH", len(tf.times) - 1)
    lyr = build_contour_layer_fn(tf, vals, geom)
    assert lyr is not None
    assert isinstance(lyr, dict)


def test_vectorized_analysis(tf):
    from analysis import (
        compute_mesh_quality, compute_slope, compute_courant_number,
        compute_element_area, find_boundary_nodes,
    )
    npoin = tf.npoin2
    vals = tf.get_data_value("WATER DEPTH", len(tf.times) - 1)

    mq = compute_mesh_quality(tf)
    assert mq.shape == (npoin,)

    sl = compute_slope(tf, vals)
    assert sl.shape == (npoin,)

    cfl = compute_courant_number(tf, len(tf.times) - 1)
    assert cfl is not None and cfl.shape == (npoin,)

    ea = compute_element_area(tf)
    assert ea.shape == (npoin,)

    bn = find_boundary_nodes(tf)
    assert len(bn) > 0


def test_derived_variables(tf):
    from analysis import get_available_derived, compute_derived
    derived = get_available_derived(tf)
    assert "VELOCITY MAGNITUDE" in derived
    assert "FROUDE NUMBER" in derived
    assert "VORTICITY" in derived
    for d in derived:
        result = compute_derived(tf, d, 0)
        assert result.shape[0] == tf.npoin2


def test_expression_parser(tf):
    from analysis import evaluate_expression
    r = evaluate_expression(tf, 0, "sqrt(VELOCITY_U**2 + VELOCITY_V**2)")
    assert r.shape[0] == tf.npoin2
    assert np.all(np.isfinite(r))


def test_particle_tracing(tf, geom):
    from analysis import compute_particle_paths, generate_seed_grid
    seeds = generate_seed_grid(tf, n_target=20)
    paths = compute_particle_paths(tf, seeds, geom["x_off"], geom["y_off"])
    assert isinstance(paths, list)
    # Each path (if any) should be list of [x, y, time] triplets
    for p in paths:
        assert len(p) > 1
        assert len(p[0]) == 3


def test_temporal_stats(tf):
    from analysis import compute_temporal_stats
    stats = compute_temporal_stats(tf, "WATER DEPTH")
    assert stats is not None
    npoin = tf.npoin2
    # min <= mean <= max at every node
    assert np.all(stats["min"][:npoin] <= stats["mean"][:npoin] + 1e-6)
    assert np.all(stats["mean"][:npoin] <= stats["max"][:npoin] + 1e-6)
