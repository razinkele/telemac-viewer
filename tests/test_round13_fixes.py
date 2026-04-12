"""Round 13 regression tests: builder, analysis, parser_1d, writer_cas."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest


# --- Bug #1: builder._buffer_alignment single-point alignment ---

def test_buffer_alignment_single_point():
    """Single-point alignment should return empty polygon, not crash."""
    from telemac_tools.domain.builder import _buffer_alignment
    alignment = np.array([[100.0, 200.0]])
    result = _buffer_alignment(alignment, 50.0)
    assert result.shape == (0, 2)


def test_buffer_alignment_empty():
    """Empty alignment should return empty polygon."""
    from telemac_tools.domain.builder import _buffer_alignment
    alignment = np.empty((0, 2))
    result = _buffer_alignment(alignment, 50.0)
    assert result.shape == (0, 2)


def test_buffer_alignment_two_points():
    """Two-point alignment should produce a valid polygon."""
    from telemac_tools.domain.builder import _buffer_alignment
    alignment = np.array([[0.0, 0.0], [10.0, 0.0]])
    result = _buffer_alignment(alignment, 5.0)
    assert result.shape[0] > 0
    assert result.shape[1] == 2


# --- Bug #2: cross_section_profile empty points ---

def test_cross_section_profile_empty_polyline(fake_tf):
    """Derived cross-section profile with empty polyline should not crash."""
    from analysis import cross_section_profile, DERIVED_VARIABLES, get_available_derived

    derived = get_available_derived(fake_tf)
    if not derived:
        pytest.skip("No derived variables available in fake_tf")

    varname = derived[0]
    # Polyline that doesn't intersect the mesh at all (far away)
    polyline_far = [[1e6, 1e6], [1e6 + 1, 1e6 + 1]]
    abscissa, values = cross_section_profile(fake_tf, varname, 0, polyline_far)
    # Should return arrays without crashing
    assert isinstance(abscissa, np.ndarray)
    assert isinstance(values, np.ndarray)


# --- Bug #3: parser_1d missing dataset check ---

def test_parser_1d_missing_se_datasets(tmp_path):
    """Parser should skip XS gracefully when required datasets are missing."""
    h5py = pytest.importorskip("h5py")
    from telemac_tools.hecras.parser_1d import parse_hecras_1d

    f = tmp_path / "test.hdf"
    with h5py.File(str(f), "w") as hf:
        geo = hf.create_group("Geometry")
        xs = geo.create_group("Cross Sections")
        # Only Attributes, missing Station Elevation Info etc.
        dt = np.dtype([("River Name", "S20"), ("Reach Name", "S20"), ("Station", "S10")])
        attrs = np.array([(b"River1", b"Reach1", b"100")], dtype=dt)
        xs.create_dataset("Attributes", data=attrs)

    model = parse_hecras_1d(str(f))
    # Should parse without crash; no cross sections created
    assert len(model.rivers) == 0 or all(
        len(r.cross_sections) == 0 for r in model.rivers
    )


# --- Bug #4: parser_1d invalid offset/count ---

def test_parser_1d_oob_offset_skipped(tmp_path):
    """Parser should skip cross-sections with OOB SE offsets."""
    h5py = pytest.importorskip("h5py")
    from telemac_tools.hecras.parser_1d import parse_hecras_1d

    f = tmp_path / "test.hdf"
    with h5py.File(str(f), "w") as hf:
        geo = hf.create_group("Geometry")
        xs = geo.create_group("Cross Sections")

        dt = np.dtype([("River Name", "S20"), ("Reach Name", "S20"), ("Station", "S10")])
        attrs = np.array([(b"River1", b"Reach1", b"100")], dtype=dt)
        xs.create_dataset("Attributes", data=attrs)

        # SE Info with offset beyond values array
        xs.create_dataset("Station Elevation Info", data=np.array([[999, 5]]))
        xs.create_dataset("Station Elevation Values", data=np.array([[0.0, 1.0], [1.0, 2.0]]))
        xs.create_dataset("Polyline Info", data=np.array([[0, 2]]))
        xs.create_dataset("Polyline Points", data=np.array([[0.0, 0.0], [1.0, 1.0]]))

    model = parse_hecras_1d(str(f))
    # Should not crash; XS skipped due to invalid offset
    total_xs = sum(len(r.cross_sections) for r in model.rivers)
    assert total_xs == 0


def test_parser_1d_negative_offset_skipped(tmp_path):
    """Parser should skip cross-sections with negative SE offsets."""
    h5py = pytest.importorskip("h5py")
    from telemac_tools.hecras.parser_1d import parse_hecras_1d

    f = tmp_path / "test.hdf"
    with h5py.File(str(f), "w") as hf:
        geo = hf.create_group("Geometry")
        xs = geo.create_group("Cross Sections")

        dt = np.dtype([("River Name", "S20"), ("Reach Name", "S20"), ("Station", "S10")])
        attrs = np.array([(b"River1", b"Reach1", b"100")], dtype=dt)
        xs.create_dataset("Attributes", data=attrs)

        # Negative offset
        xs.create_dataset("Station Elevation Info", data=np.array([[-1, 2]]))
        xs.create_dataset("Station Elevation Values", data=np.array([[0.0, 1.0], [1.0, 2.0]]))
        xs.create_dataset("Polyline Info", data=np.array([[0, 2]]))
        xs.create_dataset("Polyline Points", data=np.array([[0.0, 0.0], [1.0, 1.0]]))

    model = parse_hecras_1d(str(f))
    total_xs = sum(len(r.cross_sections) for r in model.rivers)
    assert total_xs == 0


# --- Bug #5: writer_cas quote escaping ---

def test_writer_cas_apostrophe_escaped(tmp_path):
    """String values with apostrophes should be escaped in .cas output."""
    from telemac_tools.telemac.writer_cas import write_cas

    out = tmp_path / "test.cas"
    write_cas(str(out), overrides={"TITLE": "River's Flow"})
    content = out.read_text()
    assert "River''s Flow" in content or "River\\'s Flow" in content
    # Find the override TITLE line specifically
    override_lines = [l for l in content.splitlines() if "TITLE" in l and "River" in l]
    assert len(override_lines) == 1
    line = override_lines[0]
    quote_count = line.count("'")
    assert quote_count % 2 == 0, f"Unbalanced quotes in: {line}"


def test_writer_cas_normal_string(tmp_path):
    """Normal string without quotes should be auto-quoted."""
    from telemac_tools.telemac.writer_cas import write_cas

    out = tmp_path / "test.cas"
    write_cas(str(out), overrides={"TITLE": "Simple Title"})
    content = out.read_text()
    assert "'Simple Title'" in content
