"""Round 9 regression tests — parser_2d n_fp fix, writer_cas overrides."""

import pathlib
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# parser_2d: n_fp must be defined before the OOB guard (line 97)
# ---------------------------------------------------------------------------

def test_parser2d_oob_filter_uses_n_fp():
    """The OOB face-point index filter must reference the actual face_points length."""
    from telemac_tools.hecras import parser_2d as mod
    import inspect
    src = inspect.getsource(mod)
    assign_pos = src.index("n_fp = len(face_points)")
    use_pos = src.index("idx >= n_fp")
    assert assign_pos < use_pos, "n_fp must be defined before its first use"


def test_parser2d_oob_skips_bad_cells(tmp_path):
    """Cells referencing out-of-range face-point indices are skipped."""
    import h5py, numpy as np
    from telemac_tools.hecras.parser_2d import parse_hecras_2d

    hdf = tmp_path / "test.hdf"
    with h5py.File(hdf, "w") as f:
        geo = f.create_group("Geometry/2D Flow Areas/TestArea")
        fp = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        geo.create_dataset("FacePoints Coordinate", data=fp)
        cfpi = np.array([[0, 1, 2, -1], [0, 1, 99, -1]])
        geo.create_dataset("Cells FacePoint Indexes", data=cfpi)
        cc = np.array([[0.33, 0.33], [0.5, 0.5]])
        geo.create_dataset("Cells Center Coordinate", data=cc)
        f.create_dataset(
            "Geometry/2D Flow Areas/Attributes",
            data=np.array([("TestArea",)], dtype=[("Name", "S64")]),
        )

    model = parse_hecras_2d(str(hdf))
    area = model.areas_2d[0]
    assert len(area.cells) == 1, "OOB cell should be skipped"
    assert area.cells[0].face_point_indices == [0, 1, 2]


# ---------------------------------------------------------------------------
# writer_cas: None values skipped, strings auto-quoted
# ---------------------------------------------------------------------------

def test_writer_cas_none_values_skipped(tmp_path):
    """Override entries with None value should be omitted entirely."""
    from telemac_tools.telemac.writer_cas import write_cas

    out = tmp_path / "test.cas"
    write_cas(str(out), name="test", duration=100.0, overrides={"FRICTION COEFFICIENT": None})
    content = out.read_text()
    assert "FRICTION COEFFICIENT" not in content


def test_writer_cas_string_auto_quoted(tmp_path):
    """String override values should be wrapped in single quotes."""
    from telemac_tools.telemac.writer_cas import write_cas

    out = tmp_path / "test.cas"
    write_cas(str(out), name="test", duration=100.0, overrides={"EQUATIONS": "SAINT-VENANT FV"})
    content = out.read_text()
    assert "EQUATIONS = 'SAINT-VENANT FV'" in content


def test_writer_cas_prequoted_string_unchanged(tmp_path):
    """Already-quoted strings should not be double-quoted."""
    from telemac_tools.telemac.writer_cas import write_cas

    out = tmp_path / "test.cas"
    write_cas(str(out), name="test", duration=100.0, overrides={"EQUATIONS": "'SAINT-VENANT FV'"})
    content = out.read_text()
    assert "EQUATIONS = 'SAINT-VENANT FV'" in content
    assert "EQUATIONS = ''SAINT-VENANT FV''" not in content


def test_writer_cas_numeric_values_unquoted(tmp_path):
    """Numeric overrides should not be quoted."""
    from telemac_tools.telemac.writer_cas import write_cas

    out = tmp_path / "test.cas"
    write_cas(str(out), name="test", duration=100.0, overrides={"FRICTION COEFFICIENT": 30})
    content = out.read_text()
    assert "FRICTION COEFFICIENT = 30" in content
    assert "'" not in content.split("FRICTION COEFFICIENT")[1].split("\n")[0]


def test_writer_cas_mixed_overrides(tmp_path):
    """Mix of None, string, pre-quoted, and numeric overrides."""
    from telemac_tools.telemac.writer_cas import write_cas

    out = tmp_path / "test.cas"
    write_cas(str(out), name="test", duration=100.0, overrides={
        "SKIP_ME": None,
        "EQUATIONS": "SAINT-VENANT FV",
        "ALREADY": "'QUOTED'",
        "NUMBER_OF_PROCESSORS": 4,
    })
    content = out.read_text()
    assert "SKIP_ME" not in content
    assert "EQUATIONS = 'SAINT-VENANT FV'" in content
    assert "ALREADY = 'QUOTED'" in content
    assert "NUMBER_OF_PROCESSORS = 4" in content
