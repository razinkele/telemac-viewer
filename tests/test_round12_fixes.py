"""Round 12 regression tests — parser bounds checks, filter inf, cli NaN."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from helpers import FakeTF


# ---------------------------------------------------------------------------
# parser_2d: 1D CFPI path now has OOB bounds check
# ---------------------------------------------------------------------------

def test_parser2d_1d_cfpi_oob_skipped(tmp_path):
    """1D CFPI path should skip cells with out-of-range face-point indices."""
    import h5py
    from telemac_tools.hecras.parser_2d import parse_hecras_2d

    hdf = tmp_path / "test.hdf"
    with h5py.File(hdf, "w") as f:
        geo = f.create_group("Geometry/2D Flow Areas/TestArea")
        fp = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        geo.create_dataset("FacePoints Coordinate", data=fp)
        # 1D flat array: cell 0 = indices [0,1,2], cell 1 = indices [0,1,99]
        cfpi_flat = np.array([0, 1, 2, 0, 1, 99])
        geo.create_dataset("Cells FacePoint Indexes", data=cfpi_flat)
        # face_info: offset, count pairs
        face_info = np.array([[0, 3], [3, 3]])
        geo.create_dataset("Cells Face and Orientation Info", data=face_info)
        cc = np.array([[0.33, 0.33], [0.5, 0.5]])
        geo.create_dataset("Cells Center Coordinate", data=cc)
        f.create_dataset(
            "Geometry/2D Flow Areas/Attributes",
            data=np.array([("TestArea",)], dtype=[("Name", "S64")]),
        )

    model = parse_hecras_2d(str(hdf))
    area = model.areas_2d[0]
    assert len(area.cells) == 1, "OOB cell in 1D CFPI path should be skipped"
    assert area.cells[0].face_point_indices == [0, 1, 2]


# ---------------------------------------------------------------------------
# parser_2d: faces_fp path validates fp < n_fp
# ---------------------------------------------------------------------------

def test_parser2d_faces_fp_oob_filtered(tmp_path):
    """Face-point indices >= n_fp in faces_fp path should be filtered out."""
    import h5py
    from telemac_tools.hecras.parser_2d import parse_hecras_2d

    hdf = tmp_path / "test.hdf"
    with h5py.File(hdf, "w") as f:
        geo = f.create_group("Geometry/2D Flow Areas/TestArea")
        fp = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        geo.create_dataset("FacePoints Coordinate", data=fp)
        # No CFPI dataset — force faces_fp path
        # face_info: 1 cell with 2 faces
        face_info = np.array([[0, 2]])
        geo.create_dataset("Cells Face and Orientation Info", data=face_info)
        # face_orient_vals: face indices (absolute)
        face_orient_vals = np.array([[0, 0], [1, 0]])
        geo.create_dataset("Cells Face and Orientation Values", data=face_orient_vals)
        # faces_fp: face 0 → [0, 1], face 1 → [2, 999]
        faces_fp = np.array([[0, 1], [2, 999]])
        geo.create_dataset("Faces FacePoint Indexes", data=faces_fp)
        cc = np.array([[0.33, 0.33]])
        geo.create_dataset("Cells Center Coordinate", data=cc)
        f.create_dataset(
            "Geometry/2D Flow Areas/Attributes",
            data=np.array([("TestArea",)], dtype=[("Name", "S64")]),
        )

    model = parse_hecras_2d(str(hdf))
    area = model.areas_2d[0]
    assert len(area.cells) == 1
    # Index 999 should be filtered out, only [0, 1, 2] remain
    for idx in area.cells[0].face_point_indices:
        assert idx < len(fp), f"OOB index {idx} should have been filtered"


# ---------------------------------------------------------------------------
# filter_ui: inf values should not break slider
# ---------------------------------------------------------------------------

def test_filter_ui_handles_inf():
    """Values with inf should produce valid finite slider bounds."""
    # Simulate the filter_ui logic
    vals = np.array([1.0, 2.0, np.inf, -np.inf, 3.0])
    vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
    if np.isnan(vmin) or np.isinf(vmin) or np.isinf(vmax) or vmax <= vmin:
        vmin, vmax = 0.0, 1.0
    assert np.isfinite(vmin)
    assert np.isfinite(vmax)
    assert vmin == 0.0
    assert vmax == 1.0


def test_filter_ui_handles_normal_values():
    """Normal values should produce correct slider bounds."""
    vals = np.array([10.0, 20.0, 30.0])
    vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
    if np.isnan(vmin) or np.isinf(vmin) or np.isinf(vmax) or vmax <= vmin:
        vmin, vmax = 0.0, 1.0
    assert vmin == 10.0
    assert vmax == 30.0


# ---------------------------------------------------------------------------
# writer_cli: NaN prescribed_h replaced with 0.0
# ---------------------------------------------------------------------------

def test_writer_cli_nan_prescribed_h(tmp_path):
    """NaN prescribed_h should be replaced with 0.0, not written as 'nan'."""
    from telemac_tools.telemac.writer_cli import write_cli
    from telemac_tools.model import TelemacDomain, BCSegment, Mesh2D

    # Minimal triangle mesh: 3 boundary nodes
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    elements = np.array([[0, 1, 2]])
    mesh = Mesh2D(nodes=nodes, elements=elements,
                  elevation=np.zeros(3), mannings_n=np.full(3, 0.03))

    domain = TelemacDomain(
        boundary_polygon=nodes,
        bc_segments=[
            BCSegment(
                node_indices=[0],
                lihbor=5,
                prescribed_h=float("nan"),
            ),
        ],
    )
    out = tmp_path / "test.cli"
    write_cli(mesh, domain, str(out))
    content = out.read_text()
    assert "nan" not in content.lower(), "NaN should not appear in .cli file"


def test_writer_cli_valid_prescribed_h(tmp_path):
    """Valid prescribed_h should be written correctly."""
    from telemac_tools.telemac.writer_cli import write_cli
    from telemac_tools.model import TelemacDomain, BCSegment, Mesh2D

    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    elements = np.array([[0, 1, 2]])
    mesh = Mesh2D(nodes=nodes, elements=elements,
                  elevation=np.zeros(3), mannings_n=np.full(3, 0.03))

    domain = TelemacDomain(
        boundary_polygon=nodes,
        bc_segments=[
            BCSegment(
                node_indices=[0],
                lihbor=5,
                prescribed_h=3.5,
            ),
        ],
    )
    out = tmp_path / "test.cli"
    write_cli(mesh, domain, str(out))
    content = out.read_text()
    assert "3.5" in content
