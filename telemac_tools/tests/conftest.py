"""Shared test fixtures for telemac_tools."""
from __future__ import annotations
import numpy as np
import pytest
import h5py


@pytest.fixture
def hdf_1d(tmp_path):
    """Create a minimal synthetic HEC-RAS 1D geometry HDF5 file.

    Contains one reach with 3 cross-sections spaced 100m apart.
    River alignment runs along x-axis from (0,0) to (300,0).
    Cross-sections are perpendicular (along y-axis), 100m wide.
    """
    path = tmp_path / "test.g01.hdf"
    with h5py.File(path, "w") as f:
        geo = f.create_group("Geometry")

        # --- River Centerlines ---
        rc = geo.create_group("River Centerlines")
        rc_attrs = np.array(
            [("River1", "Reach1")],
            dtype=[("River", "S40"), ("Reach", "S40")],
        )
        rc.create_dataset("Attributes", data=rc_attrs)
        rc.create_dataset("Polyline Info", data=np.array([[0, 4]]))
        rc.create_dataset("Polyline Points", data=np.array([
            [0.0, 0.0], [100.0, 0.0], [200.0, 0.0], [300.0, 0.0],
        ]))

        # --- Cross Sections ---
        xs = geo.create_group("Cross Sections")
        xs_attrs = np.array(
            [
                ("River1", "Reach1", 0.0),
                ("River1", "Reach1", 100.0),
                ("River1", "Reach1", 200.0),
            ],
            dtype=[("River", "S40"), ("Reach", "S40"), ("Station", "f8")],
        )
        xs.create_dataset("Attributes", data=xs_attrs)

        # Each XS: 5 station-elevation points spanning 100m
        stations = np.array([0, 25, 50, 75, 100], dtype=np.float64)
        elevations_0 = np.array([5, 2, 0, 2, 5], dtype=np.float64)
        elevations_1 = np.array([5, 2, -1, 2, 5], dtype=np.float64)
        elevations_2 = np.array([5, 2, -2, 2, 5], dtype=np.float64)

        se_values = np.concatenate([
            np.column_stack([stations, elevations_0]),
            np.column_stack([stations, elevations_1]),
            np.column_stack([stations, elevations_2]),
        ])
        se_info = np.array([[0, 5], [5, 5], [10, 5]])
        xs.create_dataset("Station Elevation Info", data=se_info)
        xs.create_dataset("Station Elevation Values", data=se_values)

        # Polyline Points: world-coord XS lines (perpendicular to river)
        poly_pts = np.array([
            [0, -50], [0, 50],
            [100, -50], [100, 50],
            [200, -50], [200, 50],
        ], dtype=np.float64)
        poly_info = np.array([[0, 2], [2, 2], [4, 2]])
        xs.create_dataset("Polyline Info", data=poly_info)
        xs.create_dataset("Polyline Points", data=poly_pts)

        # Manning's n: 3 values per XS (left overbank, channel, right overbank)
        manning_vals = np.array([
            0.06, 0.035, 0.06,
            0.06, 0.035, 0.06,
            0.06, 0.035, 0.06,
        ])
        manning_info = np.array([[0, 3], [3, 3], [6, 3]])
        xs.create_dataset("Manning's n Info", data=manning_info)
        xs.create_dataset("Manning's n Values", data=manning_vals)

        # Bank stations: left=25, right=75 for all XS
        xs.create_dataset("Bank Stations", data=np.array([
            [25.0, 75.0], [25.0, 75.0], [25.0, 75.0],
        ]))

        # --- Boundary Condition Lines ---
        bc = geo.create_group("Boundary Condition Lines")
        bc_attrs = np.array(
            [
                ("Upstream", "flow"),
                ("Downstream", "normal_depth"),
            ],
            dtype=[("Name", "S40"), ("Type", "S40")],
        )
        bc.create_dataset("Attributes", data=bc_attrs)
        bc.create_dataset("Polyline Info", data=np.array([[0, 2], [2, 2]]))
        bc.create_dataset("Polyline Points", data=np.array([
            [0, -50], [0, 50],
            [200, -50], [200, 50],
        ], dtype=np.float64))

    return str(path)


@pytest.fixture
def hdf_2d(tmp_path):
    """Create a minimal synthetic HEC-RAS 2D geometry HDF5 file.

    Contains one 2D flow area with 4 square cells arranged in a 2x2 grid.
    Each cell is a square with 4 face points.
    Cell size = 100m. Domain = 200x200m.
    """
    path = tmp_path / "test_2d.g01.hdf"
    with h5py.File(path, "w") as f:
        geo = f.create_group("Geometry")
        areas = geo.create_group("2D Flow Areas")
        area = areas.create_group("TestArea")

        # 9 face points forming a 3x3 grid
        fp = np.array([
            [0, 0], [100, 0], [200, 0],
            [0, 100], [100, 100], [200, 100],
            [0, 200], [100, 200], [200, 200],
        ], dtype=np.float64)
        area.create_dataset("FacePoints Coordinate", data=fp)

        # 4 cell centers
        cc = np.array([
            [50, 50], [150, 50], [50, 150], [150, 150],
        ], dtype=np.float64)
        area.create_dataset("Cell Points", data=cc)

        # Cell face point indices: padded 2D array (N_cells, max_faces)
        # with -1 fill for unused slots (matches real HDF5 format)
        cell_fp_idx = np.array([
            [0, 1, 4, 3],
            [1, 2, 5, 4],
            [3, 4, 7, 6],
            [4, 5, 8, 7],
        ], dtype=np.int32)
        area.create_dataset("Cells FacePoint Indexes", data=cell_fp_idx)

        # Cells Face and Orientation Info: (N_cells, 2) with [offset, count]
        face_info = np.array([
            [0, 4], [4, 4], [8, 4], [12, 4],
        ], dtype=np.int32)
        area.create_dataset("Cells Face and Orientation Info", data=face_info)

        # Elevation per cell center
        area.create_dataset("Cells Minimum Elevation", data=np.array([
            1.0, 0.5, 1.5, 1.0,
        ]))

        # Area attributes
        areas.attrs["Names"] = np.bytes_("TestArea")
        areas.attrs["Cell Size"] = 100.0
        areas.attrs["Manning's n"] = 0.035

    return str(path)


@pytest.fixture
def fake_dem(tmp_path):
    """Create a minimal GeoTIFF DEM: 300x200m at 10m resolution.

    Flat at elevation 5.0m. Origin at (-50, 100), covers the 1D fixture's domain.
    """
    width, height = 30, 20  # 10m pixels
    origin_x, origin_y = -50.0, 100.0  # top-left corner
    pixel_size = 10.0

    data = np.full((height, width), 5.0, dtype=np.float32)

    path = tmp_path / "dem.tif"
    import tifffile
    tifffile.imwrite(
        str(path), data,
        extratags=[
            (33922, 12, 6, (0.0, 0.0, 0.0, origin_x, origin_y, 0.0)),  # ModelTiepointTag
            (33550, 12, 3, (pixel_size, pixel_size, 0.0)),  # ModelPixelScaleTag
        ],
    )
    return str(path)


@pytest.fixture
def hdf_unsteady(tmp_path):
    """Synthetic HEC-RAS unsteady flow file with BC time series."""
    path = tmp_path / "test.u01.hdf"
    with h5py.File(path, "w") as f:
        ufd = f.create_group("Unsteady Flow Data")
        bcs = ufd.create_group("Boundary Conditions")
        us = bcs.create_group("Upstream")
        fh = us.create_group("Flow Hydrograph")
        fh.create_dataset("Flow", data=np.array([10.0, 50.0, 100.0, 50.0, 10.0]))
        fh.create_dataset("Time", data=np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
        ds = bcs.create_group("Downstream")
        sh = ds.create_group("Stage Hydrograph")
        sh.create_dataset("Stage", data=np.array([2.0, 2.5, 3.0, 2.5, 2.0]))
        sh.create_dataset("Time", data=np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    return str(path)
