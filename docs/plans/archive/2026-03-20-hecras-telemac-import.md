# HEC-RAS → TELEMAC Import Tool — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python library (`telemac_tools`) that converts HEC-RAS 6.x geometry HDF5 files into ready-to-run TELEMAC-2D simulations (.slf mesh + .cli boundary + .cas steering).

**Architecture:** Four-layer pipeline (Parser → Domain Builder → Mesh Generator → TELEMAC Writer), each independently testable. Synthetic HDF5 fixtures for unit tests; real FEMA FFRD fixtures for integration. The viewer "Import" tab is a separate follow-up plan.

**Tech Stack:** Python 3.12+, h5py, numpy, scipy, tifffile, triangle. Gmsh optional (lazy import).

**Spec:** `docs/plans/2026-03-20-hecras-telemac-import-design.md`

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `telemac_tools/__init__.py` | Convenience `hecras_to_telemac()` entry point | Create |
| `telemac_tools/model.py` | All dataclasses (CrossSection, Reach, Mesh2D, etc.) | Create |
| `telemac_tools/hecras/__init__.py` | Re-export `parse_hecras`, `parse_hecras_1d`, `parse_hecras_2d` | Create |
| `telemac_tools/hecras/parser_1d.py` | 1D geometry from HDF5 (cross-sections, reaches, BCs) | Create |
| `telemac_tools/hecras/parser_2d.py` | 2D area mesh from HDF5 (Voronoi → triangles) | Create |
| `telemac_tools/hecras/manning.py` | Manning's n extraction from HDF5 | Create |
| `telemac_tools/domain/__init__.py` | Re-export `build_domain_1d`, `build_domain_2d` | Create |
| `telemac_tools/domain/builder.py` | Combine geometry + DEM → TelemacDomain | Create |
| `telemac_tools/domain/channel_carve.py` | Station-to-world transform, thalweg interpolation | Create |
| `telemac_tools/meshing/__init__.py` | Re-export `generate_mesh` | Create |
| `telemac_tools/meshing/base.py` | Abstract MeshBackend interface | Create |
| `telemac_tools/meshing/triangle_mesh.py` | Triangle backend | Create |
| `telemac_tools/telemac/__init__.py` | Re-export `write_telemac` | Create |
| `telemac_tools/telemac/writer_slf.py` | Write .slf mesh file (via Selafin) | Create |
| `telemac_tools/telemac/writer_cli.py` | Write .cli boundary conditions | Create |
| `telemac_tools/telemac/writer_cas.py` | Write .cas steering file template | Create |
| `telemac_tools/tests/conftest.py` | Shared fixtures (synthetic HDF5, fake DEM) | Create |
| `telemac_tools/tests/test_model.py` | Tests for data model construction | Create |
| `telemac_tools/tests/test_parser_1d.py` | Tests for 1D HDF5 parsing | Create |
| `telemac_tools/tests/test_parser_2d.py` | Tests for 2D HDF5 parsing | Create |
| `telemac_tools/tests/test_manning.py` | Tests for Manning's extraction | Create |
| `telemac_tools/tests/test_channel_carve.py` | Tests for station→world and thalweg interpolation | Create |
| `telemac_tools/tests/test_domain_builder.py` | Tests for domain building | Create |
| `telemac_tools/tests/test_mesh_generator.py` | Tests for Triangle meshing | Create |
| `telemac_tools/tests/test_writers.py` | Tests for .slf/.cli/.cas output | Create |
| `telemac_tools/tests/test_integration.py` | End-to-end pipeline test | Create |

All paths are relative to `/home/razinka/telemac/telemac-viewer/`.

---

### Task 1: Scaffold package and install dependencies

**Files:**
- Create: `telemac_tools/__init__.py`
- Create: `telemac_tools/model.py`
- Create: `telemac_tools/hecras/__init__.py`
- Create: `telemac_tools/domain/__init__.py`
- Create: `telemac_tools/meshing/__init__.py`
- Create: `telemac_tools/telemac/__init__.py`
- Create: `telemac_tools/tests/__init__.py`
- Create: `telemac_tools/tests/conftest.py`

- [ ] **Step 1: Install tifffile dependency**

Run: `/opt/micromamba/envs/shiny/bin/pip install tifffile`
Expected: Successfully installed tifffile

- [ ] **Step 2: Create package skeleton with empty `__init__.py` files**

```python
# telemac_tools/__init__.py
"""HEC-RAS → TELEMAC import tools."""
```

```python
# telemac_tools/hecras/__init__.py
"""HEC-RAS HDF5 parsers."""
```

```python
# telemac_tools/domain/__init__.py
"""Domain building: geometry + DEM → TELEMAC domain."""
```

```python
# telemac_tools/meshing/__init__.py
"""Mesh generation backends."""
```

```python
# telemac_tools/telemac/__init__.py
"""TELEMAC file writers (.slf, .cli, .cas)."""
```

```python
# telemac_tools/tests/__init__.py
```

```python
# telemac_tools/tests/conftest.py
"""Shared test fixtures for telemac_tools."""
from __future__ import annotations
import pytest
```

- [ ] **Step 3: Create data model with all dataclasses**

```python
# telemac_tools/model.py
"""Intermediate data model for HEC-RAS → TELEMAC pipeline."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class CrossSection:
    station: float                          # chainage along river (meters)
    coords: np.ndarray                      # (N, 3) x, y, z in world coordinates
    mannings_n: list[float]                 # [left_overbank, channel, right_overbank]
    bank_stations: tuple[float, float]      # left/right bank station (local)
    bank_coords: np.ndarray                 # (2, 2) left/right bank x, y world


@dataclass
class Reach:
    name: str
    alignment: np.ndarray                   # (M, 2) river centerline x, y
    cross_sections: list[CrossSection] = field(default_factory=list)


@dataclass
class BoundaryCondition:
    bc_type: str                            # "flow", "stage", "normal_depth", "rating_curve"
    location: str                           # "upstream" / "downstream" / reach name
    line_coords: np.ndarray | None = None   # (L, 2) BC line geometry


@dataclass
class HecRasCell:
    face_point_indices: list[int]           # indices into face-points array


@dataclass
class HecRas2DArea:
    name: str
    face_points: np.ndarray                 # (F, 2)
    cell_centers: np.ndarray                # (C, 2)
    cells: list[HecRasCell] = field(default_factory=list)
    elevation: np.ndarray | None = None     # (C,) bed elevation per cell
    mannings_n_constant: float = 0.035
    mannings_n_raster: str | None = None


@dataclass
class Mesh2D:
    nodes: np.ndarray                       # (N, 2) x, y
    elements: np.ndarray                    # (E, 3) triangles (0-based)
    elevation: np.ndarray                   # (N,) bed elevation
    mannings_n: np.ndarray                  # (N,) Manning's n


@dataclass
class HecRasModel:
    rivers: list[Reach] = field(default_factory=list)
    boundaries: list[BoundaryCondition] = field(default_factory=list)
    areas_2d: list[HecRas2DArea] = field(default_factory=list)
    crs: str | None = None


@dataclass
class BCSegment:
    node_indices: list[int]
    lihbor: int                             # 2=wall, 4=free, 5=prescribed
    prescribed_h: float | None = None
    prescribed_q: float | None = None
    _line_coords: np.ndarray | None = None  # BC line geometry for post-mesh matching


@dataclass
class TelemacDomain:
    boundary_polygon: np.ndarray            # (P, 2)
    refinement_zones: list[dict] = field(default_factory=list)
    channel_points: np.ndarray | None = None  # (C, 3) x, y, z
    channel_segments: np.ndarray | None = None  # (S, 2)
    mannings_regions: list[dict] = field(default_factory=list)
    bc_segments: list[BCSegment] = field(default_factory=list)
    _dem_data: np.ndarray | None = None     # DEM raster for elevation sampling
    _dem_transform: dict | None = None      # DEM geotransform


class HecRasParseError(Exception):
    """Raised when HDF5 structure is invalid or missing expected groups."""
```

- [ ] **Step 4: Write data model tests**

```python
# telemac_tools/tests/test_model.py
"""Tests for telemac_tools.model dataclasses."""
from __future__ import annotations
import numpy as np
import pytest
from telemac_tools.model import (
    CrossSection, Reach, BoundaryCondition, HecRasCell, HecRas2DArea,
    Mesh2D, HecRasModel, BCSegment, TelemacDomain, HecRasParseError,
)


class TestDataclasses:
    def test_cross_section(self):
        xs = CrossSection(
            station=100.0,
            coords=np.zeros((5, 3)),
            mannings_n=[0.06, 0.035, 0.06],
            bank_stations=(20.0, 80.0),
            bank_coords=np.zeros((2, 2)),
        )
        assert xs.station == 100.0
        assert len(xs.mannings_n) == 3

    def test_reach(self):
        r = Reach(name="River1", alignment=np.zeros((10, 2)))
        assert r.name == "River1"
        assert len(r.cross_sections) == 0

    def test_mesh2d(self):
        m = Mesh2D(
            nodes=np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64),
            elements=np.array([[0, 1, 2]], dtype=np.int32),
            elevation=np.array([0.0, 1.0, 0.5]),
            mannings_n=np.array([0.035, 0.035, 0.035]),
        )
        assert m.nodes.shape == (3, 2)
        assert m.elements.shape == (1, 3)

    def test_hecras_model_defaults(self):
        m = HecRasModel()
        assert m.rivers == []
        assert m.boundaries == []
        assert m.areas_2d == []

    def test_bc_segment(self):
        bc = BCSegment(node_indices=[1, 2, 3], lihbor=5, prescribed_h=2.0)
        assert bc.lihbor == 5

    def test_parse_error(self):
        with pytest.raises(HecRasParseError):
            raise HecRasParseError("Missing group")
```

- [ ] **Step 5: Run tests to verify**

Run: `cd /home/razinka/telemac/telemac-viewer && /opt/micromamba/envs/shiny/bin/python3.13 -m pytest telemac_tools/tests/test_model.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add telemac_tools/
git commit -m "feat(telemac_tools): scaffold package with data model"
```

---

### Task 2: Synthetic HDF5 fixtures for parser tests

**Files:**
- Modify: `telemac_tools/tests/conftest.py`

The HEC-RAS HDF5 structure is complex. We create minimal synthetic fixtures that mirror the real structure closely enough for unit testing.

- [ ] **Step 1: Write fixture builders in conftest.py**

```python
# telemac_tools/tests/conftest.py
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
        # XS at x=0: from (0,-50) to (0,50)
        # XS at x=100: from (100,-50) to (100,50)
        # XS at x=200: from (200,-50) to (200,50)
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
            [0, -50], [0, 50],      # upstream
            [200, -50], [200, 50],   # downstream
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

        # 9 face points forming a 3x3 grid (vertices of 4 cells)
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

        # Each cell has 4 face points (square)
        # Cell 0 (bottom-left): fp 0,1,4,3
        # Cell 1 (bottom-right): fp 1,2,5,4
        # Cell 2 (top-left): fp 3,4,7,6
        # Cell 3 (top-right): fp 4,5,8,7
        face_counts = np.array([4, 4, 4, 4])
        face_values = np.array([
            0, 1, 4, 3,
            1, 2, 5, 4,
            3, 4, 7, 6,
            4, 5, 8, 7,
        ])
        area.create_dataset("Cells Face and Orientation Info", data=face_counts)
        area.create_dataset("Faces FacePoint Indexes", data=face_values)

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

    Flat at elevation 5.0m (river channel will be carved lower).
    Origin at (-50, -100), covers the 1D fixture's domain.
    """
    import struct

    width, height = 30, 20  # 10m pixels
    origin_x, origin_y = -50.0, 100.0  # top-left corner
    pixel_size = 10.0

    # Flat DEM at 5.0m
    data = np.full((height, width), 5.0, dtype=np.float32)

    path = tmp_path / "dem.tif"
    # Write minimal GeoTIFF with tifffile
    import tifffile
    # ModelTiepointTag + ModelPixelScaleTag for georeference
    tifffile.imwrite(
        str(path), data,
        metadata={
            'ModelTiepointTag': (0, 0, 0, origin_x, origin_y, 0),
            'ModelPixelScaleTag': (pixel_size, pixel_size, 0),
        },
    )
    return str(path)
```

- [ ] **Step 2: Run conftest to verify fixture creation**

Run: `cd /home/razinka/telemac/telemac-viewer && /opt/micromamba/envs/shiny/bin/python3.13 -c "from telemac_tools.tests.conftest import *; print('Fixtures OK')"`
Expected: No errors (fixtures are only created when pytest calls them)

- [ ] **Step 3: Commit**

```bash
git add telemac_tools/tests/conftest.py
git commit -m "test(telemac_tools): add synthetic HDF5 and DEM fixtures"
```

---

### Task 3: Layer 1a — 1D HEC-RAS parser

**Files:**
- Create: `telemac_tools/hecras/parser_1d.py`
- Create: `telemac_tools/tests/test_parser_1d.py`

- [ ] **Step 1: Write failing tests for `parse_hecras_1d`**

```python
# telemac_tools/tests/test_parser_1d.py
"""Tests for 1D HEC-RAS geometry parsing."""
from __future__ import annotations
import numpy as np
import pytest
from telemac_tools.hecras.parser_1d import parse_hecras_1d
from telemac_tools.model import HecRasModel, HecRasParseError


class TestParseHecras1d:
    def test_returns_model(self, hdf_1d):
        model = parse_hecras_1d(hdf_1d)
        assert isinstance(model, HecRasModel)

    def test_one_reach(self, hdf_1d):
        model = parse_hecras_1d(hdf_1d)
        assert len(model.rivers) == 1
        assert model.rivers[0].name == "River1/Reach1"

    def test_three_cross_sections(self, hdf_1d):
        model = parse_hecras_1d(hdf_1d)
        xs_list = model.rivers[0].cross_sections
        assert len(xs_list) == 3

    def test_cross_section_stations(self, hdf_1d):
        model = parse_hecras_1d(hdf_1d)
        stations = [xs.station for xs in model.rivers[0].cross_sections]
        assert stations == [0.0, 100.0, 200.0]

    def test_cross_section_coords_shape(self, hdf_1d):
        model = parse_hecras_1d(hdf_1d)
        xs = model.rivers[0].cross_sections[0]
        assert xs.coords.shape == (5, 3)

    def test_mannings_n(self, hdf_1d):
        model = parse_hecras_1d(hdf_1d)
        xs = model.rivers[0].cross_sections[0]
        assert xs.mannings_n == [0.06, 0.035, 0.06]

    def test_bank_stations(self, hdf_1d):
        model = parse_hecras_1d(hdf_1d)
        xs = model.rivers[0].cross_sections[0]
        assert xs.bank_stations == (25.0, 75.0)

    def test_alignment_shape(self, hdf_1d):
        model = parse_hecras_1d(hdf_1d)
        assert model.rivers[0].alignment.shape == (4, 2)

    def test_boundary_conditions(self, hdf_1d):
        model = parse_hecras_1d(hdf_1d)
        assert len(model.boundaries) == 2
        assert model.boundaries[0].bc_type == "flow"
        assert model.boundaries[1].bc_type == "normal_depth"

    def test_bc_line_coords(self, hdf_1d):
        model = parse_hecras_1d(hdf_1d)
        assert model.boundaries[0].line_coords.shape == (2, 2)

    def test_invalid_file_raises(self, tmp_path):
        import h5py
        bad = tmp_path / "empty.hdf"
        with h5py.File(bad, "w") as f:
            f.create_group("NoGeometry")
        with pytest.raises(HecRasParseError):
            parse_hecras_1d(str(bad))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/razinka/telemac/telemac-viewer && /opt/micromamba/envs/shiny/bin/python3.13 -m pytest telemac_tools/tests/test_parser_1d.py -v`
Expected: FAIL — `ImportError: cannot import name 'parse_hecras_1d'`

- [ ] **Step 3: Implement `parse_hecras_1d`**

```python
# telemac_tools/hecras/parser_1d.py
"""Parse 1D geometry from HEC-RAS 6.x HDF5 file."""
from __future__ import annotations
import warnings
import h5py
import numpy as np
from telemac_tools.model import (
    CrossSection, Reach, BoundaryCondition, HecRasModel, HecRasParseError,
)


def parse_hecras_1d(path: str) -> HecRasModel:
    """Parse 1D geometry (reaches, cross-sections, BCs) from HEC-RAS HDF5.

    Args:
        path: Path to .g##.hdf file.

    Returns:
        HecRasModel with rivers and boundaries populated.

    Raises:
        HecRasParseError: If required HDF5 groups are missing.
    """
    with h5py.File(path, "r") as f:
        if "Geometry" not in f:
            raise HecRasParseError(f"No 'Geometry' group in {path}")
        geo = f["Geometry"]

        rivers = _parse_reaches(geo)
        boundaries = _parse_boundary_conditions(geo)

    return HecRasModel(rivers=rivers, boundaries=boundaries)


def _parse_reaches(geo: h5py.Group) -> list[Reach]:
    """Parse river centerlines and their cross-sections."""
    if "River Centerlines" not in geo:
        return []

    rc = geo["River Centerlines"]
    rc_attrs = rc["Attributes"][:]
    rc_info = rc["Polyline Info"][:]
    rc_pts = rc["Polyline Points"][:]

    # Build reach objects with alignments
    reaches: dict[str, Reach] = {}
    for i, attr in enumerate(rc_attrs):
        river = attr["River"].decode().strip()
        reach_name = attr["Reach"].decode().strip()
        key = f"{river}/{reach_name}"
        off, count = int(rc_info[i, 0]), int(rc_info[i, 1])
        alignment = rc_pts[off:off + count]
        reaches[key] = Reach(name=key, alignment=alignment)

    # Parse cross-sections and attach to reaches
    if "Cross Sections" in geo:
        _attach_cross_sections(geo["Cross Sections"], reaches)

    return list(reaches.values())


def _attach_cross_sections(xs_grp: h5py.Group, reaches: dict[str, Reach]) -> None:
    """Parse cross-sections and attach to their parent reach."""
    attrs = xs_grp["Attributes"][:]
    se_info = xs_grp["Station Elevation Info"][:]
    se_vals = xs_grp["Station Elevation Values"][:]
    poly_info = xs_grp["Polyline Info"][:]
    poly_pts = xs_grp["Polyline Points"][:]

    # Optional datasets
    manning_info = xs_grp["Manning's n Info"][:] if "Manning's n Info" in xs_grp else None
    manning_vals = xs_grp["Manning's n Values"][:] if "Manning's n Values" in xs_grp else None
    bank_data = xs_grp["Bank Stations"][:] if "Bank Stations" in xs_grp else None

    for i, attr in enumerate(attrs):
        river = attr["River"].decode().strip()
        reach_name = attr["Reach"].decode().strip()
        station = float(attr["Station"])
        key = f"{river}/{reach_name}"

        if key not in reaches:
            warnings.warn(f"Cross-section references unknown reach {key}, skipping")
            continue

        # Station-elevation values
        se_off, se_cnt = int(se_info[i, 0]), int(se_info[i, 1])
        se = se_vals[se_off:se_off + se_cnt]
        local_stations = se[:, 0]
        elevations = se[:, 1]

        # World-coordinate polyline
        p_off, p_cnt = int(poly_info[i, 0]), int(poly_info[i, 1])
        poly = poly_pts[p_off:p_off + p_cnt]

        # Interpolate world coords for each station point along the XS polyline
        coords = _interpolate_xs_world_coords(local_stations, poly, elevations)

        # Manning's n
        if manning_info is not None and manning_vals is not None:
            m_off, m_cnt = int(manning_info[i, 0]), int(manning_info[i, 1])
            n_vals = manning_vals[m_off:m_off + m_cnt].tolist()
            # Simplify to [left, channel, right]
            if len(n_vals) >= 3:
                mannings = [n_vals[0], n_vals[1], n_vals[2]]
            else:
                mannings = [0.035, 0.035, 0.035]
        else:
            mannings = [0.035, 0.035, 0.035]

        # Bank stations
        if bank_data is not None:
            banks = (float(bank_data[i, 0]), float(bank_data[i, 1]))
            # Bank world coords: interpolate along polyline
            bank_coords = _interpolate_xs_world_coords(
                np.array([banks[0], banks[1]]), poly, np.zeros(2),
            )[:, :2]
        else:
            banks = (local_stations[0], local_stations[-1])
            bank_coords = np.array([[poly[0, 0], poly[0, 1]],
                                     [poly[-1, 0], poly[-1, 1]]])

        xs = CrossSection(
            station=station,
            coords=coords,
            mannings_n=mannings,
            bank_stations=banks,
            bank_coords=bank_coords,
        )
        reaches[key].cross_sections.append(xs)


def _interpolate_xs_world_coords(
    local_stations: np.ndarray,
    polyline: np.ndarray,
    elevations: np.ndarray,
) -> np.ndarray:
    """Map local station distances to world x,y coordinates along the XS polyline.

    Linearly interpolate along the polyline using cumulative distance.
    Returns (N, 3) array of x, y, z.
    """
    # Compute cumulative distance along polyline
    diffs = np.diff(polyline, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cum_dist = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_length = cum_dist[-1]

    if total_length < 1e-10:
        # Degenerate polyline — all points at same location
        x = np.full(len(local_stations), polyline[0, 0])
        y = np.full(len(local_stations), polyline[0, 1])
    else:
        # Normalize local stations to polyline distance
        s_min, s_max = local_stations[0], local_stations[-1]
        s_range = s_max - s_min
        if s_range < 1e-10:
            frac = np.zeros(len(local_stations))
        else:
            frac = (local_stations - s_min) / s_range
        dist_along = frac * total_length

        x = np.interp(dist_along, cum_dist, polyline[:, 0])
        y = np.interp(dist_along, cum_dist, polyline[:, 1])

    return np.column_stack([x, y, elevations])


def _parse_boundary_conditions(geo: h5py.Group) -> list[BoundaryCondition]:
    """Parse boundary condition lines."""
    if "Boundary Condition Lines" not in geo:
        return []

    bc_grp = geo["Boundary Condition Lines"]
    attrs = bc_grp["Attributes"][:]
    info = bc_grp["Polyline Info"][:]
    pts = bc_grp["Polyline Points"][:]

    bcs = []
    for i, attr in enumerate(attrs):
        name = attr["Name"].decode().strip()
        bc_type = attr["Type"].decode().strip()
        off, cnt = int(info[i, 0]), int(info[i, 1])
        line = pts[off:off + cnt]

        location = "upstream" if "upstream" in name.lower() else "downstream"
        bcs.append(BoundaryCondition(
            bc_type=bc_type,
            location=location,
            line_coords=line,
        ))
    return bcs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/razinka/telemac/telemac-viewer && /opt/micromamba/envs/shiny/bin/python3.13 -m pytest telemac_tools/tests/test_parser_1d.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add telemac_tools/hecras/parser_1d.py telemac_tools/tests/test_parser_1d.py
git commit -m "feat(hecras): add 1D geometry parser from HDF5"
```

---

### Task 4: Layer 1b — 2D HEC-RAS parser

**Files:**
- Create: `telemac_tools/hecras/parser_2d.py`
- Create: `telemac_tools/tests/test_parser_2d.py`

- [ ] **Step 1: Write failing tests for `parse_hecras_2d`**

```python
# telemac_tools/tests/test_parser_2d.py
"""Tests for 2D HEC-RAS geometry parsing."""
from __future__ import annotations
import numpy as np
import pytest
from telemac_tools.hecras.parser_2d import parse_hecras_2d
from telemac_tools.model import HecRasModel, Mesh2D, HecRasParseError


class TestParseHecras2d:
    def test_returns_model(self, hdf_2d):
        model = parse_hecras_2d(hdf_2d)
        assert isinstance(model, HecRasModel)

    def test_one_2d_area(self, hdf_2d):
        model = parse_hecras_2d(hdf_2d)
        assert len(model.areas_2d) == 1
        assert model.areas_2d[0].name == "TestArea"

    def test_face_points(self, hdf_2d):
        model = parse_hecras_2d(hdf_2d)
        area = model.areas_2d[0]
        assert area.face_points.shape == (9, 2)

    def test_cell_centers(self, hdf_2d):
        model = parse_hecras_2d(hdf_2d)
        area = model.areas_2d[0]
        assert area.cell_centers.shape == (4, 2)

    def test_four_cells(self, hdf_2d):
        model = parse_hecras_2d(hdf_2d)
        area = model.areas_2d[0]
        assert len(area.cells) == 4

    def test_cell_face_point_count(self, hdf_2d):
        model = parse_hecras_2d(hdf_2d)
        area = model.areas_2d[0]
        for cell in area.cells:
            assert len(cell.face_point_indices) == 4

    def test_elevation(self, hdf_2d):
        model = parse_hecras_2d(hdf_2d)
        area = model.areas_2d[0]
        assert area.elevation is not None
        assert len(area.elevation) == 4

    def test_invalid_file_raises(self, tmp_path):
        import h5py
        bad = tmp_path / "empty.hdf"
        with h5py.File(bad, "w") as f:
            f.create_group("Geometry")
        with pytest.raises(HecRasParseError):
            parse_hecras_2d(str(bad))


class TestVoronoiToTriangles:
    """Test the Voronoi→triangle conversion for 2D meshes."""

    def test_triangulate_2d_area(self, hdf_2d):
        from telemac_tools.hecras.parser_2d import triangulate_2d_area
        model = parse_hecras_2d(hdf_2d)
        area = model.areas_2d[0]
        mesh = triangulate_2d_area(area)
        assert isinstance(mesh, Mesh2D)
        assert mesh.nodes.shape[1] == 2
        assert mesh.elements.shape[1] == 3

    def test_triangles_cover_cells(self, hdf_2d):
        from telemac_tools.hecras.parser_2d import triangulate_2d_area
        model = parse_hecras_2d(hdf_2d)
        mesh = triangulate_2d_area(model.areas_2d[0])
        # 4 square cells → fan triangulation from centroid: 4 triangles per cell = 16
        assert mesh.elements.shape[0] >= 4

    def test_elevation_interpolated(self, hdf_2d):
        from telemac_tools.hecras.parser_2d import triangulate_2d_area
        model = parse_hecras_2d(hdf_2d)
        mesh = triangulate_2d_area(model.areas_2d[0])
        assert len(mesh.elevation) == mesh.nodes.shape[0]
        assert not np.any(np.isnan(mesh.elevation))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/razinka/telemac/telemac-viewer && /opt/micromamba/envs/shiny/bin/python3.13 -m pytest telemac_tools/tests/test_parser_2d.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement `parse_hecras_2d` and `triangulate_2d_area`**

```python
# telemac_tools/hecras/parser_2d.py
"""Parse 2D geometry from HEC-RAS 6.x HDF5 file."""
from __future__ import annotations
import h5py
import numpy as np
from scipy.interpolate import NearestNDInterpolator
from telemac_tools.model import (
    HecRasCell, HecRas2DArea, HecRasModel, Mesh2D, HecRasParseError,
)


def parse_hecras_2d(path: str) -> HecRasModel:
    """Parse 2D flow areas from HEC-RAS HDF5.

    Args:
        path: Path to .g##.hdf file.

    Returns:
        HecRasModel with areas_2d populated.

    Raises:
        HecRasParseError: If required groups are missing.
    """
    with h5py.File(path, "r") as f:
        if "Geometry" not in f:
            raise HecRasParseError(f"No 'Geometry' group in {path}")
        geo = f["Geometry"]
        if "2D Flow Areas" not in geo:
            raise HecRasParseError(f"No '2D Flow Areas' group in {path}")

        areas_grp = geo["2D Flow Areas"]
        areas = []
        for name in areas_grp:
            if not isinstance(areas_grp[name], h5py.Group):
                continue
            area = _parse_one_area(name, areas_grp[name], areas_grp)
            areas.append(area)

        if not areas:
            raise HecRasParseError("No 2D flow area sub-groups found")

    return HecRasModel(areas_2d=areas)


def _parse_one_area(name: str, grp: h5py.Group, parent: h5py.Group) -> HecRas2DArea:
    """Parse a single 2D flow area."""
    face_points = grp["FacePoints Coordinate"][:]
    cell_centers = grp["Cell Points"][:]

    # Cell-to-face-point connectivity
    face_counts = grp["Cells Face and Orientation Info"][:]
    face_indices = grp["Faces FacePoint Indexes"][:]

    cells = []
    offset = 0
    for count in face_counts:
        n = int(count)
        indices = face_indices[offset:offset + n].tolist()
        cells.append(HecRasCell(face_point_indices=indices))
        offset += n

    # Elevation
    elevation = None
    if "Cells Minimum Elevation" in grp:
        elevation = grp["Cells Minimum Elevation"][:]

    # Manning's n
    n_constant = 0.035
    if "Manning's n" in parent.attrs:
        n_constant = float(parent.attrs["Manning's n"])

    return HecRas2DArea(
        name=name,
        face_points=face_points,
        cell_centers=cell_centers,
        cells=cells,
        elevation=elevation,
        mannings_n_constant=n_constant,
    )


def triangulate_2d_area(area: HecRas2DArea) -> Mesh2D:
    """Convert HEC-RAS 2D Voronoi cells to a triangular mesh.

    Fan-triangulates each polygon cell from its centroid to its face points.
    Shared face points become shared triangle vertices.

    Args:
        area: Parsed HecRas2DArea.

    Returns:
        Mesh2D with triangulated mesh.
    """
    fp = area.face_points
    cc = area.cell_centers

    # Build nodes: face points + cell centers
    n_fp = len(fp)
    nodes = np.vstack([fp, cc])  # face points first, then centers

    triangles = []
    for ci, cell in enumerate(area.cells):
        center_idx = n_fp + ci
        fp_idx = cell.face_point_indices
        n = len(fp_idx)
        for j in range(n):
            a = fp_idx[j]
            b = fp_idx[(j + 1) % n]
            triangles.append([center_idx, a, b])

    elements = np.array(triangles, dtype=np.int32)

    # Interpolate elevation from cell centers to all nodes
    if area.elevation is not None:
        interp = NearestNDInterpolator(cc, area.elevation)
        elevation = interp(nodes)
    else:
        elevation = np.zeros(len(nodes))

    mannings = np.full(len(nodes), area.mannings_n_constant)

    return Mesh2D(
        nodes=nodes[:, :2],
        elements=elements,
        elevation=elevation,
        mannings_n=mannings,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/razinka/telemac/telemac-viewer && /opt/micromamba/envs/shiny/bin/python3.13 -m pytest telemac_tools/tests/test_parser_2d.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add telemac_tools/hecras/parser_2d.py telemac_tools/tests/test_parser_2d.py
git commit -m "feat(hecras): add 2D geometry parser with Voronoi→triangle conversion"
```

---

### Task 5: Layer 1c — Manning's n extraction + unified parse_hecras

**Files:**
- Create: `telemac_tools/hecras/manning.py`
- Create: `telemac_tools/tests/test_manning.py`
- Modify: `telemac_tools/hecras/__init__.py`

- [ ] **Step 1: Write failing tests**

```python
# telemac_tools/tests/test_manning.py
"""Tests for Manning's n extraction."""
from __future__ import annotations
import numpy as np
import pytest
from telemac_tools.hecras.manning import extract_mannings_1d


class TestExtractMannings1d:
    def test_returns_three_values_per_xs(self, hdf_1d):
        from telemac_tools.hecras.parser_1d import parse_hecras_1d
        model = parse_hecras_1d(hdf_1d)
        for xs in model.rivers[0].cross_sections:
            assert len(xs.mannings_n) == 3
            assert all(n > 0 for n in xs.mannings_n)

    def test_default_on_missing(self, tmp_path):
        """When Manning's data is missing, defaults to 0.035."""
        import h5py
        path = tmp_path / "no_manning.hdf"
        with h5py.File(path, "w") as f:
            geo = f.create_group("Geometry")
            xs = geo.create_group("Cross Sections")
            xs.create_dataset("Attributes", data=np.array(
                [("R", "R", 0.0)],
                dtype=[("River", "S40"), ("Reach", "S40"), ("Station", "f8")],
            ))
            xs.create_dataset("Station Elevation Info", data=np.array([[0, 2]]))
            xs.create_dataset("Station Elevation Values", data=np.array([[0, 1], [10, 0]]))
            xs.create_dataset("Polyline Info", data=np.array([[0, 2]]))
            xs.create_dataset("Polyline Points", data=np.array([[0, 0], [10, 0]], dtype=np.float64))
            # No Manning's datasets
            rc = geo.create_group("River Centerlines")
            rc.create_dataset("Attributes", data=np.array(
                [("R", "R")], dtype=[("River", "S40"), ("Reach", "S40")],
            ))
            rc.create_dataset("Polyline Info", data=np.array([[0, 2]]))
            rc.create_dataset("Polyline Points", data=np.array([[0, 0], [10, 0]], dtype=np.float64))

        n_values = extract_mannings_1d(str(path))
        assert n_values == [[0.035, 0.035, 0.035]]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/razinka/telemac/telemac-viewer && /opt/micromamba/envs/shiny/bin/python3.13 -m pytest telemac_tools/tests/test_manning.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement manning.py and update hecras/__init__.py**

```python
# telemac_tools/hecras/manning.py
"""Manning's n extraction from HEC-RAS HDF5."""
from __future__ import annotations
from telemac_tools.hecras.parser_1d import parse_hecras_1d


def extract_mannings_1d(path: str) -> list[list[float]]:
    """Extract Manning's n values from all cross-sections.

    Returns list of [left, channel, right] per cross-section.
    """
    model = parse_hecras_1d(path)
    result = []
    for reach in model.rivers:
        for xs in reach.cross_sections:
            result.append(xs.mannings_n)
    return result
```

```python
# telemac_tools/hecras/__init__.py
"""HEC-RAS HDF5 parsers."""
from telemac_tools.hecras.parser_1d import parse_hecras_1d
from telemac_tools.hecras.parser_2d import parse_hecras_2d, triangulate_2d_area
from telemac_tools.model import HecRasModel, HecRasParseError
import h5py


def parse_hecras(path: str) -> HecRasModel:
    """Auto-detect 1D/2D content and parse accordingly.

    Returns a combined HecRasModel with both 1D and 2D data if present.
    """
    with h5py.File(path, "r") as f:
        has_1d = "Geometry" in f and "Cross Sections" in f["Geometry"]
        has_2d = "Geometry" in f and "2D Flow Areas" in f["Geometry"]

    if not has_1d and not has_2d:
        raise HecRasParseError(f"No 1D or 2D geometry found in {path}")

    model = HecRasModel()
    if has_1d:
        m1d = parse_hecras_1d(path)
        model.rivers = m1d.rivers
        model.boundaries = m1d.boundaries
    if has_2d:
        m2d = parse_hecras_2d(path)
        model.areas_2d = m2d.areas_2d

    return model
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/razinka/telemac/telemac-viewer && /opt/micromamba/envs/shiny/bin/python3.13 -m pytest telemac_tools/tests/test_manning.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add telemac_tools/hecras/
git commit -m "feat(hecras): add Manning's extraction and unified parse_hecras"
```

---

### Task 6: Layer 2a — Channel carving (station→world transform + thalweg interpolation)

**Files:**
- Create: `telemac_tools/domain/channel_carve.py`
- Create: `telemac_tools/tests/test_channel_carve.py`

- [ ] **Step 1: Write failing tests**

```python
# telemac_tools/tests/test_channel_carve.py
"""Tests for channel carving and station→world coordinate transforms."""
from __future__ import annotations
import numpy as np
import pytest
from telemac_tools.domain.channel_carve import (
    interpolate_thalweg, build_channel_points,
)
from telemac_tools.model import CrossSection, Reach


def _make_reach():
    """Simple reach: 200m along x-axis, 3 XS perpendicular (y-axis)."""
    alignment = np.array([[0, 0], [100, 0], [200, 0]], dtype=np.float64)
    xs0 = CrossSection(
        station=0.0,
        coords=np.array([[-0, -50, 5], [0, -25, 2], [0, 0, 0], [0, 25, 2], [0, 50, 5]]),
        mannings_n=[0.06, 0.035, 0.06],
        bank_stations=(25.0, 75.0),
        bank_coords=np.array([[0, -25], [0, 25]]),
    )
    xs1 = CrossSection(
        station=100.0,
        coords=np.array([[100, -50, 5], [100, -25, 2], [100, 0, -1], [100, 25, 2], [100, 50, 5]]),
        mannings_n=[0.06, 0.035, 0.06],
        bank_stations=(25.0, 75.0),
        bank_coords=np.array([[100, -25], [100, 25]]),
    )
    xs2 = CrossSection(
        station=200.0,
        coords=np.array([[200, -50, 5], [200, -25, 2], [200, 0, -2], [200, 25, 2], [200, 50, 5]]),
        mannings_n=[0.06, 0.035, 0.06],
        bank_stations=(25.0, 75.0),
        bank_coords=np.array([[200, -25], [200, 25]]),
    )
    return Reach(name="test", alignment=alignment, cross_sections=[xs0, xs1, xs2])


class TestInterpolateThalweg:
    def test_returns_xyz(self):
        reach = _make_reach()
        pts = interpolate_thalweg(reach, spacing=50.0)
        assert pts.shape[1] == 3

    def test_spacing(self):
        reach = _make_reach()
        pts = interpolate_thalweg(reach, spacing=50.0)
        # 200m at 50m spacing → ~5 points (0, 50, 100, 150, 200)
        assert pts.shape[0] >= 4

    def test_elevation_interpolated(self):
        reach = _make_reach()
        pts = interpolate_thalweg(reach, spacing=50.0)
        # Thalweg z at station 0 = 0, at 100 = -1, at 200 = -2
        # Should be monotonically decreasing
        assert pts[0, 2] > pts[-1, 2]


class TestBuildChannelPoints:
    def test_returns_points_and_segments(self):
        reach = _make_reach()
        points, segments = build_channel_points(reach, spacing=50.0)
        assert points.shape[1] == 3
        assert segments.shape[1] == 2
        assert segments.shape[0] >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/razinka/telemac/telemac-viewer && /opt/micromamba/envs/shiny/bin/python3.13 -m pytest telemac_tools/tests/test_channel_carve.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement channel_carve.py**

```python
# telemac_tools/domain/channel_carve.py
"""Station→world coordinate transforms and thalweg interpolation."""
from __future__ import annotations
import numpy as np
from telemac_tools.model import CrossSection, Reach


def interpolate_thalweg(reach: Reach, spacing: float = 10.0) -> np.ndarray:
    """Interpolate thalweg (deepest channel point) between cross-sections.

    Args:
        reach: Reach with cross-sections.
        spacing: Distance between interpolated points (meters).

    Returns:
        (N, 3) array of x, y, z points along the thalweg.
    """
    xs_list = sorted(reach.cross_sections, key=lambda xs: xs.station)
    if len(xs_list) < 2:
        if xs_list:
            # Single XS: find minimum elevation point
            xs = xs_list[0]
            min_idx = int(np.argmin(xs.coords[:, 2]))
            return xs.coords[min_idx:min_idx + 1]
        return np.zeros((0, 3))

    # Extract thalweg point (minimum z) from each XS
    thalweg_stations = []
    thalweg_xyz = []
    for xs in xs_list:
        min_idx = int(np.argmin(xs.coords[:, 2]))
        thalweg_xyz.append(xs.coords[min_idx])
        thalweg_stations.append(xs.station)

    thalweg_xyz = np.array(thalweg_xyz)
    thalweg_stations = np.array(thalweg_stations)

    # Interpolate at regular spacing
    total = thalweg_stations[-1] - thalweg_stations[0]
    n_pts = max(int(total / spacing) + 1, 2)
    interp_stations = np.linspace(thalweg_stations[0], thalweg_stations[-1], n_pts)

    x = np.interp(interp_stations, thalweg_stations, thalweg_xyz[:, 0])
    y = np.interp(interp_stations, thalweg_stations, thalweg_xyz[:, 1])
    z = np.interp(interp_stations, thalweg_stations, thalweg_xyz[:, 2])

    return np.column_stack([x, y, z])


def build_channel_points(
    reach: Reach, spacing: float = 10.0
) -> tuple[np.ndarray, np.ndarray]:
    """Build channel constraint points and segments for PSLG meshing.

    Returns:
        points: (N, 3) channel bed points x, y, z.
        segments: (N-1, 2) segment index pairs for constrained edges.
    """
    points = interpolate_thalweg(reach, spacing)
    n = len(points)
    if n < 2:
        return points, np.zeros((0, 2), dtype=np.int32)
    segments = np.column_stack([np.arange(n - 1), np.arange(1, n)]).astype(np.int32)
    return points, segments
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/razinka/telemac/telemac-viewer && /opt/micromamba/envs/shiny/bin/python3.13 -m pytest telemac_tools/tests/test_channel_carve.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add telemac_tools/domain/channel_carve.py telemac_tools/tests/test_channel_carve.py
git commit -m "feat(domain): add channel carving with thalweg interpolation"
```

---

### Task 7: Layer 2b — Domain builder

**Files:**
- Create: `telemac_tools/domain/builder.py`
- Create: `telemac_tools/tests/test_domain_builder.py`
- Modify: `telemac_tools/domain/__init__.py`

- [ ] **Step 1: Write failing tests**

```python
# telemac_tools/tests/test_domain_builder.py
"""Tests for domain builder (geometry + DEM → TelemacDomain)."""
from __future__ import annotations
import numpy as np
import pytest
from telemac_tools.domain.builder import build_domain_1d, build_domain_2d
from telemac_tools.model import TelemacDomain, HecRasModel


class TestBuildDomain1d:
    def test_returns_domain(self, hdf_1d, fake_dem):
        from telemac_tools.hecras.parser_1d import parse_hecras_1d
        model = parse_hecras_1d(hdf_1d)
        domain = build_domain_1d(model, dem_path=fake_dem, floodplain_width=80.0)
        assert isinstance(domain, TelemacDomain)

    def test_boundary_polygon_closed(self, hdf_1d, fake_dem):
        from telemac_tools.hecras.parser_1d import parse_hecras_1d
        model = parse_hecras_1d(hdf_1d)
        domain = build_domain_1d(model, dem_path=fake_dem, floodplain_width=80.0)
        poly = domain.boundary_polygon
        assert poly.shape[1] == 2
        assert np.allclose(poly[0], poly[-1])  # closed ring

    def test_has_channel_points(self, hdf_1d, fake_dem):
        from telemac_tools.hecras.parser_1d import parse_hecras_1d
        model = parse_hecras_1d(hdf_1d)
        domain = build_domain_1d(model, dem_path=fake_dem, floodplain_width=80.0)
        assert domain.channel_points is not None
        assert domain.channel_points.shape[1] == 3

    def test_has_bc_segments(self, hdf_1d, fake_dem):
        from telemac_tools.hecras.parser_1d import parse_hecras_1d
        model = parse_hecras_1d(hdf_1d)
        domain = build_domain_1d(model, dem_path=fake_dem, floodplain_width=80.0)
        assert len(domain.bc_segments) >= 1

    def test_dem_loaded(self, hdf_1d, fake_dem):
        from telemac_tools.hecras.parser_1d import parse_hecras_1d
        model = parse_hecras_1d(hdf_1d)
        domain = build_domain_1d(model, dem_path=fake_dem, floodplain_width=80.0)
        assert domain._dem_data is not None
        assert domain._dem_transform is not None

    def test_sample_dem(self, hdf_1d, fake_dem):
        from telemac_tools.hecras.parser_1d import parse_hecras_1d
        from telemac_tools.domain.builder import sample_dem
        model = parse_hecras_1d(hdf_1d)
        domain = build_domain_1d(model, dem_path=fake_dem, floodplain_width=80.0)
        # Sample at a point inside the DEM
        z = sample_dem(np.array([50.0]), np.array([0.0]),
                       domain._dem_data, domain._dem_transform)
        assert z[0] == pytest.approx(5.0, abs=0.5)  # flat DEM at 5m


class TestBuildDomain2d:
    def test_returns_domain(self, hdf_2d):
        from telemac_tools.hecras.parser_2d import parse_hecras_2d
        model = parse_hecras_2d(hdf_2d)
        domain = build_domain_2d(model)
        assert isinstance(domain, TelemacDomain)

    def test_boundary_from_mesh(self, hdf_2d):
        from telemac_tools.hecras.parser_2d import parse_hecras_2d
        model = parse_hecras_2d(hdf_2d)
        domain = build_domain_2d(model)
        assert domain.boundary_polygon.shape[1] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/razinka/telemac/telemac-viewer && /opt/micromamba/envs/shiny/bin/python3.13 -m pytest telemac_tools/tests/test_domain_builder.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement builder.py**

```python
# telemac_tools/domain/builder.py
"""Domain builder: combine HEC-RAS geometry + DEM → TelemacDomain."""
from __future__ import annotations
import numpy as np
from telemac_tools.model import (
    HecRasModel, TelemacDomain, BCSegment,
)
from telemac_tools.domain.channel_carve import build_channel_points


def build_domain_1d(
    model: HecRasModel,
    dem_path: str,
    floodplain_width: float = 500.0,
    channel_refinement: float = 10.0,
    floodplain_refinement: float = 200.0,
    boundary_polygon: np.ndarray | None = None,
) -> TelemacDomain:
    """Build TELEMAC domain from 1D HEC-RAS model + DEM.

    Args:
        model: Parsed HecRasModel with rivers.
        dem_path: Path to GeoTIFF DEM.
        floodplain_width: Buffer distance each side of river (meters).
        channel_refinement: Max triangle area in channel zone (m²).
        floodplain_refinement: Max triangle area in floodplain (m²).
        boundary_polygon: Optional user-provided domain boundary (P, 2).

    Returns:
        TelemacDomain ready for mesh generation.
    """
    if not model.rivers:
        raise ValueError("Model has no 1D rivers")

    reach = model.rivers[0]

    # Load DEM
    dem_data, dem_transform = _read_dem(dem_path)

    # Build domain boundary: buffer the alignment
    if boundary_polygon is None:
        boundary_polygon = _buffer_alignment(reach.alignment, floodplain_width)

    # Channel points
    channel_pts, channel_segs = build_channel_points(reach, spacing=10.0)

    # Refinement zones
    channel_zone = _buffer_alignment(reach.alignment, 30.0)
    refinement_zones = [
        {"polygon": channel_zone, "max_area": channel_refinement},
        {"polygon": boundary_polygon, "max_area": floodplain_refinement},
    ]

    # Manning's regions from cross-sections
    mannings_regions = _build_mannings_regions(reach)

    # BC segments from model boundary conditions (with line geometry for post-mesh matching)
    bc_segments = []
    for bc in model.boundaries:
        lihbor = 5 if bc.location == "upstream" else 4
        bc_segments.append(BCSegment(
            node_indices=[],  # populated by assign_bc_nodes() after meshing
            lihbor=lihbor,
            _line_coords=bc.line_coords,  # stash for post-meshing node matching
        ))

    return TelemacDomain(
        boundary_polygon=boundary_polygon,
        refinement_zones=refinement_zones,
        channel_points=channel_pts,
        channel_segments=channel_segs,
        mannings_regions=mannings_regions,
        bc_segments=bc_segments,
        _dem_data=dem_data,
        _dem_transform=dem_transform,
    )


def build_domain_2d(model: HecRasModel) -> TelemacDomain:
    """Build TELEMAC domain from 2D HEC-RAS model.

    Extracts boundary polygon from mesh topology (edges shared by one cell)
    and maps BC segments.
    """
    if not model.areas_2d:
        raise ValueError("Model has no 2D areas")

    area = model.areas_2d[0]
    boundary = _extract_boundary_polygon(area)

    return TelemacDomain(
        boundary_polygon=boundary,
        bc_segments=[BCSegment(node_indices=[], lihbor=2)],  # default wall
    )


def _read_dem(dem_path: str) -> tuple[np.ndarray, dict]:
    """Read GeoTIFF DEM and extract geotransform.

    Returns:
        data: 2D elevation array.
        transform: dict with origin_x, origin_y, pixel_size_x, pixel_size_y.
    """
    import tifffile
    data = tifffile.imread(dem_path)
    if data.ndim == 3:
        data = data[0]  # take first band

    # Try to read geotransform from TIFF tags
    with tifffile.TiffFile(dem_path) as tif:
        page = tif.pages[0]
        tags = page.tags
        # ModelTiepointTag: (I, J, K, X, Y, Z)
        tp = tags.get("ModelTiepointTag")
        ps = tags.get("ModelPixelScaleTag")
        if tp and ps:
            tp_val = tp.value
            ps_val = ps.value
            transform = {
                "origin_x": float(tp_val[3]),
                "origin_y": float(tp_val[4]),
                "pixel_size_x": float(ps_val[0]),
                "pixel_size_y": float(ps_val[1]),
            }
        else:
            # Fallback: assume 1m pixels at origin
            transform = {
                "origin_x": 0.0, "origin_y": float(data.shape[0]),
                "pixel_size_x": 1.0, "pixel_size_y": 1.0,
            }

    return data.astype(np.float64), transform


def sample_dem(x: np.ndarray, y: np.ndarray, dem_data: np.ndarray,
               transform: dict) -> np.ndarray:
    """Sample DEM elevation at given x, y coordinates using bilinear interpolation."""
    from scipy.interpolate import RegularGridInterpolator

    ny, nx = dem_data.shape
    ox, oy = transform["origin_x"], transform["origin_y"]
    dx, dy = transform["pixel_size_x"], transform["pixel_size_y"]

    # Build coordinate arrays (row 0 = top of raster = max y)
    x_coords = ox + np.arange(nx) * dx + dx / 2
    y_coords = oy - np.arange(ny) * dy - dy / 2  # y decreases downward

    interp = RegularGridInterpolator(
        (y_coords[::-1], x_coords),  # must be ascending
        dem_data[::-1],
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    return interp(np.column_stack([y, x]))


def _extract_boundary_polygon(area) -> np.ndarray:
    """Extract boundary polygon from 2D area mesh topology.

    Boundary edges are face-point edges shared by only one cell.
    """
    edge_count: dict[tuple[int, int], int] = {}
    for cell in area.cells:
        fp = cell.face_point_indices
        n = len(fp)
        for j in range(n):
            a, b = fp[j], fp[(j + 1) % n]
            edge = (min(a, b), max(a, b))
            edge_count[edge] = edge_count.get(edge, 0) + 1

    # Collect boundary edges (count == 1)
    boundary_edges = [(a, b) for (a, b), c in edge_count.items() if c == 1]

    # Walk boundary ring
    if not boundary_edges:
        # Fallback to convex hull
        from scipy.spatial import ConvexHull
        hull = ConvexHull(area.face_points)
        pts = area.face_points[hull.vertices]
        return np.vstack([pts, pts[0:1]])

    adj: dict[int, list[int]] = {}
    for a, b in boundary_edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    # Walk from first boundary node
    start = boundary_edges[0][0]
    ring = [start]
    visited = {start}
    current = start
    while True:
        neighbors = adj.get(current, [])
        next_node = None
        for n in neighbors:
            if n not in visited:
                next_node = n
                break
        if next_node is None:
            break
        ring.append(next_node)
        visited.add(next_node)
        current = next_node

    pts = area.face_points[ring]
    return np.vstack([pts, pts[0:1]])  # close ring


def assign_bc_nodes(mesh, domain: TelemacDomain, tolerance: float = 50.0) -> None:
    """Assign boundary node indices to BC segments after meshing.

    Matches mesh boundary nodes to BC line geometry by proximity.
    Modifies domain.bc_segments in place.
    """
    from telemac_tools.telemac.writer_cli import _find_boundary_nodes
    boundary_nodes = _find_boundary_nodes(mesh)

    for seg in domain.bc_segments:
        line = getattr(seg, '_line_coords', None)
        if line is None or len(line) == 0:
            continue
        # Find boundary nodes within tolerance of the BC line
        matched = []
        for nidx in boundary_nodes:
            pt = mesh.nodes[nidx]
            # Distance to line segment
            dists = np.sqrt(((line - pt) ** 2).sum(axis=1))
            if dists.min() < tolerance:
                matched.append(nidx)
        seg.node_indices = matched


def _buffer_alignment(alignment: np.ndarray, distance: float) -> np.ndarray:
    """Create a simple rectangular buffer around a polyline.

    Returns a closed polygon (first point == last point).
    """
    if len(alignment) < 2:
        raise ValueError("Alignment must have at least 2 points")

    # Compute perpendicular offsets
    left_pts = []
    right_pts = []
    for i in range(len(alignment)):
        if i == 0:
            direction = alignment[1] - alignment[0]
        elif i == len(alignment) - 1:
            direction = alignment[-1] - alignment[-2]
        else:
            direction = alignment[i + 1] - alignment[i - 1]
        norm = np.sqrt(direction[0] ** 2 + direction[1] ** 2)
        if norm < 1e-10:
            perp = np.array([0, 1])
        else:
            perp = np.array([-direction[1], direction[0]]) / norm
        left_pts.append(alignment[i] + perp * distance)
        right_pts.append(alignment[i] - perp * distance)

    # Close polygon: left forward + right reversed + close
    polygon = np.vstack(left_pts + right_pts[::-1] + [left_pts[0]])
    return polygon


def _build_mannings_regions(reach) -> list[dict]:
    """Build Manning's n spatial regions from cross-section bank positions."""
    regions = []
    for xs in reach.cross_sections:
        # Simple: one region for the channel zone at each XS
        regions.append({
            "center": xs.coords[len(xs.coords) // 2, :2],
            "n": xs.mannings_n[1],  # channel value
        })
    return regions
```

Update `telemac_tools/domain/__init__.py`:
```python
# telemac_tools/domain/__init__.py
"""Domain building: geometry + DEM → TELEMAC domain."""
from telemac_tools.domain.builder import build_domain_1d, build_domain_2d
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/razinka/telemac/telemac-viewer && /opt/micromamba/envs/shiny/bin/python3.13 -m pytest telemac_tools/tests/test_domain_builder.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add telemac_tools/domain/
git commit -m "feat(domain): add 1D and 2D domain builder"
```

---

### Task 8: Layer 3 — Triangle mesh generator

**Files:**
- Create: `telemac_tools/meshing/base.py`
- Create: `telemac_tools/meshing/triangle_mesh.py`
- Create: `telemac_tools/tests/test_mesh_generator.py`
- Modify: `telemac_tools/meshing/__init__.py`

- [ ] **Step 1: Write failing tests**

```python
# telemac_tools/tests/test_mesh_generator.py
"""Tests for mesh generation."""
from __future__ import annotations
import numpy as np
import pytest
from telemac_tools.meshing import generate_mesh
from telemac_tools.model import TelemacDomain, Mesh2D, BCSegment


def _simple_domain():
    """A 100x100m square domain with no channel."""
    poly = np.array([
        [0, 0], [100, 0], [100, 100], [0, 100], [0, 0],
    ], dtype=np.float64)
    return TelemacDomain(
        boundary_polygon=poly,
        bc_segments=[BCSegment(node_indices=[], lihbor=2)],
    )


class TestGenerateMesh:
    def test_returns_mesh2d(self):
        domain = _simple_domain()
        mesh = generate_mesh(domain, backend="triangle")
        assert isinstance(mesh, Mesh2D)

    def test_has_triangles(self):
        domain = _simple_domain()
        mesh = generate_mesh(domain, backend="triangle")
        assert mesh.elements.shape[1] == 3
        assert mesh.elements.shape[0] > 0

    def test_nodes_inside_domain(self):
        domain = _simple_domain()
        mesh = generate_mesh(domain, backend="triangle")
        assert mesh.nodes[:, 0].min() >= -1
        assert mesh.nodes[:, 0].max() <= 101
        assert mesh.nodes[:, 1].min() >= -1
        assert mesh.nodes[:, 1].max() <= 101

    def test_elevation_assigned(self):
        domain = _simple_domain()
        mesh = generate_mesh(domain, backend="triangle")
        assert len(mesh.elevation) == mesh.nodes.shape[0]

    def test_mannings_assigned(self):
        domain = _simple_domain()
        mesh = generate_mesh(domain, backend="triangle")
        assert len(mesh.mannings_n) == mesh.nodes.shape[0]
        assert np.all(mesh.mannings_n > 0)

    def test_invalid_backend(self):
        domain = _simple_domain()
        with pytest.raises(ValueError):
            generate_mesh(domain, backend="invalid")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/razinka/telemac/telemac-viewer && /opt/micromamba/envs/shiny/bin/python3.13 -m pytest telemac_tools/tests/test_mesh_generator.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement mesh generator**

```python
# telemac_tools/meshing/base.py
"""Abstract mesh generation interface."""
from __future__ import annotations
from abc import ABC, abstractmethod
from telemac_tools.model import TelemacDomain, Mesh2D


class MeshBackend(ABC):
    @abstractmethod
    def generate(self, domain: TelemacDomain, min_angle: float = 30.0,
                 max_area: float | None = None) -> Mesh2D:
        ...
```

```python
# telemac_tools/meshing/triangle_mesh.py
"""Triangle-based mesh generation."""
from __future__ import annotations
import numpy as np
import triangle as tr
from telemac_tools.meshing.base import MeshBackend
from telemac_tools.model import TelemacDomain, Mesh2D


class TriangleBackend(MeshBackend):
    def generate(self, domain: TelemacDomain, min_angle: float = 30.0,
                 max_area: float | None = None) -> Mesh2D:
        poly = domain.boundary_polygon
        # Remove closing point if present
        if np.allclose(poly[0], poly[-1]):
            poly = poly[:-1]
        n = len(poly)

        vertices = poly.tolist()
        segments = [[i, (i + 1) % n] for i in range(n)]

        pslg = {
            "vertices": np.array(vertices),
            "segments": np.array(segments),
        }

        # Add channel constraint points if available
        if domain.channel_points is not None and len(domain.channel_points) > 0:
            ch_xy = domain.channel_points[:, :2]
            base_n = len(vertices)
            extra_verts = ch_xy.tolist()
            pslg["vertices"] = np.vstack([pslg["vertices"], ch_xy])

            if domain.channel_segments is not None and len(domain.channel_segments) > 0:
                ch_segs = domain.channel_segments + base_n
                pslg["segments"] = np.vstack([pslg["segments"], ch_segs])

        # Build Triangle options string
        opts = f"pq{min_angle}"
        if max_area is not None:
            opts += f"a{max_area}"
        elif domain.refinement_zones:
            # Use largest refinement area as default
            areas = [z.get("max_area", 200) for z in domain.refinement_zones]
            opts += f"a{max(areas)}"

        result = tr.triangulate(pslg, opts)

        nodes = result["vertices"]
        elements = result["triangles"]

        # Sample DEM elevation for all nodes
        if domain._dem_data is not None and domain._dem_transform is not None:
            from telemac_tools.domain.builder import sample_dem
            elevation = sample_dem(nodes[:, 0], nodes[:, 1],
                                   domain._dem_data, domain._dem_transform)
        else:
            elevation = np.zeros(len(nodes))

        mannings = np.full(len(nodes), 0.035)

        # Override channel node elevations with carved cross-section values
        if domain.channel_points is not None and len(domain.channel_points) > 0:
            ch_pts = domain.channel_points
            for i in range(len(nodes)):
                dists = np.sqrt(((ch_pts[:, :2] - nodes[i]) ** 2).sum(axis=1))
                if dists.min() < 1.0:  # within 1m of a channel point
                    elevation[i] = ch_pts[np.argmin(dists), 2]

        return Mesh2D(
            nodes=nodes,
            elements=elements.astype(np.int32),
            elevation=elevation,
            mannings_n=mannings,
        )
```

```python
# telemac_tools/meshing/__init__.py
"""Mesh generation backends."""
from telemac_tools.model import TelemacDomain, Mesh2D


def generate_mesh(domain: TelemacDomain, backend: str = "triangle",
                  min_angle: float = 30.0, max_area: float | None = None) -> Mesh2D:
    """Generate triangular mesh from domain.

    Args:
        domain: TelemacDomain with boundary and constraints.
        backend: "triangle" or "gmsh".
        min_angle: Minimum triangle angle (degrees).
        max_area: Maximum triangle area (m²). If None, uses refinement zones.

    Returns:
        Mesh2D with nodes, elements, elevation, Manning's n.
    """
    if backend == "triangle":
        from telemac_tools.meshing.triangle_mesh import TriangleBackend
        return TriangleBackend().generate(domain, min_angle, max_area)
    elif backend == "gmsh":
        raise NotImplementedError("Gmsh backend not yet implemented")
    else:
        raise ValueError(f"Unknown backend: {backend}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/razinka/telemac/telemac-viewer && /opt/micromamba/envs/shiny/bin/python3.13 -m pytest telemac_tools/tests/test_mesh_generator.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add telemac_tools/meshing/
git commit -m "feat(meshing): add Triangle mesh generation backend"
```

---

### Task 9: Layer 4 — TELEMAC writers (.slf, .cli, .cas)

**Files:**
- Create: `telemac_tools/telemac/writer_slf.py`
- Create: `telemac_tools/telemac/writer_cli.py`
- Create: `telemac_tools/telemac/writer_cas.py`
- Create: `telemac_tools/tests/test_writers.py`
- Modify: `telemac_tools/telemac/__init__.py`

- [ ] **Step 1: Write failing tests**

```python
# telemac_tools/tests/test_writers.py
"""Tests for TELEMAC file writers."""
from __future__ import annotations
import os
import numpy as np
import pytest
from telemac_tools.model import Mesh2D, TelemacDomain, BCSegment
from telemac_tools.telemac.writer_slf import write_slf
from telemac_tools.telemac.writer_cli import write_cli
from telemac_tools.telemac.writer_cas import write_cas


def _simple_mesh():
    return Mesh2D(
        nodes=np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64),
        elements=np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32),
        elevation=np.array([0.0, 1.0, 0.5, 1.5]),
        mannings_n=np.array([0.035, 0.035, 0.035, 0.035]),
    )


def _simple_domain():
    poly = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=np.float64)
    return TelemacDomain(
        boundary_polygon=poly,
        bc_segments=[
            BCSegment(node_indices=[0, 2], lihbor=5, prescribed_h=2.0),
            BCSegment(node_indices=[1, 3], lihbor=4),
        ],
    )


class TestWriteSlf:
    def test_creates_file(self, tmp_path):
        mesh = _simple_mesh()
        path = tmp_path / "test.slf"
        write_slf(mesh, str(path))
        assert path.exists()
        assert path.stat().st_size > 0

    def test_roundtrip_with_telemac_file(self, tmp_path):
        """Write .slf and read back with TelemacFile."""
        mesh = _simple_mesh()
        path = tmp_path / "test.slf"
        write_slf(mesh, str(path))

        from data_manip.extraction.telemac_file import TelemacFile
        tf = TelemacFile(str(path))
        assert tf.npoin2 == 4
        assert tf.nelem2 == 2
        assert "BOTTOM" in tf.varnames
        tf.close()


class TestWriteCli:
    def test_creates_file(self, tmp_path):
        mesh = _simple_mesh()
        domain = _simple_domain()
        path = tmp_path / "test.cli"
        write_cli(mesh, domain, str(path))
        assert path.exists()

    def test_correct_columns(self, tmp_path):
        mesh = _simple_mesh()
        domain = _simple_domain()
        path = tmp_path / "test.cli"
        write_cli(mesh, domain, str(path))
        lines = path.read_text().strip().split("\n")
        # Each line should have 13 space-separated values
        for line in lines:
            parts = line.split()
            assert len(parts) == 13


class TestWriteCas:
    def test_creates_file(self, tmp_path):
        path = tmp_path / "test.cas"
        write_cas(str(path), name="project")
        assert path.exists()

    def test_contains_geometry(self, tmp_path):
        path = tmp_path / "test.cas"
        write_cas(str(path), name="project")
        text = path.read_text()
        assert "GEOMETRY FILE" in text
        assert "project.slf" in text

    def test_overrides_applied(self, tmp_path):
        path = tmp_path / "test.cas"
        write_cas(str(path), name="project", overrides={"TURBULENCE MODEL": 4})
        text = path.read_text()
        assert "TURBULENCE MODEL = 4" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/razinka/telemac/telemac-viewer && /opt/micromamba/envs/shiny/bin/python3.13 -m pytest telemac_tools/tests/test_writers.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement writer_slf.py**

```python
# telemac_tools/telemac/writer_slf.py
"""Write TELEMAC .slf mesh file using Selafin."""
from __future__ import annotations
import numpy as np
from telemac_tools.model import Mesh2D


def write_slf(mesh: Mesh2D, path: str, initial_depth: float = 0.1) -> None:
    """Write mesh as TELEMAC .slf file with BOTTOM, FRICTION, WATER DEPTH, FREE SURFACE.

    Args:
        mesh: Triangulated mesh with elevation and Manning's n.
        path: Output .slf file path.
        initial_depth: Initial water depth for cold start (meters).
    """
    from data_manip.extraction.selafin import Selafin

    npoin = mesh.nodes.shape[0]
    nelem = mesh.elements.shape[0]

    slf = Selafin("")
    slf.title = "HEC-RAS Import".ljust(80)[:80]
    slf.nbv1 = 4
    slf.varnames = [
        "BOTTOM".ljust(32),
        "FRICTION COEFFICIENT".ljust(32),
        "WATER DEPTH".ljust(32),
        "FREE SURFACE".ljust(32),
    ]
    slf.varunits = ["M".ljust(32)] * 4
    slf.nplan = 0
    slf.ndp2 = 3
    slf.ndp3 = 3
    slf.npoin2 = npoin
    slf.npoin3 = npoin
    slf.nelem2 = nelem
    slf.nelem3 = nelem
    slf.iparam = [0] * 10

    # 1-based connectivity
    slf.ikle2 = mesh.elements + 1
    slf.ikle3 = slf.ikle2

    slf.meshx = mesh.nodes[:, 0].astype(np.float64)
    slf.meshy = mesh.nodes[:, 1].astype(np.float64)

    # Variables at timestep 0
    bottom = mesh.elevation.astype(np.float64)
    friction = mesh.mannings_n.astype(np.float64)
    depth = np.full(npoin, initial_depth, dtype=np.float64)
    free_surface = bottom + depth

    slf.tags = {"times": [0.0]}
    slf.values = {0: [bottom, friction, depth, free_surface]}

    slf.fole = {"name": path}
    slf.fole.update({"endian": ">", "float": ("f", 4)})
    slf.append_header_slf()
    slf.append_core_time_slf(0)
```

- [ ] **Step 4: Implement writer_cli.py**

```python
# telemac_tools/telemac/writer_cli.py
"""Write TELEMAC .cli boundary conditions file."""
from __future__ import annotations
import numpy as np
from telemac_tools.model import Mesh2D, TelemacDomain


def write_cli(mesh: Mesh2D, domain: TelemacDomain, path: str) -> None:
    """Write .cli boundary conditions file.

    Identifies boundary nodes from mesh topology, assigns LIHBOR codes
    from domain.bc_segments.

    Args:
        mesh: Triangulated mesh.
        domain: Domain with BC segment definitions.
        path: Output .cli file path.
    """
    boundary_nodes = _find_boundary_nodes(mesh)

    # Build LIHBOR lookup from domain BC segments
    lihbor_map = {}
    for seg in domain.bc_segments:
        for nidx in seg.node_indices:
            lihbor_map[nidx] = (seg.lihbor, seg.prescribed_h or 0.0)

    lines = []
    for k, node in enumerate(boundary_nodes):
        lihbor, h = lihbor_map.get(node, (2, 0.0))

        if lihbor == 2:  # wall
            liubor, livbor = 0, 0
            litbor = 0
        elif lihbor == 4:  # free
            liubor, livbor = 4, 4
            litbor = 4
        elif lihbor == 5:  # prescribed
            liubor, livbor = 5, 5
            litbor = 5
        else:
            liubor, livbor = 0, 0
            litbor = 0

        # LIHBOR LIUBOR LIVBOR HBOR UBOR VBOR AUBOR LITBOR TBOR ATBOR BTBOR N_GLOBAL K_BOUND
        line = (f"{lihbor} {liubor} {livbor}  {h:.1f} 0.0 0.0 0.0  "
                f"{litbor}  0.0 0.0 0.0  {node + 1}  {k + 1}")
        lines.append(line)

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _find_boundary_nodes(mesh: Mesh2D) -> list[int]:
    """Find boundary nodes (nodes on edges shared by only one triangle)."""
    edge_count: dict[tuple[int, int], int] = {}
    for tri in mesh.elements:
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i + 1) % 3])
            edge = (min(a, b), max(a, b))
            edge_count[edge] = edge_count.get(edge, 0) + 1

    boundary = set()
    for (a, b), count in edge_count.items():
        if count == 1:
            boundary.add(a)
            boundary.add(b)

    return sorted(boundary)
```

- [ ] **Step 5: Implement writer_cas.py**

```python
# telemac_tools/telemac/writer_cas.py
"""Write TELEMAC .cas steering file."""
from __future__ import annotations

_TEMPLATE = """\
/ TELEMAC-2D steering file — generated by telemac_tools
/
TITLE = 'Imported from HEC-RAS'
GEOMETRY FILE = {name}.slf
BOUNDARY CONDITIONS FILE = {name}.cli
RESULTS FILE = r2d_{name}.slf
/
EQUATIONS = 'SAINT-VENANT FV'
FINITE VOLUME SCHEME = 5
VARIABLE TIME-STEP = YES
DESIRED COURANT NUMBER = 0.8
/
LAW OF BOTTOM FRICTION = 4
/
INITIAL CONDITIONS = 'CONSTANT DEPTH'
INITIAL DEPTH = 0.1
/
TIDAL FLATS = YES
CONTINUITY CORRECTION = YES
TREATMENT OF NEGATIVE DEPTHS = 2
/
DURATION = {duration:.1f}
GRAPHIC PRINTOUT PERIOD = 60
VARIABLES FOR GRAPHIC PRINTOUTS = 'U,V,H,S,B'
"""


def write_cas(
    path: str,
    name: str = "project",
    duration: float = 3600.0,
    overrides: dict[str, object] | None = None,
) -> None:
    """Write .cas steering file with safe defaults.

    Args:
        path: Output .cas file path.
        name: Project name (used for file references).
        duration: Simulation duration in seconds.
        overrides: Dict of keyword→value overrides to append.
    """
    text = _TEMPLATE.format(name=name, duration=duration)

    if overrides:
        text += "/\n/ User overrides\n"
        for key, val in overrides.items():
            text += f"{key} = {val}\n"

    with open(path, "w") as f:
        f.write(text)
```

Update `telemac_tools/telemac/__init__.py`:
```python
# telemac_tools/telemac/__init__.py
"""TELEMAC file writers (.slf, .cli, .cas)."""
from telemac_tools.telemac.writer_slf import write_slf
from telemac_tools.telemac.writer_cli import write_cli
from telemac_tools.telemac.writer_cas import write_cas


def write_telemac(mesh, domain, output_dir, name="project",
                  duration=3600.0, cas_overrides=None):
    """Write all TELEMAC files (.slf, .cli, .cas) to output directory."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    write_slf(mesh, os.path.join(output_dir, f"{name}.slf"))
    write_cli(mesh, domain, os.path.join(output_dir, f"{name}.cli"))
    write_cas(os.path.join(output_dir, f"{name}.cas"), name=name,
              duration=duration, overrides=cas_overrides)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /home/razinka/telemac/telemac-viewer && /opt/micromamba/envs/shiny/bin/python3.13 -m pytest telemac_tools/tests/test_writers.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add telemac_tools/telemac/
git commit -m "feat(telemac): add .slf, .cli, .cas writers"
```

---

### Task 10: Convenience entry point + integration test

**Files:**
- Modify: `telemac_tools/__init__.py`
- Create: `telemac_tools/tests/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
# telemac_tools/tests/test_integration.py
"""End-to-end integration test: HEC-RAS HDF5 → TELEMAC files."""
from __future__ import annotations
import os
import numpy as np
import pytest


class TestFullPipeline1d:
    def test_hecras_to_telemac(self, hdf_1d, fake_dem, tmp_path):
        from telemac_tools import hecras_to_telemac
        out = str(tmp_path / "output")
        hecras_to_telemac(
            hecras_path=hdf_1d,
            dem_path=fake_dem,
            output_dir=out,
            floodplain_width=80.0,
        )
        assert os.path.isfile(os.path.join(out, "project.slf"))
        assert os.path.isfile(os.path.join(out, "project.cli"))
        assert os.path.isfile(os.path.join(out, "project.cas"))

    def test_slf_readable(self, hdf_1d, fake_dem, tmp_path):
        from telemac_tools import hecras_to_telemac
        out = str(tmp_path / "output")
        hecras_to_telemac(hecras_path=hdf_1d, dem_path=fake_dem, output_dir=out,
                          floodplain_width=80.0)

        from data_manip.extraction.telemac_file import TelemacFile
        tf = TelemacFile(os.path.join(out, "project.slf"))
        assert tf.npoin2 > 0
        assert tf.nelem2 > 0
        assert "BOTTOM" in tf.varnames
        assert "FRICTION COEFFICIENT" in tf.varnames
        tf.close()

    def test_cli_has_boundary_nodes(self, hdf_1d, fake_dem, tmp_path):
        from telemac_tools import hecras_to_telemac
        out = str(tmp_path / "output")
        hecras_to_telemac(hecras_path=hdf_1d, dem_path=fake_dem, output_dir=out,
                          floodplain_width=80.0)

        cli_text = open(os.path.join(out, "project.cli")).read()
        lines = cli_text.strip().split("\n")
        assert len(lines) > 0
        # Each line has 13 fields
        for line in lines:
            assert len(line.split()) == 13


class TestFullPipeline2d:
    def test_2d_to_telemac(self, hdf_2d, tmp_path):
        from telemac_tools.hecras import parse_hecras_2d, triangulate_2d_area
        from telemac_tools.domain import build_domain_2d
        from telemac_tools.telemac import write_telemac

        model = parse_hecras_2d(hdf_2d)
        mesh = triangulate_2d_area(model.areas_2d[0])
        domain = build_domain_2d(model)

        out = str(tmp_path / "output_2d")
        write_telemac(mesh, domain, out, name="hecras2d")

        assert os.path.isfile(os.path.join(out, "hecras2d.slf"))
        assert os.path.isfile(os.path.join(out, "hecras2d.cli"))
        assert os.path.isfile(os.path.join(out, "hecras2d.cas"))
```

- [ ] **Step 2: Implement convenience function**

```python
# telemac_tools/__init__.py
"""HEC-RAS → TELEMAC import tools."""
from __future__ import annotations


def hecras_to_telemac(
    hecras_path: str,
    dem_path: str | None = None,
    output_dir: str = ".",
    name: str = "project",
    floodplain_width: float = 500.0,
    backend: str = "triangle",
    duration: float = 3600.0,
    cas_overrides: dict | None = None,
) -> None:
    """Convert HEC-RAS geometry to TELEMAC simulation files.

    Args:
        hecras_path: Path to .g##.hdf file.
        dem_path: Path to GeoTIFF DEM (required for 1D).
        output_dir: Output directory for .slf/.cli/.cas files.
        name: Project name for output files.
        floodplain_width: Buffer distance each side of river (m).
        backend: Meshing backend ("triangle" or "gmsh").
        duration: Simulation duration (seconds).
        cas_overrides: Additional .cas keyword overrides.
    """
    from telemac_tools.hecras import parse_hecras
    from telemac_tools.domain import build_domain_1d, build_domain_2d
    from telemac_tools.meshing import generate_mesh
    from telemac_tools.telemac import write_telemac
    from telemac_tools.hecras.parser_2d import triangulate_2d_area

    model = parse_hecras(hecras_path)

    if model.areas_2d:
        # 2D workflow: direct mesh conversion
        mesh = triangulate_2d_area(model.areas_2d[0])
        domain = build_domain_2d(model)
    elif model.rivers:
        # 1D workflow: DEM + channel carving + meshing
        if dem_path is None:
            raise ValueError("DEM path required for 1D→2D conversion")
        domain = build_domain_1d(model, dem_path, floodplain_width=floodplain_width)
        mesh = generate_mesh(domain, backend=backend)
        # Assign BC nodes after meshing
        from telemac_tools.domain.builder import assign_bc_nodes
        assign_bc_nodes(mesh, domain)
    else:
        raise ValueError("No 1D or 2D geometry found in HEC-RAS file")

    write_telemac(mesh, domain, output_dir, name=name,
                  duration=duration, cas_overrides=cas_overrides)
```

- [ ] **Step 3: Run all tests**

Run: `cd /home/razinka/telemac/telemac-viewer && /opt/micromamba/envs/shiny/bin/python3.13 -m pytest telemac_tools/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add telemac_tools/
git commit -m "feat(telemac_tools): add hecras_to_telemac convenience function and integration tests"
```

---

## Summary

| Task | Component | Tests | Lines (est.) |
|------|-----------|-------|--------------|
| 1 | Package scaffold + data model | 6 | ~120 |
| 2 | Synthetic HDF5 fixtures | — | ~120 |
| 3 | 1D parser | 11 | ~180 |
| 4 | 2D parser + Voronoi→triangles | 10 | ~120 |
| 5 | Manning's extraction + unified API | 2 | ~40 |
| 6 | Channel carving | 4 | ~80 |
| 7 | Domain builder (1D + 2D) | 6 | ~120 |
| 8 | Triangle mesh generator | 6 | ~100 |
| 9 | .slf/.cli/.cas writers | 8 | ~150 |
| 10 | Convenience function + integration | 5 | ~60 |
| **Total** | | **~58 tests** | **~1090 lines** |
