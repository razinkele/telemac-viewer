# Remaining Work — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete 5 remaining items: Gmsh mesh backend, real HEC-RAS fixture testing, BC time series import, basemap tile verification, and parametrized layer origin tests.

**Architecture:** Five independent tasks touching different subsystems. No dependencies between them — can be executed in any order or in parallel.

**Tech Stack:** Python 3.12+, gmsh 4.13.1, h5py, shiny-deckgl, triangle, pyproj.

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `telemac_tools/meshing/gmsh_mesh.py` | Gmsh mesh generation backend | Create |
| `telemac_tools/tests/test_gmsh_mesh.py` | Tests for Gmsh backend | Create |
| `telemac_tools/meshing/__init__.py` | Wire Gmsh backend into `generate_mesh` | Modify |
| `telemac_tools/hecras/parser_bc.py` | Parse BC time series from `.u##` unsteady flow files | Create |
| `telemac_tools/tests/test_parser_bc.py` | Tests for BC time series parser | Create |
| `telemac_tools/tests/conftest.py` | Add `hdf_1d_with_bc` fixture | Modify |
| `telemac_tools/model.py` | Add `timeseries` field to BoundaryCondition | Modify |
| `telemac_tools/telemac/writer_cas.py` | Add liquid boundary file reference when BC has timeseries | Modify |
| `tests/test_layers.py` | Parametrize origin tests across all 10 builders | Modify |
| `tests/test_crs.py` | Add basemap verification test (manual) | — |

---

### Task 1: Gmsh mesh generation backend

**Files:**
- Create: `telemac_tools/meshing/gmsh_mesh.py`
- Create: `telemac_tools/tests/test_gmsh_mesh.py`
- Modify: `telemac_tools/meshing/__init__.py`

- [ ] **Step 1: Write failing tests**

```python
# telemac_tools/tests/test_gmsh_mesh.py
"""Tests for Gmsh mesh generation backend."""
from __future__ import annotations
import numpy as np
import pytest
from telemac_tools.meshing import generate_mesh
from telemac_tools.model import TelemacDomain, Mesh2D, BCSegment


def _simple_domain():
    poly = np.array([
        [0, 0], [100, 0], [100, 100], [0, 100], [0, 0],
    ], dtype=np.float64)
    return TelemacDomain(
        boundary_polygon=poly,
        bc_segments=[BCSegment(node_indices=[], lihbor=2)],
    )


class TestGmshBackend:
    def test_returns_mesh2d(self):
        mesh = generate_mesh(_simple_domain(), backend="gmsh")
        assert isinstance(mesh, Mesh2D)

    def test_has_triangles(self):
        mesh = generate_mesh(_simple_domain(), backend="gmsh")
        assert mesh.elements.shape[1] == 3
        assert mesh.elements.shape[0] > 0

    def test_nodes_inside_domain(self):
        mesh = generate_mesh(_simple_domain(), backend="gmsh")
        assert mesh.nodes[:, 0].min() >= -1
        assert mesh.nodes[:, 0].max() <= 101

    def test_elevation_assigned(self):
        mesh = generate_mesh(_simple_domain(), backend="gmsh")
        assert len(mesh.elevation) == mesh.nodes.shape[0]

    def test_mannings_assigned(self):
        mesh = generate_mesh(_simple_domain(), backend="gmsh")
        assert len(mesh.mannings_n) == mesh.nodes.shape[0]
        assert np.all(mesh.mannings_n > 0)

    def test_max_area_reduces_elements(self):
        coarse = generate_mesh(_simple_domain(), backend="gmsh", max_area=500)
        fine = generate_mesh(_simple_domain(), backend="gmsh", max_area=50)
        assert fine.elements.shape[0] > coarse.elements.shape[0]

    def test_with_channel_constraints(self):
        domain = _simple_domain()
        domain.channel_points = np.array([
            [50, 0, -1], [50, 50, -2], [50, 100, -3],
        ], dtype=np.float64)
        domain.channel_segments = np.array([[0, 1], [1, 2]], dtype=np.int32)
        mesh = generate_mesh(domain, backend="gmsh")
        assert mesh.elements.shape[0] > 0
        # Channel nodes should have negative elevation
        assert mesh.elevation.min() < 0
```

- [ ] **Step 2: Run tests — verify FAIL (NotImplementedError)**

Run: `cd /home/razinka/telemac/telemac-viewer && /opt/micromamba/envs/shiny/bin/python3.13 -m pytest telemac_tools/tests/test_gmsh_mesh.py -v`
Expected: FAIL — `NotImplementedError: Gmsh backend not yet implemented`

- [ ] **Step 3: Implement GmshBackend**

```python
# telemac_tools/meshing/gmsh_mesh.py
"""Gmsh-based mesh generation backend."""
from __future__ import annotations
import numpy as np
from telemac_tools.meshing.base import MeshBackend
from telemac_tools.model import Mesh2D, TelemacDomain


class GmshBackend(MeshBackend):
    """Mesh generator using the Gmsh library.

    Offers more control than Triangle: distance-based size fields,
    embedded curves for channel constraints, and advanced quality control.
    """

    def generate(self, domain: TelemacDomain, min_angle: float = 30.0,
                 max_area: float | None = None) -> Mesh2D:
        import gmsh

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)  # suppress output
        gmsh.model.add("hecras_import")

        poly = domain.boundary_polygon
        # Remove closing point if present
        if np.allclose(poly[0], poly[-1]):
            poly = poly[:-1]
        n = len(poly)

        # Create boundary points and lines
        point_tags = []
        for i, (x, y) in enumerate(poly):
            tag = gmsh.model.occ.addPoint(float(x), float(y), 0)
            point_tags.append(tag)

        line_tags = []
        for i in range(n):
            tag = gmsh.model.occ.addLine(point_tags[i], point_tags[(i + 1) % n])
            line_tags.append(tag)

        loop = gmsh.model.occ.addCurveLoop(line_tags)
        surface = gmsh.model.occ.addPlaneSurface([loop])

        # Embed channel constraint lines if available
        if domain.channel_points is not None and len(domain.channel_points) >= 2:
            ch_xy = domain.channel_points[:, :2]
            ch_pt_tags = []
            for x, y in ch_xy:
                tag = gmsh.model.occ.addPoint(float(x), float(y), 0)
                ch_pt_tags.append(tag)
            ch_line_tags = []
            for i in range(len(ch_pt_tags) - 1):
                tag = gmsh.model.occ.addLine(ch_pt_tags[i], ch_pt_tags[i + 1])
                ch_line_tags.append(tag)

            gmsh.model.occ.synchronize()
            # Embed points and lines in the surface
            gmsh.model.mesh.embed(0, ch_pt_tags, 2, surface)
            gmsh.model.mesh.embed(1, ch_line_tags, 2, surface)
        else:
            gmsh.model.occ.synchronize()

        # Mesh size
        if max_area is not None:
            # Approximate element size from max area (equilateral triangle)
            target_size = np.sqrt(4 * max_area / np.sqrt(3))
        else:
            extent = max(poly[:, 0].max() - poly[:, 0].min(),
                         poly[:, 1].max() - poly[:, 1].min(), 1.0)
            target_size = extent / 20  # reasonable default

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", target_size * 0.3)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", target_size)
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay

        # Generate 2D mesh
        gmsh.model.mesh.generate(2)

        # Extract nodes and elements
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes(dim=2)
        coords = node_coords.reshape(-1, 3)[:, :2]

        # Build tag-to-index map
        tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
        triangles = []
        for etype, etags, enodes in zip(elem_types, elem_tags, elem_node_tags):
            if etype == 2:  # 3-node triangle
                enodes = enodes.reshape(-1, 3)
                for tri in enodes:
                    triangles.append([tag_to_idx[int(t)] for t in tri])

        gmsh.finalize()

        nodes = coords.astype(np.float64)
        elements = np.array(triangles, dtype=np.int32)

        # Elevation: sample DEM if available, else zero
        if domain._dem_data is not None and domain._dem_transform is not None:
            from telemac_tools.domain.builder import sample_dem
            elevation = sample_dem(nodes[:, 0], nodes[:, 1],
                                   domain._dem_data, domain._dem_transform)
        else:
            elevation = np.zeros(len(nodes))

        # Override channel node elevations
        if domain.channel_points is not None and len(domain.channel_points) > 0:
            ch_pts = domain.channel_points
            for i in range(len(nodes)):
                dists = np.sqrt(((ch_pts[:, :2] - nodes[i]) ** 2).sum(axis=1))
                if dists.min() < target_size * 0.5:
                    elevation[i] = ch_pts[np.argmin(dists), 2]

        mannings = np.full(len(nodes), 0.035)

        # Apply Manning's regions
        if domain.mannings_regions:
            from telemac_tools.meshing.triangle_mesh import _point_in_polygon
            for region in domain.mannings_regions:
                if "polygon" not in region:
                    continue
                rpoly = region["polygon"]
                n_val = region["n"]
                for i in range(len(nodes)):
                    if _point_in_polygon(nodes[i, 0], nodes[i, 1], rpoly):
                        mannings[i] = n_val

        return Mesh2D(
            nodes=nodes,
            elements=elements,
            elevation=elevation,
            mannings_n=mannings,
        )
```

- [ ] **Step 4: Update `telemac_tools/meshing/__init__.py`**

Replace the `elif backend == "gmsh"` block:
```python
    elif backend == "gmsh":
        from telemac_tools.meshing.gmsh_mesh import GmshBackend
        return GmshBackend().generate(domain, min_angle, max_area)
```

- [ ] **Step 5: Check if triangle_mesh.py has `_point_in_polygon` — if not, extract it**

Read `telemac_tools/meshing/triangle_mesh.py` and check if `_point_in_polygon` exists. If not, add a simple ray-casting implementation:

```python
def _point_in_polygon(px, py, polygon):
    """Ray-casting point-in-polygon test."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside
```

- [ ] **Step 6: Run tests — verify all PASS**

Run: `cd /home/razinka/telemac/telemac-viewer && /opt/micromamba/envs/shiny/bin/python3.13 -m pytest telemac_tools/tests/test_gmsh_mesh.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add telemac_tools/meshing/ telemac_tools/tests/test_gmsh_mesh.py && git commit -m "feat(meshing): add Gmsh mesh generation backend"
```

---

### Task 2: Real HEC-RAS fixture testing

**Files:**
- Modify: `telemac_tools/tests/test_real_hecras.py` (already exists with auto-skip)

The test harness already exists and auto-skips when no `.hdf` files are in `telemac_tools/tests/fixtures/`. This task attempts to download a real fixture via `rashdf`'s GitHub, or if that fails, creates a more realistic synthetic fixture that exercises edge cases the simple fixture doesn't.

- [ ] **Step 1: Try to clone rashdf test data with git LFS**

```bash
cd /tmp && git clone --depth 1 https://github.com/fema-ffrd/rashdf.git rashdf_repo 2>&1 | tail -5
ls -lh /tmp/rashdf_repo/tests/data/*.hdf 2>/dev/null | head -5
```

If files exist and are real HDF5 (not LFS pointers):
```bash
cp /tmp/rashdf_repo/tests/data/*.hdf /home/razinka/telemac/telemac-viewer/telemac_tools/tests/fixtures/
```

If they're LFS pointers (small files, ~130 bytes):
```bash
cd /tmp/rashdf_repo && git lfs pull 2>&1
```

- [ ] **Step 2: If real fixtures available, run tests**

Run: `cd /home/razinka/telemac/telemac-viewer && /opt/micromamba/envs/shiny/bin/python3.13 -m pytest telemac_tools/tests/test_real_hecras.py -v`

If tests fail due to parser assumptions about HDF5 structure, fix the parser to handle real-world HDF5 paths. Common differences from our synthetic fixtures:
- Dataset names may vary (e.g., `"Cells Minimum Elevation"` vs `"Cell Elevation Info"`)
- Compound dataset field names may differ
- Multiple reaches or 2D areas

- [ ] **Step 3: If no real fixtures obtainable, create a richer synthetic fixture**

Add to `telemac_tools/tests/conftest.py` a `hdf_1d_complex` fixture with:
- 2 reaches (not 1)
- 5 cross-sections (not 3)
- Non-perpendicular cross-sections (angled)
- Missing Manning's data on one XS
- A 2D area alongside the 1D data

- [ ] **Step 4: Add parser test for the complex fixture**

- [ ] **Step 5: Commit**

```bash
git add telemac_tools/tests/ && git commit -m "test: add real or complex HEC-RAS fixture testing"
```

---

### Task 3: BC time series import from `.u##` unsteady flow files

**Files:**
- Create: `telemac_tools/hecras/parser_bc.py`
- Create: `telemac_tools/tests/test_parser_bc.py`
- Modify: `telemac_tools/model.py` — add `timeseries` field
- Modify: `telemac_tools/tests/conftest.py` — add fixture
- Modify: `telemac_tools/telemac/writer_cas.py` — reference liquid boundary file

**Background:** HEC-RAS stores BC time series in `.u##` unsteady flow files (also HDF5). The structure is:

```
Unsteady Flow Data/
  Boundary Conditions/
    <BC Name>/
      DSS Data                    → may reference external DSS
      Flow Hydrograph/
        Flow                      → float array
        Time                      → float array (hours since reference)
      Stage Hydrograph/
        Stage                     → float array
        Time                      → float array
```

- [ ] **Step 1: Write failing tests**

```python
# telemac_tools/tests/test_parser_bc.py
"""Tests for BC time series parsing from HEC-RAS unsteady flow files."""
from __future__ import annotations
import numpy as np
import pytest
from telemac_tools.hecras.parser_bc import parse_bc_timeseries
from telemac_tools.model import BoundaryCondition


class TestParseBcTimeseries:
    def test_returns_list(self, hdf_unsteady):
        bcs = parse_bc_timeseries(hdf_unsteady)
        assert isinstance(bcs, list)

    def test_flow_hydrograph(self, hdf_unsteady):
        bcs = parse_bc_timeseries(hdf_unsteady)
        upstream = [bc for bc in bcs if bc.location == "upstream"][0]
        assert upstream.timeseries is not None
        assert "time" in upstream.timeseries
        assert "values" in upstream.timeseries
        assert len(upstream.timeseries["time"]) == len(upstream.timeseries["values"])

    def test_stage_hydrograph(self, hdf_unsteady):
        bcs = parse_bc_timeseries(hdf_unsteady)
        downstream = [bc for bc in bcs if bc.location == "downstream"][0]
        assert downstream.timeseries is not None

    def test_no_unsteady_returns_empty(self, tmp_path):
        import h5py
        path = tmp_path / "no_bc.hdf"
        with h5py.File(path, "w") as f:
            f.create_group("Geometry")
        bcs = parse_bc_timeseries(str(path))
        assert bcs == []
```

- [ ] **Step 2: Add `hdf_unsteady` fixture to conftest.py**

```python
@pytest.fixture
def hdf_unsteady(tmp_path):
    """Create a synthetic HEC-RAS unsteady flow HDF5 file with BC time series."""
    path = tmp_path / "test.u01.hdf"
    with h5py.File(path, "w") as f:
        ufd = f.create_group("Unsteady Flow Data")
        bcs = ufd.create_group("Boundary Conditions")

        # Upstream flow hydrograph
        us = bcs.create_group("Upstream")
        fh = us.create_group("Flow Hydrograph")
        fh.create_dataset("Flow", data=np.array([10.0, 50.0, 100.0, 50.0, 10.0]))
        fh.create_dataset("Time", data=np.array([0.0, 1.0, 2.0, 3.0, 4.0]))  # hours

        # Downstream stage hydrograph
        ds = bcs.create_group("Downstream")
        sh = ds.create_group("Stage Hydrograph")
        sh.create_dataset("Stage", data=np.array([2.0, 2.5, 3.0, 2.5, 2.0]))
        sh.create_dataset("Time", data=np.array([0.0, 1.0, 2.0, 3.0, 4.0]))

    return str(path)
```

- [ ] **Step 3: Add `timeseries` field to BoundaryCondition in model.py**

```python
@dataclass
class BoundaryCondition:
    bc_type: str
    location: str
    line_coords: np.ndarray | None = None
    timeseries: dict | None = None  # {"time": array, "values": array, "unit": str}
```

- [ ] **Step 4: Implement parser_bc.py**

```python
# telemac_tools/hecras/parser_bc.py
"""Parse BC time series from HEC-RAS unsteady flow files (.u##.hdf)."""
from __future__ import annotations
import h5py
import numpy as np
from telemac_tools.model import BoundaryCondition


def parse_bc_timeseries(path: str) -> list[BoundaryCondition]:
    """Parse boundary condition time series from HEC-RAS unsteady flow HDF5.

    Args:
        path: Path to .u##.hdf file.

    Returns:
        List of BoundaryCondition with timeseries populated.
    """
    bcs = []
    with h5py.File(path, "r") as f:
        bc_grp = f.get("Unsteady Flow Data/Boundary Conditions")
        if bc_grp is None:
            return []

        for name in bc_grp:
            if not isinstance(bc_grp[name], h5py.Group):
                continue
            grp = bc_grp[name]

            # Determine BC type and extract timeseries
            ts = None
            bc_type = "unknown"

            if "Flow Hydrograph" in grp:
                fh = grp["Flow Hydrograph"]
                if "Flow" in fh and "Time" in fh:
                    ts = {
                        "time": fh["Time"][:] * 3600,  # hours → seconds
                        "values": fh["Flow"][:],
                        "unit": "m3/s",
                    }
                    bc_type = "flow"

            elif "Stage Hydrograph" in grp:
                sh = grp["Stage Hydrograph"]
                if "Stage" in sh and "Time" in sh:
                    ts = {
                        "time": sh["Time"][:] * 3600,
                        "values": sh["Stage"][:],
                        "unit": "m",
                    }
                    bc_type = "stage"

            location = "upstream" if "upstream" in name.lower() else "downstream"
            bcs.append(BoundaryCondition(
                bc_type=bc_type,
                location=location,
                timeseries=ts,
            ))

    return bcs
```

- [ ] **Step 5: Run tests — verify all PASS**

Run: `cd /home/razinka/telemac/telemac-viewer && /opt/micromamba/envs/shiny/bin/python3.13 -m pytest telemac_tools/tests/test_parser_bc.py -v`

- [ ] **Step 6: Update writer_cas.py — add liquid boundary file reference**

When BC timeseries are present, the .cas file should reference a TELEMAC liquid boundary file. Add to the template:

```python
# In write_cas, after writing the base template:
if any(seg.prescribed_h is not None for seg in domain.bc_segments if hasattr(seg, 'timeseries')):
    text += "/\n/ Boundary time series (edit liquid boundary file manually)\n"
    text += f"LIQUID BOUNDARIES FILE = {name}.liq\n"
```

This is a placeholder — actual `.liq` file writing is a follow-up task.

- [ ] **Step 7: Commit**

```bash
git add telemac_tools/ && git commit -m "feat(hecras): add BC time series parser from unsteady flow files"
```

---

### Task 4: Parametrize layer origin tests

**Files:**
- Modify: `tests/test_layers.py`

- [ ] **Step 1: Replace TestOriginParameter class with parametrized tests**

```python
# Replace the existing TestOriginParameter class in tests/test_layers.py

class TestOriginParameter:
    """Verify all layer builders pass origin through to coordinateOrigin."""

    def test_mesh_layer(self, fake_geom):
        values = np.array([0.1, 0.5, 0.5, 1.0])
        lyr, _, _, _ = build_mesh_layer(fake_geom, values, "Viridis", origin=[24.0, 55.0])
        assert lyr["coordinateOrigin"] == [24.0, 55.0]

    def test_velocity_layer(self, fake_tf, fake_geom):
        lyr = build_velocity_layer(fake_tf, 0, fake_geom, origin=[24.0, 55.0])
        if lyr is not None:
            assert lyr["coordinateOrigin"] == [24.0, 55.0]

    def test_contour_layer(self, fake_tf, fake_geom):
        values = fake_tf.get_data_value("WATER DEPTH", 0)
        lyr = build_contour_layer_fn(fake_tf, values, fake_geom, origin=[24.0, 55.0])
        if lyr is not None:
            assert lyr["coordinateOrigin"] == [24.0, 55.0]

    def test_marker_layer(self):
        lyr = build_marker_layer(0.0, 0.0, origin=[24.0, 55.0])
        assert lyr["coordinateOrigin"] == [24.0, 55.0]

    def test_cross_section_layer(self):
        lyr = build_cross_section_layer([[0, 0], [1, 1]], origin=[24.0, 55.0])
        assert lyr["coordinateOrigin"] == [24.0, 55.0]

    def test_particle_layer(self):
        paths = [[[0, 0, 0], [1, 1, 1]]]
        lyr = build_particle_layer(paths, 0.5, 1.0, origin=[24.0, 55.0])
        assert lyr["coordinateOrigin"] == [24.0, 55.0]

    def test_wireframe_layer(self, fake_tf, fake_geom):
        lyr = build_wireframe_layer(fake_tf, fake_geom, origin=[24.0, 55.0])
        assert lyr["coordinateOrigin"] == [24.0, 55.0]

    def test_extrema_markers(self, fake_tf, fake_geom):
        from analysis import find_extrema
        values = fake_tf.get_data_value("WATER DEPTH", 0)
        extrema = find_extrema(fake_tf, values)
        layers = build_extrema_markers(extrema, fake_geom["x_off"], fake_geom["y_off"], origin=[24.0, 55.0])
        for lyr in layers:
            assert lyr["coordinateOrigin"] == [24.0, 55.0]

    def test_measurement_layer(self):
        pts = [[0, 0], [1, 1]]
        layers = build_measurement_layer(pts, origin=[24.0, 55.0])
        for lyr in layers:
            assert lyr["coordinateOrigin"] == [24.0, 55.0]

    def test_boundary_layer(self, fake_tf, fake_geom):
        from analysis import find_boundary_nodes
        bnodes = find_boundary_nodes(fake_tf)
        layers = build_boundary_layer(fake_tf, fake_geom, bnodes, origin=[24.0, 55.0])
        for lyr in layers:
            assert lyr["coordinateOrigin"] == [24.0, 55.0]

    def test_default_origin_is_zero(self, fake_geom):
        values = np.array([0.1, 0.5, 0.5, 1.0])
        lyr, _, _, _ = build_mesh_layer(fake_geom, values, "Viridis")
        assert lyr["coordinateOrigin"] == [0, 0]
```

- [ ] **Step 2: Run tests — verify all PASS**

Run: `cd /home/razinka/telemac/telemac-viewer && /opt/micromamba/envs/shiny/bin/python3.13 -m pytest tests/test_layers.py::TestOriginParameter -v`
Expected: 11 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_layers.py && git commit -m "test(layers): parametrize origin tests across all 10 layer builders"
```

---

### Task 5: Basemap tile verification

This is a manual/semi-automated verification task. Basemap tiles require internet access to the tile server. The test confirms that when CRS + offset are set, the view_state has real WGS84 coordinates.

**Files:**
- No new files — this is a verification script

- [ ] **Step 1: Write a standalone verification script**

```bash
cd /home/razinka/telemac/telemac-viewer && /opt/micromamba/envs/shiny/bin/python3.13 << 'PYEOF'
"""Verify CRS produces correct WGS84 coordinates for basemap alignment."""
from geometry import build_mesh_geometry
from crs import crs_from_epsg
from data_manip.extraction.telemac_file import TelemacFile
from constants import EXAMPLES

# Test 1: Curonian Lagoon with LKS94 offset
tf = TelemacFile(EXAMPLES["Curonian Lagoon (24h)"])
crs = crs_from_epsg(3346)
geom = build_mesh_geometry(tf, crs=crs, origin_offset=(309424, 6132619))
print(f"Curonian Lagoon with CRS+offset:")
print(f"  lon_off={geom['lon_off']:.4f}, lat_off={geom['lat_off']:.4f}")
assert 20.0 < geom["lon_off"] < 22.0, f"lon_off={geom['lon_off']} out of range"
assert 54.5 < geom["lat_off"] < 56.0, f"lat_off={geom['lat_off']} out of range"
print(f"  PASS — coordinates are in Lithuania")
tf.close()

# Test 2: Gouttedo with no CRS
tf = TelemacFile(EXAMPLES["Gouttedo (raindrop)"])
geom = build_mesh_geometry(tf)
print(f"\nGouttedo without CRS:")
print(f"  lon_off={geom['lon_off']}, lat_off={geom['lat_off']}")
assert geom["lon_off"] == 0.0
assert geom["lat_off"] == 0.0
print(f"  PASS — centered at (0,0)")
tf.close()

print("\n=== BASEMAP ALIGNMENT VERIFICATION PASSED ===")
PYEOF
```

- [ ] **Step 2: If basemap tiles load (internet available), start the app and visually verify**

```bash
cd /home/razinka/telemac/telemac-viewer && /opt/micromamba/envs/shiny/bin/shiny run app.py --port 8765
```

Open browser, select Curonian Lagoon, set EPSG 3346, set offset (309424, 6132619), switch basemap to "CartoDB Dark" or "Satellite" — verify mesh overlays the Curonian Lagoon in Lithuania.

- [ ] **Step 3: Document result**

No commit needed — this is a verification step. If tiles don't load (no internet), note it as a known environment limitation.

---

## Summary

| Task | What | Tests | Effort |
|------|------|-------|--------|
| 1 | Gmsh mesh backend | 7 | Medium |
| 2 | Real HEC-RAS fixtures | varies | Small-Medium |
| 3 | BC time series parser | 4 | Medium |
| 4 | Layer origin tests | 11 | Small |
| 5 | Basemap verification | manual | Small |
