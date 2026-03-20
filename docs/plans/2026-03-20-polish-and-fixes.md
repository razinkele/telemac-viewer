# Polish & Fixes Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all broken tests, enable Curonian Lagoon basemap alignment, add Import tab preview layers, and prepare for real HEC-RAS testing.

**Architecture:** Four independent tasks touching different areas. Can be done in any order.

**Tech Stack:** Python Shiny, shiny-deckgl, pyproj, h5py, triangle.

---

## File Structure

| File | Action | Task |
|------|--------|------|
| `tests/test_geometry.py` | Fix 5 tests — update for binary-encoded positions | 1 |
| `tests/test_layers.py` | Fix 1 test — boundary layer returns list, not dict | 1 |
| `tests/test_integration.py` | Fix 1 test — positions are binary-encoded | 1 |
| `crs.py` | Add `crs_with_offset()` for pre-centered meshes | 2 |
| `app.py` | Add manual X/Y offset inputs in CRS UI, wire into geometry | 2 |
| `tests/test_crs.py` | Add offset transform tests | 2 |
| `app.py` | Add Import preview map with alignment/XS/boundary layers | 3 |
| `telemac_tools/tests/test_real_hecras.py` | Fixture-based test for real HEC-RAS files | 4 |

---

### Task 1: Fix 7 pre-existing test failures

**Root causes:**
- 5 geometry tests: `encode_binary_attribute()` returns a dict `{"@@binary": True, "dtype": ..., "value": ...}`, not a flat array. Tests index into it as if it were an array.
- 1 boundary test: `build_boundary_layer()` returns `list[dict]`, test asserts `isinstance(result, dict)`.
- 1 integration test: `len(geom["positions"])` counts dict keys (4), not array elements.

**Files:**
- Modify: `tests/test_geometry.py`
- Modify: `tests/test_layers.py`
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Fix geometry tests**

The positions/indices are binary-encoded dicts. Tests that need raw values should decode them or test the dict structure instead. The simplest fix: test the dict has the right keys (`@@binary`, `dtype`, `value`) and verify the base64-encoded size.

For `test_positions_length`: positions is a binary dict with `value` containing base64-encoded float32 data. 4 nodes × 3 coords × 4 bytes = 48 bytes → 64 base64 chars.

```python
# tests/test_geometry.py — fix test_positions_length
def test_positions_length(self, fake_tf):
    geom = build_mesh_geometry(fake_tf)
    pos = geom["positions"]
    assert pos["@@binary"] is True
    assert pos["dtype"] == "float32"
    # 4 nodes × 3 coords × 4 bytes = 48 bytes → 64 base64 chars
    import base64
    raw = base64.b64decode(pos["value"])
    assert len(raw) == 4 * 3 * 4
```

Similarly fix `test_indices_length`, `test_positions_are_centered`, `test_z_values_applied`, `test_z_none_is_flat`.

For `test_positions_are_centered` and 3D tests: decode the base64, interpret as float32 numpy array, then check values.

- [ ] **Step 2: Fix boundary layer test**

```python
# tests/test_layers.py — fix test_boundary
def test_boundary(self, fake_tf, fake_geom):
    bnodes = find_boundary_nodes(fake_tf)
    result = build_boundary_layer(fake_tf, fake_geom, bnodes)
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(r, dict) for r in result)
```

- [ ] **Step 3: Fix integration geometry test**

```python
# tests/test_integration.py — fix test_geometry
def test_geometry(tf, geom):
    assert geom["npoin"] == tf.npoin2
    assert geom["positions"]["@@binary"] is True
    import base64
    raw = base64.b64decode(geom["positions"]["value"])
    assert len(raw) == tf.npoin2 * 3 * 4  # float32
    assert geom["zoom"] > 0
```

- [ ] **Step 4: Run all tests**

Run: `cd /home/razinka/telemac/telemac-viewer && /opt/micromamba/envs/shiny/bin/python3.13 -m pytest tests/ telemac_tools/ -v`
Expected: 0 failures

- [ ] **Step 5: Commit**

```bash
git add tests/ && git commit -m "fix(tests): update geometry/layer/integration tests for binary-encoded positions"
```

---

### Task 2: Curonian Lagoon CRS alignment — manual origin offset

**Problem:** The Curonian Lagoon mesh has pre-centered coordinates (X: -31k to +18k, Y: -46k to +48k). The SELAFIN file doesn't store the absolute LKS94 origin. Without the offset, CRS transforms produce wrong lon/lat.

**Solution:** Add manual X/Y origin offset inputs to the CRS UI. When set, these offsets are added to the mesh center before the CRS transform. The Curonian Lagoon's offset is approximately **(309424, 6132619)** in LKS94.

**Files:**
- Modify: `app.py` — add offset inputs to CRS UI
- Modify: `geometry.py` — apply manual offset to x_off/y_off before CRS transform

- [ ] **Step 1: Add offset inputs to Data accordion (after CRS status)**

In `app.py`, after the `crs_status_ui` output, add:

```python
ui.tags.details(
    ui.tags.summary("CRS Origin Offset"),
    ui.input_numeric("crs_x_offset", "X offset (m)", value=0, step=1000),
    ui.input_numeric("crs_y_offset", "Y offset (m)", value=0, step=1000),
),
```

- [ ] **Step 2: Update `mesh_geom()` to pass offset**

In `mesh_geom()`, read the offset and pass to `build_mesh_geometry`:

```python
x_offset = input.crs_x_offset() or 0
y_offset = input.crs_y_offset() or 0
```

- [ ] **Step 3: Update `geometry.py` — add `origin_offset` parameter**

Add `origin_offset: tuple[float, float] = (0, 0)` to `build_mesh_geometry`. Apply it to `x_off`/`y_off` before the CRS transform:

```python
x_off += origin_offset[0]
y_off += origin_offset[1]
```

This shifts the mesh center to absolute CRS coordinates before converting to WGS84.

- [ ] **Step 4: Add test**

```python
# tests/test_crs.py
def test_offset_transform():
    """Manual offset should shift mesh center to absolute CRS coords."""
    from geometry import build_mesh_geometry
    from tests.helpers import FakeTF
    from crs import crs_from_epsg
    tf = FakeTF()
    crs = crs_from_epsg(3346)
    geom = build_mesh_geometry(tf, crs=crs, origin_offset=(309424, 6132619))
    # Mesh center (0.5, 0.5) + offset → (309424.5, 6132619.5) in LKS94
    # Should map to approximately 20.9°E, 55.3°N
    assert 20.0 < geom["lon_off"] < 22.0
    assert 54.5 < geom["lat_off"] < 56.0
```

- [ ] **Step 5: Run tests, commit**

```bash
git add app.py geometry.py tests/test_crs.py && git commit -m "feat(crs): add manual origin offset for pre-centered meshes"
```

---

### Task 3: Import tab preview map

**Problem:** The Import tab shows a text log but no map preview. The spec calls for river alignment, cross-sections, domain boundary, and BC segments rendered on a map.

**Files:**
- Modify: `app.py` — replace text preview with deck.gl map showing parsed geometry

- [ ] **Step 1: Replace preview_info output with a MapWidget**

Add a second `MapWidget` for the import preview. After parsing, build layers from the model:

- River alignment → `path_layer` (cyan `[0, 200, 255, 200]`)
- Cross-section lines → `line_layer` (yellow `[255, 220, 0, 180]`)
- Domain boundary → `path_layer` (white dashed `[255, 255, 255, 150]`)
- BC lines → `line_layer` (blue=inflow `[40, 120, 255]`, green=outflow `[0, 200, 80]`)

After converting, add:
- Generated mesh → wireframe `line_layer` (gray `[100, 100, 100, 80]`)

- [ ] **Step 2: Build preview layers from HecRasModel**

Create helper function `_build_import_preview_layers(model)` that converts model geometry to deck.gl layer dicts. All coordinates in METER_OFFSETS relative to the model's center.

- [ ] **Step 3: Wire into import handlers**

In `handle_import_preview`: after parsing, compute center, build layers, update preview map.
In `handle_import_convert`: after meshing, add wireframe layer.

- [ ] **Step 4: Test manually — verify preview renders**
- [ ] **Step 5: Commit**

```bash
git add app.py && git commit -m "feat(app): add preview map to Import tab with alignment/XS/boundary layers"
```

---

### Task 4: Real HEC-RAS file test harness

**Problem:** Can't download FEMA FFRD fixtures (Git LFS). Need a test harness that works when real files are provided manually.

**Files:**
- Create: `telemac_tools/tests/test_real_hecras.py`

- [ ] **Step 1: Create test that skips when no fixture**

```python
# telemac_tools/tests/test_real_hecras.py
"""Tests with real HEC-RAS files — skipped when fixtures not available.

To run: place .g##.hdf files in telemac_tools/tests/fixtures/
Download from: https://github.com/fema-ffrd/rashdf (requires git lfs)
"""
from __future__ import annotations
import os
import glob
import numpy as np
import pytest
from telemac_tools.hecras import parse_hecras
from telemac_tools.model import HecRasModel

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
HDF_FILES = glob.glob(os.path.join(FIXTURE_DIR, "*.hdf"))

pytestmark = pytest.mark.skipif(
    not HDF_FILES,
    reason="No real HEC-RAS fixtures in telemac_tools/tests/fixtures/",
)


@pytest.fixture(params=HDF_FILES, ids=lambda p: os.path.basename(p))
def hdf_path(request):
    return request.param


class TestRealHecRasFiles:
    def test_parses_without_error(self, hdf_path):
        model = parse_hecras(hdf_path)
        assert isinstance(model, HecRasModel)
        assert model.rivers or model.areas_2d

    def test_has_geometry(self, hdf_path):
        model = parse_hecras(hdf_path)
        if model.rivers:
            for r in model.rivers:
                assert r.alignment.shape[1] == 2
                for xs in r.cross_sections:
                    assert xs.coords.shape[1] == 3
        if model.areas_2d:
            for a in model.areas_2d:
                assert a.face_points.shape[1] == 2
                assert len(a.cells) > 0

    def test_full_pipeline(self, hdf_path, tmp_path):
        """Full conversion if 2D, or parse-only if 1D (no DEM)."""
        model = parse_hecras(hdf_path)
        if model.areas_2d:
            from telemac_tools import hecras_to_telemac
            out = str(tmp_path / "output")
            hecras_to_telemac(hdf_path, output_dir=out)
            assert os.path.isfile(os.path.join(out, "project.slf"))
```

- [ ] **Step 2: Create fixtures directory with README**

```bash
mkdir -p telemac_tools/tests/fixtures/
cat > telemac_tools/tests/fixtures/README.md << 'EOF'
# HEC-RAS Test Fixtures

Place real HEC-RAS geometry HDF5 files here (.g##.hdf).

Source: https://github.com/fema-ffrd/rashdf (requires git lfs clone)

Tests in test_real_hecras.py auto-discover all .hdf files in this directory.
When no files are present, tests are skipped.
EOF
```

- [ ] **Step 3: Run tests (should skip)**

Run: `cd /home/razinka/telemac/telemac-viewer && /opt/micromamba/envs/shiny/bin/python3.13 -m pytest telemac_tools/tests/test_real_hecras.py -v`
Expected: All skipped ("No real HEC-RAS fixtures")

- [ ] **Step 4: Commit**

```bash
git add telemac_tools/tests/test_real_hecras.py telemac_tools/tests/fixtures/ && git commit -m "test: add real HEC-RAS file test harness (auto-skip when no fixtures)"
```
