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

Fix all 5 tests by decoding the base64 binary:

```python
import base64

def _decode_binary(d):
    """Decode shiny-deckgl binary-encoded attribute to numpy array."""
    raw = base64.b64decode(d["value"])
    return np.frombuffer(raw, dtype=d["dtype"])

def test_indices_length(self, fake_tf):
    geom = build_mesh_geometry(fake_tf)
    idx = _decode_binary(geom["indices"])
    assert len(idx) == 2 * 3

def test_positions_are_centered(self, fake_tf):
    geom = build_mesh_geometry(fake_tf)
    pos = _decode_binary(geom["positions"])
    # Node 0: (0-0.5, 0-0.5, 0) = (-0.5, -0.5, 0)
    assert pos[0] == pytest.approx(-0.5, abs=0.01)
    assert pos[1] == pytest.approx(-0.5, abs=0.01)
    assert pos[2] == pytest.approx(0.0, abs=0.01)
```

3D tests — same pattern:
```python
def test_z_values_applied(self, fake_tf):
    z = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    geom = build_mesh_geometry(fake_tf, z_values=z, z_scale=10)
    pos = _decode_binary(geom["positions"])
    assert pos[2] == pytest.approx(0.0)      # node0 z
    assert pos[11] == pytest.approx(30.0)     # node3 z = 3*10

def test_z_none_is_flat(self, fake_tf):
    geom = build_mesh_geometry(fake_tf, z_values=None)
    pos = _decode_binary(geom["positions"])
    z_vals = [pos[i * 3 + 2] for i in range(4)]
    assert all(z == 0.0 for z in z_vals)
```

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
- Modify: `app.py` — add MapWidget to Import tab, build preview layers from HecRasModel

- [ ] **Step 1: Add MapWidget to Import tab UI**

In the Import tab's right panel card (currently shows "Preview" header + text log), replace the card content. The card should contain both a MapWidget and the text log below it.

Find the Import tab's right-panel card in `app.py` (around line 690):
```python
ui.card(
    ui.card_header("Preview"),
    ui.output_ui("import_preview_info"),
    ui.tags.pre(
        ui.output_text_verbatim("import_log"),
        class_="sim-console",
    ),
    class_="ocean-card",
),
```

Replace with:
```python
ui.card(
    ui.card_header("Preview"),
    output_widget("import_map", height="400px"),
    ui.tags.pre(
        ui.output_text_verbatim("import_log"),
        class_="sim-console",
        style="max-height: 200px; overflow-y: auto;",
    ),
    class_="ocean-card",
),
```

- [ ] **Step 2: Initialize import MapWidget in server**

At the start of the Import server section (after `import_log_text = reactive.value("")`), add:

```python
import_map_widget = MapWidget(
    view_state={"longitude": 0, "latitude": 0, "zoom": 0},
    style="data:application/json;charset=utf-8,%7B%22version%22%3A8%2C%22sources%22%3A%7B%7D%2C%22layers%22%3A%5B%7B%22id%22%3A%22bg%22%2C%22type%22%3A%22background%22%2C%22paint%22%3A%7B%22background-color%22%3A%22%230f1923%22%7D%7D%5D%7D",
)

@render_widget
def import_map():
    return import_map_widget
```

- [ ] **Step 3: Create `_build_import_preview_layers` helper**

Add this function in the Import server section (before the handlers):

```python
_COORD_METER_OFFSETS = 2  # deck.gl coordinate system constant

def _build_import_preview_layers(model, x_off, y_off):
    """Build deck.gl layers from parsed HecRasModel for preview."""
    layers = []

    # River alignment (cyan path)
    for reach in model.rivers:
        path = [[float(p[0] - x_off), float(p[1] - y_off)] for p in reach.alignment]
        layers.append(path_layer(
            f"alignment-{reach.name}",
            [{"path": path}],
            getPath="@@=d.path",
            getColor=[0, 200, 255, 200],
            getWidth=4,
            widthMinPixels=2,
            widthMaxPixels=6,
            pickable=False,
            coordinateSystem=_COORD_METER_OFFSETS,
            coordinateOrigin=[0, 0],
        ))

        # Cross-section lines (yellow)
        xs_lines = []
        for xs in reach.cross_sections:
            if xs.coords.shape[0] >= 2:
                xs_lines.append({
                    "sourcePosition": [float(xs.coords[0, 0] - x_off), float(xs.coords[0, 1] - y_off)],
                    "targetPosition": [float(xs.coords[-1, 0] - x_off), float(xs.coords[-1, 1] - y_off)],
                })
        if xs_lines:
            layers.append(line_layer(
                "cross-sections",
                xs_lines,
                getColor=[255, 220, 0, 180],
                getWidth=2,
                widthMinPixels=1,
                widthMaxPixels=3,
                pickable=False,
                coordinateSystem=_COORD_METER_OFFSETS,
                coordinateOrigin=[0, 0],
            ))

    # BC lines (blue=inflow, green=outflow)
    for i, bc in enumerate(model.boundaries):
        if bc.line_coords is not None and len(bc.line_coords) >= 2:
            color = [40, 120, 255, 220] if bc.location == "upstream" else [0, 200, 80, 220]
            layers.append(line_layer(
                f"bc-{i}",
                [{"sourcePosition": [float(bc.line_coords[0, 0] - x_off), float(bc.line_coords[0, 1] - y_off)],
                  "targetPosition": [float(bc.line_coords[-1, 0] - x_off), float(bc.line_coords[-1, 1] - y_off)]}],
                getColor=color,
                getWidth=4,
                widthMinPixels=2,
                widthMaxPixels=5,
                pickable=False,
                coordinateSystem=_COORD_METER_OFFSETS,
                coordinateOrigin=[0, 0],
            ))

    # 2D area face points (scatter)
    for area in model.areas_2d:
        pts = [{"position": [float(p[0] - x_off), float(p[1] - y_off)]} for p in area.face_points[::max(1, len(area.face_points)//2000)]]
        layers.append(scatterplot_layer(
            f"2d-{area.name}",
            pts,
            getPosition="@@=d.position",
            getColor=[0, 200, 255, 120],
            getRadius=3,
            radiusMinPixels=1,
            radiusMaxPixels=4,
            pickable=False,
            coordinateSystem=_COORD_METER_OFFSETS,
            coordinateOrigin=[0, 0],
        ))

    return layers
```

- [ ] **Step 4: Wire into `handle_import_preview`**

After `import_model.set(model)` and the log messages, add:

```python
# Compute center for METER_OFFSETS
all_x, all_y = [], []
for r in model.rivers:
    all_x.extend(r.alignment[:, 0].tolist())
    all_y.extend(r.alignment[:, 1].tolist())
for a in model.areas_2d:
    all_x.extend(a.face_points[:, 0].tolist())
    all_y.extend(a.face_points[:, 1].tolist())
if all_x:
    x_off = (min(all_x) + max(all_x)) / 2
    y_off = (min(all_y) + max(all_y)) / 2
    extent = max(max(all_x) - min(all_x), max(all_y) - min(all_y), 1.0)
    import math
    zoom = math.log2(600 * 360 / (256 * (extent / 111320))) if extent > 0 else 10

    preview_layers = _build_import_preview_layers(model, x_off, y_off)
    import asyncio
    asyncio.ensure_future(import_map_widget.update(
        session,
        layers=preview_layers,
        view_state={"longitude": 0, "latitude": 0, "zoom": zoom},
    ))
```

- [ ] **Step 5: Verify imports are available**

The Import tab server section needs `path_layer`, `line_layer`, `scatterplot_layer` — these are already imported at the top of app.py from `shiny_deckgl`. Verify they're in scope.

- [ ] **Step 6: Run app, switch to Import tab, upload a file, click Preview — verify map renders**

- [ ] **Step 7: Commit**

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
