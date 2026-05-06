# CRS Awareness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add full coordinate reference system support so TELEMAC meshes render on real basemaps, coordinates display in both native CRS and WGS84, and exports include CRS metadata.

**Architecture:** New `crs.py` module handles all pyproj transforms. `geometry.py` converts mesh center to WGS84 for deck.gl origin. All layer `coordinateOrigin` values shift from `[0,0]` to `[lon_off, lat_off]`. Click coordinates convert back via pyproj inverse. All analysis stays in native CRS meters.

**Tech Stack:** pyproj 3.7.1, Python Shiny 1.5.1, shiny-deckgl 1.0.1, deck.gl METER_OFFSETS coordinate system.

**Spec:** `docs/specs/2026-03-20-crs-awareness-design.md`

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `crs.py` | CRS dataclass, pyproj transforms, .cas detection, coord heuristic, click conversion | Create |
| `tests/test_crs.py` | Unit tests for all crs.py functions | Create |
| `geometry.py` | Add `crs` param, compute `lon_off`/`lat_off`, check SELAFIN IPARAM offsets | Modify |
| `layers.py` | Add `origin` param to all 10 layer builders, pass through to `coordinateOrigin` | Modify |
| `analysis.py` | Remove `coord_to_meters` (moved to crs.py) | Modify |
| `app.py` | CRS UI, reactive, click handler rewiring, coordinate display, CSV export headers | Modify |
| `tests/test_geometry.py` | Add tests for CRS-aware geometry building | Modify |
| `tests/test_layers.py` | Add tests for origin parameter pass-through | Modify |

---

### Task 1: Create `crs.py` — core CRS module

**Files:**
- Create: `crs.py`
- Create: `tests/test_crs.py`

- [ ] **Step 1: Write failing tests for `crs_from_epsg`**

```python
# tests/test_crs.py
"""Tests for crs.py — coordinate reference system transforms."""
from __future__ import annotations
import numpy as np
import pytest
from crs import CRS, crs_from_epsg, native_to_wgs84, wgs84_to_native


class TestCrsFromEpsg:
    def test_lks94(self):
        crs = crs_from_epsg(3346)
        assert crs.epsg == 3346
        assert "LKS" in crs.name or "Lithuania" in crs.name

    def test_utm_north(self):
        crs = crs_from_epsg(32634)
        assert crs.epsg == 32634

    def test_wgs84(self):
        crs = crs_from_epsg(4326)
        assert crs.epsg == 4326

    def test_invalid_raises(self):
        with pytest.raises(Exception):
            crs_from_epsg(99999)


class TestTransforms:
    def test_lks94_to_wgs84(self):
        crs = crs_from_epsg(3346)
        lon, lat = native_to_wgs84(500000, 6100000, crs)
        assert 23.0 < lon < 25.0
        assert 54.5 < lat < 56.0

    def test_roundtrip(self):
        crs = crs_from_epsg(3346)
        x_orig, y_orig = 500000.0, 6100000.0
        lon, lat = native_to_wgs84(x_orig, y_orig, crs)
        x_back, y_back = wgs84_to_native(lon, lat, crs)
        assert x_back == pytest.approx(x_orig, abs=0.01)
        assert y_back == pytest.approx(y_orig, abs=0.01)

    def test_wgs84_identity(self):
        """WGS84 CRS should be near-identity transform."""
        crs = crs_from_epsg(4326)
        lon, lat = native_to_wgs84(24.0, 55.0, crs)
        assert lon == pytest.approx(24.0, abs=0.001)
        assert lat == pytest.approx(55.0, abs=0.001)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/razinka/telemac/telemac-viewer && python -m pytest tests/test_crs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'crs'`

- [ ] **Step 3: Implement `CRS` dataclass, `crs_from_epsg`, `native_to_wgs84`, `wgs84_to_native`**

```python
# crs.py
"""Coordinate Reference System transforms for TELEMAC Viewer.

Single source of truth for all CRS operations: EPSG lookup,
native↔WGS84 transforms, .cas auto-detection, click conversion.
"""
from __future__ import annotations
from dataclasses import dataclass
from pyproj import Transformer, CRS as ProjCRS
from constants import _M2D


@dataclass
class CRS:
    epsg: int
    name: str
    transformer: Transformer      # native → WGS84
    inv_transformer: Transformer   # WGS84 → native


def crs_from_epsg(code: int) -> CRS:
    """Create CRS from EPSG code. Raises pyproj.exceptions.CRSError for invalid codes."""
    proj_crs = ProjCRS.from_epsg(code)
    fwd = Transformer.from_crs(code, 4326, always_xy=True)
    inv = Transformer.from_crs(4326, code, always_xy=True)
    return CRS(epsg=code, name=proj_crs.name, transformer=fwd, inv_transformer=inv)


def native_to_wgs84(x: float, y: float, crs: CRS) -> tuple[float, float]:
    """Transform native CRS coordinates to WGS84 (lon, lat)."""
    return crs.transformer.transform(x, y)


def wgs84_to_native(lon: float, lat: float, crs: CRS) -> tuple[float, float]:
    """Transform WGS84 (lon, lat) to native CRS coordinates."""
    return crs.inv_transformer.transform(lon, lat)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/razinka/telemac/telemac-viewer && python -m pytest tests/test_crs.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add crs.py tests/test_crs.py
git commit -m "feat(crs): add CRS dataclass with pyproj transforms"
```

---

### Task 2: Add `.cas` detection and coordinate heuristic to `crs.py`

**Files:**
- Modify: `crs.py`
- Modify: `tests/test_crs.py`

- [ ] **Step 1: Write failing tests for `detect_crs_from_cas` and `guess_crs_from_coords`**

Add to `tests/test_crs.py`:

```python
from crs import detect_crs_from_cas, guess_crs_from_coords


class TestDetectCrsFromCas:
    def test_lambert_zone1(self, tmp_path):
        cas = tmp_path / "test.cas"
        cas.write_text(
            "TITLE = 'TEST'\n"
            "GEOGRAPHIC SYSTEM : 4\n"
            "ZONE NUMBER IN GEOGRAPHIC SYSTEM : 1\n"
        )
        crs = detect_crs_from_cas(str(cas))
        assert crs is not None
        assert crs.epsg == 27561

    def test_utm_north_zone34(self, tmp_path):
        cas = tmp_path / "test.cas"
        cas.write_text(
            "GEOGRAPHIC SYSTEM = 2\n"
            "ZONE NUMBER IN GEOGRAPHIC SYSTEM = 34\n"
        )
        crs = detect_crs_from_cas(str(cas))
        assert crs is not None
        assert crs.epsg == 32634

    def test_no_keywords_returns_none(self, tmp_path):
        cas = tmp_path / "test.cas"
        cas.write_text("TITLE = 'NO CRS'\nTIME STEP = 10\n")
        crs = detect_crs_from_cas(str(cas))
        assert crs is None

    def test_comments_stripped(self, tmp_path):
        cas = tmp_path / "test.cas"
        cas.write_text(
            "/ This is a comment\n"
            "GEOGRAPHIC SYSTEM = 1 / WGS84\n"
        )
        crs = detect_crs_from_cas(str(cas))
        assert crs is not None
        assert crs.epsg == 4326

    def test_lambert_zone93(self, tmp_path):
        """Zone 93 is RGF93 Lambert (different datum from zones 1-4)."""
        cas = tmp_path / "test.cas"
        cas.write_text(
            "GEOGRAPHIC SYSTEM = 4\n"
            "ZONE NUMBER IN GEOGRAPHIC SYSTEM = 93\n"
        )
        crs = detect_crs_from_cas(str(cas))
        assert crs is not None
        assert crs.epsg == 2154

    def test_french_keyword(self, tmp_path):
        cas = tmp_path / "test.cas"
        cas.write_text("SYSTEME GEOGRAPHIQUE : 5\n")
        crs = detect_crs_from_cas(str(cas))
        assert crs is not None
        assert crs.epsg == 3395

    def test_french_keyword_with_zone(self, tmp_path):
        cas = tmp_path / "test.cas"
        cas.write_text(
            "SYSTEME GEOGRAPHIQUE : 2\n"
            "NUMERO DE ZONE DU SYSTEME GEOGRAPHIQUE : 34\n"
        )
        crs = detect_crs_from_cas(str(cas))
        assert crs is not None
        assert crs.epsg == 32634

    def test_geosyst_minus1_returns_none(self, tmp_path):
        cas = tmp_path / "test.cas"
        cas.write_text("GEOGRAPHIC SYSTEM = -1\n")
        crs = detect_crs_from_cas(str(cas))
        assert crs is None

    def test_geosyst_zero_returns_none(self, tmp_path):
        cas = tmp_path / "test.cas"
        cas.write_text("GEOGRAPHIC SYSTEM = 0\n")
        crs = detect_crs_from_cas(str(cas))
        assert crs is None

    def test_real_tide_cas(self):
        """Test with actual TELEMAC tide example if available."""
        import os
        cas_path = os.path.join(
            os.environ.get("HOMETEL", ""),
            "examples/telemac2d/tide/t2d_tide_local-jmj_type_gen.cas",
        )
        if not os.path.isfile(cas_path):
            pytest.skip("Tide example not available")
        crs = detect_crs_from_cas(cas_path)
        assert crs is not None
        assert crs.epsg == 27561  # Lambert zone I


class TestGuessCrsFromCoords:
    def test_lks94_range(self):
        x = np.array([400000, 500000, 600000], dtype=np.float64)
        y = np.array([6000000, 6100000, 6200000], dtype=np.float64)
        crs = guess_crs_from_coords(x, y)
        assert crs is not None
        assert crs.epsg == 3346

    def test_wgs84_range(self):
        """Real geographic extent — must span >1 degree to trigger."""
        x = np.array([20.0, 25.0, 30.0], dtype=np.float64)
        y = np.array([50.0, 55.0, 60.0], dtype=np.float64)
        crs = guess_crs_from_coords(x, y)
        assert crs is not None
        assert crs.epsg == 4326

    def test_small_local_returns_none(self):
        """Small lab/flume model (e.g. Gouttedo 0-1m) must NOT trigger WGS84."""
        x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        y = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        crs = guess_crs_from_coords(x, y)
        assert crs is None

    def test_gouttedo_scale_returns_none(self):
        """Gouttedo-scale mesh (~100m) must not be mistaken for WGS84."""
        x = np.array([0.0, 50.0, 100.0], dtype=np.float64)
        y = np.array([0.0, 50.0, 100.0], dtype=np.float64)
        crs = guess_crs_from_coords(x, y)
        assert crs is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/razinka/telemac/telemac-viewer && python -m pytest tests/test_crs.py::TestDetectCrsFromCas tests/test_crs.py::TestGuessCrsFromCoords -v`
Expected: FAIL — `ImportError: cannot import name 'detect_crs_from_cas'`

- [ ] **Step 3: Implement `detect_crs_from_cas` and `guess_crs_from_coords`**

Add to `crs.py`:

```python
import re
from numpy import ndarray


# TELEMAC GEOGRAPHIC SYSTEM → EPSG mapping
_LAMBERT_ZONES = {1: 27561, 2: 27562, 3: 27563, 4: 27564, 93: 2154}

_GEOSYST_KEYWORDS = re.compile(
    r'(?:GEOGRAPHIC SYSTEM|SYSTEME GEOGRAPHIQUE)\s*[:=]\s*(-?\d+)',
    re.IGNORECASE,
)
_NUMZONE_KEYWORDS = re.compile(
    r'(?:ZONE NUMBER IN GEOGRAPHIC SYSTEM|NUMERO DE ZONE DU SYSTEME GEOGRAPHIQUE)\s*[:=]\s*(\d+)',
    re.IGNORECASE,
)


def detect_crs_from_cas(cas_path: str) -> CRS | None:
    """Parse TELEMAC .cas file for GEOGRAPHIC SYSTEM and ZONE NUMBER.

    Returns CRS if detected, None otherwise.
    """
    try:
        with open(cas_path) as f:
            text = f.read()
    except (OSError, IOError):
        return None

    # Strip comments (everything after / on each line)
    lines = []
    for line in text.split('\n'):
        idx = line.find('/')
        if idx >= 0:
            line = line[:idx]
        lines.append(line)
    clean = '\n'.join(lines)

    geo_match = _GEOSYST_KEYWORDS.search(clean)
    if not geo_match:
        return None

    geosyst = int(geo_match.group(1))
    zone_match = _NUMZONE_KEYWORDS.search(clean)
    numzone = int(zone_match.group(1)) if zone_match else 0

    epsg = _geosyst_to_epsg(geosyst, numzone)
    if epsg is None:
        return None

    try:
        return crs_from_epsg(epsg)
    except Exception:
        return None


def _geosyst_to_epsg(geosyst: int, numzone: int) -> int | None:
    """Map TELEMAC GEOGRAPHIC SYSTEM code to EPSG."""
    if geosyst == 1:
        return 4326
    if geosyst == 2:
        return 32600 + numzone  # UTM North
    if geosyst == 3:
        return 32700 + numzone  # UTM South
    if geosyst == 4:
        return _LAMBERT_ZONES.get(numzone)
    if geosyst == 5:
        return 3395
    return None


def guess_crs_from_coords(x: ndarray, y: ndarray) -> CRS | None:
    """Heuristic CRS guess from coordinate ranges.

    Returns a CRS suggestion or None if indeterminate.
    """
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    # WGS84 lon/lat — require minimum extent (>1°) to avoid false-positive
    # on small lab/flume models whose coords happen to fall in [-180,180]
    x_extent = x_max - x_min
    y_extent = y_max - y_min
    if (-180 <= x_min and x_max <= 180 and -90 <= y_min and y_max <= 90
            and x_extent > 1.0 and y_extent > 1.0):
        return crs_from_epsg(4326)

    # LKS94 (Lithuania) — check before general UTM
    if 300_000 <= x_min and x_max <= 700_000 and 5_950_000 <= y_min and y_max <= 6_250_000:
        return crs_from_epsg(3346)

    # General UTM range
    if 100_000 <= x_min and x_max <= 900_000 and 0 <= y_min and y_max <= 10_000_000:
        # Estimate UTM zone from approximate longitude
        # x≈500000 is zone center; this is a rough suggestion
        return None  # too ambiguous without more info

    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/razinka/telemac/telemac-viewer && python -m pytest tests/test_crs.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add crs.py tests/test_crs.py
git commit -m "feat(crs): add .cas auto-detection and coordinate heuristic"
```

---

### Task 3: Add `click_to_native` and `meters_to_wgs84` to `crs.py`

**Files:**
- Modify: `crs.py`
- Modify: `tests/test_crs.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_crs.py`:

```python
from crs import click_to_native, meters_to_wgs84


class TestClickToNative:
    def test_with_crs_roundtrip(self):
        """Round-trip: native → wgs84 → click_to_native should recover original."""
        crs = crs_from_epsg(3346)
        x_orig, y_orig = 500000.0, 6100000.0
        lon, lat = native_to_wgs84(x_orig, y_orig, crs)
        geom = {"x_off": x_orig, "y_off": y_orig, "crs": crs}
        x, y = click_to_native(lon, lat, geom)
        assert x == pytest.approx(x_orig, abs=1.0)
        assert y == pytest.approx(y_orig, abs=1.0)

    def test_without_crs_fallback(self):
        """Without CRS, should match old coord_to_meters behavior."""
        from constants import _M2D
        geom = {"x_off": 100.0, "y_off": 200.0, "crs": None}
        lon, lat = 0.001, 0.002
        x, y = click_to_native(lon, lat, geom)
        assert x == pytest.approx(lon * _M2D + 100.0)
        assert y == pytest.approx(lat * _M2D + 200.0)


class TestMetersToWgs84:
    def test_with_crs(self):
        crs = crs_from_epsg(3346)
        geom = {"crs": crs}
        result = meters_to_wgs84(500000, 6100000, geom)
        assert result is not None
        lon, lat = result
        assert 23.0 < lon < 25.0
        assert 54.5 < lat < 56.0

    def test_without_crs(self):
        geom = {"crs": None}
        result = meters_to_wgs84(100, 200, geom)
        assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/razinka/telemac/telemac-viewer && python -m pytest tests/test_crs.py::TestClickToNative tests/test_crs.py::TestMetersToWgs84 -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement `click_to_native` and `meters_to_wgs84`**

Add to `crs.py`:

```python
def click_to_native(lon: float, lat: float, geom: dict) -> tuple[float, float]:
    """Convert deck.gl click coordinates to native CRS meters.

    When CRS is set, deck.gl reports real lon/lat — use pyproj inverse.
    When CRS is None, fall back to _M2D approximation (old behavior).
    """
    crs = geom.get("crs")
    if crs is not None:
        return wgs84_to_native(lon, lat, crs)
    return float(lon) * _M2D + geom["x_off"], float(lat) * _M2D + geom["y_off"]


def meters_to_wgs84(x_m: float, y_m: float, geom: dict) -> tuple[float, float] | None:
    """Convert native CRS meters to WGS84 for display. Returns None if no CRS."""
    crs = geom.get("crs")
    if crs is None:
        return None
    return native_to_wgs84(x_m, y_m, crs)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/razinka/telemac/telemac-viewer && python -m pytest tests/test_crs.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add crs.py tests/test_crs.py
git commit -m "feat(crs): add click_to_native and meters_to_wgs84"
```

---

### Task 4: Update `geometry.py` — CRS-aware mesh center

**Files:**
- Modify: `geometry.py`
- Modify: `tests/test_geometry.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_geometry.py`:

```python
from crs import crs_from_epsg


class TestBuildMeshGeometryCRS:
    def test_no_crs_defaults_zero(self, fake_tf):
        geom = build_mesh_geometry(fake_tf)
        assert geom["lon_off"] == 0.0
        assert geom["lat_off"] == 0.0
        assert geom["crs"] is None

    def test_with_crs_converts_center(self):
        """LKS94 mesh center should map to real lon/lat."""
        from tests.helpers import FakeTF
        tf = FakeTF()
        # Override with LKS94-range coordinates
        tf.meshx = np.array([490000, 510000, 490000, 510000], dtype=np.float64)
        tf.meshy = np.array([6090000, 6090000, 6110000, 6110000], dtype=np.float64)
        crs = crs_from_epsg(3346)
        geom = build_mesh_geometry(tf, crs=crs)
        assert 23.0 < geom["lon_off"] < 25.0
        assert 54.5 < geom["lat_off"] < 56.0
        assert geom["crs"] is crs
        # x_off/y_off still in native meters
        assert geom["x_off"] == pytest.approx(500000)
        assert geom["y_off"] == pytest.approx(6100000)

    def test_existing_tests_unchanged(self, fake_tf):
        """crs=None path produces same results as before."""
        geom = build_mesh_geometry(fake_tf, crs=None)
        assert geom["x_off"] == pytest.approx(0.5)
        assert geom["y_off"] == pytest.approx(0.5)
        assert geom["npoin"] == 4

    def test_iparam_offsets_applied(self):
        """SELAFIN I_ORIG/J_ORIG offsets should shift x_off/y_off."""
        from tests.helpers import FakeTF
        tf = FakeTF()
        tf.iparam = [0, 0, 1000, 2000, 0, 0, 0, 0, 0, 0]  # I_ORIG=1000, J_ORIG=2000
        geom = build_mesh_geometry(tf)
        assert geom["x_off"] == pytest.approx(0.5 + 1000)
        assert geom["y_off"] == pytest.approx(0.5 + 2000)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/razinka/telemac/telemac-viewer && python -m pytest tests/test_geometry.py::TestBuildMeshGeometryCRS -v`
Expected: FAIL — `geom["lon_off"]` KeyError

- [ ] **Step 3: Modify `geometry.py`**

Update `build_mesh_geometry` signature and add CRS conversion:

```python
# geometry.py — add import at top
from crs import CRS as CRSType, native_to_wgs84

def build_mesh_geometry(tf: Any, crs: CRSType | None = None,
                        z_values: np.ndarray | None = None,
                        z_scale: float = 1) -> dict[str, Any]:
    # ... existing code for x, y, npoin, ikle ...

    # Check for SELAFIN origin offsets
    i_orig = 0
    j_orig = 0
    if hasattr(tf, 'iparam') and hasattr(tf.iparam, '__len__') and len(tf.iparam) > 3:
        i_orig = int(tf.iparam[2])
        j_orig = int(tf.iparam[3])

    x_off = float((x.min() + x.max()) / 2) + i_orig
    y_off = float((y.min() + y.max()) / 2) + j_orig

    # ... existing positions/indices/zoom code (unchanged) ...

    # CRS: convert mesh center to WGS84
    if crs is not None:
        lon_off, lat_off = native_to_wgs84(x_off, y_off, crs)
    else:
        lon_off, lat_off = 0.0, 0.0

    return {
        "npoin": npoin,
        "positions": encode_binary_attribute(positions.flatten()),
        "indices": encode_binary_attribute(indices),
        "x_off": x_off, "y_off": y_off,
        "lon_off": lon_off, "lat_off": lat_off,
        "crs": crs,
        "extent_m": extent_m,
        "zoom": zoom,
    }
```

- [ ] **Step 4: Run ALL geometry + existing tests**

Run: `cd /home/razinka/telemac/telemac-viewer && python -m pytest tests/test_geometry.py -v`
Expected: All PASS (new + existing)

- [ ] **Step 5: Commit**

```bash
git add geometry.py tests/test_geometry.py
git commit -m "feat(geometry): add CRS-aware mesh center conversion"
```

---

### Task 5: Update `layers.py` — add `origin` parameter

**Files:**
- Modify: `layers.py`
- Modify: `tests/test_layers.py`

- [ ] **Step 1: Write failing test for origin pass-through**

Add to `tests/test_layers.py`:

```python
class TestOriginParameter:
    def test_mesh_layer_default_origin(self, fake_tf, fake_geom):
        from layers import build_mesh_layer
        values = np.array([0.1, 0.5, 0.5, 1.0])
        lyr, _, _, _ = build_mesh_layer(fake_geom, values, "Viridis")
        assert lyr["coordinateOrigin"] == [0, 0]

    def test_mesh_layer_custom_origin(self, fake_tf, fake_geom):
        from layers import build_mesh_layer
        values = np.array([0.1, 0.5, 0.5, 1.0])
        lyr, _, _, _ = build_mesh_layer(fake_geom, values, "Viridis", origin=[24.0, 55.0])
        assert lyr["coordinateOrigin"] == [24.0, 55.0]

    def test_marker_layer_custom_origin(self):
        from layers import build_marker_layer
        lyr = build_marker_layer(0.0, 0.0, origin=[24.0, 55.0])
        assert lyr["coordinateOrigin"] == [24.0, 55.0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/razinka/telemac/telemac-viewer && python -m pytest tests/test_layers.py::TestOriginParameter -v`
Expected: FAIL — `unexpected keyword argument 'origin'`

- [ ] **Step 3: Add `origin` parameter to all 10 layer builders**

For each function in `layers.py`, add `origin: list[float] | None = None` parameter and replace `coordinateOrigin=[0, 0]` with `coordinateOrigin=origin or [0, 0]`.

The 10 functions to modify (each one gets the same pattern):

1. `build_mesh_layer(geom, values, palette_id, ..., origin=None)` — line 24
2. `build_velocity_layer(tf, time_idx, geom, origin=None)` — line 80
3. `build_contour_layer_fn(tf, values, geom, ..., origin=None)` — line 132
4. `build_marker_layer(x_m, y_m, layer_id="marker", origin=None)` — line 233
5. `build_cross_section_layer(points_m, origin=None)` — line 249
6. `build_particle_layer(paths, current_time, trail_length, origin=None)` — line 268
7. `build_wireframe_layer(tf, geom, origin=None)` — line 287
8. `build_extrema_markers(extrema, x_off, y_off, origin=None)` — line 330
9. `build_measurement_layer(points_m, origin=None)` — line 356
10. `build_boundary_layer(tf, geom, boundary_nodes, ..., origin=None)` — line 387

In each function, every occurrence of `coordinateOrigin=[0, 0]` becomes `coordinateOrigin=origin or [0, 0]`.

For `build_extrema_markers` and `build_measurement_layer` which return lists, the `origin` is passed to each sub-layer.

For `build_boundary_layer` which also returns a list, pass `origin` to each `line_layer` and `scatterplot_layer` call inside.

- [ ] **Step 4: Run ALL layer tests**

Run: `cd /home/razinka/telemac/telemac-viewer && python -m pytest tests/test_layers.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add layers.py tests/test_layers.py
git commit -m "feat(layers): add origin parameter to all layer builders"
```

---

### Task 6: Remove `coord_to_meters` from `analysis.py`

**Files:**
- Modify: `analysis.py`
- Modify: `tests/test_analysis.py`

- [ ] **Step 1: Check existing test for `coord_to_meters`**

Look at `tests/test_analysis.py` for any tests referencing `coord_to_meters`. These must be updated to import from `crs.py` instead, or use `click_to_native`.

- [ ] **Step 2: Remove `coord_to_meters` function from `analysis.py`**

Delete the function at `analysis.py:490-497`:
```python
# DELETE this function:
def coord_to_meters(lon, lat, x_off, y_off):
    ...
```

- [ ] **Step 3: Update `tests/test_analysis.py` import**

Change:
```python
from analysis import coord_to_meters, ...
```
To:
```python
from crs import click_to_native
```

Update any test that called `coord_to_meters(lon, lat, x_off, y_off)` to use `click_to_native(lon, lat, {"x_off": x_off, "y_off": y_off, "crs": None})` instead.

- [ ] **Step 4: Run ALL analysis tests**

Run: `cd /home/razinka/telemac/telemac-viewer && python -m pytest tests/test_analysis.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add analysis.py tests/test_analysis.py
git commit -m "refactor: move coord_to_meters to crs.py as click_to_native"
```

---

### Task 7: Wire CRS into `app.py` — UI, reactives, and click handlers

**Files:**
- Modify: `app.py`

This is the largest task. It has 4 sub-steps.

- [ ] **Step 1: Add imports and CRS UI**

At top of `app.py`, add import:
```python
from crs import crs_from_epsg, detect_crs_from_cas, guess_crs_from_coords, click_to_native, meters_to_wgs84
```

Remove import of `coord_to_meters` from the `analysis` import block.

In the Data accordion panel (after basemap selector, around line 658), add:
```python
ui.input_text("epsg_input", "CRS (EPSG)", placeholder="e.g. 3346"),
ui.input_switch("auto_crs", "Auto-detect CRS", value=True),
ui.output_ui("crs_status_ui"),
```

- [ ] **Step 2: Add CRS reactive and resolution logic**

In the server function, add:
```python
current_crs = reactive.value(None)

@reactive.effect
def resolve_crs():
    """Auto-detect or manually set CRS when file or EPSG input changes."""
    epsg_text = input.epsg_input() if input.epsg_input() else ""

    # Manual EPSG overrides auto-detect
    if epsg_text.strip():
        try:
            code = int(epsg_text.strip())
            current_crs.set(crs_from_epsg(code))
            return
        except Exception:
            current_crs.set(None)
            return

    # Auto-detect
    if not input.auto_crs():
        current_crs.set(None)
        return

    # Try .cas file detection
    uploaded = input.upload()
    if not (uploaded and use_upload.get()):
        path = EXAMPLES.get(input.example(), "")
        cas_files = find_cas_files(path)
        for cas_path in cas_files.values():
            detected = detect_crs_from_cas(cas_path)
            if detected:
                current_crs.set(detected)
                ui.update_text("epsg_input", value=str(detected.epsg))
                return

    # Try coordinate heuristic
    tf = tel_file()
    detected = guess_crs_from_coords(tf.meshx, tf.meshy)
    if detected:
        current_crs.set(detected)
        ui.update_text("epsg_input", value=str(detected.epsg))
        return

    current_crs.set(None)

@output
@render.ui
def crs_status_ui():
    crs = current_crs.get()
    if crs:
        return ui.span(f"EPSG:{crs.epsg} — {crs.name}", class_="small text-success")
    return ui.span("No CRS — basemap alignment disabled", class_="small text-muted")
```

- [ ] **Step 3: Update `mesh_geom()` to pass CRS**

Change:
```python
@reactive.calc
def mesh_geom():
    return build_mesh_geometry(tel_file())
```
To:
```python
@reactive.calc
def mesh_geom():
    return build_mesh_geometry(tel_file(), crs=current_crs.get())
```

Also add to the `on_file_change` handler where `current_crs` is reset:
```python
# In the file-change handler (around line 998):
# current_crs is handled by resolve_crs reactive, no manual reset needed
```

- [ ] **Step 4: Replace all `coord_to_meters` calls with `click_to_native`**

There are 5 call sites in `app.py`. Each one changes from:
```python
x_m, y_m = coord_to_meters(coord[0], coord[1], geom["x_off"], geom["y_off"])
```
To:
```python
x_m, y_m = click_to_native(coord[0], coord[1], geom)
```

Call sites (by line number in current app.py):
1. Line 1464 — `handle_map_click`
2. Line 1509 — `handle_drawn_features` (polygon)
3. Line 1521 — `handle_drawn_features` (line)
4. Line 1728 — measurement click handler
5. Line 1946 — hover handler

- [ ] **Step 5: Update `update_map` — view_state and layer origins**

In `update_map()` (around line 2569), change:
```python
kwargs["view_state"] = {
    "longitude": 0,
    "latitude": 0,
    "zoom": geom["zoom"],
}
```
To:
```python
kwargs["view_state"] = {
    "longitude": geom["lon_off"],
    "latitude": geom["lat_off"],
    "zoom": geom["zoom"],
}
```

For every layer builder call in `update_map`, add `origin=[geom["lon_off"], geom["lat_off"]]`:

```python
origin = [geom["lon_off"], geom["lat_off"]]

# Line ~2465:
lyr, vmin, vmax, log_applied = build_mesh_layer(geom, values, palette_id, ..., origin=origin)

# Line ~2478:
layers.append(build_wireframe_layer(tf, geom, origin=origin))

# Line ~2482:
layers.extend(build_boundary_layer(tf, geom, ..., origin=origin))

# Line ~2487:
layers.extend(build_extrema_markers(extrema, geom["x_off"], geom["y_off"], origin=origin))

# Line ~2490:
vlyr = build_velocity_layer(tf, tidx, geom, origin=origin)

# Line ~2495:
clyr = build_contour_layer_fn(tf, values, geom, origin=origin)

# Line ~2511:
clyr2 = build_contour_layer_fn(tf, compare_vals, geom, ..., origin=origin)

# Line ~2521:
layers.append(build_marker_layer(mx, my, layer_id=f"marker-{i}", origin=origin))

# Line ~2527:
layers.append(build_cross_section_layer(path_centered, origin=origin))

# Line ~2534:
layers.append(build_particle_layer(paths, current_time, trail, origin=origin))

# Line ~2540:
layers.extend(build_measurement_layer(mpts_centered, origin=origin))
```

- [ ] **Step 6: Commit**

```bash
git add app.py
git commit -m "feat(app): wire CRS into UI, reactives, click handlers, and layer origins"
```

---

### Task 8: Add coordinate display and CSV export metadata

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add WGS84 to hover/click coordinate display**

In the click handler that builds hover/node info (around line 1464), after computing `x_m, y_m`:
```python
wgs84 = meters_to_wgs84(x_m, y_m, geom)
if wgs84:
    lon, lat = wgs84
    coord_text = f"Native: {x_m:.0f}, {y_m:.0f}  |  WGS84: {lon:.6f}°E, {lat:.6f}°N"
else:
    coord_text = f"X: {x_m:.1f}  Y: {y_m:.1f}"
```

Apply similar logic to the hover info display and the probe table rows.

- [ ] **Step 2: Add CRS metadata to CSV exports**

In the download handlers (around line 2190-2230), add CRS header comment:
```python
crs = current_crs.get()
if crs:
    buf.write(f"# CRS: EPSG:{crs.epsg} ({crs.name})\n")
```

For point exports, add lon/lat columns:
```python
if crs:
    lon, lat = meters_to_wgs84(pt[0], pt[1], geom)
    # Add to column headers and data rows
```

- [ ] **Step 3: Run existing tests to verify nothing broke**

Run: `cd /home/razinka/telemac/telemac-viewer && python -m pytest -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat(app): add WGS84 coordinate display and CRS metadata in exports"
```

---

### Task 9: Manual verification

- [ ] **Step 1: Start the viewer**

```bash
cd /home/razinka/telemac/telemac-viewer && /opt/micromamba/envs/shiny/bin/shiny run app.py --port 8765
```

- [ ] **Step 2: Test Curonian Lagoon with CRS**

1. Select "Curonian Lagoon (24h)" from examples
2. Type `3346` in the CRS EPSG field
3. Verify: mesh overlays on OSM/satellite basemap at correct location (Curonian Lagoon, Lithuania)
4. Click on mesh → verify coordinates show both LKS94 and WGS84
5. Switch basemap to "CartoDB Dark" → verify alignment holds

- [ ] **Step 3: Test non-CRS example**

1. Select "Gouttedo (raindrop)"
2. Clear EPSG field
3. Verify: mesh renders centered at (0,0), basemap shows ocean, current behavior preserved

- [ ] **Step 4: Test auto-detection**

1. Select tide example (if available) → verify CRS auto-detects as Lambert
2. Upload a .slf file → verify coordinate heuristic suggests CRS if applicable

- [ ] **Step 5: Test exports**

1. With CRS set, click a point and download CSV
2. Verify header contains `# CRS: EPSG:3346`
3. Verify lon/lat columns are present
