# CRS Awareness for TELEMAC Viewer

**Date**: 2026-03-20
**Status**: Approved design (rev 2 — post spec review)
**Goal**: Add full coordinate reference system support so meshes render on real basemaps, coordinates display in both native CRS and WGS84, and exports include CRS metadata.

---

## Problem

TELEMAC meshes use metric coordinate systems (LKS94, UTM, Lambert, etc.) but the viewer renders all meshes centered at `(longitude=0, latitude=0)` using deck.gl's METER_OFFSETS. Basemaps never align with the mesh. Coordinate displays show raw meter values with no geographic context.

## Architecture

**Core principle**: All analysis stays in native CRS meters. The rendering origin and all layer `coordinateOrigin` values shift to real WGS84 coordinates. Click coordinates are converted back to native CRS via pyproj inverse transform (not the `_M2D` approximation).

```
Native CRS (e.g. LKS94 / EPSG:3346)
    │
    ├── analysis.py: all computations in native meters (unchanged)
    │   coord_to_meters() updated: uses pyproj inverse when CRS set
    │
    ├── layers.py: builds layers in centered meters, coordinateOrigin
    │   updated from [0,0] to [lon_off, lat_off]
    │
    ▼
geometry.py: converts mesh center (x_off, y_off) → (lon_off, lat_off) via pyproj
    │         positions array unchanged (still centered meter offsets)
    │
    ▼
deck.gl METER_OFFSETS: view_state + coordinateOrigin at real lon/lat
    → basemaps align automatically
```

**Why this works**: deck.gl METER_OFFSETS interprets position data as meter offsets from `coordinateOrigin`. The centered positions are the same regardless of whether the origin is `(0, 0)` or `(24.0, 55.1)`. Moving both `coordinateOrigin` and `view_state` to real coordinates is all that's needed for basemap alignment.

## New Module: `crs.py`

~120 lines. Single source of truth for coordinate transforms.

### Data Model

```python
@dataclass
class CRS:
    epsg: int                    # e.g. 3346
    name: str                    # e.g. "LKS94 / Lithuania TM"
    transformer: Transformer     # native → WGS84
    inv_transformer: Transformer # WGS84 → native
```

### Functions

- `crs_from_epsg(code: int) -> CRS` — create CRS from EPSG code via pyproj
- `detect_crs_from_cas(cas_path: str) -> CRS | None` — parse .cas file for GEOGRAPHIC SYSTEM + ZONE NUMBER
- `guess_crs_from_coords(x: ndarray, y: ndarray) -> CRS | None` — heuristic from coordinate ranges
- `native_to_wgs84(x: float, y: float, crs: CRS) -> tuple[float, float]` — single point transform
- `wgs84_to_native(lon: float, lat: float, crs: CRS) -> tuple[float, float]` — inverse transform
- `click_to_native(lon: float, lat: float, geom: dict) -> tuple[float, float]` — convert deck.gl click coordinates to native CRS meters. Uses pyproj inverse when CRS is set, falls back to `_M2D` approximation when CRS is None.

### .cas Detection: GEOGRAPHIC SYSTEM Mapping

TELEMAC's `GEOGRAPHIC SYSTEM` keyword maps to EPSG:

| Code | Meaning | EPSG |
|------|---------|------|
| -1 | No default | None |
| 0 | User-defined | None (prompt user) |
| 1 | WGS84 lon/lat | 4326 |
| 2 | WGS84 UTM North | 326{NUMZONE} |
| 3 | WGS84 UTM South | 327{NUMZONE} |
| 4 | Lambert | NUMZONE→EPSG: 1→27561, 2→27562, 3→27563, 4→27564, 93→2154 |
| 5 | Mercator | 3395 |

Note: Lambert zones 1-4 are NTF (Clarke 1880 ellipsoid), zone 93 is RGF93 (GRS80 ellipsoid) — different datums.

Parser reads lines matching `GEOGRAPHIC SYSTEM` and `ZONE NUMBER IN GEOGRAPHIC SYSTEM`. Handles TELEMAC .cas syntax: strips comments (after `/`), supports both `=` and `:` separators, handles French keyword equivalents (`SYSTEME GEOGRAPHIQUE`).

### Coordinate Range Heuristic

When no .cas file is available (uploaded .slf files):

- x ∈ [100k–900k], y ∈ [0–10M] → likely UTM (suggest UTM zone from longitude estimate)
- x ∈ [300k–700k], y ∈ [5.95M–6.25M] → likely LKS94 (EPSG:3346)
- x ∈ [-180, 180], y ∈ [-90, 90] → already WGS84 (EPSG:4326)
- Otherwise → None (no CRS, current behavior)

The heuristic populates the EPSG field as a suggestion; the user can override.

### SELAFIN Origin Offsets

SELAFIN files can store coordinate origin offsets in `iparam[2]` (I_ORIG) and `iparam[3]` (J_ORIG). If present and non-zero, these must be added to `(x_off, y_off)` before the native-to-WGS84 conversion. `geometry.py` checks for these when building the mesh geometry.

## geometry.py Changes

### `build_mesh_geometry(tf, crs=None, z_values=None, z_scale=1)`

New `crs` parameter. When provided:

```python
# Check for SELAFIN origin offsets
i_orig = tf.iparam[2] if hasattr(tf, 'iparam') and len(tf.iparam) > 3 else 0
j_orig = tf.iparam[3] if hasattr(tf, 'iparam') and len(tf.iparam) > 3 else 0

x_off = float((x.min() + x.max()) / 2) + i_orig
y_off = float((y.min() + y.max()) / 2) + j_orig

if crs:
    lon_off, lat_off = native_to_wgs84(x_off, y_off, crs)
else:
    lon_off, lat_off = 0.0, 0.0
```

Return dict adds:
- `"lon_off"`: float — WGS84 longitude of mesh center
- `"lat_off"`: float — WGS84 latitude of mesh center
- `"crs"`: CRS | None — for downstream coordinate display and click conversion

Positions array, indices, zoom calculation — all unchanged.

## layers.py Changes

Every layer builder currently hardcodes `coordinateOrigin=[0, 0]`. All 12 layer calls must change to accept and use the real origin.

**Approach**: Add `origin: list[float] = [0, 0]` parameter to each layer builder function. In `app.py`, pass `origin=[geom["lon_off"], geom["lat_off"]]` from the geom dict. This is a mechanical change — ~12 lines modified (one per function), plus ~12 lines at call sites in `app.py`.

Affected functions:
- `build_mesh_layer`
- `build_velocity_layer`
- `build_contour_layer_fn`
- `build_marker_layer`
- `build_cross_section_layer`
- `build_particle_layer`
- `build_wireframe_layer`
- `build_extrema_markers`
- `build_measurement_layer`
- `build_boundary_layer`

When `crs=None`, `origin` defaults to `[0, 0]` — preserving current behavior.

## analysis.py Changes

### `coord_to_meters` — replaced by `click_to_native` in `crs.py`

The current `coord_to_meters()` uses `_M2D = 111320` to convert deck.gl pseudo-lon/lat back to native meters. This approximation is only valid at the equator — at latitude 55° (Lithuania), it introduces ~43% error in the X dimension.

**New approach**: `click_to_native(lon, lat, geom)` in `crs.py`:
- **When CRS is set**: deck.gl reports clicks as real lon/lat near `(lon_off, lat_off)`. Use `wgs84_to_native(lon, lat, crs)` to get exact native CRS coordinates.
- **When CRS is None**: fall back to current `_M2D` formula: `(lon * _M2D + x_off, lat * _M2D + y_off)`.

All callers of `coord_to_meters()` in `app.py` switch to `click_to_native()`. The old function is removed.

### `meters_to_wgs84(x_m, y_m, geom) -> tuple[float, float] | None`

Convenience wrapper for display. Returns `(lon, lat)` if `geom["crs"]` is set, else `None`. Uses `native_to_wgs84()`.

## app.py Changes

### New Reactives

```python
current_crs = reactive.value(None)  # CRS object or None
```

### UI Addition (Data accordion, after basemap selector)

```python
ui.input_text("epsg_input", "CRS (EPSG)", placeholder="e.g. 3346"),
ui.input_switch("auto_crs", "Auto-detect CRS", value=True),
ui.output_ui("crs_status_ui"),
```

`crs_status_ui` displays: `"EPSG:3346 — LKS94 / Lithuania TM"` or `"No CRS — basemap alignment disabled"`.

### CRS Resolution Logic

On file change:
1. If auto-detect is on: try `detect_crs_from_cas()`, fall back to `guess_crs_from_coords()`
2. If detected, populate `epsg_input` with the code
3. If user has typed a manual EPSG code, that overrides auto-detect
4. `current_crs` reactive updates → triggers `mesh_geom()` recalc → triggers `update_map()`

### mesh_geom() Change

```python
@reactive.calc
def mesh_geom():
    return build_mesh_geometry(tel_file(), crs=current_crs.get())
```

### update_map() Changes

View state uses real coordinates:
```python
kwargs["view_state"] = {
    "longitude": geom["lon_off"],
    "latitude": geom["lat_off"],
    "zoom": geom["zoom"],
}
```

All layer calls pass `origin=[geom["lon_off"], geom["lat_off"]]`.

### Click/Hover Handlers

All calls to `coord_to_meters()` replaced with `click_to_native(lon, lat, geom)` from `crs.py`.

### Coordinate Display

Hover info and click handlers gain WGS84 display when CRS is set:
```
LKS94: 500123, 6123456  |  WGS84: 24.0012°E, 55.1234°N
```

Probe table adds Longitude/Latitude columns.

### CSV Export Changes

- Header comment: `# CRS: EPSG:3346 (LKS94 / Lithuania TM)`
- Point exports: add `Longitude,Latitude` columns alongside `X,Y`
- Cross-section export: add start/end WGS84 in header comment

## What Doesn't Change

- All analysis functions — cross-sections, particles, flood mapping, integrals, polygon stats (all work in native meters)
- Positions array construction in geometry.py (still centered meter offsets)
- All existing tests — `crs=None` path preserves current behavior exactly
- Simulation launcher, animation, recording
- Constants, palettes, telemac_defaults

## Estimated Scope

| File | Change |
|------|--------|
| `crs.py` (new) | ~120 lines |
| `geometry.py` | ~20 lines modified |
| `layers.py` | ~25 lines modified (origin param + pass-through) |
| `app.py` | ~80 lines modified (UI + reactives + display + click handlers) |
| `analysis.py` | ~5 lines removed (coord_to_meters moved to crs.py) |
| `tests/test_crs.py` (new) | ~70 lines |

Total: ~320 lines new/modified. No breaking changes when `crs=None`.

## Test Plan

1. **Unit tests for `crs.py`**:
   - `crs_from_epsg(3346)` returns correct name and transforms known LKS94 point
   - `crs_from_epsg(99999)` raises ValueError for invalid code
   - `detect_crs_from_cas()` with tide example (.cas has GEOGRAPHIC SYSTEM = 4, NUMZONE = 1) → Lambert I
   - `detect_crs_from_cas()` with no CRS keywords → None
   - `guess_crs_from_coords()` with LKS94-range arrays → EPSG:3346
   - `guess_crs_from_coords()` with UTM-range arrays → UTM zone suggestion
   - `guess_crs_from_coords()` with small local coords → None
   - Round-trip: `native → wgs84 → native` within 0.01m tolerance
   - `click_to_native()` with CRS set: verify against known point
   - `click_to_native()` with CRS None: verify matches old `coord_to_meters` behavior

2. **Integration test**:
   - Load Curonian Lagoon with EPSG:3346, verify `lon_off ≈ 21.1`, `lat_off ≈ 55.5`
   - Verify basemap tiles load for the correct geographic area
   - Load Gouttedo (no CRS) → `lon_off = 0, lat_off = 0`, behaves as before
   - All existing tests pass unchanged with `crs=None` path

3. **Manual verification**:
   - Open viewer with Curonian Lagoon + CRS 3346 → mesh overlays correctly on OSM/satellite
   - Click on a point → coordinates show both LKS94 and WGS84
   - Export CSV → header contains CRS, columns include lon/lat
   - Load a non-CRS example (Gouttedo) → no CRS detected, centered at (0,0), current behavior preserved
   - Switch CRS on/off → mesh re-renders correctly in both modes
