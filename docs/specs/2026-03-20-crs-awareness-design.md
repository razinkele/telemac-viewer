# CRS Awareness for TELEMAC Viewer

**Date**: 2026-03-20
**Status**: Approved design
**Goal**: Add full coordinate reference system support so meshes render on real basemaps, coordinates display in both native CRS and WGS84, and exports include CRS metadata.

---

## Problem

TELEMAC meshes use metric coordinate systems (LKS94, UTM, Lambert, etc.) but the viewer renders all meshes centered at `(longitude=0, latitude=0)` using deck.gl's METER_OFFSETS. Basemaps never align with the mesh. Coordinate displays show raw meter values with no geographic context.

## Architecture

**Core principle**: All analysis stays in native CRS meters. Only the rendering origin shifts to real WGS84 coordinates.

```
Native CRS (e.g. LKS94 / EPSG:3346)
    │
    ├── analysis.py: all computations in native meters (unchanged)
    ├── layers.py: builds layers in centered meters (unchanged)
    │
    ▼
geometry.py: converts mesh center (x_off, y_off) → (lon_off, lat_off) via pyproj
    │         positions array unchanged (still centered meter offsets)
    │         deck.gl view_state uses (lon_off, lat_off) as origin
    │
    ▼
deck.gl METER_OFFSETS: renders at real lon/lat → basemaps align automatically
```

**Why this works**: deck.gl METER_OFFSETS interprets position data as meter offsets from the view_state origin. The centered positions are the same regardless of whether the origin is `(0, 0)` or `(24.0, 55.1)`. Moving the origin to real coordinates is the only change needed for basemap alignment.

## New Module: `crs.py`

~100 lines. Single source of truth for coordinate transforms.

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
- `wgs84_to_native(lon: float, lat: float, crs: CRS) -> tuple[float, float]` — inverse

### .cas Detection: GEOGRAPHIC SYSTEM Mapping

TELEMAC's `GEOGRAPHIC SYSTEM` keyword maps to EPSG:

| Code | Meaning | EPSG |
|------|---------|------|
| -1 | No default | None |
| 0 | User-defined | None (prompt user) |
| 1 | WGS84 lon/lat | 4326 |
| 2 | WGS84 UTM North | 326{NUMZONE} |
| 3 | WGS84 UTM South | 327{NUMZONE} |
| 4 | Lambert | NUMZONE→EPSG: 1→27561, 2→27562, 3→27563, 93→2154 |
| 5 | Mercator | 3395 |

Parser reads lines matching `GEOGRAPHIC SYSTEM` and `ZONE NUMBER IN GEOGRAPHIC SYSTEM`, strips comments (after `/`), extracts integer values.

### Coordinate Range Heuristic

When no .cas file is available (uploaded .slf files):

- x ∈ [100k–900k], y ∈ [0–10M] → likely UTM (suggest UTM zone from longitude estimate)
- x ∈ [100k–700k], y ∈ [6M–6.5M] → likely LKS94 (EPSG:3346)
- x ∈ [-180, 180], y ∈ [-90, 90] → already WGS84 (EPSG:4326)
- Otherwise → None (no CRS, current behavior)

The heuristic populates the EPSG field as a suggestion; the user can override.

## geometry.py Changes

### `build_mesh_geometry(tf, crs=None, z_values=None, z_scale=1)`

New `crs` parameter. When provided:

```python
if crs:
    lon_off, lat_off = native_to_wgs84(x_off, y_off, crs)
else:
    lon_off, lat_off = 0.0, 0.0
```

Return dict adds:
- `"lon_off"`: float — WGS84 longitude of mesh center
- `"lat_off"`: float — WGS84 latitude of mesh center
- `"crs"`: CRS | None — for downstream coordinate display

Positions array, indices, zoom calculation — all unchanged.

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

### update_map() Change

```python
kwargs["view_state"] = {
    "longitude": geom["lon_off"],
    "latitude": geom["lat_off"],
    "zoom": geom["zoom"],
}
```

### Coordinate Display

Hover info and click handlers gain WGS84 display:
```
LKS94: 500123, 6123456  |  WGS84: 24.0012°E, 55.1234°N
```

Probe table adds Longitude/Latitude columns.

### CSV Export Changes

- Header comment: `# CRS: EPSG:3346 (LKS94 / Lithuania TM)`
- Point exports: add `Longitude,Latitude` columns alongside `X,Y`
- Cross-section export: add start/end WGS84 in header comment

## analysis.py Changes

### `coord_to_meters` — unchanged

The function converts deck.gl pseudo-lon/lat (click coordinates) back to centered meters using `_M2D`, then adds `(x_off, y_off)` to get native CRS coordinates. This is still correct because METER_OFFSETS reports clicks the same way regardless of map center.

### New: `meters_to_wgs84(x_m, y_m, geom) -> tuple[float, float] | None`

Convenience wrapper for display. Returns `(lon, lat)` if `geom["crs"]` is set, else `None`.

## What Doesn't Change

- All layer builders in `layers.py` — still receive centered meter positions
- All analysis functions — cross-sections, particles, flood mapping, integrals, polygon stats
- `coord_to_meters()` — same pipeline
- All existing tests — `crs=None` path preserves current behavior exactly
- Simulation launcher, animation, recording

## Estimated Scope

| File | Change |
|------|--------|
| `crs.py` (new) | ~100 lines |
| `geometry.py` | ~15 lines modified |
| `app.py` | ~60 lines added (UI + reactives + display) |
| `analysis.py` | ~10 lines added (meters_to_wgs84 helper) |
| `tests/test_crs.py` (new) | ~50 lines |

Total: ~235 lines new/modified. No breaking changes.

## Test Plan

1. **Unit tests for `crs.py`**:
   - `crs_from_epsg(3346)` returns correct name and transforms known LKS94 point
   - `detect_crs_from_cas()` with tide example (.cas has GEOGRAPHIC SYSTEM = 4, NUMZONE = 1)
   - `guess_crs_from_coords()` with LKS94-range and UTM-range coordinate arrays
   - Round-trip: `native → wgs84 → native` within 0.01m tolerance

2. **Integration test**:
   - Load Curonian Lagoon with EPSG:3346, verify `lon_off ≈ 21.1`, `lat_off ≈ 55.5` (approximate center of Curonian Lagoon)
   - Verify basemap tiles load for the correct geographic area

3. **Manual verification**:
   - Open viewer with Curonian Lagoon + CRS 3346 → mesh overlays correctly on OSM/satellite
   - Click on a point → coordinates show both LKS94 and WGS84
   - Export CSV → header contains CRS, columns include lon/lat
   - Load a non-CRS example (Gouttedo) → no CRS detected, centered at (0,0), behaves as before
