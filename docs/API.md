# API Reference

Module-level API documentation for the TELEMAC Viewer.

---

## `viewer_types` — Shared Types

### `MeshGeometry`

```python
@dataclass(frozen=True)
class MeshGeometry:
    npoin: int                  # Number of 2D mesh nodes
    positions: dict[str, Any]   # Binary-encoded vertex positions for deck.gl
    indices: dict[str, Any]     # Binary-encoded triangle indices for deck.gl
    x_off: float                # Mesh center X in native CRS (meters)
    y_off: float                # Mesh center Y in native CRS (meters)
    lon_off: float              # WGS84 longitude of mesh center (0.0 if no CRS)
    lat_off: float              # WGS84 latitude of mesh center (0.0 if no CRS)
    crs: CRS | None             # Coordinate reference system
    extent_m: float             # Mesh bounding box extent in meters
    zoom: float                 # Suggested deck.gl zoom level
```

Produced by `build_mesh_geometry()`, consumed by all layer builders and analysis functions. Positions and indices are base64-encoded for efficient WebSocket transport.

### `TelemacFileProtocol`

```python
class TelemacFileProtocol(Protocol):
    meshx: np.ndarray           # X coordinates of mesh nodes
    meshy: np.ndarray           # Y coordinates of mesh nodes
    npoin2: int                 # Number of 2D nodes
    nelem2: int                 # Number of 2D elements
    ikle2: np.ndarray           # Connectivity table (nelem2 x 3)
    varnames: list[str]         # Variable names in the file
    times: Sequence[float]      # Timestep values (seconds)
    nplan: int                  # Number of vertical planes (1 for 2D)

    def get_data_value(self, varname: str, tidx: int) -> np.ndarray: ...
    def get_z_name(self) -> str: ...
    def get_timeseries_on_points(self, varname: str, points: list) -> list: ...
    def get_data_on_points(self, varname: str, tidx: int, points: list) -> list: ...
    def get_data_on_polyline(self, varname: str, tidx: int, polyline: list) -> tuple: ...
    def close(self) -> None: ...
```

Structural type matching `TelemacFile` from the TELEMAC Python API. Used for type hints throughout the codebase.

---

## `crs` — Coordinate Reference System

### `CRS`

```python
@dataclass(frozen=True)
class CRS:
    epsg: int                       # EPSG code (e.g. 2154 for Lambert-93)
    name: str                       # Human-readable name
    transformer: Transformer        # Native CRS -> WGS84
    inv_transformer: Transformer    # WGS84 -> Native CRS
```

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `crs_from_epsg` | `(code: int) -> CRS` | Create CRS from EPSG code. Raises `CRSError` for invalid codes. |
| `native_to_wgs84` | `(x, y, crs) -> (lon, lat)` | Transform native CRS coordinates to WGS84. |
| `wgs84_to_native` | `(lon, lat, crs) -> (x, y)` | Transform WGS84 to native CRS coordinates. |
| `detect_crs_from_cas` | `(cas_path: str) -> CRS \| None` | Parse TELEMAC `.cas` file for GEOGRAPHIC SYSTEM and ZONE NUMBER keywords. Returns `None` if not found. |
| `guess_crs_from_coords` | `(x, y) -> CRS \| None` | Heuristic CRS detection from coordinate ranges. Currently detects LKS94 (Lithuania). |
| `click_to_native` | `(lon, lat, geom) -> (x_m, y_m)` | Convert deck.gl click coordinates to native CRS meters. |
| `meters_to_wgs84` | `(x_m, y_m, geom) -> (lon, lat) \| None` | Convert native meters to WGS84 for display. Returns `None` if no CRS. |

---

## `geometry` — Mesh Geometry Builder

### `build_mesh_geometry`

```python
def build_mesh_geometry(
    tf: TelemacFileProtocol,
    crs: CRS | None = None,
    z_values: np.ndarray | None = None,
    z_scale: float = 1,
    origin_offset: tuple[float, float] = (0, 0),
) -> MeshGeometry
```

Builds mesh geometry for deck.gl `SimpleMeshLayer`. Transforms TELEMAC mesh coordinates into centered-meter coordinates by subtracting the bounding box center. When a CRS is provided, the mesh center is converted to WGS84 for basemap alignment. Handles SELAFIN `I_ORIG`/`J_ORIG` offsets from `iparam[2:4]`.

---

## `layers` — deck.gl Layer Builders

All layer builders return dict-based layer specifications for shiny-deckgl's `MapWidget`.

| Function | Description |
|----------|-------------|
| `build_mesh_layer(geom, values, palette_id, ...)` | `SimpleMeshLayer` with per-vertex coloring, optional value filter, log scale, and palette reversal. Returns `(layer_dict, vmin, vmax, is_bipolar)`. |
| `build_velocity_layer(tf, time_idx, geom, origin, ...)` | Arrow layer showing velocity vectors. Auto-detects U/V variable pair. |
| `build_contour_layer_fn(tf, values, geom, origin, levels)` | Returns a factory function that produces `LineLayer` contour isolines at specified levels. |
| `build_marker_layer(x_m, y_m, layer_id)` | Single-point `ScatterplotLayer` marker. |
| `build_cross_section_layer(points_m, layer_id)` | `PathLayer` polyline for cross-section display. |
| `build_particle_layer(paths, current_time, trail_length, layer_id)` | `TripsLayer` for animated particle trails. |
| `build_wireframe_layer(tf, geom, origin)` | `LineLayer` showing mesh edges. |
| `build_extrema_markers(extrema, x_off, y_off, layer_id)` | `ScatterplotLayer` showing min/max value locations. |
| `build_measurement_layer(points_m, layer_id)` | `ScatterplotLayer` + `PathLayer` for distance measurement. |
| `build_boundary_layer(tf, geom, boundary_nodes, origin)` | `ScatterplotLayer` color-coded by boundary condition type from `.cli` file. |

---

## `analysis` — Spatial and Temporal Analysis

### Derived Variables

| Function | Signature | Description |
|----------|-----------|-------------|
| `get_available_derived` | `(tf) -> list[str]` | List derived variables available for this file (velocity magnitude, Froude, vorticity). |
| `compute_derived` | `(tf, varname, tidx) -> ndarray` | Compute a derived variable at a given timestep. |

### Spatial Analysis

| Function | Signature | Description |
|----------|-----------|-------------|
| `nearest_node` | `(tf, x_m, y_m) -> (idx, x, y)` | Find nearest mesh node to a point. |
| `time_series_at_point` | `(tf, varname, x_m, y_m) -> (times, values)` | Extract time series at a point. Raw variables use TELEMAC's built-in interpolation; derived variables (since 3.2.0) use barycentric interpolation within the enclosing triangle — consistent with the map layer. Falls back to nearest node outside the mesh. |
| `cross_section_profile` | `(tf, varname, record, polyline_m) -> (abscissa, values)` | Interpolate values along a polyline. |
| `vertical_profile_at_point` | `(tf, varname, tidx, x_m, y_m) -> (elevations, values, layer_name)` | Extract vertical profile for 3D results. |
| `polygon_zonal_stats` | `(tf, values, polygon_m) -> dict` | Compute min, max, mean, std, area within a polygon. |
| `find_boundary_nodes` | `(tf) -> list[int]` | Identify boundary nodes from mesh topology. |
| `find_boundary_edges` | `(tf) -> (keys, nodes_a, nodes_b)` | Find boundary edges with node indices. |
| `find_extrema` | `(tf, values) -> dict` | Locate min and max value positions on mesh. |
| `mesh_identity_hash` | `(tf) -> str` | 12-char SHA1 digest over `(x, y, ikle)`. Identical meshes produce identical hashes; any difference — including float noise from re-export — yields a different hash. Used by compare-overlay to reject non-identical geometry (added 3.2.0). |

### Discharge and Volume

| Function | Signature | Description |
|----------|-----------|-------------|
| `compute_discharge` | `(tf, tidx, polyline_m) -> dict` | Integrate flow rate (Q) across a polyline. Returns dict with `discharge`, `width`, `mean_depth`, `mean_velocity`. |
| `compute_mesh_integral` | `(tf, values, threshold) -> dict` | Area-weighted integral over mesh. Returns `integral`, `area`, `mean`. |
| `compute_volume_timeseries` | `(tf, compute_integral_fn) -> (times, volumes)` | Water volume at each timestep (in `validation.py`). Searches `WATER DEPTH` / `HAUTEUR D'EAU` / `WATER DEPTH M`; falls back to `FREE SURFACE − BOTTOM` (logged once) if no depth variable is present. Never treats `FREE SURFACE` alone as depth — that is bottom + depth, not depth (since 3.2.0). |

### Mesh Diagnostics

| Function | Signature | Description |
|----------|-----------|-------------|
| `compute_courant_number` | `(tf, tidx) -> ndarray \| None` | CFL number per vertex. Returns `None` if velocity variables missing. |
| `compute_element_area` | `(tf) -> ndarray` | Per-vertex average element area. |
| `compute_mesh_quality` | `(tf) -> ndarray` | Mesh quality (0-1) based on element aspect ratio. |
| `compute_slope` | `(tf, values) -> ndarray` | Gradient magnitude of a scalar field. |

### Flood Mapping

| Function | Signature | Description |
|----------|-----------|-------------|
| `compute_flood_envelope` | `(tf, varname, threshold)` | Maximum value over all timesteps (flood extent). |
| `compute_flood_arrival` | `(tf, varname, threshold)` | First timestep exceeding threshold at each node. |
| `compute_flood_duration` | `(tf, varname, threshold)` | Duration above threshold at each node. |

### Temporal Statistics

| Function | Signature | Description |
|----------|-----------|-------------|
| `compute_temporal_stats` | `(tf, varname) -> dict \| None` | Min, max, mean, std over all timesteps for one variable. |
| `compute_all_temporal_stats` | `(tf, variables_list) -> dict` | Single-pass computation of temporal stats for multiple variables. |

### Particle Tracing

| Function | Signature | Description |
|----------|-----------|-------------|
| `compute_particle_paths` | `(tf, seed_points, x_off, y_off) -> list` | Lagrangian particle trajectories through velocity field. |
| `generate_seed_grid` | `(tf, n_target) -> list` | Generate regular grid of seed points within mesh bounds. |
| `distribute_seeds_along_line` | `(polyline_m, n_seeds) -> list` | Distribute seed points evenly along a polyline. |

### Data I/O and Expressions

| Function | Signature | Description |
|----------|-----------|-------------|
| `evaluate_expression` | `(tf, tidx, expression) -> ndarray` | Evaluate safe math expression using variable names (AST-based, no eval). |
| `export_timeseries_csv` | `(times, values, varname) -> str` | Format time series as CSV string. |
| `export_crosssection_csv` | `(abscissa, values, varname) -> str` | Format cross-section as CSV string. |
| `export_all_variables_csv` | `(tf, tidx, x_m, y_m) -> str` | Export all variables at a point as CSV string. |
| `compute_difference` | `(tf, varname, tidx, ref_tidx) -> ndarray` | Compute difference between two timesteps. |
| `read_cli_file` | `(cli_path) -> dict[int, int] \| None` | Parse `.cli` boundary condition file. Returns node-to-type mapping. |
| `extract_layer_2d` | `(values_3d, npoin2, layer_k) -> ndarray` | Extract a 2D layer from 3D result array. |
| `find_cas_files` | `(example_path) -> dict[str, str]` | Locate `.cas`, `.cli`, `.liq` files associated with a result file. |
| `detect_module_from_path` | `(cas_path) -> str` | Identify TELEMAC module from `.cas` file path. |

---

## `constants` — Configuration and Palettes

| Name | Type | Description |
|------|------|-------------|
| `EXAMPLES` | `dict[str, str]` | Map of example name to file path (only existing files). |
| `EXAMPLE_GROUPS` | `dict[str, dict]` | Examples grouped by TELEMAC module. |
| `EXAMPLE_CHOICES` | `dict[str, dict]` | Grouped choices for UI dropdown (existing files only). |
| `PALETTES` | `dict[str, list]` | 10 named color palettes from shiny-deckgl. |
| `PALETTE_DIVERGING` | `list` | Blue-white-red diverging palette for difference mode. |
| `cached_palette_arr(palette_id, reverse)` | `-> ndarray` | LRU-cached 256-color array for per-vertex coloring. |
| `cached_gradient_colors(palette_id, reverse)` | `-> list` | Cached gradient color stops for legends. |
| `format_time(seconds)` | `-> str` | Human-readable time formatting (`"3 h 15 min"`). |

---

## `telemac_defaults` — Variable Semantics

| Function | Signature | Description |
|----------|-----------|-------------|
| `suggest_palette` | `(varname) -> str \| None` | Recommend palette by variable name (e.g., "WATER DEPTH" -> "Ocean"). |
| `is_bipolar` | `(varname) -> bool` | Check if variable has positive/negative semantics (e.g., bed evolution). |
| `detect_module_from_vars` | `(varnames) -> str` | Identify TELEMAC module from variable names. |
| `find_velocity_pair` | `(varnames) -> (u, v) \| None` | Find U/V velocity pair from variable list. |

---

## `validation` — Model Validation

| Function | Signature | Description |
|----------|-----------|-------------|
| `parse_observation_csv` | `(file_path) -> (times, values, varname)` | Read 2-column CSV with time and observed values. |
| `compute_rmse` | `(model, observed) -> float` | Root Mean Square Error. |
| `compute_nse` | `(model, observed) -> float` | Nash-Sutcliffe Efficiency (1.0 = perfect). |
| `compute_volume_timeseries` | `(tf, compute_integral_fn) -> (times, volumes)` | Water volume per timestep using area-weighted integration. |
| `parse_liq_file` | `(liq_path) -> dict \| None` | Parse TELEMAC `.liq` liquid boundary file. Returns column name -> `{times, values, unit}`. |

---

## `telemac_tools` — HEC-RAS Import Pipeline

### Top-level API

```python
def hecras_to_telemac(
    hecras_path: str,
    dem_path: str | None = None,
    output_dir: str = ".",
    name: str = "project",
    floodplain_width: float = 200.0,
    backend: str = "triangle",
    duration: float = 86400.0,
    cas_overrides: dict | None = None,
) -> None
```

End-to-end conversion from HEC-RAS `.hdf` to TELEMAC input files (`.slf`, `.cli`, `.cas`).

### `telemac_tools.model` — Data Model

| Class | Description |
|-------|-------------|
| `BCType` | Enum: `FLOW`, `STAGE`, `NORMAL_DEPTH`, `RATING_CURVE`, `UNKNOWN` |
| `LIHBOR` | IntEnum: `WALL=2`, `FREE=4`, `PRESCRIBED=5` |
| `CrossSection` | Station geometry with Manning's n and bank positions |
| `Reach` | Named river reach with alignment and cross-sections |
| `BoundaryCondition` | BC type, location, line coordinates, and timeseries |
| `HecRasCell` | 2D cell face point indices |
| `HecRas2DArea` | 2D area with face points, cell centers, elevation, Manning's n |
| `Mesh2D` | Triangle mesh with nodes, elements, elevation, Manning's n (validated on init) |
| `HecRasModel` | Complete parsed model: rivers, boundaries, 2D areas, CRS |
| `BCSegment` | Boundary segment with node indices and prescribed values |
| `TelemacDomain` | Domain definition: boundary polygon, refinement zones, BCs |
| `HecRasParseError` | Exception for parse failures |

### `telemac_tools.hecras` — HEC-RAS Parsers

| Function | Description |
|----------|-------------|
| `parse_hecras(path)` | Auto-detect 1D or 2D and parse HDF5 file. Returns `HecRasModel`. |
| `parse_hecras_1d(path)` | Parse 1D geometry (reaches, cross-sections, alignment). |
| `parse_hecras_2d(path)` | Parse 2D geometry (face points, cells, elevation). |
| `triangulate_2d_area(area)` | Convert 2D area Voronoi cells to triangle mesh. Returns `Mesh2D`. |
| `parse_bc_timeseries(path)` | Parse unsteady flow file for BC time series. |
| `extract_mannings_1d(path)` | Extract Manning's roughness from 1D cross-sections. |

### `telemac_tools.domain` — Domain Building

| Function | Description |
|----------|-------------|
| `build_domain_1d(model, dem_path, floodplain_width)` | Build TELEMAC domain from 1D model with DEM sampling and channel carving. |
| `build_domain_2d(model)` | Build TELEMAC domain from 2D model geometry. |

### `telemac_tools.meshing` — Mesh Generation

| Function | Description |
|----------|-------------|
| `generate_mesh(domain, backend, max_area)` | Generate triangle mesh. `backend` is `"triangle"` or `"gmsh"`. |

### `telemac_tools.telemac` — File Writers

| Function | Description |
|----------|-------------|
| `write_telemac(mesh, domain, output_dir, name, duration, cas_overrides)` | Write all TELEMAC input files. |
| `write_slf(...)` | Write SELAFIN binary file with mesh and initial conditions. |
| `write_cli(...)` | Write `.cli` boundary condition file. |
| `write_cas(...)` | Write `.cas` steering file. |
