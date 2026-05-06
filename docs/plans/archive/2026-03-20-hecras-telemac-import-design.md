# HEC-RAS → TELEMAC Import Tool — Design Specification

**Date:** 2026-03-20
**Status:** Draft (rev 2 — post spec review)
**Scope:** Sub-project 1 of TELEMAC Model Creation Tools

## 1. Purpose

Build a Python library (`telemac_tools`) and viewer integration that converts HEC-RAS 6.x models into ready-to-run TELEMAC-2D simulations. Supports both 1D→2D conversion (cross-sections + DEM → triangular mesh) and 2D→2D conversion (HEC-RAS 2D Voronoi mesh → TELEMAC triangular mesh).

## 2. Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| HEC-RAS versions | 6.x only (HDF5) | Clean structured format, `h5py` parsing, covers current standard |
| 1D→2D mesh strategy | DEM overlay + channel carving | Most practical for real projects; users almost always have a DEM |
| Meshing backends | Triangle (default) + Gmsh (optional) | Triangle is simple/fast for default; Gmsh for advanced control |
| Coordinate systems | Assume projected CRS (meters) | Avoids GDAL/pyproj dependency; HEC-RAS 6.x models are typically projected |
| Architecture | Layered pipeline | Parser → Domain Builder → Mesh Generator → TELEMAC Writer |
| Delivery | Standalone library + Viewer tab | Library is testable/scriptable; viewer is a thin UI consumer |
| HEC-RAS input file | Geometry HDF (`.g##.hdf`) | Geometry file contains mesh, cross-sections, Manning's; plan file has results only |

## 3. Architecture

```
HEC-RAS .g##.hdf ──► Parser ──► HecRasModel ──► Domain Builder ──► TelemacDomain ──► Mesh Generator ──► Mesh2D ──► TELEMAC Writer
                                                       ▲                                                              │
                                                 DEM (GeoTIFF)                                              .slf + .cli + .cas
```

Four layers, each independently testable with clear input/output contracts.

**Important:** The input is the HEC-RAS **geometry** HDF file (`.g##.hdf`), not the plan file (`.p##.hdf`). The geometry file contains cross-sections, 2D mesh structure, Manning's n, and boundary condition line geometry. BC flow/stage time series are out of scope for v1 (they live in separate `.u##` unsteady flow files or the plan HDF).

## 4. Package Structure

```
telemac_tools/
├── __init__.py                 # Convenience: hecras_to_telemac()
├── model.py                    # Intermediate data model (dataclasses)
├── hecras/
│   ├── __init__.py
│   ├── parser_1d.py            # 1D geometry from HDF5
│   ├── parser_2d.py            # 2D area mesh from HDF5 (Voronoi → triangles)
│   └── manning.py              # Manning's n region extraction
├── domain/
│   ├── __init__.py
│   ├── builder.py              # Combine geometry + DEM → domain
│   └── channel_carve.py        # Interpolate cross-sections into DEM
├── meshing/
│   ├── __init__.py
│   ├── base.py                 # Abstract mesher interface
│   ├── triangle_mesh.py        # Triangle backend
│   └── gmsh_mesh.py            # Gmsh backend
├── telemac/
│   ├── __init__.py
│   ├── writer_slf.py           # Write .slf mesh file (via Selafin)
│   ├── writer_cli.py           # Write .cli boundary conditions
│   └── writer_cas.py           # Write .cas steering file template
└── tests/
    ├── test_parser_1d.py
    ├── test_parser_2d.py
    ├── test_domain_builder.py
    ├── test_mesh_generator.py
    ├── test_writers.py
    └── fixtures/               # Small real HEC-RAS model files (from FEMA FFRD)
```

## 5. Intermediate Data Model

```python
@dataclass
class CrossSection:
    station: float                  # chainage along river (meters)
    coords: np.ndarray              # (N, 3) — x, y, z points in world coordinates
    mannings_n: list[float]         # [left_overbank, channel, right_overbank]
    bank_stations: tuple[float, float]  # left/right bank station positions (local)
    bank_coords: np.ndarray         # (2, 2) — left/right bank x, y in world coords

@dataclass
class Reach:
    name: str
    alignment: np.ndarray           # (M, 2) — river centerline x, y
    cross_sections: list[CrossSection]

@dataclass
class BoundaryCondition:
    bc_type: str                    # "flow", "stage", "normal_depth", "rating_curve"
    location: str                   # "upstream" / "downstream" / reach name
    line_coords: np.ndarray | None  # (L, 2) — BC line geometry in world coords

@dataclass
class HecRasCell:
    """Single HEC-RAS 2D cell (polygon with 3-8 sides)."""
    face_point_indices: list[int]   # indices into the face-points array

@dataclass
class HecRas2DArea:
    name: str
    face_points: np.ndarray         # (F, 2) — face point coordinates
    cell_centers: np.ndarray        # (C, 2) — cell center coordinates
    cells: list[HecRasCell]         # polygon cells (Voronoi)
    elevation: np.ndarray           # (C,) — bed elevation per cell center
    mannings_n_constant: float      # per-area constant (default 0.035)
    mannings_n_raster: str | None   # path to external land cover GeoTIFF (if any)

@dataclass
class Mesh2D:
    """Triangular mesh (output of triangulation)."""
    nodes: np.ndarray               # (N, 2) x, y coordinates
    elements: np.ndarray            # (E, 3) triangle connectivity (0-based)
    elevation: np.ndarray           # (N,) bed elevation per node
    mannings_n: np.ndarray          # (N,) Manning's n per node

@dataclass
class HecRasModel:
    rivers: list[Reach]             # 1D geometry
    boundaries: list[BoundaryCondition]
    areas_2d: list[HecRas2DArea]    # 2D areas (may be multiple)
    crs: str | None                 # reserved for future CRS reprojection support

@dataclass
class BCSegment:
    """Boundary condition segment on the TELEMAC domain boundary."""
    node_indices: list[int]         # boundary node indices
    lihbor: int                     # 2=wall, 4=free, 5=prescribed
    prescribed_h: float | None      # water level (for LIHBOR=5)
    prescribed_q: float | None      # unit discharge (for LIHBOR=5)

@dataclass
class TelemacDomain:
    boundary_polygon: np.ndarray    # (P, 2) outer domain boundary
    refinement_zones: list[dict]    # [{"polygon": ndarray, "max_area": float}]
    channel_points: np.ndarray      # (C, 3) carved channel bed points x, y, z
    channel_segments: np.ndarray    # (S, 2) thalweg segment indices for PSLG constraints
    mannings_regions: list[dict]    # [{"polygon": ndarray, "n": float}]
    bc_segments: list[BCSegment]
```

## 6. Layer 1 — HEC-RAS Parser

### HDF5 Structure (HEC-RAS 6.x Geometry File `.g##.hdf`)

**Note:** Paths must be verified against a real file before implementation. The paths below are based on the `rashdf` library (FEMA FFRD) and HEC-RAS documentation. Treat as best-effort until a fixture file is inspected with `h5py`.

**1D Geometry:**
```
Geometry/
  Cross Sections/
    Attributes                → compound dataset: river, reach, station
    Polyline Info              → offset/count into Polyline Points
    Polyline Points            → (x, y) station coordinates
    Station Elevation Info     → offset/count into Station Elevation Values
    Station Elevation Values   → (station, elevation) pairs per XS
    Manning's n Info           → offset/count into Manning's n Values
    Manning's n Values         → per-segment roughness
    Bank Stations              → left/right bank stations per XS
  River Centerlines/
    Attributes                → river/reach names
    Polyline Info              → offset/count
    Polyline Points            → (x, y) alignment
  Boundary Condition Lines/
    Attributes                → BC names, types
    Polyline Info              → offset/count
    Polyline Points            → (x, y) line geometry
```

**2D Geometry:**
```
Geometry/
  2D Flow Areas/
    <area_name>/
      Cell Points              → (C, 2) cell center coordinates
      FacePoints Coordinate    → (F, 2) face point coordinates
      Faces FacePoint Indexes  → face-to-facepoint linkage
      Cells Face and Orientation Info    → cell-to-face counts
      Cells Face and Orientation Values  → cell-to-face indices
      Cell Info                → per-cell metadata
    Attributes                → area names, cell size, Manning's n
```

**Terrain:** HEC-RAS typically stores terrain as an external GeoTIFF referenced via `Geometry.attrs['Terrain Filename']`, not as inline cell elevations. The parser should check for both inline and external terrain sources.

### 2D Mesh Conversion (Voronoi → Triangles)

HEC-RAS 2D uses a **Voronoi polygon mesh** (cells with 3–8 sides), not triangles. The parser must convert this:

1. Read face-point coordinates and cell-to-face connectivity
2. For each cell: reconstruct the polygon from its face points
3. Fan-triangulate each polygon from its centroid to its face points
4. Merge shared nodes at face boundaries
5. Interpolate elevation from cell centers to triangle vertices

This produces a `Mesh2D` with proper triangular connectivity for TELEMAC.

### API

```python
from telemac_tools.hecras import parse_hecras, parse_hecras_1d, parse_hecras_2d

model = parse_hecras("/path/to/project.g01.hdf")      # auto-detect 1D/2D
model = parse_hecras_1d("/path/to/project.g01.hdf")    # 1D only
model = parse_hecras_2d("/path/to/project.g01.hdf")    # 2D only
```

### Error Handling

- `HecRasParseError` with descriptive message if expected HDF5 groups/datasets are missing
- Warns on optional missing data (e.g., no Manning's n → defaults to 0.035)
- Validates HDF5 structure before extraction

### Dependencies

`h5py`, `numpy` only.

## 7. Layer 2 — Domain Builder

### 1D → 2D Workflow

1. **Load DEM** — Read GeoTIFF with `tifffile` (handles compressed TIFFs). Extract geotransform for coordinate mapping.
2. **Define domain boundary** — Buffer the river alignment by user-specified floodplain width (default 500m each side), or use a user-provided polygon.
3. **Channel carving** — Convert cross-section station/elevation profiles to world coordinates using the reach alignment and chainage. Interpolate bed elevations between cross-sections along the thalweg at configurable spacing (default 10m). **Carving is limited to the channel zone** (between bank stations in world coordinates) — overbank cross-section points do NOT override the DEM.
4. **Manning's n regions** — Create spatial polygons from cross-section bank station world coordinates:
   - Channel zone (between banks): uses HEC-RAS channel Manning's n
   - Left/right overbank zones: uses HEC-RAS overbank Manning's n
   - Floodplain beyond cross-sections: user-specified default (0.06)
5. **Refinement zones** — Channel corridor gets fine mesh (default max area 10 m²), floodplain gets coarse mesh (default 200 m²).
6. **Boundary segments** — Map HEC-RAS BC line geometry to domain boundary edges:
   - Upstream inflow → LIHBOR 5 (prescribed)
   - Downstream outflow → LIHBOR 4 (free/Neumann)
   - All other edges → LIHBOR 2 (wall)

### Station-to-World Coordinate Transform

Cross-section coordinates in HEC-RAS are stored as (station, elevation) pairs in the cross-section's local coordinate system. The world-coordinate positions are reconstructed by:

1. Finding the cross-section's intersection point on the reach alignment using its chainage
2. Computing the cross-section direction (perpendicular to the alignment at that point)
3. Offsetting each station point along the cross-section direction by (station - center_station) distance

This transform is implemented in `channel_carve.py`.

### 2D → 2D Workflow

1. Use the already-triangulated mesh from `parser_2d.py`
2. Identify boundary edges from mesh topology
3. Match boundary edges to HEC-RAS BC line geometry
4. For spatially varying Manning's n: if `mannings_n_raster` path exists, read it with `tifffile` and interpolate onto mesh nodes; otherwise use the constant value

### API

```python
from telemac_tools.domain import build_domain_1d, build_domain_2d

domain = build_domain_1d(
    model,
    dem_path="/path/to/dem.tif",
    floodplain_width=500.0,
    boundary_polygon=None,
    channel_refinement=10.0,
    floodplain_refinement=200.0,
)

domain = build_domain_2d(model)
```

### Dependencies

`numpy`, `scipy`, `tifffile`.

## 8. Layer 3 — Mesh Generator

### Interface

```python
class MeshBackend:
    def generate(self, domain: TelemacDomain) -> Mesh2D:
        ...
```

### Triangle Backend (Default)

1. Define PSLG from `domain.boundary_polygon`
2. Add channel thalweg as **constrained segment sequences** in the PSLG (not free points — this guarantees Triangle places nodes along the channel)
3. Add `domain.channel_points` as additional constrained vertices
4. Set per-region max area from `domain.refinement_zones`
5. Call `triangle.triangulate(pslg, "pq30a...")` — quality mesh with 30° minimum angle
6. Interpolate DEM elevation onto mesh nodes (bilinear from raster grid)
7. Override channel node elevations with carved cross-section values
8. Assign Manning's n per node via point-in-polygon against `domain.mannings_regions`

### Gmsh Backend (Optional)

1. Build geometry with `gmsh.model.occ` — boundary polygon, channel lines as embedded curves
2. Set mesh size fields: distance-based grading from channel
3. Generate with `gmsh.model.mesh.generate(2)`
4. Same elevation/Manning's interpolation

### For 2D Direct Import

Mesh is already triangulated by `parser_2d.py`. Generator validates element quality and optionally refines poor elements.

### API

```python
from telemac_tools.meshing import generate_mesh

mesh = generate_mesh(domain, backend="triangle", min_angle=30.0)
mesh = generate_mesh(domain, backend="gmsh", min_angle=30.0)
```

### Dependencies

- `triangle` (pip install triangle) — required
- `gmsh` (pip install gmsh) — optional, imported lazily

## 9. Layer 4 — TELEMAC Writer

### `.slf` Mesh File

Uses the existing `Selafin` class from the TELEMAC Python API.

Written variables:
- `BOTTOM` — bed elevation
- `FRICTION COEFFICIENT` — Manning's n per node
- `WATER DEPTH` — initial depth (default 0.1 m for wet-start stability)
- `FREE SURFACE` — initial water level (bottom + depth)

### `.cli` Boundary File

**Column format** (TELEMAC-2D standard, space-separated per boundary node):

```
LIHBOR LIUBOR LIVBOR HBOR UBOR VBOR AUBOR LITBOR TBOR ATBOR BTBOR N_GLOBAL K_BOUND
```

Where:
- Wall (LIHBOR=2): `2 0 0  0.0 0.0 0.0 0.0  0  0.0 0.0 0.0  <node>  <idx>`
- Free (LIHBOR=4): `4 4 4  0.0 0.0 0.0 0.0  4  0.0 0.0 0.0  <node>  <idx>`
- Prescribed (LIHBOR=5): `5 5 5  <H> 0.0 0.0 0.0  5  0.0 0.0 0.0  <node>  <idx>`

Steps:
1. Find boundary edges (edges in one element only)
2. Order into connected segments (walk the boundary ring)
3. Assign LIHBOR codes from `domain.bc_segments`
4. Write exact column format per TELEMAC-2D specification

### `.cas` Steering File

Template with safe defaults for cold-start stability:

```
/ TELEMAC-2D steering file — generated by telemac_tools
/
TITLE = 'Imported from HEC-RAS'
GEOMETRY FILE = project.slf
BOUNDARY CONDITIONS FILE = project.cli
RESULTS FILE = r2d_project.slf
/
EQUATIONS = 'SAINT-VENANT FV'
FINITE VOLUME SCHEME = 5
VARIABLE TIME-STEP = YES
DESIRED COURANT NUMBER = 0.8
/
LAW OF BOTTOM FRICTION = 4
/ Note: spatially varying Manning's n is read from BOTTOM FRICTION DATA FILE
/ or from FRICTION COEFFICIENT variable in the geometry .slf
/
INITIAL CONDITIONS = 'CONSTANT DEPTH'
INITIAL DEPTH = 0.1
/ Cold-start with 0.1m wet depth for numerical stability
/
TIDAL FLATS = YES
CONTINUITY CORRECTION = YES
TREATMENT OF NEGATIVE DEPTHS = 2
/
DURATION = 3600.
GRAPHIC PRINTOUT PERIOD = 60
VARIABLES FOR GRAPHIC PRINTOUTS = 'U,V,H,S,B'
```

User can override any parameter via `cas_overrides` dict.

### API

```python
from telemac_tools.telemac import write_telemac

write_telemac(
    mesh, domain,
    output_dir="/path/to/output/",
    name="project",
    time_step=1.0,
    duration=3600.0,
    scheme="finite_volume",
    cas_overrides={"TURBULENCE MODEL": 4},
)
```

## 10. Convenience Function

```python
from telemac_tools import hecras_to_telemac

hecras_to_telemac(
    hecras_path="/path/to/project.g01.hdf",
    dem_path="/path/to/dem.tif",
    output_dir="/path/to/output/",
    floodplain_width=500.0,
    backend="triangle",
    scheme="finite_volume",
)
# Produces: output/project.slf, output/project.cli, output/project.cas
```

## 11. Viewer Integration

### New "Import" Tab

Added after "Run" in `page_navbar`.

**Left panel (4 cols):** Import setup
- Source dropdown (HEC-RAS 6.x)
- HEC-RAS file upload (.hdf)
- DEM file upload (.tif)
- 1D options: floodplain width, channel/floodplain refinement
- Mesher dropdown (Triangle / Gmsh)
- Scheme dropdown (Finite Volume / Finite Element)
- Preview button, Convert button
- Download buttons for each output file (.slf, .cli, .cas)
- Status output

**Right panel (8 cols):** Preview map
- River alignment as `path_layer` (cyan)
- Cross-sections as `line_layer` (yellow)
- Domain boundary as `path_layer` (white dashed)
- Generated mesh preview as wireframe
- BC segments color-coded (wall=gray, inflow=blue, outflow=green)

**Workflow:**
1. Upload HEC-RAS `.g##.hdf` + DEM `.tif`
2. Click **Preview** → parser runs, geometry renders on map
3. Adjust parameters
4. Click **Convert** → full pipeline runs
5. Download output files via download buttons
6. Auto-load output `.slf` into Map tab (local deployment only)

**Implementation:** Thin UI calling `telemac_tools` library. Download handlers via Shiny `download_button` + `download_handler`.

## 12. Dependencies

| Package | Required | Purpose |
|---------|----------|---------|
| `numpy` | Yes | Array operations |
| `scipy` | Yes | DEM interpolation, spatial queries |
| `h5py` | Yes | HEC-RAS HDF5 parsing |
| `tifffile` | Yes | GeoTIFF reading (handles compression) |
| `triangle` | Yes | Default meshing backend |
| `gmsh` | Optional | Advanced meshing backend |

No GDAL, rasterio, or pyproj required.

## 13. Testing Strategy

- **Fixtures:** Use small real HEC-RAS models from FEMA FFRD public datasets or generate minimal models in HEC-RAS. Synthetic HDF5 fixtures are impractical due to complex compound dataset structure.
- **Unit tests per layer:** Parser, domain builder, mesh generator, writer each tested independently
- **Integration test:** Full pipeline from HEC-RAS fixture to validated TELEMAC files
- **Mesh quality:** Verify minimum angle > 25°, no degenerate elements, boundary node ordering
- **Round-trip:** Write `.slf` → read back with `TelemacFile` → verify coordinates/variables match
- **`.cli` validation:** Compare generated `.cli` against reference files from TELEMAC examples

## 14. Known Limitations (v1)

- HEC-RAS versions 4.x/5.x ASCII format not supported (HDF5 only)
- No automatic CRS reprojection — user must ensure projected coordinates
- BC flow/stage time series not imported (require `.u##` unsteady flow file — future work)
- Spatially varying Manning's n from external GeoTIFF is optional (requires `tifffile`)
- Single 2D flow area processed at a time; multi-area support is partial (processes first area, warns on others)
- Viewer auto-load is local deployment only; download buttons work in all deployments

## 15. References

- [rashdf: Read data from HEC-RAS HDF files](https://github.com/fema-ffrd/rashdf) — FEMA FFRD
- [HEC-RAS 2D Output File (HDF5)](https://www.hec.usace.army.mil/confluence/rasdocs/r2dum/6.2/viewing-2d-or-1d-2d-output-using-hec-ras-mapper/2d-output-file-hdf5-binary-file)
- [pyHMT2D RAS_2D_Data.py](https://github.com/psu-efd/pyHMT2D) — HEC-RAS 2D mesh parsing reference
- [TELEMAC-2D User Manual](https://www.opentelemac.org/downloads/MANUALS/TELEMAC-2D/)
- [raspy: Python automation for HEC-RAS](https://github.com/quantum-dan/raspy) — COM-based, Windows-only (not used, but referenced)
