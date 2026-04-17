# Changelog

All notable changes to the TELEMAC Viewer are documented in this file.

## [Unreleased]

## [3.2.0] - 2026-04-17

### Added
- **Mesh identity hash** (`mesh_identity_hash`) for compare-overlay: SHA1 over node coordinates and connectivity rejects files whose mesh differs beyond count, preventing silent rendering of file B values on file A geometry.
- **Barycentric point sampling for derived variables** in `time_series_at_point`: computes the derived field (e.g. VELOCITY MAGNITUDE) at all mesh nodes per timestep, then barycentric-interpolates at the point — matching map-layer semantics. Falls back to nearest node when the point lies outside the mesh.
- **Volume fallback** `depth = FREE SURFACE − BOTTOM`: when no water-depth variable is present, `compute_volume_timeseries` now computes depth from available FS and BOTTOM variables (logged once). Previously FREE SURFACE was incorrectly used as depth directly.
- **French depth variable fallback** (`HAUTEUR D'EAU`) for volume conservation on French TELEMAC outputs (Round 11).
- **`.liq` liquid boundary file writer** for the HEC-RAS import pipeline.
- **`release.py` CLI** for version bumping, commit parsing, and git tagging (supersedes legacy `bump_version.py`).

### Fixed
- FREE SURFACE no longer treated as water depth in volume computations or polygon flooded-fraction stats.
- `compute_flood_duration` no longer double-counts the last interval (Round 11).
- Parser bounds and OOB guards across HEC-RAS 1D/2D parsers (Rounds 7, 9, 12): `n_fp` undefined fix in `parser_2d`, 1D CFPI path bounds check, `fp < n_fp` in faces reconstruction, required-dataset validation, polyline offset/count bounds.
- Alignment guard against fewer-than-2 points; cross-section profile guard against empty/sparse polylines (Round 13).
- Apostrophes in `.cas` string values now properly escaped (Round 13).
- NaN prescribed head replaced with 0.0 in `.cli` writer to avoid invalid output (Round 12).
- Filter UI handles infinite values gracefully — falls back to 0–1 (Round 12).
- Mesh edge cases hardened: empty mesh, all-NaN extrema, empty time-series, empty polygon zonal stats (Round 8).
- Notification leaks, empty-dict guards, ConvexHull OOB, out-of-bounds filter behavior (Round 7).
- Path-traversal fix in file-selection; element-area-weighted flooded stats; list-growth cap (Round 6).
- `writer_cas` now skips `None` overrides and auto-quotes unquoted string values (Round 9).
- Rounds 2–5 also addressed 48 additional issues across the viewer (detail preserved in commit history).

### Changed
- `_silent_div` / `_silent_mod` / `_silent_floordiv` wrap `np.errstate` so user expressions like `Q / 0` yield `inf` silently instead of emitting `RuntimeWarning`.
- `polygon_zonal_stats` silences expected all-NaN and empty-slice warnings where the existing fallback is correct.
- `build_mesh_layer` silences expected all-NaN slice warning when coloring inactive/empty meshes.
- `bump_version.py` removed (superseded by `release.py`).

### Tests
- Added `tests/test_round14_review_fixes.py`: 8 tests covering mesh identity hash, volume FS/BOTTOM fallback, and barycentric-vs-nearest derived point sampling.
- Total suite: **429 tests**. `pytest -W error::RuntimeWarning` is now clean apart from one external `shinywidgets` DeprecationWarning.

## [3.1.0] - 2026-04-09

### Refactored
- Extract server logic into dedicated modules: `server_core.py`, `server_analysis.py`, `server_playback.py`, `server_simulation.py`, `server_import.py`
- Extract helpers from `update_map()` and chart mode dispatch
- Deduplicate edge-encoding between `analysis.py` and `layers.py`
- Remove dead `geom` parameter from `polygon_zonal_stats`
- Minor cleanups: rename collisions, hoist imports, dict lookups

### Added
- `compute_all_temporal_stats()` for single-pass temporal computation
- `BCType` enum and `LIHBOR` IntEnum for boundary condition types
- `Mesh2D.__post_init__` validation (node/element consistency)
- `TelemacFileProtocol` for structural typing of TELEMAC file objects
- `MeshGeometry` frozen dataclass with field validation
- `CRS` frozen dataclass with `__post_init__` validation

### Fixed
- Register playback handlers and clean up unused imports after module extraction
- Wrap `resolve_crs` text updates in `reactive.isolate()`
- Skip timesteps instead of appending 0.0 in rating curve
- Notify user when 3D layer extraction fails
- Narrow 20 bare `except Exception` blocks to specific exception types
- Add `finally` block for particle notification cleanup
- Close previous `compare_tf` before replacing
- Clean up temp directory from import conversion on error
- Make `handle_import_preview` async instead of `ensure_future`
- Move side-effects out of `tel_file()` reactive.calc
- Initialize `elev_label` before loop to prevent `NameError`
- Log warning when `_sanitize_result` replaces NaN/Inf values
- `generate_seed_grid` returns empty list on triangulation failure
- Check `use_upload` flag when determining `current_path` in `update_map`
- Chained comparisons in expression parser
- Destructure `nearest_node` tuple in `coord_readout_ui`

### Tests
- Add expression parser, `extract_layer_2d`, and discharge tests
- Add `build_mesh_layer` edge cases and boundary layer `bc_types` tests
- Add tests for `constants` and `telemac_defaults` modules
- Add coverage for `read_cli_file` and `polygon_zonal_stats`
- Add coverage for flood envelope, arrival, and duration
- Parametrize origin tests across all 10 layer builders

## [3.0.0] - 2026-03-20

Major rewrite with modular architecture.

### Added
- **CRS awareness**: auto-detection from `.cas` files, coordinate transforms via pyproj, basemap alignment
- **HEC-RAS import pipeline**: parse 1D/2D HDF5 models, build domains, generate meshes, write SELAFIN/CLI/CAS
- **Import tab** in UI with preview map showing alignment, cross-sections, and boundaries
- **Gmsh mesh generation** backend (alternative to Triangle)
- **BC time series parser** from HEC-RAS unsteady flow files
- **Manual origin offset** for pre-centered meshes
- **UI redesign** with tabbed layout, progressive disclosure sidebar (Tier 1 always visible, Tier 2 collapsed)
- **Contextual UI** — `compare_upload_ui` and `crs_offset_ui` render functions
- **Basemap selector** (CartoDB Dark, Satellite, Light)

### Changed
- Upgraded to shiny-deckgl 1.0.1 with binary-encoded mesh transport
- Sidebar reorganized with progressive disclosure

## [2.5.0] - 2026-03-19

### Added
- **Validation module**: observation CSV parsing, RMSE, Nash-Sutcliffe Efficiency
- **Volume conservation** tracking over time
- **3D layer extraction** for TELEMAC-3D results
- **Animation playback** controls with adjustable speed
- **Overlay layers**: wireframe, velocity arrows, contour isolines
- **Polygon zonal statistics** (draw polygon, get min/max/mean/std/area)
- **`.liq` liquid boundary file** parser
- **Curonian Lagoon** case study added to examples

### Changed
- Smart defaults: auto-detect TELEMAC module from variables, suggest palettes, resolve velocity pairs

## [2.0.0] - 2026-03-15

### Added
- **Tier 9 hydraulic engineering tools**: discharge computation, Courant number, mesh quality, slope analysis
- **Flood mapping**: envelope (max depth), arrival time, duration above threshold
- **Particle tracing**: Lagrangian trajectories with seed grids and animated trails
- **FEM isocontouring** with configurable levels
- **Batch particle tracing** with line-distributed seeds
- **Custom expression evaluator** with safe AST-based parsing
- **Temporal statistics**: min/max/mean/std over all timesteps
- **Rating curve** computation
- **Boundary node** visualization with `.cli` file integration
- **Min/max extrema** markers on map
- **Measurement tool** with multi-point distance

### Changed
- Vectorized hot paths for performance
- Secure expression evaluation (AST whitelist, no exec/eval)
- Thread safety for concurrent reactive calculations
- NaN/Inf sanitization in all result arrays

## [1.0.0] - 2026-03-06

### Added
- **Derived variables**: velocity magnitude, Froude number, vorticity
- **File inspector** panel with mesh statistics
- **Multi-point time series** with CSV export
- **Cross-section profiles** along user-drawn polylines
- **Dark mode** toggle
- **Difference mode** between timesteps with diverging color palette

### Changed
- Modular architecture: separate `geometry.py`, `layers.py`, `analysis.py`, `constants.py`
- Expanded examples from 4 to 30+ across 6 TELEMAC modules (2D, 3D, GAIA, TOMAWAC, ARTEMIS, KHIONE)

## [0.2.0] - 2026-03-05

### Added
- Performance improvements (v2): faster mesh rendering, lazy data loading

### Fixed
- Safe input validation when switching between SELAFIN files

## [0.1.0] - 2026-03-05

Initial release.

### Added
- TELEMAC SELAFIN file viewer with deck.gl mesh rendering
- Variable selector and timestep slider
- Per-vertex coloring with configurable palettes
- Map-click point probing with coordinate readout
- Upload custom `.slf` files or select from built-in examples
