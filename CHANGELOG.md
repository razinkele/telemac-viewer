# Changelog

All notable changes to the TELEMAC Viewer are documented in this file.

## [Unreleased]

## [3.4.4] - 2026-04-25

### Performance
- **Velocity arrow layer cached** via a new `velocity_layer_cached` `@reactive.calc`. Closes the cache-on-palette-change pattern for every overlay — every dynamic overlay (mesh, contour, compare-contour, particle, velocity) now follows the same `*_layer_cached` reactive.calc + fast-path emission shape. Velocity colors are hardcoded so palette changes don't invalidate; only `vectors` toggle, `tel_file`, `current_tidx`, and `mesh_geom` do. The unused `build_velocity_patch` import is dropped from `app.py`; the helper stays exported for downstream consumers and remains covered by the wire-format contract tests.

### Tests
- **Extracted the remaining 6 chart builders** (begun in v3.3.1 with timeseries + crosssection). Pure module-level functions for `build_vertprofile_chart`, `build_histogram_chart`, `build_multivar_chart`, `build_rating_chart`, `build_volume_chart`, `build_boundary_ts_chart`. Each closure inside `register_analysis_handlers` reduces to a 1-3 line shim. `build_rating_chart` returns `(fig, skipped, ntimes)` so the shim performs the >50%-skipped notification side-effect; tests assert the count directly.
- 12 new tests in `tests/test_chart_builders.py` (empty-input + populated-output per builder). **509 tests total** (up from 497).

## [3.4.3] - 2026-04-25

### Fixed
- **Particle-trips animation now updates on timestep scrub.** The fast-path `update_map` (taken when the structural signature is unchanged) previously didn't emit any particle-layer patch. The deck.gl `TripsLayer` reads `currentTime` from the layer dict, so without a per-tick patch the trail animation froze at whatever `currentTime` the last full update set. Same class of bug as v3.3.4's compare-contour stale-on-scrub fix, just on a different overlay.

### Performance
- **Cached particle layer** via a new `particle_layer_cached` `@reactive.calc`. Palette changes / log-scale toggles / diagnostic flips no longer rebuild the trips-layer dict — only real deps invalidate (particles toggle, particle_paths re-seed, tidx, trail length, mesh_geom). Closes the cache-on-palette-change pattern for the third (and last user-visible) overlay type.

## [3.4.2] - 2026-04-25

### Added
- **`.liq` upload companion support** completes the upload-companions trio (`.cas` shipped in v3.4.0, `.cli` in v3.4.1, `.liq` here). `liq_data()` in `server_analysis.py` now scans the upload batch for a `.liq` file via `_find_uploaded_by_ext` and parses it through the existing `parse_liq_file`, restoring boundary-time-series overlays for uploaded files. Without it, the boundary-TS analysis chart was always empty for uploads.
- The upload widget's `accept` list now spans `[.slf, .cas, .cli, .liq]` and the label reads "Or upload .slf (+ optional .cas / .cli / .liq companions)".

### Changed
- The `_reset_state_on_new_file` upload notification refactored to a small table-driven enumeration of `(ext, label)` companion pairs. Three states: all three companions present → "all companion features available"; some present → list active + remaining; none → single hint listing all three.

## [3.4.1] - 2026-04-25

### Added
- **`.cli` upload companion support.** Extends v3.4.0's multi-file upload to recognize a `.cli` boundary-conditions file alongside the `.slf`. Uploaded files previously rendered every boundary edge as "Wall (inferred)" because `cli_data()` was hard-coded to return `None` for uploads; now it scans the upload batch via `_find_uploaded_by_ext(uploaded, ".cli")` and parses through the existing `read_cli_file`, producing the same wall / free-Neumann / prescribed-H/Q color coding example files have always had.
- The upload widget's `accept` list expands to `[.slf, .cas, .cli]` and the label updates to "Or upload .slf (+ optional .cas / .cli companions)".

### Changed
- The `_reset_state_on_new_file` upload notification is now three-way: lists active companions when any are present, lists missing ones with their unlocked features, and falls back to a single hint when only the bare `.slf` was uploaded.

### Total suite
**497 tests** (unchanged — the new wiring is reactive-scope-only; the underlying `_find_uploaded_by_ext` and `read_cli_file` helpers are already covered).

## [3.4.0] - 2026-04-24

### Added
- **CRS auto-detection for uploaded files.** The upload widget now accepts `.slf` + optional `.cas` together (`multiple=True`). When the user uploads both, the CRS resolver scans the `.cas` file's `GEOGRAPHIC SYSTEM` keyword just like it does for example files — previously uploads were hard-coded to skip `.cas`-based detection. The label updates to "Or upload .slf (+ optional .cas for CRS auto-detect)" and the upload-notification splits into two branches: "`.cas` CRS detection active" when a companion file is found, or the original "companion features unavailable" hint (now with an actionable "upload a `.cas` alongside the `.slf`" suggestion) when not.
- `server_core._find_uploaded_by_ext(uploaded, ext)` helper — case-insensitive extension match on the first matching entry in Shiny's upload list. Used both for picking the `.slf` among a multi-file upload and for spotting the companion `.cas`.

### Changed
- `_pick_file_path` and the `tel_file()` reactive calc now pick the `.slf` explicitly via `_find_uploaded_by_ext` rather than trusting `uploaded[0]`, which could be a `.cas` if the user dragged files in a different order.

### Tests
- 7 new tests: 2 in `TestPickFilePath` (multi-file picks `.slf` not position 0, case-insensitive) + 5 in new `TestFindUploadedByExt` class (None handling, first-match by ext, case insensitivity, missing-ext, missing-name graceful). Total suite now **497 tests**.

## [3.3.4] - 2026-04-24

### Fixed
- **Compare-contour overlay now refreshes on timestep scrub.** The fast-path `update_map` (taken when the structural signature is unchanged) previously emitted only the primary-contour patch. With a compare variable active, scrubbing tidx left the compare-contour layer showing the previous tidx's iso-values. The fast path now calls `compare_contour_layer_cached()` and emits the patch when it's non-`None`, keeping the compare overlay in sync.

### Performance
- **Cached compare-contour overlay** via a new `compare_contour_layer_cached` `@reactive.calc`. Same rationale as the primary `contour_layer_cached` shipped in v3.3.3 — palette / log-scale / diagnostic flips produce cache hits because the colors are hardcoded. Only the real deps (compare_var, compare_tf, tidx, tel_file, mesh_geom) invalidate.

## [3.3.3] - 2026-04-24

### Performance
- **Cached primary contour layer** via a new `contour_layer_cached` `@reactive.calc` matching the existing `wireframe_layer_cached` / `boundary_layers_cached` pattern. `build_contour_layer_fn` runs marching-triangles over the whole mesh on every call (~20-50 ms CPU on a 100k-triangle mesh × 6 thresholds). The previous code re-ran it on every reactive fire, including palette changes that don't actually affect contour appearance (colors are hardcoded). Now the rebuild only fires when the mesh, the toggle, or the underlying values change — palette changes, log-scale toggles, and any other non-contour-affecting reactive input produce a cache hit. Both the full-path `_build_overlay_layers` and the fast-path `update_map` branch route through the same cache.

## [3.3.2] - 2026-04-24

### Performance
- **Binary-encoded overlay layers**: `build_velocity_layer` and `build_contour_layer_fn` now emit their deck.gl `LineLayer` in binary-attribute mode (`data={length: N}` with `getSourcePosition`/`getTargetPosition` as `encode_binary_attribute(Float32Array)`) instead of a list of `{sourcePosition, targetPosition}` dicts (~85 bytes each).
  - Velocity: ~63 KB → ~12 KB per tick on a Malpasset-scale mesh (~750 arrows). **5.3× smaller.**
  - Contour: ~101 KB → ~19 KB per tick on the same mesh (~1200 segments). **5.3× smaller.**
  - Combined per-tick overhead when both overlays are enabled drops from ~480 KB to ~347 KB — roughly **28% smaller WebSocket payload** with overlays on.
- Finishes the deliberately-pass-through velocity and contour patch helpers from v3.3.0; the `build_velocity_patch` / `build_contour_patch` wrappers unchanged (they continue to delegate to the full builders).

### Tests
- 4 new wire-format tests pinning the binary-attribute contract for both velocity and contour layers, including two end-to-end assertions that drive `MapWidget.partial_update` through a `FakeSession` and verify the emitted `deck_partial_update` payload shape.
- 2 existing contour tests updated from `len(result["data"])` to `result["data"]["length"]` for the new `{length: N}` shape.

### Total suite
**490 tests** (up from 484, +6), all passing under `pytest`.

## [3.3.1] - 2026-04-24

### Changed
- **Extracted `_pick_file_path` helper** in `server_core.py` — factors the `uploaded[0]["datapath"] if ... else EXAMPLES.get(input.example(), "")` pattern from 2 inlined sites in `app.py` (`_structural_sig`, `update_map` full-path branch) into one reusable helper. `tel_file()` keeps its existing `EXAMPLES[k]` direct indexing (strict semantics surface missing keys with a clearer traceback than letting `TelemacFile("")` fail downstream).

### Tests
- **Extracted `_resolve_crs_from_inputs`** (`server_core.py`) as a pure function covering all 4 CRS decision branches (manual EPSG > auto-disabled > `.cas` scan > coord heuristic > None). The reactive `resolve_crs` effect becomes a thin shim that drives the 3 side-effects (`current_crs.set`, `ui.notification_show`, `ui.update_text`). 9 new tests pin the contract.
- **Extracted `build_timeseries_chart` + `build_crosssection_chart`** (`server_analysis.py`) as module-level pure functions, reachable from tests without a Shiny session. The reactive state (`clicked_points.get()`, `obs_data.get()`, `cross_section_points.get()`) becomes explicit kwargs. The RMSE/NSE observation-overlay block is preserved verbatim. 6 new tests. Remaining 6 chart builders stay as closures (extract-on-demand).
- **Promoted `FakeSession`** to `tests/helpers.py` (from `tests/test_map_dispatch.py`'s private `RecordingSession`). Adds `messages_of_type()` and `clear()` helpers. `tests/conftest.py` gains a `fake_session` fixture.

### Total suite
**484 tests** (up from 464, +20), all passing under `pytest`. Coverage of the two reactive modules stays at ~10% line coverage overall — the extracted pure helpers are covered at ~100%, but the reactive-handler bodies (80-90% of each file) remain uncovered because `pytest` doesn't boot Shiny. Full integration coverage would need either Playwright or a reactive-runtime mock; neither was in scope.

## [3.3.0] - 2026-04-24

### Added
- **Full-vs-partial map update dispatcher**: when only palette, timestep, or values change (no structural changes), the viewer sends a mesh color-buffer patch via `map_widget.partial_update` + `set_widgets` instead of a full `update`. For a ~100k-node mesh this drops the per-tick WebSocket payload roughly 3-4× (positions + indices are preserved by deck.gl's JS-side cache).
- `build_mesh_color_patch()` in `layers.py` — produces a colors-only patch dict ( `id` + `_meshColors` only; positions/indices/coordinate-system deliberately omitted so the cache preserves them).
- `build_velocity_patch()` and `build_contour_patch()` in `layers.py` — pass-through wrappers today; seams ready for sparse attribute updates later without touching `app.py`.
- `app_dispatch.decide_dispatch(prev_sig, curr_sig)` — pure decision function, unit-testable outside the Shiny reactive scope.
- Default map background switched from dark to light; dark option remains in the Background dropdown.

### Changed
- **`_structural_sig` reactive calc** (`app.py`): collects every input that forces a full map rebuild (file, 7 layer toggles, 3D mode, basemap, compare-var, user-drawn overlay counts) into one hashable tuple. Consumed by the full-vs-partial dispatcher.
- **Import-tab map widget** (`server_import.py`): moved to module scope and embedded via `import_map_widget.ui(...)` directly, replacing the broken `shinywidgets.render_widget` wrapper (`MapWidget` is a native Shiny component, not an ipywidget).
- **Basemap styles and tooltip config** factored into `constants.py` (`MAP_BG_DARK`, `MAP_BG_LIGHT`, `BASEMAP_STYLES`, `MAP_TOOLTIP`, `solid_bg_style()`) — previously duplicated as URL-encoded data-URI strings in three places.
- **Static-layer reactive caches** (`app.py`): `wireframe_layer_cached()` and `boundary_layers_cached()` now memoized via `@reactive.calc`, so `compute_unique_edges()` and `find_boundary_edges()` no longer re-run on every timestep scrub or palette change — only on file load or toggle flip.

### Fixed
- **Perpetual ".cli not found" notification** (`app.py`): previously re-emitted every ~4 seconds during any reactive tick when boundary overlays were on, and permanently for uploaded files (where `cli_data()` is hard-coded to `None`). Now gated on a reactive flag reset per file load, and skipped entirely for uploaded files (where the existing `upload_notice` already explains the absence of companion files).
- **Silent `TelemacFile.close()` failures** leaked file descriptors across example switches. Centralised via `server_core._safe_close(tf, context)` which logs through `_logger.warning(..., exc_info=True)`. Three call sites updated (two in `server_core.py`, one in `server_analysis.py`).
- **Invalid EPSG input silently disabled basemap alignment** (`server_core.resolve_crs`). Typos like `"4326x"` or non-existent codes like `99999` now raise a user-visible warning notification with the offending text and the underlying `pyproj.CRSError`. Narrowed `except Exception` to `(ValueError, pyproj.exceptions.CRSError)` so `KeyboardInterrupt` / `SystemExit` propagate.
- **3D view silently fell back to a flat (z=0) plate** when the z-variable lookup failed, misleadingly suggesting flat bathymetry. `server_core.mesh_geom` now switches back to 2D via `is_3d_mode.set(False)` and raises a clear warning.
- **Simulation module name whitelisted** (`server_simulation.py`): an unknown module string (from a malformed `.cas` path) could launch `python3 .py case.cas` and hang for 120 s waiting on stdin. Added `_ALLOWED_TELEMAC_MODULES = frozenset({...})` validation; unknown names abort with a clear error notification.
- **`evaluate_expression` error handling narrowed** (`server_analysis.py`) from `except Exception` to `(ValueError, TypeError, ArithmeticError, SyntaxError, KeyError, np.linalg.LinAlgError)` so `MemoryError` and `KeyboardInterrupt` propagate.
- **HEC-RAS import handlers narrowed** (`server_import.py`): both `import_preview` and `import_convert` previously caught `Exception`, masking `KeyboardInterrupt` and `MemoryError`. Restricted to `(OSError, ValueError, KeyError, RuntimeError)` with `_logger.exception()` so tracebacks reach server logs even when the in-UI log panel is dismissed.
- **`read_cli_file` and `parse_liq_file` None-returns logged** (`analysis.py`, `validation.py`): each early-return (`OSError`, too-few-lines, no valid data rows) now logs the path and reason so operators can distinguish "file missing" from "file unreadable" from "file truncated".
- **`tel_file()` open-before-close** (`server_core.py`): previously closed the old file before trying to open the new one; if the new open failed, the reactive chain halted with no file at all. Now opens new first, then closes old only on success.
- **`cas_file` init-race catch logged** (`server_simulation.py`): `_logger.debug` now distinguishes "widget not yet rendered" from "user didn't pick a file" even though the user-facing message stays the same.
- **Degenerate geometry guards** (`analysis.py`, `layers.py`): replaced `+1e-30` divide-by-zero tricks in circumradius and contour edge-crossing math with explicit `np.where(denom > eps, ..., np.nan)` masks. Collinear triangles now trigger `_sanitize_result`'s warning log; previously they were hidden as tiny-but-finite values that slipped through.
- **Log-scale on uniform-valued mesh** (`layers.py`): detect `vmin == vmax` and skip the log branch instead of producing a silently all-black mesh. Existing `"use_log and not log_applied"` notification tells the user.
- **Rating-curve skipped-timestep warning** (`server_analysis.py`): notify when more than half the timesteps in the h-Q curve lack FREE SURFACE / WATER DEPTH data.
- **Particle-tracing empty-result notification** (`server_analysis.py`): warn when `compute_particle_paths` returns an empty list (seed grid empty, all seeds outside mesh) instead of silently clearing the particle overlay.
- **Diverging-palette all-zero notification** (`app.py`): when a bipolar/diff-mode palette is active on an all-zero field, the map renders as a uniform midpoint color — now flagged with a message-level notification so the user doesn't interpret uniform color as real structure.
- **`release.py` git subprocess stderr captured**: previously users running `release.py` got a Python traceback but not the actual git error (e.g. pre-commit hook rejection). New `_run_git(args)` helper uses `capture_output=True` and raises `RuntimeError` with both stdout and stderr in the message.
- **`polygon_zonal_stats` returns `None` for empty intersection** (`analysis.py`): a zero-dict (`area=0`, `mean=0`) was indistinguishable from a polygon over genuinely zero values. Now returns `None`; the handler renders an explicit "no mesh nodes inside polygon" notification.

### Tests
- `TestWireFormatContract` in `tests/test_map_dispatch.py` pins the wire shape of `deck_partial_update` (only `_meshColors`, no `_meshPositions`/`_meshIndices`) and `deck_set_widgets`. Guards against accidentally re-adding position keys to the mesh color patch.
- `TestParseObservationCSV` in `tests/test_validation.py` covers empty file, malformed row, and valid roundtrip — locks in the current `(times, values, varname)` 3-tuple contract.
- `TestEvaluateExpressionCorrectness` and `TestEvaluateExpressionSecurity` in `tests/test_analysis.py` lock in arithmetic behaviour and verify `__import__('os').system(...)` and dunder-attribute access are rejected by the AST sandbox.

### Documentation
- README.md modernized (cover image, architecture notes, clearer quick-start).

### Tier-0 chores (bundled)
- Removed unused `from constants import _M2D` in `analysis.py`.
- Moved `validation.py::_logger` declaration to module top; added `from __future__ import annotations`.
- Renamed `time_idx` → `tidx` in `build_velocity_layer`/`build_velocity_patch` for consistency with the rest of the codebase (`analysis.py`, `server_core.py`, protocol, tests).
- Collapsed duplicate comment block above `_COORD_METER_OFFSETS` in `layers.py`.
- Added `id=` to every remaining `ui.notification_show` call in `server_core.py`; wrapped bare `input.X()` reads in `server_playback.py` with the codebase-standard `try/except (TypeError, AttributeError, KeyError)` pattern; added notification when playback speed is clamped.

### Total suite
**464 tests** (up from 429 at the previous release, +35), all passing under `pytest`.

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
