# Sidebar Progressive Disclosure — Design Specification

**Date:** 2026-03-20
**Status:** Draft (rev 2 — post review)

## 1. Purpose

Reduce sidebar cognitive load by showing only essential controls by default. The current sidebar has ~25 controls in 3 expanded accordions, requiring scrolling to find common actions. The redesign uses progressive disclosure: essential controls always visible, everything else collapsed.

## 2. Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Tier 1 placement | Bare elements, no accordion | Core workflow controls should never be hidden |
| Accordion default state | All collapsed via `ui.accordion(..., open=False)` | User expands only what they need |
| Basemap location | Tier 1 (always visible) | Used frequently enough to justify top-level placement |
| Time step slider | Tier 1 (always visible) | Primary interaction for temporal data |
| Playback advanced controls | Collapsed accordion | Speed, loop, go-to-time are secondary to the slider |
| Compare file upload | Contextual (diff mode only) | No reason to show upload when diff mode is off |
| CRS origin offset | Contextual (EPSG set only) | Offset is meaningless without a CRS code |
| Input IDs | Unchanged | Preserves all existing server wiring |

## 3. Layout

### Tier 1 — Always Visible

`ui.h6("TELEMAC Viewer", ...)` title stays at the very top (unchanged).

Bare elements below the title (no accordion wrapper):

1. `input_select("example", ...)` — Example case dropdown
2. `input_file("upload", ...)` — Upload .slf file
3. `output_ui("clear_upload_ui")` — Clear upload button (when file uploaded)
4. `input_select("basemap", ...)` — Background/basemap selector
5. `output_ui("var_select_ui")` — Variable dropdown (rendered dynamically)
6. `output_ui("time_slider_ui")` — Time step slider (rendered dynamically)

### Tier 2 — Collapsed Accordions

Four accordions below Tier 1, all collapsed via `ui.accordion(..., open=False)`:

**Note on Shiny API:** Use `ui.accordion(..., open=False)` at the accordion level to collapse all panels. Do NOT use `open=False` on individual `ui.accordion_panel()` — it doesn't accept that parameter in Shiny 1.5.x.

**Display**
- `input_select("palette", ...)` — Color palette
- `input_switch("log_scale", ...)` — Log scale
- `input_switch("reverse_palette", ...)` — Reverse palette
- `input_switch("diff_mode", ...)` — Difference mode
- `output_ui("ref_timestep_ui")` — Reference timestep (contextual: diff mode on)
- `output_ui("color_range_ui")` — Custom color range toggle + inputs
- `output_ui("filter_ui")` — Value filter slider

**Overlays & Analysis**
- Overlay toggles: `input_switch("vectors")`, `input_switch("contours")`, `input_switch("wireframe")`, `input_switch("boundary_nodes")`, `input_switch("show_extrema")`, `input_switch("particles")`
- `output_ui("compare_var_ui")` — Compare variable dropdown
- `output_ui("compare_upload_ui")` — Compare file upload (contextual: diff mode on) — **new `@render.ui` wrapper needed**
- `input_select("diagnostic", ...)` — Analysis tools dropdown (mesh quality, slope, courant, element area)
- `input_action_button("draw_xsec", ...)` — Draw cross-section button
- `output_ui("clear_xsec_ui")` — Clear cross-section
- `output_ui("discharge_ui")` — Discharge calculator
- `output_ui("rating_curve_ui")` — Rating curve
- `input_action_button("measure_btn", ...)` — Measurement tool
- `output_ui("measure_info_ui")` — Measurement info display
- `input_action_button("draw_polygon", ...)` — Draw polygon button
- `output_ui("polygon_stats_ui")` — Polygon zonal stats
- `input_text("expr_input", ...)` + `input_action_button("eval_expr", ...)` — Expression evaluator
- `output_ui("toggle_3d_ui")` — 3D mode toggle
- `output_ui("particle_seed_ui")` — Particle seed configuration

**CRS**
- `input_text("epsg_input", ...)` — EPSG code
- `input_switch("auto_crs", ...)` — Auto-detect toggle
- `output_ui("crs_status_ui")` — CRS status display
- `output_ui("crs_offset_ui")` — Origin offset X/Y (contextual: EPSG non-empty) — **new `@render.ui` wrapper needed**

**Playback**
- Play / Record buttons (`input_action_button("play_btn")`, `input_action_button("record_btn")`)
- `input_numeric("goto_time")` + `input_action_button("goto_btn")` — Go-to-time
- `input_slider("speed", ...)` — Speed slider
- `input_switch("loop", ...)` — Loop toggle
- `output_ui("trail_length_ui")` — Particle trail length slider

### Tier 3 — Contextual Visibility

These controls require `@render.ui` wrappers that return the control or `ui.div()` (empty) based on condition:

| Control | Shown when | Notes |
|---------|-----------|-------|
| `compare_upload_ui` | `input.diff_mode()` is True | **New** `@render.ui` — wraps existing `input_file("compare_upload")` |
| `crs_offset_ui` | `input.epsg_input()` is non-empty | **New** `@render.ui` — wraps existing `crs_x_offset` / `crs_y_offset` inputs |
| `ref_timestep_ui` | `input.diff_mode()` is True | Already exists as `@render.ui` |

## 4. What Moves Where

| Control | Current Location | New Location |
|---------|-----------------|--------------|
| `h6` title | Top of sidebar | Top of sidebar (unchanged) |
| Example case | Data accordion | Tier 1 (top) |
| Upload .slf | Data accordion | Tier 1 |
| Clear upload button | Data accordion | Tier 1 |
| Basemap | Data accordion | Tier 1 |
| Compare file upload | Data accordion | Overlays & Analysis (contextual) |
| Variable dropdown | Visualization accordion | Tier 1 |
| Time step slider | Playback accordion | Tier 1 |
| EPSG input | Data accordion | CRS accordion |
| Auto-detect CRS | Data accordion | CRS accordion |
| CRS status | Data accordion | CRS accordion |
| CRS origin offset | Data accordion | CRS accordion (contextual) |
| Palette | Visualization accordion | Display accordion |
| Log/reverse/diff | Visualization accordion | Display accordion |
| Color range | Visualization accordion | Display accordion |
| Filter (`filter_ui`) | Visualization accordion | Display accordion |
| Ref timestep (`ref_timestep_ui`) | Visualization accordion | Display accordion (contextual) |
| Overlay toggles (vectors, contours, wireframe, boundary, extrema, particles) | Visualization > Overlays/Analysis details | Overlays & Analysis accordion |
| Compare variable (`compare_var_ui`) | Visualization > Overlays details | Overlays & Analysis accordion |
| Draw XS / Measure / Polygon buttons | Visualization > Analysis details | Overlays & Analysis accordion |
| Clear XS (`clear_xsec_ui`) | Visualization > Analysis details | Overlays & Analysis accordion |
| Discharge / Rating / Polygon stats (`discharge_ui`, `rating_curve_ui`, `polygon_stats_ui`) | Visualization > Analysis details | Overlays & Analysis accordion |
| Measure info (`measure_info_ui`) | Visualization > Analysis details | Overlays & Analysis accordion |
| Expression evaluator | Visualization > Analysis details | Overlays & Analysis accordion |
| 3D mode toggle | Visualization > Analysis details | Overlays & Analysis accordion |
| Particle seed config | Visualization > Analysis details | Overlays & Analysis accordion |
| Play/Record | Playback accordion | Playback accordion |
| Speed/Loop | Playback accordion | Playback accordion |
| Go-to-time | Playback accordion | Playback accordion |
| Trail length | Playback accordion | Playback accordion |

## 5. Implementation Notes

- **Mostly UI-only.** All `input.*` IDs stay the same. All existing `@reactive` and `@render` functions are unchanged. The 210 tests are unaffected.
- **Two new `@render.ui` functions needed** in the server for contextual controls:
  1. `compare_upload_ui` — returns `input_file("compare_upload", ...)` when `input.diff_mode()` else `ui.div()`
  2. `crs_offset_ui` — returns offset inputs when `input.epsg_input()` is non-empty else `ui.div()`
- Use `ui.accordion(..., open=False)` at the accordion level. Do NOT use `open=False` on individual `accordion_panel()`.
- The time step slider and variable dropdown are already rendered via `output_ui`. They simply move from inside an accordion to bare sidebar elements — render functions stay the same.

## 6. What Doesn't Change

- No new files created
- No input ID renames
- No test changes needed
- Import tab layout unchanged (it has its own panel)
- All existing functionality preserved
- Only 2 small new `@render.ui` functions added to server
