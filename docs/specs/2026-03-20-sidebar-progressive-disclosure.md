# Sidebar Progressive Disclosure — Design Specification

**Date:** 2026-03-20
**Status:** Draft

## 1. Purpose

Reduce sidebar cognitive load by showing only essential controls by default. The current sidebar has ~20 controls in 3 expanded accordions, requiring scrolling to find common actions. The redesign uses progressive disclosure: essential controls always visible, everything else collapsed.

## 2. Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Tier 1 placement | Bare elements, no accordion | Core workflow controls should never be hidden |
| Accordion default state | All collapsed (`open=False`) | User expands only what they need |
| Basemap location | Tier 1 (always visible) | Used frequently enough to justify top-level placement |
| Time step slider | Tier 1 (always visible) | Primary interaction for temporal data |
| Playback advanced controls | Collapsed accordion | Speed, loop, go-to-time are secondary to the slider |
| Compare file upload | Contextual (diff mode only) | No reason to show upload when diff mode is off |
| CRS origin offset | Contextual (EPSG set only) | Offset is meaningless without a CRS code |
| Input IDs | Unchanged | No server logic changes needed |

## 3. Layout

### Tier 1 — Always Visible

Bare elements at the top of the sidebar (no accordion wrapper):

1. `input_select("example", ...)` — Example case dropdown
2. `input_file("upload", ...)` — Upload .slf file
3. `output_ui("clear_upload_ui")` — Clear upload button (when file uploaded)
4. `input_select("basemap", ...)` — Background/basemap selector
5. `output_ui("var_select_ui")` — Variable dropdown (rendered dynamically)
6. Time step slider + associated output (rendered dynamically from `output_ui`)

### Tier 2 — Collapsed Accordions

Four accordions below Tier 1, all with `open=False`:

**Display**
- `input_select("palette", ...)` — Color palette
- `input_switch("log_scale", ...)` — Log scale
- `input_switch("reverse_palette", ...)` — Reverse palette
- `input_switch("diff_mode", ...)` — Difference mode
- `output_ui("ref_timestep_ui")` — Reference timestep (contextual: diff mode on)
- `output_ui("color_range_ui")` — Custom color range toggle + inputs
- Value filter slider

**Overlays & Analysis**
- Overlay toggles: vectors, contours, wireframe, boundary, extrema, particles
- Compare variable dropdown
- `input_file("compare_upload", ...)` — Compare file (contextual: diff mode on)
- Analysis tools dropdown (mesh quality, slope, courant, element area)
- Expression input + eval button

**CRS**
- `input_text("epsg_input", ...)` — EPSG code
- `input_switch("auto_crs", ...)` — Auto-detect toggle
- `output_ui("crs_status_ui")` — CRS status display
- CRS origin offset X/Y (contextual: EPSG non-empty)

**Playback**
- Play / Record buttons
- Go-to-time input + button
- Speed slider
- Loop toggle

### Tier 3 — Contextual Visibility

These controls are rendered via `@render.ui` conditionals:

| Control | Shown when |
|---------|-----------|
| Compare file upload | `input.diff_mode()` is True |
| CRS origin offset inputs | `input.epsg_input()` is non-empty |
| Reference timestep selector | `input.diff_mode()` is True |

## 4. What Moves Where

| Control | Current Location | New Location |
|---------|-----------------|--------------|
| Example case | Data accordion | Tier 1 (top) |
| Upload .slf | Data accordion | Tier 1 |
| Clear upload button | Data accordion | Tier 1 |
| Basemap | Data accordion | Tier 1 |
| Variable dropdown | Visualization accordion | Tier 1 |
| Time step slider | Playback accordion | Tier 1 |
| Compare file upload | Data accordion | Overlays & Analysis (contextual) |
| EPSG input | Data accordion | CRS accordion |
| Auto-detect CRS | Data accordion | CRS accordion |
| CRS status | Data accordion | CRS accordion |
| CRS origin offset | Data accordion | CRS accordion (contextual) |
| Palette | Visualization accordion | Display accordion |
| Log/reverse/diff | Visualization accordion | Display accordion |
| Color range | Visualization accordion | Display accordion |
| Filter | Visualization accordion | Display accordion |
| Overlays section | Visualization accordion | Overlays & Analysis accordion |
| Analysis tools | Visualization accordion | Overlays & Analysis accordion |
| Play/Record | Playback accordion | Playback accordion |
| Speed/Loop | Playback accordion | Playback accordion |
| Go-to-time | Playback accordion | Playback accordion |

## 5. Implementation Notes

- **UI-only change.** All `input.*` IDs remain the same. All `@reactive` and `@render` functions in the server are unchanged. The 210 existing tests are unaffected.
- The time step slider and variable dropdown are currently rendered dynamically via `output_ui`. They move from inside an accordion to bare sidebar elements, but the render functions stay the same.
- The `open=False` on accordion panels is set via `ui.accordion_panel(..., open=False)` or `ui.accordion(..., open=False)` depending on the Shiny version. Check the existing pattern in `app.py`.
- Contextual visibility for compare upload and CRS offset: wrap in `@render.ui` that returns the control or `ui.div()` (empty) based on the condition.

## 6. What Doesn't Change

- No new files created
- No server logic changes
- No input ID renames
- No test changes needed
- Import tab layout unchanged (it has its own panel)
- All existing functionality preserved
