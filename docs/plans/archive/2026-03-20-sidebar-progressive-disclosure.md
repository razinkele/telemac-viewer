# Sidebar Progressive Disclosure — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the sidebar to show only essential controls by default — 5 always-visible controls at top, 4 collapsed accordions below, and contextual controls that appear based on state.

**Architecture:** Single-file UI rearrangement in `app.py`. Replace the sidebar block (lines 701–805) with the new tiered layout. Add 2 new `@render.ui` functions for contextual controls. All input IDs unchanged, all server logic unchanged.

**Tech Stack:** Python Shiny 1.5.1, existing `ui.sidebar()`, `ui.accordion()`.

**Spec:** `docs/specs/2026-03-20-sidebar-progressive-disclosure.md`

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `app.py` (lines 701–805) | Sidebar UI block | Replace entirely |
| `app.py` (server section) | Two new `@render.ui` functions | Add |

---

### Task 1: Replace sidebar UI block

**Files:**
- Modify: `app.py:701-805` — replace entire sidebar content

- [ ] **Step 1: Replace sidebar block**

Find the sidebar block in `app.py` (from `sidebar=ui.sidebar(` through `width="280px",` and its closing `),`). Replace the entire block from line 701 to line 805 with the new tiered layout.

The old block starts with:
```python
    sidebar=ui.sidebar(
        ui.h6("TELEMAC Viewer", style="margin:0 0 8px; color: var(--coastal-text);"),
        ui.accordion(
            ui.accordion_panel(
                "Data",
```

And ends with:
```python
            open=["Data", "Visualization", "Playback"],
            multiple=True,
        ),
        width="280px",
    ),
```

Replace with:

```python
    sidebar=ui.sidebar(
        # ── Tier 1: Always visible ──
        ui.h6("TELEMAC Viewer", style="margin:0 0 8px; color: var(--coastal-text);"),
        ui.input_select("example", "Example case", choices=EXAMPLE_CHOICES),
        ui.input_file("upload", "Or upload .slf file", accept=[".slf"]),
        ui.output_ui("clear_upload_ui"),
        ui.input_select("basemap", "Background", choices={
            "dark": "Dark (ocean)",
            "light": "Light (blank)",
            "osm": "CartoDB Dark",
            "satellite": "Satellite (ESRI)",
        }),
        ui.output_ui("var_select_ui"),
        ui.output_ui("time_slider_ui"),
        # ── Tier 2: Collapsed accordions ──
        ui.accordion(
            ui.accordion_panel(
                "Display",
                ui.input_select("palette", "Color palette", choices=list(PALETTES.keys())),
                ui.input_switch("log_scale", "Log scale coloring", value=False),
                ui.input_switch("reverse_palette", "Reverse palette", value=False),
                ui.input_switch("diff_mode", "Difference mode", value=False),
                ui.output_ui("ref_timestep_ui"),
                ui.output_ui("color_range_ui"),
                ui.output_ui("filter_ui"),
            ),
            ui.accordion_panel(
                "Overlays & Analysis",
                ui.input_switch("vectors", "Velocity vectors", value=False),
                ui.input_switch("contours", "Contour lines", value=False),
                ui.input_switch("wireframe", "Mesh wireframe", value=False),
                ui.input_switch("boundary_nodes", "Boundary conditions", value=False),
                ui.input_switch("show_extrema", "Min/max locations", value=False),
                ui.input_switch("particles", "Particle traces", value=False),
                ui.output_ui("compare_var_ui"),
                ui.output_ui("compare_upload_ui"),
                ui.input_select("diagnostic", "Mesh diagnostic", choices={
                    "none": "None",
                    "mesh_quality": "Mesh quality",
                    "slope": "Slope/gradient",
                    "courant": "Courant number",
                    "elem_area": "Element area",
                }),
                ui.input_action_button("draw_xsec", "Draw Cross-Section",
                                       class_="btn-sm btn-outline-primary w-100 mb-1"),
                ui.output_ui("clear_xsec_ui"),
                ui.output_ui("discharge_ui"),
                ui.output_ui("rating_curve_ui"),
                ui.input_action_button("measure_btn", "Measure Distance",
                                       class_="btn-sm btn-outline-warning w-100 mb-1"),
                ui.output_ui("measure_info_ui"),
                ui.input_action_button("draw_polygon", "Draw Polygon",
                                       class_="btn-sm btn-outline-success w-100 mb-1"),
                ui.output_ui("polygon_stats_ui"),
                ui.div(
                    ui.input_text("expr_input", "Custom expression", placeholder="VELOCITY_U**2 + VELOCITY_V**2"),
                    ui.input_action_button("eval_expr", "Eval",
                                           class_="btn-sm btn-outline-secondary"),
                    class_="d-flex align-items-end gap-1 mb-1",
                ),
                ui.output_ui("toggle_3d_ui"),
                ui.output_ui("particle_seed_ui"),
            ),
            ui.accordion_panel(
                "CRS",
                ui.input_text("epsg_input", "CRS (EPSG)", placeholder="e.g. 3346"),
                ui.input_switch("auto_crs", "Auto-detect CRS", value=True),
                ui.output_ui("crs_status_ui"),
                ui.output_ui("crs_offset_ui"),
            ),
            ui.accordion_panel(
                "Playback",
                ui.div(
                    ui.input_action_button(
                        "play_btn", "Play", class_="btn-sm btn-primary w-100"
                    ),
                    class_="mb-2",
                ),
                ui.input_action_button("record_btn", "Record", class_="btn-sm btn-outline-danger w-100 mb-1"),
                ui.div(
                    ui.input_numeric("goto_time", "Go to time (s)", value=0, min=0, step=1),
                    ui.input_action_button("goto_btn", "Go",
                                           class_="btn-sm btn-outline-primary"),
                    class_="d-flex align-items-end gap-1 mb-2",
                ),
                ui.input_slider(
                    "speed", "Speed (s/frame)",
                    min=0.1, max=2.0, value=0.5, step=0.1,
                ),
                ui.input_switch("loop", "Loop animation", value=True),
                ui.output_ui("trail_length_ui"),
            ),
            id="sidebar_accordion",
            open=False,
            multiple=True,
        ),
        width="280px",
    ),
```

Key changes from old layout:
- `ui.h6` title stays at top
- 6 Tier 1 controls placed as bare elements (example, upload, clear, basemap, variable, time step)
- `compare_upload` replaced with `output_ui("compare_upload_ui")` (contextual)
- CRS origin offset replaced with `output_ui("crs_offset_ui")` (contextual)
- 4 accordions (Display, Overlays & Analysis, CRS, Playback) replace 3 (Data, Visualization, Playback)
- `open=False` replaces `open=["Data", "Visualization", "Playback"]`
- Time step slider moved from Playback accordion to Tier 1

- [ ] **Step 2: Verify syntax**

Run: `cd /home/razinka/telemac/telemac-viewer && /opt/micromamba/envs/shiny/bin/python3.13 -c "import ast; ast.parse(open('app.py').read()); print('Syntax OK')"`
Expected: `Syntax OK`

- [ ] **Step 3: Verify import**

Run: `cd /home/razinka/telemac/telemac-viewer && /opt/micromamba/envs/shiny/bin/python3.13 -c "from app import *; print('Import OK')"`
Expected: `Import OK`

- [ ] **Step 4: Run all tests**

Run: `cd /home/razinka/telemac/telemac-viewer && /opt/micromamba/envs/shiny/bin/python3.13 -m pytest tests/ telemac_tools/ -q --tb=short`
Expected: 210 passed, 3 skipped, 0 failed

- [ ] **Step 5: Commit**

```bash
git add app.py && git commit -m "refactor(ui): reorganize sidebar with progressive disclosure — Tier 1 always visible, Tier 2 collapsed"
```

---

### Task 2: Add contextual `@render.ui` functions for compare upload and CRS offset

**Files:**
- Modify: `app.py` (server section) — add 2 new `@render.ui` functions

- [ ] **Step 1: Find the server section for compare and CRS**

The compare upload was previously a bare `input_file`. We need a new `@render.ui` function that shows it only when diff mode is on.

The CRS offset inputs were in a `<details>` block. We need a new `@render.ui` that shows them only when EPSG is entered.

Search for existing render functions near the compare/CRS area:
- `def clear_upload_ui()` — around line 940
- `def crs_status_ui()` — around line 1060

- [ ] **Step 2: Add `compare_upload_ui` render function**

Find `def clear_upload_ui()` in the server section. Add the new function AFTER it:

```python
    @output
    @render.ui
    def compare_upload_ui():
        if input.diff_mode():
            return ui.input_file("compare_upload", "Compare file (.slf)", accept=[".slf"])
        return ui.div()
```

**Note:** When diff mode is off, the `compare_upload` file input is removed from the DOM. Existing server code that reads `input.compare_upload()` already guards against `None` (it checks `uploaded and use_upload.get()` patterns). No additional server changes needed.

- [ ] **Step 3: Add `crs_offset_ui` render function**

Find `def crs_status_ui()` in the server section. Add the new function AFTER it:

```python
    @output
    @render.ui
    def crs_offset_ui():
        if input.epsg_input() and input.epsg_input().strip():
            return ui.div(
                ui.input_numeric("crs_x_offset", "X offset (m)", value=0, step=1000),
                ui.input_numeric("crs_y_offset", "Y offset (m)", value=0, step=1000),
            )
        return ui.div()
```

- [ ] **Step 4: Verify syntax and import**

Run: `cd /home/razinka/telemac/telemac-viewer && /opt/micromamba/envs/shiny/bin/python3.13 -c "import ast; ast.parse(open('app.py').read()); print('Syntax OK')"`

Run: `cd /home/razinka/telemac/telemac-viewer && /opt/micromamba/envs/shiny/bin/python3.13 -c "from app import *; print('Import OK')"`

- [ ] **Step 5: Run all tests**

Run: `cd /home/razinka/telemac/telemac-viewer && /opt/micromamba/envs/shiny/bin/python3.13 -m pytest tests/ telemac_tools/ -q --tb=short`
Expected: 210 passed, 3 skipped, 0 failed

- [ ] **Step 6: Commit**

```bash
git add app.py && git commit -m "feat(ui): add contextual compare_upload_ui and crs_offset_ui render functions"
```

---

### Task 3: Manual verification — start app and check layout

- [ ] **Step 1: Start the app**

Run: `cd /home/razinka/telemac/telemac-viewer && /opt/micromamba/envs/shiny/bin/shiny run app.py --port 8765`

- [ ] **Step 2: Verify Tier 1 controls are visible without expanding anything**

Check that Example case, Upload, Basemap, Variable, and Time step are visible at sidebar top without clicking any accordion.

- [ ] **Step 3: Verify all 4 accordions are collapsed by default**

Display, Overlays & Analysis, CRS, Playback should all be collapsed (closed).

- [ ] **Step 4: Verify contextual controls**

- With diff mode OFF: compare file upload should NOT be visible
- Toggle diff mode ON: compare file upload should appear in Overlays & Analysis
- With EPSG empty: CRS offset inputs should NOT be visible
- Type "3346" in EPSG: CRS offset X/Y inputs should appear in CRS accordion

- [ ] **Step 5: Verify all functionality still works**

- Select Curonian Lagoon → mesh renders
- Change variable → colors update
- Expand Display, change palette → works
- Expand Overlays, toggle wireframe → works
- Expand Playback, click Play → animation works

---

## Summary

| Task | What | Steps |
|------|------|-------|
| 1 | Replace sidebar UI block (single edit) | 5 |
| 2 | Add 2 contextual `@render.ui` functions | 6 |
| 3 | Manual verification | 5 |
