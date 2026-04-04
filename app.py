# app.py — TELEMAC Viewer v3
import io
import numpy as np
import plotly.graph_objects as go
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget
from shiny_deckgl import (
    MapWidget,
    head_includes,
    zoom_widget,
    fullscreen_widget,
    scale_widget,
    screenshot_widget,
    compass_widget,
    layer_legend_widget,
    reset_view_widget,
    loading_widget,
    gimbal_widget,
    first_person_view,
    ambient_light,
    directional_light,
    lighting_effect,
)

from constants import (
    EXAMPLES, EXAMPLE_CHOICES, PALETTES,
    cached_gradient_colors, format_time,
)
from geometry import build_mesh_geometry
from layers import (
    build_mesh_layer,
    build_velocity_layer,
    build_contour_layer_fn,
    build_marker_layer,
    build_cross_section_layer,
    build_particle_layer,
    build_wireframe_layer,
    build_extrema_markers,
    build_measurement_layer,
    build_boundary_layer,
)
import asyncio
import math
import shlex
import threading
import os as _os
from crs import (
    crs_from_epsg, detect_crs_from_cas, guess_crs_from_coords,
    click_to_native, meters_to_wgs84,
)
from analysis import (
    nearest_node,
    time_series_at_point,
    cross_section_profile,
    compute_particle_paths,
    generate_seed_grid,
    distribute_seeds_along_line,
    get_available_derived,
    compute_derived,
    export_crosssection_csv,
    compute_mesh_quality,
    find_cas_files,
    detect_module,
    vertical_profile_at_point,
    compute_difference,
    compute_temporal_stats,
    find_extrema,
    find_boundary_nodes,
    find_boundary_edges,
    compute_slope,
    export_all_variables_csv,
    compute_mesh_integral,
    evaluate_expression,
    compute_discharge,
    compute_courant_number,
    compute_element_area,
    compute_flood_envelope,
    compute_flood_arrival,
    compute_flood_duration,
    read_cli_file,
    extract_layer_2d,
    polygon_zonal_stats,
)
from telemac_defaults import (
    suggest_palette, is_bipolar,
    detect_module as detect_module_vars, find_velocity_pair,
)
from validation import (
    parse_observation_csv, compute_rmse, compute_nse,
    compute_volume_timeseries, parse_liq_file,
)
from data_manip.extraction.telemac_file import TelemacFile

# ---------------------------------------------------------------------------
# Map widget
# ---------------------------------------------------------------------------

map_widget = MapWidget(
    "map",
    view_state={"longitude": 0, "latitude": 0, "zoom": 0},
    style="data:application/json;charset=utf-8,%7B%22version%22%3A8%2C%22sources%22%3A%7B%7D%2C%22layers%22%3A%5B%7B%22id%22%3A%22bg%22%2C%22type%22%3A%22background%22%2C%22paint%22%3A%7B%22background-color%22%3A%22%230f1923%22%7D%7D%5D%7D",
    cooperative_gestures=True,
    tooltip={
        "html": "<b>{layerType}</b><br/>{info}",
        "style": {
            "backgroundColor": "rgba(15, 25, 35, 0.92)",
            "color": "#c8dce8",
            "fontSize": "12px",
            "borderRadius": "6px",
            "border": "1px solid rgba(13, 115, 119, 0.5)",
            "padding": "6px 10px",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.3)",
        },
    },
)

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

_HELP_MODAL = ui.modal(
    ui.navset_tab(
        ui.nav_panel(
            "About TELEMAC",
            ui.div(
                ui.h5("What is TELEMAC?"),
                ui.p(
                    "TELEMAC-MASCARET is an open-source suite of solvers for free-surface flow, "
                    "wave propagation, sediment transport, and water quality modelling. "
                    "Developed since the 1980s by EDF R&D (France) and the TELEMAC-MASCARET Consortium, "
                    "it is widely used in hydraulic engineering, coastal and river studies, flood risk analysis, "
                    "and environmental impact assessment."
                ),
                ui.h6("Modules"),
                ui.tags.dl(
                    ui.tags.dt("TELEMAC-2D"),
                    ui.tags.dd("Depth-averaged shallow water equations — river flows, dam breaks, flood propagation."),
                    ui.tags.dt("TELEMAC-3D"),
                    ui.tags.dd("Three-dimensional hydrodynamics — estuaries, stratified flows, thermal plumes."),
                    ui.tags.dt("TOMAWAC"),
                    ui.tags.dd("Spectral wave modelling — nearshore wave transformation, refraction, breaking."),
                    ui.tags.dt("ARTEMIS"),
                    ui.tags.dd("Harbour wave agitation — short-wave diffraction and reflection in port areas."),
                    ui.tags.dt("SISYPHE / GAIA"),
                    ui.tags.dd("Sediment transport and morphodynamics — bedload, suspended load, bed evolution."),
                    ui.tags.dt("WAQTEL"),
                    ui.tags.dd("Water quality — pollutant dispersion, dissolved oxygen, thermal modelling."),
                ),
                ui.h6("SELAFIN format"),
                ui.p(
                    "TELEMAC stores results in the SELAFIN (.slf) binary format. "
                    "Each file contains an unstructured triangular mesh, a list of variables "
                    "(e.g. WATER DEPTH, VELOCITY U/V, FREE SURFACE, BOTTOM), "
                    "and time-varying solution data at each node. "
                    "This viewer reads SELAFIN files directly using the TelemacFile Python API."
                ),
                ui.h6("Finite Element Mesh"),
                ui.p(
                    "TELEMAC uses unstructured triangular finite element meshes. "
                    "Nodes carry the solution values; elements (triangles) connect the nodes. "
                    "Mesh density varies: finer in areas of interest (channels, structures), "
                    "coarser in calm zones, allowing efficient computation over large domains."
                ),
                ui.h6("Typical workflow"),
                ui.tags.ol(
                    ui.tags.li("Create geometry and mesh (BlueKenue, SALOME, or QGIS)"),
                    ui.tags.li("Write a steering file (.cas) with parameters and boundary conditions"),
                    ui.tags.li("Run the simulation (e.g. telemac2d.py case.cas --ncsize=4)"),
                    ui.tags.li("Visualize results in this viewer or ParaView/BlueKenue"),
                ),
                ui.h6("Learn more"),
                ui.p(
                    ui.a("opentelemac.org", href="https://www.opentelemac.org", target="_blank"),
                    " — Official documentation, tutorials, and source code.",
                ),
                style="padding:8px 4px; font-size:13px;",
            ),
        ),
        ui.nav_panel(
            "Viewer Guide",
            ui.div(
                ui.h5("Viewer Interface Guide"),
                ui.h6("Data Panel"),
                ui.tags.ul(
                    ui.tags.li(ui.tags.b("Example case"), " — Load a built-in example from TELEMAC-2D, 3D, TOMAWAC, ARTEMIS, SISYPHE, or GAIA."),
                    ui.tags.li(ui.tags.b("Upload .slf"), " — Load your own SELAFIN result file."),
                    ui.tags.li(ui.tags.b("Dark map background"), " — Toggle between light and dark map canvas."),
                ),
                ui.h6("Visualization Panel"),
                ui.tags.ul(
                    ui.tags.li(ui.tags.b("Variable"), " — Select which result variable to display (e.g. WATER DEPTH, VELOCITY U)."),
                    ui.tags.li(ui.tags.b("Color palette"), " — Choose from Viridis, Plasma, Ocean, Thermal, or Chlorophyll color maps."),
                    ui.tags.li(ui.tags.b("Velocity vectors"), " — Overlay arrow glyphs showing flow direction and magnitude."),
                    ui.tags.li(ui.tags.b("Contour lines"), " — Show iso-value contour lines on the mesh."),
                    ui.tags.li(ui.tags.b("Mesh wireframe"), " — Display the triangular mesh edges."),
                    ui.tags.li(ui.tags.b("Boundary nodes"), " — Highlight nodes on the domain boundary."),
                    ui.tags.li(ui.tags.b("Min/Max locations"), " — Mark the locations of extreme values on the mesh."),
                    ui.tags.li(ui.tags.b("Log scale"), " — Apply logarithmic color scaling for variables with large ranges."),
                    ui.tags.li(ui.tags.b("Reverse palette"), " — Invert the color ramp direction."),
                    ui.tags.li(ui.tags.b("Difference mode"), " — Show the difference between the current timestep and a reference."),
                    ui.tags.li(ui.tags.b("Color range"), " — Manually set min/max values for the color scale."),
                    ui.tags.li(ui.tags.b("Filter range"), " — Gray out values outside a specified range."),
                    ui.tags.li(ui.tags.b("Cross-Section"), " — Click two points to create a profile line; view elevation/value cross-section plots."),
                    ui.tags.li(ui.tags.b("Discharge"), " — Compute volumetric flow rate across the cross-section line."),
                    ui.tags.li(ui.tags.b("Particle traces"), " — Animate flow pathlines from a seed grid."),
                    ui.tags.li(ui.tags.b("Mesh quality"), " — Color the mesh by element quality (aspect ratio)."),
                    ui.tags.li(ui.tags.b("Slope/gradient"), " — Visualize the spatial gradient magnitude of the selected variable."),
                    ui.tags.li(ui.tags.b("Courant number"), " — Display the local CFL condition number."),
                    ui.tags.li(ui.tags.b("Element area"), " — Color-code triangles by their area."),
                    ui.tags.li(ui.tags.b("Measure Distance"), " — Click two points to measure Euclidean distance."),
                    ui.tags.li(ui.tags.b("Custom expression"), " — Compute derived fields using math expressions (e.g. ", ui.tags.code("VELOCITY_U**2 + VELOCITY_V**2"), ")."),
                    ui.tags.li(ui.tags.b("3D mode"), " — Switch to 3D perspective view for multi-layer (3D) result files."),
                ),
                ui.h6("Playback Panel"),
                ui.tags.ul(
                    ui.tags.li(ui.tags.b("Time slider"), " — Scrub through simulation timesteps."),
                    ui.tags.li(ui.tags.b("Play/Pause"), " — Animate through timesteps automatically."),
                    ui.tags.li(ui.tags.b("Go to time"), " — Jump to a specific simulation time in seconds."),
                    ui.tags.li(ui.tags.b("Speed"), " — Control animation speed (seconds per frame)."),
                    ui.tags.li(ui.tags.b("Loop"), " — Restart animation from the beginning when it reaches the end."),
                ),
                ui.h6("Statistics Panel"),
                ui.tags.ul(
                    ui.tags.li(ui.tags.b("Stats"), " — Current timestep statistics: min, max, mean, std dev."),
                    ui.tags.li(ui.tags.b("All Vars Time Series"), " — Plot all variables over time at a clicked node."),
                    ui.tags.li(ui.tags.b("Value Histogram"), " — Distribution histogram of the current variable."),
                    ui.tags.li(ui.tags.b("Compute Integral"), " — Area-weighted integral of the variable over the domain."),
                    ui.tags.li(ui.tags.b("Temporal Stats"), " — Compute min/max/mean over all timesteps."),
                ),
                ui.h6("File Info Panel"),
                ui.tags.ul(
                    ui.tags.li("View metadata: mesh size, variable count, time range, precision, module type."),
                ),
                ui.h6("Run Simulation Panel"),
                ui.tags.ul(
                    ui.tags.li("Select a .cas steering file and run TELEMAC directly from the viewer."),
                    ui.tags.li("Monitor console output in real time."),
                ),
                style="padding:8px 4px; font-size:13px;",
            ),
        ),
        ui.nav_panel(
            "Keyboard Shortcuts",
            ui.div(
                ui.h5("Keyboard Shortcuts"),
                ui.tags.table(
                    ui.tags.thead(
                        ui.tags.tr(ui.tags.th("Key", style="width:140px"), ui.tags.th("Action")),
                    ),
                    ui.tags.tbody(
                        ui.tags.tr(ui.tags.td(ui.tags.kbd("Space")), ui.tags.td("Play / Pause animation")),
                        ui.tags.tr(ui.tags.td(ui.tags.kbd("\u2192 Right Arrow")), ui.tags.td("Next timestep")),
                        ui.tags.tr(ui.tags.td(ui.tags.kbd("\u2190 Left Arrow")), ui.tags.td("Previous timestep")),
                        ui.tags.tr(ui.tags.td(ui.tags.kbd("Page Down")), ui.tags.td("Next variable")),
                        ui.tags.tr(ui.tags.td(ui.tags.kbd("Page Up")), ui.tags.td("Previous variable")),
                    ),
                    class_="table table-sm table-striped",
                    style="max-width:400px;",
                ),
                ui.h6("Map controls"),
                ui.tags.table(
                    ui.tags.thead(
                        ui.tags.tr(ui.tags.th("Action", style="width:180px"), ui.tags.th("How")),
                    ),
                    ui.tags.tbody(
                        ui.tags.tr(ui.tags.td("Pan"), ui.tags.td("Click and drag")),
                        ui.tags.tr(ui.tags.td("Zoom"), ui.tags.td("Scroll wheel or +/- buttons")),
                        ui.tags.tr(ui.tags.td("Reset view"), ui.tags.td("Reset View widget button")),
                        ui.tags.tr(ui.tags.td("Screenshot"), ui.tags.td("Screenshot widget button")),
                        ui.tags.tr(ui.tags.td("Fullscreen"), ui.tags.td("Fullscreen widget button")),
                    ),
                    class_="table table-sm table-striped",
                    style="max-width:400px;",
                ),
                style="padding:8px 4px; font-size:13px;",
            ),
        ),
    ),
    title="Help — TELEMAC Viewer",
    size="xl",
    easy_close=True,
)

# ---------------------------------------------------------------------------
# Custom CSS — Marine-depth theme (light shell + dark data zones)
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
:root {
    --coastal-bg: #f0f4f8;
    --coastal-sidebar: #e8eff5;
    --coastal-text: #1a3a5c;
    --coastal-border: #d0dbe6;
    --ocean-bg: #0f1923;
    --ocean-header: #142330;
    --ocean-text: #c8dce8;
    --ocean-muted: #8899aa;
    --accent-teal: #0d7377;
    --accent-cyan: #00b4d8;
    --accent-green: #48c78e;
    --accent-amber: #f0a040;
    --accent-coral: #ff6b6b;
    --accent-slate: #5a6f80;
}

/* Navbar */
.navbar {
    min-height: 0 !important;
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    background: var(--coastal-bg) !important;
    border-bottom: 2px solid var(--accent-teal) !important;
}
.navbar > .container-fluid { min-height: 0 !important; }
.navbar-brand { padding-top: 0 !important; padding-bottom: 0 !important; }
.navbar .nav-link {
    padding-top: 0.25rem !important;
    padding-bottom: 0.25rem !important;
    color: var(--coastal-text) !important;
    font-weight: 500;
    transition: color 0.2s, border-color 0.2s;
    border-bottom: 2px solid transparent;
}
.navbar .nav-link:hover { color: var(--accent-teal) !important; }
.navbar .nav-link.active,
.navbar .nav-item.active .nav-link {
    color: var(--accent-teal) !important;
    border-bottom: 2px solid var(--accent-teal);
}

/* Sidebar */
.sidebar {
    background-color: var(--coastal-sidebar) !important;
    border-right: 1px solid var(--coastal-border);
}
.sidebar h5, .sidebar h6 { color: var(--coastal-text); font-weight: 600; }
.sidebar label {
    color: var(--coastal-text);
    font-weight: 500;
    font-size: 0.85rem;
}
.sidebar .text-muted { color: var(--accent-slate) !important; }
.sidebar hr { border-color: var(--coastal-border); opacity: 0.6; }

/* Sidebar buttons */
.sidebar .btn-primary {
    background: var(--accent-teal) !important;
    border-color: var(--accent-teal) !important;
    border-radius: 6px;
    font-weight: 500;
    transition: background 0.2s, transform 0.1s;
}
.sidebar .btn-primary:hover {
    background: #0b6264 !important;
    transform: translateY(-1px);
}
.sidebar .btn-outline-primary {
    color: var(--accent-teal) !important;
    border-color: var(--accent-teal) !important;
    border-radius: 6px;
}
.sidebar .btn-outline-primary:hover {
    background: var(--accent-teal) !important;
    color: white !important;
}
.sidebar .btn-outline-warning { border-radius: 6px; }
.sidebar .btn-outline-success { border-radius: 6px; }
.sidebar .btn-outline-secondary { border-radius: 6px; }
.sidebar .btn-danger {
    background: var(--accent-coral) !important;
    border-color: var(--accent-coral) !important;
    border-radius: 6px;
}

/* Slider: teal track */
.sidebar .irs--shiny .irs-bar {
    background: var(--accent-teal) !important;
    border-color: var(--accent-teal) !important;
}
.sidebar .irs--shiny .irs-handle {
    border-color: var(--accent-teal) !important;
}
.sidebar .irs--shiny .irs-single {
    background: var(--accent-teal) !important;
}

/* Accordion styling */
.accordion-button {
    background: var(--coastal-sidebar) !important;
    color: var(--coastal-text) !important;
    font-size: 0.85rem;
    font-weight: 600;
    padding: 0.5rem 0.75rem;
}
.accordion-button:not(.collapsed) {
    background: var(--coastal-bg) !important;
    color: var(--accent-teal) !important;
}

/* Cards */
.card {
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border-radius: 8px;
    border: 1px solid var(--coastal-border);
}
.card.ocean-card {
    background: var(--ocean-bg) !important;
    border: 1px solid rgba(255,255,255,0.06);
    border-left: 3px solid var(--accent-teal);
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
.card.ocean-card .card-header {
    background: var(--ocean-header) !important;
    color: var(--accent-cyan) !important;
    font-weight: 600;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    font-size: 0.9rem;
}
.card.ocean-card .card-body { color: var(--ocean-text); }

/* Stat chips row */
.stat-row { display: flex; gap: 6px; margin-bottom: 8px; flex-wrap: wrap; }
.stat-chip {
    flex: 1 1 100px; text-align: center; padding: 4px 10px;
    border-radius: 6px; font-size: 0.75rem; line-height: 1.4;
    color: #fff; font-weight: 500;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    min-width: 0;
}
.stat-chip .stat-label { font-weight: 400; opacity: 0.85; }
.stat-chip .stat-val { font-weight: 700; font-size: 0.85rem; }
.stat-var { background: var(--accent-teal) !important; }
.stat-time { background: var(--accent-cyan) !important; color: #000 !important; }
.stat-nodes { background: var(--accent-green) !important; }
.stat-range { background: var(--accent-amber) !important; }

/* Stats tab cards */
.stats-card {
    background: var(--ocean-bg);
    border: 1px solid rgba(255,255,255,0.06);
    border-left: 3px solid var(--accent-teal);
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
    color: var(--ocean-text);
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
.stats-card h6 { color: var(--accent-cyan); font-weight: 600; margin-bottom: 8px; }

/* Simulation console */
.sim-console {
    max-height: 300px; overflow-y: auto;
    font-size: 11px; font-family: monospace;
    background: var(--ocean-bg); color: var(--accent-green);
    padding: 8px; border-radius: 6px;
    border: 1px solid rgba(255,255,255,0.06);
}

/* File info table */
.file-info-table { width: 100%; border-collapse: collapse; }
.file-info-table th, .file-info-table td {
    padding: 6px 12px; text-align: left;
    border-bottom: 1px solid var(--coastal-border);
}
.file-info-table th { color: var(--accent-teal); font-weight: 600; width: 40%; }

/* Error display */
.shiny-output-error { color: var(--accent-coral); }
.shiny-output-error:before { content: '⚠ '; }

/* Details/summary styling in sidebar */
.sidebar details summary {
    font-size: 12px; font-weight: 600; cursor: pointer;
    margin: 6px 0 4px; color: var(--coastal-text);
    transition: color 0.2s;
}
.sidebar details summary:hover { color: var(--accent-teal); }
.sidebar details[open] summary { color: var(--accent-teal); }

/* Tab content settings sub-tabs */
.navset-card-tab .nav-link { color: var(--coastal-text) !important; }
.navset-card-tab .nav-link.active {
    color: var(--accent-teal) !important;
    border-bottom-color: var(--accent-teal) !important;
}

/* Map card: remove padding for full-bleed map */
#map-container .card-body { padding: 0 !important; }
#map-container .card { border-color: rgba(13, 115, 119, 0.3) !important; }

/* Legend widget styling */
.deckgl-legend-panel {
    background: rgba(15, 25, 35, 0.88) !important;
    color: var(--ocean-text) !important;
    border: 1px solid rgba(13, 115, 119, 0.4) !important;
    border-radius: 6px !important;
    backdrop-filter: blur(4px);
    font-size: 0.8rem !important;
}
.deckgl-legend-panel .legend-title {
    color: var(--accent-cyan) !important;
    font-weight: 600;
}

/* Map widgets: subtle dark background */
.maplibregl-ctrl-group {
    background: rgba(15, 25, 35, 0.75) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 6px !important;
}
.maplibregl-ctrl-group button {
    color: var(--ocean-text) !important;
}
.maplibregl-ctrl-group button:hover {
    background: rgba(13, 115, 119, 0.3) !important;
}
.maplibregl-ctrl-group button + button {
    border-top: 1px solid rgba(255,255,255,0.08) !important;
}
"""

app_ui = ui.page_navbar(
    # ── Tab 1: Map (Dashboard) ─────────────────────────────────────────
    ui.nav_panel(
        "Map",
        head_includes(),
        # Stat chips row
        ui.div(
            ui.div(
                ui.span("Variable ", class_="stat-label"),
                ui.span(ui.output_text("stat_var_name", inline=True), class_="stat-val"),
                class_="stat-chip stat-var",
            ),
            ui.div(
                ui.span("Time ", class_="stat-label"),
                ui.span(ui.output_text("stat_time", inline=True), class_="stat-val"),
                class_="stat-chip stat-time",
            ),
            ui.div(
                ui.span("Nodes ", class_="stat-label"),
                ui.span(ui.output_text("stat_nodes", inline=True), class_="stat-val"),
                class_="stat-chip stat-nodes",
            ),
            ui.div(
                ui.span("Range ", class_="stat-label"),
                ui.span(ui.output_text("stat_range", inline=True), class_="stat-val"),
                class_="stat-chip stat-range",
            ),
            class_="stat-row",
        ),
        # Map card — explicit height like cenjas dashboard
        ui.div(
            ui.card(
                map_widget.ui(width="100%", height="calc(100vh - 200px)"),
                ui.output_ui("analysis_panel_ui"),
                ui.output_ui("coord_readout_ui"),
                full_screen=True,
                class_="ocean-card",
            ),
            id="map-container",
        ),
    ),
    # ── Tab 2: Statistics ──────────────────────────────────────────────
    ui.nav_panel(
        "Statistics",
        ui.layout_columns(
            ui.card(
                ui.card_header("Current Timestep Stats"),
                ui.output_ui("stats_ui"),
                ui.output_ui("hover_info_ui"),
                class_="ocean-card",
            ),
            ui.card(
                ui.card_header("Node Inspector"),
                ui.output_ui("node_inspector_ui"),
                class_="ocean-card",
            ),
            col_widths=[6, 6],
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("Analysis Tools"),
                ui.input_file("obs_upload", "Upload observations (.csv)", accept=[".csv"]),
                ui.div(
                    ui.input_action_button("show_multivar", "All Vars Time Series",
                                           class_="btn-sm btn-outline-primary mb-1"),
                    ui.input_action_button("show_histogram", "Value Histogram",
                                           class_="btn-sm btn-outline-primary mb-1"),
                    ui.input_action_button("compute_integral", "Compute Integral",
                                           class_="btn-sm btn-outline-info mb-1"),
                    ui.input_action_button("compute_temporal", "Temporal Stats",
                                           class_="btn-sm btn-outline-info mb-1"),
                    ui.input_action_button("compute_volume", "Volume over time",
                                           class_="btn-sm btn-outline-info mb-1"),
                    class_="d-flex flex-wrap gap-2 mb-2",
                ),
                ui.output_ui("integral_ui"),
                ui.output_ui("temporal_stats_ui"),
                ui.output_ui("liq_display_ui"),
            ),
            col_widths=[12],
        ),
    ),
    # ── Tab 3: File Info ───────────────────────────────────────────────
    ui.nav_panel(
        "File Info",
        ui.card(
            ui.card_header("SELAFIN File Metadata"),
            ui.output_ui("file_info_ui"),
            class_="ocean-card",
        ),
    ),
    # ── Tab 4: Run Simulation ──────────────────────────────────────────
    ui.nav_panel(
        "Run",
        ui.layout_columns(
            ui.card(
                ui.card_header("Simulation Setup"),
                ui.output_ui("cas_select_ui"),
                ui.input_numeric("ncores", "CPU cores", value=4, min=0, max=28, step=1),
                ui.div(
                    ui.input_action_button("run_sim", "Run",
                                           class_="btn-sm btn-success me-2"),
                    ui.input_action_button("stop_sim", "Stop",
                                           class_="btn-sm btn-danger"),
                    class_="d-flex gap-2 mb-2",
                ),
                ui.output_ui("sim_status_ui"),
            ),
            ui.card(
                ui.card_header("Console Output"),
                ui.tags.pre(
                    ui.output_text_verbatim("sim_console"),
                    class_="sim-console",
                ),
                class_="ocean-card",
            ),
            col_widths=[4, 8],
        ),
    ),
    # ── Import tab ────────────────────────────────────────────────────
    ui.nav_panel(
        "Import",
        ui.layout_columns(
            ui.card(
                ui.card_header("HEC-RAS → TELEMAC Import"),
                ui.input_select("import_source", "Source format", choices={
                    "hecras6": "HEC-RAS 6.x (.hdf)",
                }),
                ui.input_file("import_hdf", "HEC-RAS geometry file (.hdf)",
                              accept=[".hdf", ".hdf5"]),
                ui.input_file("import_dem", "DEM file (.tif)",
                              accept=[".tif", ".tiff"]),
                ui.tags.details(
                    ui.tags.summary("1D Options"),
                    ui.input_numeric("fp_width", "Floodplain width (m)",
                                     value=500, min=50, max=5000, step=50),
                    ui.input_numeric("channel_refine", "Channel refinement (m²)",
                                     value=10, min=1, max=100, step=1),
                    ui.input_numeric("floodplain_refine", "Floodplain refinement (m²)",
                                     value=200, min=10, max=1000, step=10),
                ),
                ui.input_select("import_mesher", "Mesher", choices={
                    "triangle": "Triangle (default)",
                }),
                ui.input_select("import_scheme", "Numerical scheme", choices={
                    "finite_volume": "Finite Volume (robust)",
                    "finite_element": "Finite Element (accurate)",
                }),
                ui.div(
                    ui.input_action_button("import_preview", "Preview",
                                           class_="btn-sm btn-primary me-2"),
                    ui.input_action_button("import_convert", "Convert",
                                           class_="btn-sm btn-success"),
                    class_="d-flex gap-2 mb-2",
                ),
                ui.output_ui("import_status_ui"),
                ui.tags.hr(),
                ui.tags.h6("Download Output"),
                ui.output_ui("import_download_ui"),
            ),
            ui.card(
                ui.card_header("Preview"),
                output_widget("import_map", height="400px"),
                ui.tags.pre(
                    ui.output_text_verbatim("import_log"),
                    class_="sim-console",
                    style="max-height: 200px; overflow-y: auto;",
                ),
                class_="ocean-card",
            ),
            col_widths=[4, 8],
        ),
    ),
    # ── Navbar extras ──────────────────────────────────────────────────
    ui.nav_spacer(),
    ui.nav_control(
        ui.input_action_link("help_btn", "? Help", class_="nav-link"),
    ),
    # ── Sidebar ────────────────────────────────────────────────────────
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
    title="TELEMAC Viewer",
    fillable=True,
    header=ui.TagList(
        ui.tags.link(
            rel="stylesheet",
            href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css",
        ),
        ui.tags.style(CUSTOM_CSS),
        ui.busy_indicators.use(spinners=True, pulse=True),
        # Fallback: re-init deckgl maps if shiny:connected fired before CDN loaded
        ui.tags.script("""
        (function() {
            var _retries = 0;
            function ensureMaps() {
                if (_retries > 50) return;
                _retries++;
                var inst = window.__deckgl_instances;
                if (!inst) { setTimeout(ensureMaps, 200); return; }
                var maps = document.querySelectorAll('.deckgl-map');
                var uninit = [];
                maps.forEach(function(el) { if (!inst[el.id]) uninit.push(el); });
                if (uninit.length > 0 && typeof window.maplibregl !== 'undefined' && typeof window.deck !== 'undefined') {
                    document.dispatchEvent(new Event('shiny:connected'));
                } else if (uninit.length > 0) {
                    setTimeout(ensureMaps, 200);
                }
            }
            if (document.readyState === 'complete') ensureMaps();
            else window.addEventListener('load', ensureMaps);
        })();
        """),
        ui.tags.script("""
        document.addEventListener('keydown', function(e) {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            if (e.key === 'ArrowRight') {
                Shiny.setInputValue('kb_next', Math.random());
            } else if (e.key === 'ArrowLeft') {
                Shiny.setInputValue('kb_prev', Math.random());
            } else if (e.key === ' ') {
                e.preventDefault();
                Shiny.setInputValue('kb_play', Math.random());
            } else if (e.key === 'PageUp') {
                e.preventDefault();
                Shiny.setInputValue('kb_var_prev', Math.random());
            } else if (e.key === 'PageDown') {
                e.preventDefault();
                Shiny.setInputValue('kb_var_next', Math.random());
            }
        });

        // Animation recording via MediaRecorder
        let mediaRecorder = null;
        let recordedChunks = [];
        Shiny.addCustomMessageHandler('toggle_recording', function(data) {
            if (data.start) {
                const canvas = document.querySelector('#map-container canvas');
                if (!canvas) { Shiny.setInputValue('record_error', 'No canvas found'); return; }
                try {
                    const stream = canvas.captureStream(30);
                    mediaRecorder = new MediaRecorder(stream, {mimeType: 'video/webm'});
                    recordedChunks = [];
                    mediaRecorder.ondataavailable = e => { if (e.data.size > 0) recordedChunks.push(e.data); };
                    mediaRecorder.onstop = () => {
                        const blob = new Blob(recordedChunks, {type: 'video/webm'});
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url; a.download = 'animation.webm'; a.click();
                        URL.revokeObjectURL(url);
                    };
                    mediaRecorder.start();
                    Shiny.setInputValue('recording_active', true);
                } catch(e) { Shiny.setInputValue('record_error', e.message); }
            } else {
                if (mediaRecorder && mediaRecorder.state === 'recording') {
                    mediaRecorder.stop();
                    Shiny.setInputValue('recording_active', false);
                }
            }
        });
    """),
    ),  # end TagList header
)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


def server(input, output, session):
    _tf_lock = threading.Lock()

    def _run_with_lock(fn, *args):
        """Run fn(*args) while holding the TelemacFile lock."""
        with _tf_lock:
            return fn(*args)

    # -----------------------------------------------------------------------
    # Reactive dependency chain:
    #   tel_file() -> mesh_geom(), current_var(), current_tidx()
    #   current_var() + current_tidx() -> current_values()
    #   effective_values() wraps current_values() with override cascade:
    #     expression > mesh quality > slope > Courant > elem area >
    #     temporal stats > difference mode > current values
    #   update_map() is the terminal leaf consuming all of the above.
    #
    # Topology-only calcs (mesh_quality_values, elem_area_values,
    # boundary_nodes_cached) depend only on tel_file(), not timestep.
    #
    # Coordinate systems (see geometry.py docstring for details):
    #   - Mesh meters: original TELEMAC CRS (x_off/y_off = bounding box center)
    #   - Centered meters: mesh meters minus x_off/y_off (used by all layers)
    #   - Pseudo-degrees: centered meters as interpreted by deck.gl's
    #     METER_OFFSETS mode; map clicks return these, converted via
    #     click_to_native() back to mesh meters.
    # -----------------------------------------------------------------------
    playing = reactive.value(False)
    last_file_path = reactive.value("")
    analysis_mode = reactive.value("none")
    clicked_points = reactive.value([])  # list of (x_meters, y_meters)
    cross_section_points = reactive.value(None)  # [[x_m, y_m], ...] or None
    particle_paths = reactive.value(None)  # list of paths or None
    is_3d_mode = reactive.value(False)
    sim_process = reactive.value(None)  # asyncio.Process or None
    sim_output = reactive.value("")  # accumulated console output
    sim_running = reactive.value(False)
    temporal_stats_cache = reactive.value(None)  # dict with min/max/mean arrays
    measure_points = reactive.value([])  # list of [x_mesh_m, y_mesh_m] (max 2)
    measure_mode = reactive.value(False)  # True when waiting for measurement clicks
    use_upload = reactive.value(False)  # True when uploaded file should be used
    obs_data = reactive.value(None)  # parsed observation CSV: (times, values, varname) or None
    compare_tf = reactive.value(None)  # secondary TelemacFile for comparison
    recording = reactive.value(False)  # animation recording state
    polygon_mode = reactive.value(False)  # polygon drawing mode
    polygon_stats_data = reactive.value(None)  # polygon zonal stats result
    volume_cache = reactive.value(None)  # {"times": ndarray, "volumes": ndarray}

    # -- Help modal --
    @reactive.effect
    @reactive.event(input.help_btn)
    def _show_help():
        ui.modal_show(_HELP_MODAL)

    # -- Stat chip outputs (Map tab header) --
    @output
    @render.text
    def stat_var_name():
        try:
            return current_var()
        except Exception:
            return "—"

    @output
    @render.text
    def stat_time():
        try:
            tf = tel_file()
            tidx = current_tidx()
            return format_time(tf.times[tidx])
        except Exception:
            return "—"

    @output
    @render.text
    def stat_nodes():
        try:
            tf = tel_file()
            return f"{tf.npoin2:,}"
        except Exception:
            return "—"

    @output
    @render.text
    def stat_range():
        try:
            vals = effective_values()
            return f"{vals.min():.3g} – {vals.max():.3g}"
        except Exception:
            return "—"

    # -- Upload management --

    @reactive.effect
    @reactive.event(input.upload)
    def handle_upload_change():
        if input.upload():
            use_upload.set(True)

    @output
    @render.ui
    def clear_upload_ui():
        if not use_upload.get():
            return ui.div()
        return ui.input_action_button("clear_upload", "Clear upload (use examples)",
                                      class_="btn-sm btn-outline-danger w-100 mb-1")

    @output
    @render.ui
    def compare_upload_ui():
        if input.diff_mode():
            return ui.input_file("compare_upload", "Compare file (.slf)", accept=[".slf"])
        return ui.div()

    @reactive.effect
    @reactive.event(input.clear_upload)
    def handle_clear_upload():
        use_upload.set(False)

    # -- Observation CSV upload --

    @reactive.effect
    @reactive.event(input.obs_upload)
    def handle_obs_upload():
        uploaded = input.obs_upload()
        if not uploaded:
            obs_data.set(None)
            return
        try:
            path = uploaded[0]["datapath"]
            times, values, varname = parse_observation_csv(path)
            obs_data.set((times, values, varname))
            ui.notification_show(
                f"Loaded {len(times)} observation points for '{varname}'",
                duration=4,
            )
        except Exception as e:
            obs_data.set(None)
            ui.notification_show(f"Failed to parse observations: {e}", type="error", duration=6)

    # -- Core reactive calcs --

    _prev_tel_file = [None]  # plain list to avoid reactive dependency

    @reactive.calc
    def tel_file():
        # Close previous file to avoid leaking file descriptors
        if _prev_tel_file[0] is not None:
            with _tf_lock:
                try:
                    _prev_tel_file[0].close()
                except Exception:
                    pass
        uploaded = input.upload()
        if uploaded and use_upload.get():
            path = uploaded[0]["datapath"]
        else:
            path = EXAMPLES[input.example()]
        try:
            tf = TelemacFile(path)
        except Exception as e:
            ui.notification_show(f"Failed to open file: {e}", type="error", duration=8)
            # Re-raise to halt reactive chain — Shiny shows error state
            raise
        _prev_tel_file[0] = tf
        # Clear transient state from previous file
        particle_paths.set(None)
        cross_section_points.set(None)
        clicked_points.set([])
        temporal_stats_cache.set(None)
        integral_result.set(None)
        expr_result.set(None)
        measure_points.set([])
        measure_mode.set(False)
        analysis_mode.set("none")
        obs_data.set(None)
        compare_tf.set(None)
        volume_cache.set(None)
        polygon_stats_data.set(None)
        return tf

    # --- CRS reactive ---
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

        # Auto-detect disabled
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

    @output
    @render.ui
    def crs_offset_ui():
        if input.epsg_input() and input.epsg_input().strip():
            return ui.div(
                ui.input_numeric("crs_x_offset", "X offset (m)", value=0, step=1000),
                ui.input_numeric("crs_y_offset", "Y offset (m)", value=0, step=1000),
            )
        return ui.div()

    @reactive.calc
    def mesh_geom():
        tf = tel_file()
        crs = current_crs.get()
        try:
            x_offset = input.crs_x_offset() or 0
            y_offset = input.crs_y_offset() or 0
        except Exception:
            x_offset, y_offset = 0, 0
        origin_offset = (x_offset, y_offset)
        if is_3d_mode.get() and tf.nplan > 1:
            try:
                z_scale = input.z_scale() if input.z_scale() is not None else 10
            except Exception:
                z_scale = 10
            try:
                z_name = tf.get_z_name()
                z_vals = tf.get_data_value(z_name, current_tidx())
            except Exception as e:
                ui.notification_show(
                    f"Cannot read Z elevation: {e}. Showing flat mesh.",
                    type="warning", duration=8, id="z_warn")
                z_vals = np.zeros(tf.npoin2, dtype=np.float32)
            return build_mesh_geometry(tf, crs=crs, z_values=z_vals, z_scale=z_scale, origin_offset=origin_offset)
        return build_mesh_geometry(tf, crs=crs, origin_offset=origin_offset)

    @reactive.calc
    def current_var():
        tf = tel_file()
        try:
            var = input.variable()
        except Exception:
            var = None
        if var and var in tf.varnames:
            return var
        return tf.varnames[0]

    @reactive.calc
    def current_tidx():
        tf = tel_file()
        try:
            tidx = input.time_idx()
        except Exception:
            tidx = 0
        if tidx is None:
            tidx = 0
        return min(tidx, len(tf.times) - 1)

    @reactive.calc
    def current_values():
        tf = tel_file()
        var = current_var()
        tidx = current_tidx()
        derived = get_available_derived(tf)
        if var in derived:
            vals = compute_derived(tf, var, tidx)
        else:
            vals = tf.get_data_value(var, tidx)
        # 3D layer extraction
        try:
            layer = input.layer_select()
            if layer and layer != "all":
                nplan = getattr(tf, 'nplan', 0)
                if nplan > 1:
                    return extract_layer_2d(vals, tf.npoin2, int(layer))
        except Exception:
            pass
        return vals

    # -- Topology-only reactive calcs (independent of timestep) --

    @reactive.calc
    def mesh_quality_values():
        return compute_mesh_quality(tel_file())

    @reactive.calc
    def elem_area_values():
        return compute_element_area(tel_file())

    @reactive.calc
    def boundary_nodes_cached():
        return find_boundary_nodes(tel_file())

    @reactive.calc
    def cli_data():
        """Try to read .cli file from the same directory as the .slf."""
        uploaded = input.upload()
        if uploaded and use_upload.get():
            return None
        import glob
        path = EXAMPLES.get(input.example(), "")
        cli_files = glob.glob(_os.path.join(_os.path.dirname(path), "*.cli"))
        return read_cli_file(cli_files[0]) if cli_files else None

    # -- Dynamic UI --

    @output
    @render.ui
    def var_select_ui():
        tf = tel_file()
        file_vars = {v: v for v in tf.varnames}
        derived = get_available_derived(tf)
        if derived:
            choices = {"File variables": file_vars, "Derived": {d: d for d in derived}}
        else:
            choices = file_vars
        selected = "WATER DEPTH" if "WATER DEPTH" in tf.varnames else tf.varnames[0]
        return ui.input_select("variable", "Variable", choices=choices, selected=selected)

    @reactive.effect
    @reactive.event(input.variable)
    def auto_palette():
        """Auto-select palette based on variable semantics."""
        var = input.variable()
        if not var:
            return
        palette = suggest_palette(var)
        # _diverging is handled in update_map for bipolar vars, not via the select widget
        if palette and palette in PALETTES:
            ui.update_select("palette", selected=palette)

    @output
    @render.ui
    def time_slider_ui():
        tf = tel_file()
        n = len(tf.times)
        return ui.input_slider(
            "time_idx", "Time step", min=0, max=n - 1, value=n - 1, step=1
        )

    @output
    @render.ui
    def ref_timestep_ui():
        if not input.diff_mode():
            return ui.div()
        tf = tel_file()
        n = len(tf.times)
        return ui.input_slider(
            "ref_tidx", "Reference time step",
            min=0, max=n - 1, value=0, step=1,
        )

    @output
    @render.ui
    def color_range_ui():
        return ui.div(
            ui.input_switch("custom_range", "Custom color range", value=False),
            ui.output_ui("custom_range_inputs"),
        )

    @output
    @render.ui
    def custom_range_inputs():
        if not input.custom_range():
            return ui.div()
        vals = current_values()
        vmin, vmax = float(vals.min()), float(vals.max())
        return ui.div(
            ui.layout_columns(
                ui.input_numeric("color_min", "Min", value=round(vmin, 4), step=0.01),
                ui.input_numeric("color_max", "Max", value=round(vmax, 4), step=0.01),
                col_widths=[6, 6],
            ),
            class_="mb-1",
        )

    @output
    @render.ui
    def filter_ui():
        vals = current_values()
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        if np.isnan(vmin) or vmax <= vmin:
            vmin, vmax = 0.0, 1.0
        step = round((vmax - vmin) / 100, 4) if vmax > vmin else 0.01
        return ui.input_slider(
            "filter_range", "Value filter",
            min=round(vmin, 4), max=round(vmax, 4),
            value=[round(vmin, 4), round(vmax, 4)],
            step=step,
        )

    @output
    @render.ui
    def clear_xsec_ui():
        if cross_section_points.get() is None:
            return ui.div()
        return ui.input_action_button("clear_xsec", "Clear Cross-Section",
                                      class_="btn-sm btn-outline-danger w-100 mb-1")

    @output
    @render.ui
    def particle_seed_ui():
        if not input.particles():
            return ui.div()
        return ui.input_action_button("draw_seed", "Draw Seed Line",
                                      class_="btn-sm btn-outline-info w-100 mb-1")

    @output
    @render.ui
    def trail_length_ui():
        if not input.particles():
            return ui.div()
        tf = tel_file()
        max_trail = float(tf.times[-1] - tf.times[0]) if len(tf.times) > 1 else 1.0
        default_trail = max_trail * 0.2
        return ui.input_slider(
            "trail_length", "Trail length (s)",
            min=0, max=round(max_trail, 1), value=round(default_trail, 1),
            step=round(max_trail / 50, 2) if max_trail > 0 else 0.1,
        )

    @output
    @render.ui
    def toggle_3d_ui():
        tf = tel_file()
        if tf.nplan <= 1:
            return ui.div()
        return ui.div(
            ui.input_switch("view_3d", "3D View", value=False),
            ui.input_select("layer_select", "Display layer", choices=
                {"all": "All planes (3D)"} |
                {str(k): f"Layer {k}" + (" (bottom)" if k == 0 else " (surface)" if k == tf.nplan - 1 else "")
                 for k in range(tf.nplan)}
            ),
            ui.output_ui("z_scale_ui"),
        )

    @output
    @render.ui
    def z_scale_ui():
        if not input.view_3d():
            return ui.div()
        tf = tel_file()
        geom = mesh_geom()
        try:
            z_name = tf.get_z_name()
            z_vals = tf.get_data_value(z_name, 0)
            depth_range = float(z_vals.max() - z_vals.min())
        except Exception as e:
            ui.notification_show(
                f"Cannot determine depth range: {e}", type="warning", duration=5, id="zscale_warn")
            depth_range = 0.0
        default_scale = min(50, int(geom["extent_m"] / depth_range)) if depth_range > 0 else 10
        return ui.input_slider(
            "z_scale", "Z Scale",
            min=1, max=100, value=default_scale, step=1,
        )

    # -- Analysis panel --

    @output
    @render.ui
    def analysis_panel_ui():
        mode = analysis_mode.get()
        if mode == "none":
            return ui.div()
        titles = {"timeseries": "Time Series", "crosssection": "Cross-Section",
                  "vertprofile": "Vertical Profile", "histogram": "Value Histogram",
                  "multivar": "All Variables", "rating": "Rating Curve (h-Q)",
                  "volume": "Volume Conservation", "boundary_ts": "Boundary Time Series"}
        title = titles.get(mode, mode)
        n_pts = len(clicked_points.get()) if mode == "timeseries" else 0
        subtitle = f" ({n_pts} point{'s' if n_pts != 1 else ''})" if n_pts else ""
        return ui.div(
            ui.div(
                ui.strong(title + subtitle, style="color: #00b4d8;"),
                ui.download_button("download_csv", "CSV",
                                   class_="btn-sm btn-outline-success ms-2"),
                ui.download_button("download_all_vars", "All Vars",
                                   class_="btn-sm btn-outline-info ms-1"),
                ui.input_action_button("undo_point", "Undo",
                                       class_="btn-sm btn-outline-warning ms-1"),
                ui.input_action_button("close_analysis", "Close",
                                       class_="btn-sm btn-outline-secondary ms-1"),
                class_="d-flex align-items-center p-2",
                style="background: rgba(15, 25, 35, 0.92); border-top: 1px solid rgba(13, 115, 119, 0.3);",
            ),
            output_widget("analysis_chart"),
            style="height:250px; overflow:hidden; background: rgba(15, 25, 35, 0.85);",
        )

    @output
    @render_widget
    def analysis_chart():
        mode = analysis_mode.get()
        tf = tel_file()
        var = current_var()
        tidx = current_tidx()

        if mode == "timeseries":
            pts = clicked_points.get()
            if not pts:
                return go.Figure()
            fig = go.Figure()
            for i, pt in enumerate(pts):
                times, values = time_series_at_point(tf, var, pt[0], pt[1])
                fig.add_trace(go.Scatter(
                    x=times, y=values, mode="lines",
                    name=f"Pt {i+1} ({pt[0]:.0f}, {pt[1]:.0f})",
                ))
            if tidx < len(tf.times):
                fig.add_vline(x=tf.times[tidx], line_dash="dash", line_color="red")
            # Overlay observation data if available
            obs = obs_data.get()
            if obs is not None:
                obs_times, obs_values, obs_varname = obs
                fig.add_trace(go.Scatter(
                    x=obs_times, y=obs_values, mode="lines",
                    name=f"Obs: {obs_varname}",
                    line=dict(color="red", dash="dash"),
                ))
                # Compare with the last clicked point's model trace
                if pts:
                    last_pt = pts[-1]
                    model_times, model_values = time_series_at_point(tf, var, last_pt[0], last_pt[1])
                    model_interp = np.interp(obs_times, model_times, model_values)
                    rmse = compute_rmse(model_interp, obs_values)
                    nse = compute_nse(model_interp, obs_values)
                    fig.add_annotation(
                        text=f"RMSE={rmse:.4f}  NSE={nse:.4f}",
                        xref="paper", yref="paper",
                        x=0.02, y=0.98,
                        showarrow=False,
                        font=dict(size=11, color="red"),
                        bgcolor="rgba(255,255,255,0.8)",
                    )
            fig.update_layout(
                xaxis_title="Time (s)", yaxis_title=var,
                margin=dict(l=50, r=20, t=10, b=40), height=220,
            )
            return fig

        if mode == "crosssection":
            xsec_pts = cross_section_points.get()
            if xsec_pts is None:
                return go.Figure()
            abscissa, values = cross_section_profile(tf, var, tidx, xsec_pts)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=abscissa, y=values, mode="lines", name=var))
            fig.update_layout(
                xaxis_title="Distance (m)", yaxis_title=var,
                margin=dict(l=50, r=20, t=10, b=40), height=220,
            )
            return fig

        if mode == "vertprofile":
            pts = clicked_points.get()
            if not pts:
                return go.Figure()
            fig = go.Figure()
            for i, pt in enumerate(pts):
                elevations, values, elev_label = vertical_profile_at_point(tf, var, tidx, pt[0], pt[1])
                if len(elevations) > 0:
                    fig.add_trace(go.Scatter(
                        x=values, y=elevations, mode="lines+markers",
                        name=f"Pt {i+1} ({pt[0]:.0f}, {pt[1]:.0f})",
                    ))
            fig.update_layout(
                xaxis_title=var, yaxis_title=elev_label,
                margin=dict(l=50, r=20, t=10, b=40), height=220,
            )
            return fig

        if mode == "histogram":
            vals = effective_values()
            npoin = tf.npoin2
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=vals[:npoin], nbinsx=50, name=var))
            fig.update_layout(
                xaxis_title=var, yaxis_title="Count",
                margin=dict(l=50, r=20, t=10, b=40), height=220,
            )
            return fig

        if mode == "multivar":
            pts = clicked_points.get()
            if not pts:
                return go.Figure()
            px, py = pts[-1]
            fig = go.Figure()
            for vname in tf.varnames:
                times, values = time_series_at_point(tf, vname, px, py)
                fig.add_trace(go.Scatter(
                    x=times, y=values, mode="lines", name=vname,
                ))
            if tidx < len(tf.times):
                fig.add_vline(x=tf.times[tidx], line_dash="dash", line_color="red")
            fig.update_layout(
                xaxis_title="Time (s)", yaxis_title="Value",
                margin=dict(l=50, r=20, t=10, b=40), height=220,
            )
            return fig

        if mode == "rating":
            xsec = cross_section_points.get()
            if xsec is None:
                return go.Figure()
            tf = tel_file()
            # Compute discharge and avg water level at cross-section for each timestep
            h_values = []
            q_values = []
            for t in range(len(tf.times)):
                result = compute_discharge(tf, t, xsec)
                if result["total_q"] is not None:
                    q_values.append(result["total_q"])
                    # Average water level along the cross-section
                    try:
                        _, wl_vals = cross_section_profile(tf, "FREE SURFACE", t, xsec)
                        h_values.append(float(np.mean(wl_vals)))
                    except Exception:
                        try:
                            _, wd_vals = cross_section_profile(tf, "WATER DEPTH", t, xsec)
                            h_values.append(float(np.mean(wd_vals)))
                        except Exception:
                            h_values.append(0.0)
            fig = go.Figure()
            if h_values and q_values:
                fig.add_trace(go.Scatter(
                    x=h_values, y=q_values, mode="lines+markers",
                    name="h-Q", marker=dict(size=4),
                ))
            fig.update_layout(
                xaxis_title="Water level (m)", yaxis_title="Discharge (m\u00b3/s)",
                margin=dict(l=50, r=20, t=10, b=40), height=220,
            )
            return fig

        if mode == "volume":
            vc = volume_cache.get()
            if vc is None:
                return go.Figure()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=vc["times"], y=vc["volumes"], mode="lines", name="Volume"))
            fig.update_layout(
                xaxis_title="Time (s)", yaxis_title="Volume (m³)",
                margin=dict(l=50, r=20, t=10, b=40), height=220,
            )
            return fig

        if mode == "boundary_ts":
            data = liq_data()
            if data is None:
                return go.Figure()
            fig = go.Figure()
            for name, entry in data.items():
                label = f"{name} ({entry['unit']})" if entry.get("unit") else name
                fig.add_trace(go.Scatter(
                    x=entry["times"], y=entry["values"], mode="lines", name=label))
            fig.update_layout(
                xaxis_title="Time (s)", yaxis_title="Value",
                margin=dict(l=50, r=20, t=10, b=40), height=220,
            )
            return fig

        return go.Figure()

    # -- Click handler (time series) --

    @reactive.effect
    @reactive.event(input.map_click)
    def handle_map_click():
        click = input.map_click()
        if not click or "coordinate" not in click:
            return
        coord = click["coordinate"]
        geom = mesh_geom()
        x_m, y_m = click_to_native(coord[0], coord[1], geom)

        # Measurement mode intercepts clicks
        if measure_mode.get():
            pts = measure_points.get().copy()
            pts.append([x_m, y_m])
            measure_points.set(pts)
            if len(pts) >= 2:
                measure_mode.set(False)
            return

        pts = clicked_points.get().copy()
        pts.append((x_m, y_m))
        clicked_points.set(pts)
        tf = tel_file()
        if is_3d_mode.get() and tf.nplan > 1:
            analysis_mode.set("vertprofile")
        else:
            analysis_mode.set("timeseries")

    @reactive.effect
    @reactive.event(input.close_analysis)
    def handle_close_analysis():
        analysis_mode.set("none")
        clicked_points.set([])
        cross_section_points.set(None)

    # -- Cross-section drawing --

    @reactive.effect
    @reactive.event(input.draw_xsec)
    async def start_drawing():
        await map_widget.enable_draw(session, modes=["draw_line_string"])

    @reactive.effect
    @reactive.event(input.map_drawn_features)
    async def handle_drawn_features():
        features = input.map_drawn_features()
        if not features or "features" not in features:
            return
        for feat in features["features"]:
            geom_type = feat.get("geometry", {}).get("type")
            if geom_type == "Polygon" and polygon_mode.get():
                coords = feat["geometry"]["coordinates"][0]  # outer ring
                geom = mesh_geom()
                poly_m = [list(click_to_native(c[0], c[1], geom))
                          for c in coords]
                values = effective_values()
                stats = polygon_zonal_stats(tel_file(), values, poly_m, geom)
                polygon_stats_data.set(stats)
                polygon_mode.set(False)
                await map_widget.disable_draw(session)
                return
            if geom_type != "LineString":
                continue
            coords = feat["geometry"]["coordinates"]
            geom = mesh_geom()
            poly_m = [list(click_to_native(c[0], c[1], geom))
                      for c in coords]

            if input.particles():
                seeds = distribute_seeds_along_line(poly_m, n_seeds=100)
                tf = tel_file()
                ui.notification_show("Computing particle paths from seed line...",
                                     duration=None, id="particle_notif")
                loop = asyncio.get_running_loop()
                x_off, y_off = geom["x_off"], geom["y_off"]
                paths = await loop.run_in_executor(
                    None, _run_with_lock, compute_particle_paths, tf, seeds, x_off, y_off)
                particle_paths.set(paths)
                ui.notification_remove("particle_notif")
                ui.notification_show(f"Computed {len(paths)} particle paths", duration=3)
            else:
                # Cross-section mode
                cross_section_points.set(poly_m)
                clicked_points.set([])
                analysis_mode.set("crosssection")

            await map_widget.disable_draw(session)
            return

    @reactive.effect
    @reactive.event(input.clear_xsec)
    async def handle_clear_xsec():
        cross_section_points.set(None)
        analysis_mode.set("none")
        await map_widget.delete_drawn_features(session)

    # -- Particle tracing --

    @reactive.effect
    @reactive.event(input.particles)
    async def handle_particles_toggle():
        if input.particles():
            tf = tel_file()
            geom = mesh_geom()
            pair = find_velocity_pair(tf.varnames)
            if pair is None:
                ui.notification_show(
                    "No velocity variables found for particle tracing. "
                    "Expected: VELOCITY U/V, UX/UY, or QSBLX/QSBLY.",
                    type="warning", duration=6)
                particle_paths.set(None)
                return
            ui.notification_show("Computing particle trajectories...",
                                 duration=None, id="particle_notif")
            seeds = generate_seed_grid(tf, n_target=500)
            x_off, y_off = geom["x_off"], geom["y_off"]
            loop = asyncio.get_running_loop()
            paths = await loop.run_in_executor(
                None, _run_with_lock, compute_particle_paths, tf, seeds, x_off, y_off)
            particle_paths.set(paths)
            ui.notification_remove("particle_notif")
            ui.notification_show(f"Computed {len(paths)} particle paths", duration=3)
        else:
            particle_paths.set(None)

    @reactive.effect
    @reactive.event(input.draw_seed)
    async def start_seed_drawing():
        await map_widget.enable_draw(session, modes=["draw_line_string"])

    # -- 3D mode --

    @reactive.effect
    @reactive.event(input.view_3d)
    def sync_3d_mode():
        is_3d_mode.set(input.view_3d())

    # -- Playback --

    @reactive.effect
    @reactive.event(input.play_btn)
    def toggle_play():
        playing.set(not playing.get())
        if playing.get():
            ui.update_action_button("play_btn", label="Pause")
        else:
            ui.update_action_button("play_btn", label="Play")

    @reactive.effect
    def auto_advance():
        if not playing.get():
            return
        speed = input.speed() if input.speed() is not None else 0.5
        reactive.invalidate_later(speed)
        tf = tel_file()
        n = len(tf.times)
        with reactive.isolate():
            current = input.time_idx() if input.time_idx() is not None else 0
            loop = input.loop()
        next_idx = current + 1
        if next_idx >= n:
            if loop:
                next_idx = 0
            else:
                playing.set(False)
                ui.update_action_button("play_btn", label="Play")
                return
        ui.update_slider("time_idx", value=next_idx)

    # -- Discharge computation --

    @reactive.calc
    def discharge_result():
        xsec = cross_section_points.get()
        if xsec is None:
            return None
        return compute_discharge(tel_file(), current_tidx(), xsec)

    @output
    @render.ui
    def discharge_ui():
        result = discharge_result()
        if result is None:
            return ui.div()
        if result["total_q"] is None:
            return ui.div(
                ui.span(result.get("error", "Cannot compute discharge"),
                        class_="text-warning small"),
                class_="mb-1",
            )
        q = result["total_q"]
        skipped = result.get("skipped", 0)
        skip_text = f", {skipped} skipped" if skipped else ""
        return ui.div(
            ui.strong(f"Q = {q:.4f} m\u00b3/s", class_="small"),
            ui.span(f" ({len(result['segments'])} segments{skip_text})",
                    class_="text-muted small"),
            class_="mb-1",
        )

    # -- Rating curve --

    @output
    @render.ui
    def rating_curve_ui():
        xsec = cross_section_points.get()
        if xsec is None:
            return ui.div()
        return ui.input_action_button("show_rating", "Rating Curve (h-Q)",
                                      class_="btn-sm btn-outline-info w-100 mb-1")

    @reactive.effect
    @reactive.event(input.show_rating)
    def handle_show_rating():
        analysis_mode.set("rating")

    # -- Multi-variable time series --

    @reactive.effect
    @reactive.event(input.show_multivar)
    def handle_show_multivar():
        analysis_mode.set("multivar")

    # -- Goto time --

    @reactive.effect
    @reactive.event(input.goto_btn)
    def handle_goto_time():
        target = input.goto_time()
        if target is None:
            return
        tf = tel_file()
        times = np.array(tf.times)
        idx = int(np.argmin(np.abs(times - float(target))))
        ui.update_slider("time_idx", value=idx)

    # -- Histogram --

    @reactive.effect
    @reactive.event(input.show_histogram)
    def handle_show_histogram():
        analysis_mode.set("histogram")

    # -- Statistics --

    @output
    @render.ui
    def stats_ui():
        tf = tel_file()
        var = current_var()
        tidx = current_tidx()
        vals = current_values()
        t = tf.times[tidx]
        return ui.layout_column_wrap(
            ui.value_box("Time", format_time(t), f"Step {tidx + 1} of {len(tf.times)}"),
            ui.value_box("Min", f"{vals.min():.4f}", var),
            ui.value_box("Max", f"{vals.max():.4f}", var),
            ui.value_box("Mesh", f"{tf.nelem2:,} elements", f"{tf.npoin2:,} nodes"),
            width=1,
        )

    # -- Hover --

    @output
    @render.ui
    def hover_info_ui():
        hover = input.map_hover()
        if not hover or "coordinate" not in hover:
            return ui.p("Hover over the mesh to see values", class_="text-muted small")
        coord = hover["coordinate"]
        tf = tel_file()
        geom = mesh_geom()
        px, py = click_to_native(coord[0], coord[1], geom)
        idx, nx, ny = nearest_node(tf, px, py)
        vals = current_values()
        var = current_var()
        val = float(vals[idx])
        return ui.div(
            ui.strong(f"{var}: {val:.4f}"),
            ui.br(),
            ui.span(f"Node {idx} ({nx:.1f}, {ny:.1f})",
                     class_="text-muted small"),
        )

    # -- Temporal statistics --

    @reactive.effect
    @reactive.event(input.compute_temporal)
    async def handle_compute_temporal():
        tf = tel_file()
        var = current_var()
        ui.notification_show(f"Computing temporal stats for {var}...",
                             duration=None, id="temporal_notif")
        loop = asyncio.get_running_loop()
        def _compute_all_temporal(tf, var):
            stats = compute_temporal_stats(tf, var)
            if stats is not None:
                # Flood metrics always use WATER DEPTH when available
                flood_var = "WATER DEPTH" if "WATER DEPTH" in [v.strip() for v in tf.varnames] else var
                stats["envelope"] = compute_flood_envelope(tf, flood_var)
                stats["arrival"] = compute_flood_arrival(tf, flood_var)
                stats["duration"] = compute_flood_duration(tf, flood_var)
            return stats
        stats = await loop.run_in_executor(None, _run_with_lock, _compute_all_temporal, tf, var)
        temporal_stats_cache.set(stats)
        ui.notification_remove("temporal_notif")
        ui.notification_show("Temporal statistics computed", duration=3)

    # -- Volume conservation --

    @reactive.effect
    @reactive.event(input.compute_volume)
    async def handle_compute_volume():
        tf = tel_file()
        ui.notification_show("Computing volume over time...",
                             duration=None, id="volume_notif")
        loop = asyncio.get_running_loop()
        times, vols = await loop.run_in_executor(
            None, _run_with_lock, compute_volume_timeseries, tf, compute_mesh_integral)
        volume_cache.set({"times": times, "volumes": vols})
        ui.notification_remove("volume_notif")
        analysis_mode.set("volume")

    # -- .liq boundary display --

    @reactive.calc
    def liq_data():
        """Auto-detect and parse .liq file from the same directory as .slf."""
        uploaded = input.upload()
        if uploaded and use_upload.get():
            return None
        import glob
        path = EXAMPLES.get(input.example(), "")
        liq_files = glob.glob(_os.path.join(_os.path.dirname(path), "*.liq"))
        return parse_liq_file(liq_files[0]) if liq_files else None

    @output
    @render.ui
    def liq_display_ui():
        data = liq_data()
        if data is None:
            return ui.div()
        return ui.input_action_button("show_liq", "Show boundary time series",
                                      class_="btn-sm btn-outline-primary w-100 mb-1")

    @reactive.effect
    @reactive.event(input.show_liq)
    def handle_show_liq():
        analysis_mode.set("boundary_ts")

    @output
    @render.ui
    def temporal_stats_ui():
        stats = temporal_stats_cache.get()
        if stats is None:
            return ui.div()
        npoin = tel_file().npoin2
        return ui.div(
            ui.span("Temporal envelope:", class_="text-muted small"),
            ui.div(
                ui.span(f"Global min: {float(stats['min'][:npoin].min()):.4f}", class_="small"),
                ui.br(),
                ui.span(f"Global max: {float(stats['max'][:npoin].max()):.4f}", class_="small"),
                ui.br(),
                ui.span(f"Mean range: {float(stats['mean'][:npoin].min()):.4f} – {float(stats['mean'][:npoin].max()):.4f}", class_="small"),
            ),
            ui.input_select("temporal_display", "Show", choices={
                "none": "Current timestep",
                "min": "Temporal min",
                "max": "Temporal max",
                "mean": "Temporal mean",
                "envelope": "Flood envelope (max depth)",
                "arrival": "Flood arrival time (s)",
                "duration": "Flood duration (s)",
            }, selected="none"),
            class_="mt-2",
        )

    # -- Node inspector --

    @output
    @render.ui
    def node_inspector_ui():
        pts = clicked_points.get()
        if not pts:
            return ui.div()
        tf = tel_file()
        tidx = current_tidx()
        var = current_var()
        npoin = tf.npoin2
        x, y = tf.meshx, tf.meshy

        rows = []
        for i, (px, py) in enumerate(pts):
            idx, nx, ny = nearest_node(tf, px, py)
            val = float(tf.get_data_value(var, tidx)[idx])
            rows.append(
                ui.tags.tr(
                    ui.tags.td(f"P{i+1}", style="font-weight:bold;"),
                    ui.tags.td(f"{nx:.0f}", class_="text-muted"),
                    ui.tags.td(f"{ny:.0f}", class_="text-muted"),
                    ui.tags.td(f"{val:.4f}"),
                )
            )

        return ui.div(
            ui.strong(f"Probe points ({var})", class_="small"),
            ui.tags.table(
                ui.tags.thead(
                    ui.tags.tr(
                        ui.tags.th("#", style="width:30px;"),
                        ui.tags.th("X", style="width:60px;"),
                        ui.tags.th("Y", style="width:60px;"),
                        ui.tags.th("Value"),
                    ),
                ),
                ui.tags.tbody(*rows),
                class_="table table-sm table-striped mb-0",
                style="font-size:11px;",
            ),
            style="max-height:150px; overflow-y:auto;",
            class_="mt-2 border-top pt-2",
        )

    # -- Keyboard shortcuts --

    @reactive.effect
    @reactive.event(input.kb_next)
    def handle_kb_next():
        tf = tel_file()
        n = len(tf.times)
        current = input.time_idx() if input.time_idx() is not None else 0
        if current < n - 1:
            ui.update_slider("time_idx", value=current + 1)

    @reactive.effect
    @reactive.event(input.kb_prev)
    def handle_kb_prev():
        current = input.time_idx() if input.time_idx() is not None else 0
        if current > 0:
            ui.update_slider("time_idx", value=current - 1)

    @reactive.effect
    @reactive.event(input.kb_play)
    def handle_kb_play():
        playing.set(not playing.get())
        if playing.get():
            ui.update_action_button("play_btn", label="Pause")
        else:
            ui.update_action_button("play_btn", label="Play")

    @reactive.effect
    @reactive.event(input.kb_var_next)
    def handle_kb_var_next():
        tf = tel_file()
        all_vars = list(tf.varnames) + get_available_derived(tf)
        cur = current_var()
        if cur in all_vars:
            idx = (all_vars.index(cur) + 1) % len(all_vars)
            ui.update_select("variable", selected=all_vars[idx])

    @reactive.effect
    @reactive.event(input.kb_var_prev)
    def handle_kb_var_prev():
        tf = tel_file()
        all_vars = list(tf.varnames) + get_available_derived(tf)
        cur = current_var()
        if cur in all_vars:
            idx = (all_vars.index(cur) - 1) % len(all_vars)
            ui.update_select("variable", selected=all_vars[idx])

    # -- Coordinate readout --

    @output
    @render.ui
    def coord_readout_ui():
        hover = input.map_hover()
        _style = (
            "font-size:11px; padding:4px 10px; "
            "background: rgba(15, 25, 35, 0.85); color: #c8dce8; "
            "border-top: 1px solid rgba(13, 115, 119, 0.3); "
            "font-family: monospace;"
        )
        if not hover or "coordinate" not in hover:
            return ui.div(
                ui.span("Move cursor over mesh", style="opacity:0.5;"),
                style=_style,
            )
        coord = hover["coordinate"]
        geom = mesh_geom()
        x_m, y_m = click_to_native(coord[0], coord[1], geom)
        # Show value at nearest node if available
        try:
            tf = tel_file()
            var = current_var()
            tidx = current_tidx()
            idx, _, _ = nearest_node(tf, x_m, y_m)
            val = tf.get_data_value(var, tidx)[idx]
            val_str = f"  {var}: {val:.4g}"
        except Exception:
            val_str = ""
        wgs84 = meters_to_wgs84(x_m, y_m, geom)
        if wgs84:
            lon, lat = wgs84
            coord_text = f"{x_m:.0f}, {y_m:.0f}  |  {lon:.6f}°E, {lat:.6f}°N"
        else:
            coord_text = f"X: {x_m:.1f}   Y: {y_m:.1f}"
        return ui.div(
            ui.span(coord_text, style="color: #00b4d8;"),
            ui.span(val_str, style="color: #48c78e; margin-left: 12px;"),
            style=_style,
        )

    # -- Undo last point --

    @reactive.effect
    @reactive.event(input.undo_point)
    def handle_undo_point():
        pts = clicked_points.get().copy()
        if pts:
            pts.pop()
            clicked_points.set(pts)
            if not pts:
                analysis_mode.set("none")

    # -- Custom expression --

    expr_result = reactive.value(None)  # numpy array or None

    @reactive.effect
    @reactive.event(input.variable, input.example, input.upload)
    def clear_expr_on_change():
        expr_result.set(None)

    @reactive.effect
    @reactive.event(input.eval_expr)
    def handle_eval_expr():
        expr = input.expr_input()
        if not expr:
            return
        tf = tel_file()
        tidx = current_tidx()
        try:
            result = evaluate_expression(tf, tidx, expr)
            expr_result.set(result)
            ui.notification_show(f"Expression evaluated: {expr}", duration=3)
        except Exception as e:
            expr_result.set(None)
            ui.notification_show(f"Expression error: {e}", type="error", duration=5)

    # -- Mesh integral --

    integral_result = reactive.value(None)

    @reactive.effect
    @reactive.event(input.compute_integral)
    def handle_compute_integral():
        tf = tel_file()
        values = effective_values()
        result = compute_mesh_integral(tf, values, threshold=0.001)
        integral_result.set(result)

    @output
    @render.ui
    def integral_ui():
        result = integral_result.get()
        if result is None:
            return ui.div()
        return ui.div(
            ui.span(f"Total area: {result['total_area']:.1f} m²", class_="small"),
            ui.br(),
            ui.span(f"Integral: {result['integral']:.4g}", class_="small"),
            ui.br(),
            ui.span(f"Area mean: {result['mean']:.4f}", class_="small"),
            ui.br(),
            ui.span(f"Wetted: {result['wetted_fraction']*100:.1f}% ({result['wetted_area']:.1f} m²)",
                     class_="small"),
            class_="mt-1 mb-1",
        )

    # -- Measurement tool --

    @reactive.effect
    @reactive.event(input.measure_btn)
    def handle_measure_btn():
        measure_points.set([])
        measure_mode.set(True)
        ui.notification_show("Click two points on the mesh to measure distance", duration=3)

    @output
    @render.ui
    def measure_info_ui():
        pts = measure_points.get()
        if len(pts) < 2:
            if measure_mode.get():
                return ui.p(f"Click point {len(pts)+1} of 2...", class_="text-muted small mb-1")
            return ui.div()
        p1, p2 = pts[0], pts[1]
        dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        return ui.div(
            ui.strong(f"Distance: {dist:.2f} m", class_="small"),
            ui.input_action_button("clear_measure", "Clear",
                                   class_="btn-sm btn-outline-secondary ms-2"),
            class_="d-flex align-items-center mb-1",
        )

    @reactive.effect
    @reactive.event(input.clear_measure)
    def handle_clear_measure():
        measure_points.set([])
        measure_mode.set(False)

    # -- Animation recording --

    @reactive.effect
    @reactive.event(input.record_btn)
    async def handle_record():
        rec = not recording.get()
        recording.set(rec)
        await session.send_custom_message("toggle_recording", {"start": rec})
        if rec:
            ui.update_action_button("record_btn", label="Stop Recording")
        else:
            ui.update_action_button("record_btn", label="Record")

    @reactive.effect
    @reactive.event(input.record_error)
    def handle_record_error():
        ui.notification_show(f"Recording failed: {input.record_error()}", type="error", duration=5)

    # -- Secondary file overlay --

    @reactive.effect
    @reactive.event(input.compare_upload)
    def handle_compare_upload():
        uploaded = input.compare_upload()
        if not uploaded:
            return
        try:
            tf2 = TelemacFile(uploaded[0]["datapath"])
            if tf2.npoin2 != tel_file().npoin2:
                ui.notification_show(f"Mesh mismatch: {tf2.npoin2} vs {tel_file().npoin2} nodes", type="error", duration=5)
                return
            compare_tf.set(tf2)
            ui.notification_show(f"Comparison file loaded: {tf2.npoin2} nodes, {len(tf2.varnames)} vars", duration=3)
        except Exception as e:
            ui.notification_show(f"Failed to open comparison file: {e}", type="error", duration=5)

    # -- Polygon zonal statistics --

    @reactive.effect
    @reactive.event(input.draw_polygon)
    async def start_polygon_draw():
        polygon_mode.set(True)
        await map_widget.enable_draw(session, modes=["draw_polygon"])

    @output
    @render.ui
    def polygon_stats_ui():
        stats = polygon_stats_data.get()
        if stats is None:
            return ui.div()
        return ui.div(
            ui.strong("Polygon Statistics", class_="small"),
            ui.div(
                ui.span(f"Area: {stats['area']:.0f} m²", class_="small"), ui.br(),
                ui.span(f"Nodes inside: {stats['count']}", class_="small"), ui.br(),
                ui.span(f"Mean: {stats['mean']:.4f}", class_="small"), ui.br(),
                ui.span(f"Min: {stats['min']:.4f}  Max: {stats['max']:.4f}", class_="small"), ui.br(),
                ui.span(f"Flooded: {stats['flooded_fraction']*100:.1f}%", class_="small"),
            ),
            class_="mt-1 mb-1 border rounded p-2",
        )

    # -- Comparison variable --

    @output
    @render.ui
    def compare_var_ui():
        tf = tel_file()
        choices = {"": "None (off)"}
        choices.update({v: v for v in tf.varnames})
        ctf = compare_tf.get()
        if ctf is not None:
            for v in ctf.varnames:
                choices[f"(2) {v}"] = f"(2) {v}"
        return ui.input_select("compare_var", "Contour overlay variable",
                               choices=choices, selected="")

    # -- File info --

    @output
    @render.ui
    def file_info_ui():
        tf = tel_file()
        dt = tf.times[1] - tf.times[0] if len(tf.times) > 1 else 0
        precision = "Double" if getattr(tf, 'float_type', 'f') == 'd' else "Single"
        nplan = getattr(tf, 'nplan', 0)
        dim = "3D" if nplan > 1 else "2D"
        module = "TELEMAC-3D" if nplan > 1 else detect_module_vars(tf.varnames)
        date_info = ""
        if hasattr(tf, 'datetime') and tf.datetime is not None:
            dt_val = tf.datetime
            if hasattr(dt_val, '__len__') and len(dt_val) > 0:
                date_info = str(dt_val[0])
            else:
                date_info = str(dt_val)

        rows = [
            ("Module", module),
            ("Type", f"{dim} ({nplan} planes)" if nplan > 1 else dim),
            ("Nodes", f"{tf.npoin2:,}"),
            ("Elements", f"{tf.nelem2:,}"),
            ("Variables", f"{len(tf.varnames)}"),
            ("Time steps", f"{len(tf.times)}"),
            ("dt", f"{dt:.2f} s" if dt else "N/A"),
            ("Duration", format_time(float(tf.times[-1] - tf.times[0])) if len(tf.times) > 1 else "N/A"),
            ("Precision", precision),
        ]
        if date_info:
            rows.append(("Start date", date_info))

        items = [ui.div(
            ui.span(label, class_="text-muted", style="width:80px;display:inline-block;"),
            ui.strong(value),
            class_="small mb-1",
        ) for label, value in rows]

        var_list = ", ".join(tf.varnames)
        items.append(ui.div(
            ui.span("Vars: ", class_="text-muted small"),
            ui.span(var_list, class_="small"),
            class_="mt-1",
        ))
        return ui.div(*items)

    # -- CSV download --

    @render.download(filename=lambda: f"telemac_{analysis_mode.get()}.csv")
    def download_csv():
        mode = analysis_mode.get()
        tf = tel_file()
        var = current_var()
        tidx = current_tidx()

        if mode == "timeseries":
            pts = clicked_points.get()
            if not pts:
                yield ""
                return
            # Export all points as columns
            buf = io.StringIO()
            geom = mesh_geom()
            crs = current_crs.get()
            if crs:
                buf.write(f"# CRS: EPSG:{crs.epsg} ({crs.name})\n")
            times_arr = np.array(tf.times)
            buf.write("Time (s)")
            all_values = []
            for i, pt in enumerate(pts):
                _, values = time_series_at_point(tf, var, pt[0], pt[1])
                all_values.append(values)
                wgs84 = meters_to_wgs84(pt[0], pt[1], geom)
                if wgs84:
                    lon, lat = wgs84
                    buf.write(f",Pt{i+1} ({pt[0]:.0f} {pt[1]:.0f} | {lon:.6f}E {lat:.6f}N)")
                else:
                    buf.write(f",Pt{i+1} ({pt[0]:.0f} {pt[1]:.0f})")
            buf.write("\n")
            for j in range(len(times_arr)):
                buf.write(f"{times_arr[j]}")
                for vals in all_values:
                    buf.write(f",{vals[j]}")
                buf.write("\n")
            yield buf.getvalue()

        elif mode == "crosssection":
            xsec_pts = cross_section_points.get()
            if xsec_pts is None:
                yield ""
                return
            abscissa, values = cross_section_profile(tf, var, tidx, xsec_pts)
            yield export_crosssection_csv(abscissa, values, var)
        elif mode == "rating":
            yield ""
        else:
            yield ""

    @render.download(filename=lambda: "telemac_all_vars.csv")
    def download_all_vars():
        pts = clicked_points.get()
        if not pts:
            yield ""
            return
        tf = tel_file()
        tidx = current_tidx()
        yield export_all_variables_csv(tf, tidx, pts[-1][0], pts[-1][1])

    # -- Mesh quality override --

    @reactive.calc
    def effective_values():
        """Return expression, mesh quality, slope, difference, temporal stats, or current values."""
        er = expr_result.get()
        if er is not None:
            return er
        diag = input.diagnostic() if input.diagnostic() else "none"
        if diag == "mesh_quality":
            return mesh_quality_values()
        elif diag == "slope":
            return compute_slope(tel_file(), current_values())
        elif diag == "courant":
            cfl = compute_courant_number(tel_file(), current_tidx())
            if cfl is not None:
                return cfl
            ui.notification_show("Courant number requires VELOCITY U/V variables",
                                 type="warning", duration=4, id="cfl_warn")
        elif diag == "elem_area":
            return elem_area_values()
        # Temporal stats display
        stats = temporal_stats_cache.get()
        try:
            td = input.temporal_display() or "none"
        except Exception:
            td = "none"
        if stats is not None and td != "none":
            return stats[td]
        if input.diff_mode():
            tf = tel_file()
            var = current_var()
            tidx = current_tidx()
            try:
                ref = input.ref_tidx() if input.ref_tidx() is not None else 0
            except Exception:
                ref = 0
            return compute_difference(tf, var, tidx, ref)
        return current_values()

    # -- Simulation launcher --

    @output
    @render.ui
    def cas_select_ui():
        uploaded = input.upload()
        if uploaded:
            return ui.p("Upload mode — no .cas files available", class_="text-muted small")
        path = EXAMPLES.get(input.example(), "")
        cas_files = find_cas_files(path)
        if not cas_files:
            return ui.p("No .cas files found near this example", class_="text-muted small")
        choices = {name: name for name in cas_files}
        return ui.input_select("cas_file", "Steering file (.cas)", choices=choices)

    @output
    @render.ui
    def sim_status_ui():
        if sim_running.get():
            return ui.div(
                ui.span("Running", class_="badge bg-success"),
                class_="mb-1",
            )
        return ui.div()

    @output
    @render.text
    def sim_console():
        return sim_output.get()

    @reactive.effect
    @reactive.event(input.run_sim)
    async def handle_run_sim():
        if sim_running.get():
            ui.notification_show("Simulation already running", type="warning", duration=3)
            return
        uploaded = input.upload()
        if uploaded:
            ui.notification_show("Cannot run simulation on uploaded files", type="warning", duration=3)
            return
        path = EXAMPLES.get(input.example(), "")
        cas_files = find_cas_files(path)
        try:
            cas_name = input.cas_file() if input.cas_file() else None
        except Exception:
            cas_name = None
        if not cas_name or cas_name not in cas_files:
            ui.notification_show("No .cas file selected", type="warning", duration=3)
            return
        cas_path = cas_files[cas_name]
        module = detect_module(cas_path)
        ncores = input.ncores() if input.ncores() is not None else 4
        cas_dir = _os.path.dirname(cas_path)

        # Build command
        runner = f"{module}.py"
        cmd_display = f"{runner} {cas_name} --ncsize={ncores}"

        sim_output.set(f"$ cd {cas_dir}\n$ {cmd_display}\n\n")
        sim_running.set(True)

        try:
            # Source TELEMAC env and run (quote all paths for shell safety)
            env_script = _os.path.join(_os.environ.get("HOMETEL", ""), "configs/pysource.local.sh")
            shell_cmd = (f"source {shlex.quote(env_script)} && "
                         f"cd {shlex.quote(cas_dir)} && "
                         f"{shlex.quote(runner)} {shlex.quote(cas_name)} "
                         f"--ncsize={int(ncores)} 2>&1")
            proc = await asyncio.create_subprocess_shell(
                shell_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            sim_process.set(proc)

            # Read output incrementally, capped at 500 lines
            lines_buf = [sim_output.get()]
            line_count = 0
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                lines_buf.append(line.decode())
                line_count += 1
                if len(lines_buf) > 500:
                    lines_buf = lines_buf[-500:]
                if line_count % 10 == 0:
                    sim_output.set("".join(lines_buf))
                    await reactive.flush()
            await proc.wait()
            lines_buf.append(f"\n--- Process exited with code {proc.returncode} ---\n")
            sim_output.set("".join(lines_buf))
        except Exception as e:
            sim_output.set(sim_output.get() + f"\nERROR: {e}\n")
        finally:
            sim_running.set(False)
            sim_process.set(None)

    @reactive.effect
    @reactive.event(input.stop_sim)
    def handle_stop_sim():
        proc = sim_process.get()
        if proc and proc.returncode is None:
            proc.terminate()
            sim_output.set(sim_output.get() + "\n--- Terminated by user ---\n")
            sim_running.set(False)
            sim_process.set(None)
        else:
            ui.notification_show("No simulation running", type="info", duration=2)

    # -- Map update --

    @reactive.effect
    async def update_map():
        tf = tel_file()
        geom = mesh_geom()
        var = current_var()
        tidx = current_tidx()
        values = effective_values()
        palette_id = input.palette() if input.palette() in PALETTES else "Viridis"

        # Auto-switch to diverging palette for bipolar variables or difference mode
        diag = input.diagnostic() if input.diagnostic() else "none"
        if is_bipolar(var) and diag == "none" and not input.diff_mode():
            palette_id = "_diverging"
        elif input.diff_mode() and diag == "none":
            palette_id = "_diverging"

        # Display variable label
        try:
            td = input.temporal_display() or "none"
        except Exception:
            td = "none"
        if expr_result.get() is not None:
            display_var = f"EXPR: {input.expr_input()}"
        elif diag == "mesh_quality":
            display_var = "MESH QUALITY"
        elif diag == "slope":
            display_var = f"SLOPE ({var})"
        elif diag == "courant":
            display_var = "COURANT NUMBER"
        elif diag == "elem_area":
            display_var = "ELEMENT AREA (m²)"
        elif td != "none" and temporal_stats_cache.get() is not None:
            display_var = f"{var} (temporal {td})"
        elif input.diff_mode():
            try:
                ref = input.ref_tidx() if input.ref_tidx() is not None else 0
            except Exception:
                ref = 0
            display_var = f"Δ {var} (t{tidx}-t{ref})"
        else:
            display_var = var

        # Custom color range
        crange = None
        use_diverging = palette_id == "_diverging"
        try:
            custom_range = input.custom_range()
        except Exception:
            custom_range = False
        if use_diverging and not custom_range:
            abs_max = max(abs(float(values.min())), abs(float(values.max())))
            if abs_max > 0:
                crange = (-abs_max, abs_max)
        elif custom_range:
            try:
                cmin = input.color_min()
                cmax = input.color_max()
            except Exception:
                cmin, cmax = None, None
            if cmin is not None and cmax is not None:
                crange = (cmin, cmax)

        try:
            filt = input.filter_range()
        except Exception:
            filt = None
        try:
            use_log = input.log_scale()
        except Exception:
            use_log = False
        try:
            reverse = input.reverse_palette()
        except Exception:
            reverse = False
        origin = [geom["lon_off"], geom["lat_off"]]
        lyr, vmin, vmax, log_applied = build_mesh_layer(geom, values, palette_id,
                                           filter_range=filt,
                                           color_range_override=crange,
                                           log_scale=use_log,
                                           reverse_palette=reverse,
                                           origin=origin)
        if use_log and not log_applied:
            ui.notification_show(
                "Log scale requires positive values — using linear scale",
                type="warning", duration=3, id="log_warn")

        layers = [lyr]

        if input.wireframe():
            layers.append(build_wireframe_layer(tf, geom, origin=origin))

        # Boundary edges (color-coded by hydrodynamic type)
        if input.boundary_nodes():
            layers.extend(build_boundary_layer(tf, geom, boundary_nodes_cached(), bc_types=cli_data(), origin=origin))

        # Min/max location markers
        if input.show_extrema():
            extrema = find_extrema(tf, values)
            layers.extend(build_extrema_markers(extrema, geom["x_off"], geom["y_off"], origin=origin))

        if input.vectors():
            vlyr = build_velocity_layer(tf, tidx, geom, origin=origin)
            if vlyr is not None:
                layers.append(vlyr)

        if input.contours():
            clyr = build_contour_layer_fn(tf, values, geom, origin=origin)
            if clyr is not None:
                layers.append(clyr)

        # Comparison variable contour overlay
        try:
            compare = input.compare_var() or ""
        except Exception:
            compare = ""
        compare_vals = None
        if compare.startswith("(2) ") and compare_tf.get() is not None:
            real_name = compare[4:]
            compare_vals = compare_tf.get().get_data_value(real_name, tidx)
        elif compare and compare in tf.varnames:
            compare_vals = tf.get_data_value(compare, tidx)
        if compare_vals is not None:
            clyr2 = build_contour_layer_fn(tf, compare_vals, geom, n_contours=8,
                                           layer_id="compare-contours",
                                           contour_color=[0, 0, 180],
                                           origin=origin)
            if clyr2 is not None:
                layers.append(clyr2)

        # Markers for clicked points
        pts = clicked_points.get()
        for i, pt in enumerate(pts):
            mx, my = float(pt[0] - geom["x_off"]), float(pt[1] - geom["y_off"])
            layers.append(build_marker_layer(mx, my, layer_id=f"marker-{i}", origin=origin))

        # Cross-section path
        xsec = cross_section_points.get()
        if xsec is not None:
            path_centered = [[x - geom["x_off"], y - geom["y_off"]] for x, y in xsec]
            layers.append(build_cross_section_layer(path_centered, origin=origin))

        # Particle traces
        paths = particle_paths.get()
        if paths and input.particles():
            current_time = float(tf.times[tidx])
            trail = input.trail_length() if input.trail_length() is not None else 1.0
            layers.append(build_particle_layer(paths, current_time, trail, origin=origin))

        # Measurement line (convert mesh-meter coords to centered for layer)
        mpts = measure_points.get()
        if mpts:
            mpts_centered = [[p[0] - geom["x_off"], p[1] - geom["y_off"]] for p in mpts]
            layers.extend(build_measurement_layer(mpts_centered, origin=origin))

        gradient_colors = cached_gradient_colors(palette_id, reverse=reverse)
        legend_entries = [{
            "layer_id": "mesh",
            "label": f"{display_var}  [{vmin:.3g} - {vmax:.3g}]",
            "colors": gradient_colors,
            "shape": "gradient",
        }]
        # Add boundary type entries to legend
        if input.boundary_nodes():
            legend_entries.extend([
                {"layer_id": "boundary-wall", "label": "Wall", "color": [160, 160, 170], "shape": "line"},
                {"layer_id": "boundary-free", "label": "Free (Neumann)", "color": [0, 200, 80], "shape": "line"},
                {"layer_id": "boundary-prescribed", "label": "Prescribed (H/Q)", "color": [40, 120, 255], "shape": "line"},
            ])
        legend = layer_legend_widget(
            entries=legend_entries,
            placement="bottom-right",
            show_checkbox=False,
            title="Legend",
        )

        # Only update view_state on file change
        uploaded = input.upload()
        current_path = (uploaded[0]["datapath"] if uploaded and use_upload.get()
                        else EXAMPLES.get(input.example(), ""))
        kwargs = {}
        if current_path != last_file_path.get():
            last_file_path.set(current_path)
            kwargs["view_state"] = {
                "longitude": geom["lon_off"],
                "latitude": geom["lat_off"],
                "zoom": geom["zoom"],
            }

        # Build widgets list
        widgets = [
            zoom_widget(placement="top-right"),
            fullscreen_widget(placement="top-right"),
            screenshot_widget(placement="top-right"),
            compass_widget(placement="top-right"),
            reset_view_widget(placement="top-right"),
            scale_widget(placement="bottom-left"),
            loading_widget(placement="top-left"),
            legend,
        ]

        # Map background / basemap
        _DARK_BG = "data:application/json;charset=utf-8,%7B%22version%22%3A8%2C%22sources%22%3A%7B%7D%2C%22layers%22%3A%5B%7B%22id%22%3A%22bg%22%2C%22type%22%3A%22background%22%2C%22paint%22%3A%7B%22background-color%22%3A%22%230f1923%22%7D%7D%5D%7D"
        _LIGHT_BG = "data:application/json;charset=utf-8,%7B%22version%22%3A8%2C%22sources%22%3A%7B%7D%2C%22layers%22%3A%5B%7B%22id%22%3A%22bg%22%2C%22type%22%3A%22background%22%2C%22paint%22%3A%7B%22background-color%22%3A%22%23f0f4f8%22%7D%7D%5D%7D"
        try:
            basemap = input.basemap() or "dark"
        except Exception:
            basemap = "dark"
        _BASEMAP_STYLES = {
            "light": _LIGHT_BG,
            "dark": _DARK_BG,
            "osm": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
            "satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        }
        if basemap != "dark":
            kwargs["style"] = _BASEMAP_STYLES.get(basemap, _DARK_BG)

        # 3D mode: add gimbal widget, first-person view, and lighting
        if is_3d_mode.get():
            widgets.append(gimbal_widget())
            kwargs["views"] = [first_person_view(
                focalDistance=10,
                maxPitchAngle=80,
            )]
            kwargs["effects"] = [lighting_effect(
                ambient_light(intensity=0.6),
                directional_light(direction=[-1, -3, -1], intensity=0.8),
                directional_light(direction=[1, 2, -0.5], intensity=0.3),
            )]

        await map_widget.update(
            session,
            layers=layers,
            widgets=widgets,
            **kwargs,
        )


    # ── Import tab server logic ─────────────────────────────────────────

    import_model = reactive.value(None)     # parsed HecRasModel

    import_map_widget = MapWidget(
        "import_map",
        view_state={"longitude": 0, "latitude": 0, "zoom": 0},
        style="data:application/json;charset=utf-8,%7B%22version%22%3A8%2C%22sources%22%3A%7B%7D%2C%22layers%22%3A%5B%7B%22id%22%3A%22bg%22%2C%22type%22%3A%22background%22%2C%22paint%22%3A%7B%22background-color%22%3A%22%230f1923%22%7D%7D%5D%7D",
    )

    @render_widget
    def import_map():
        return import_map_widget
    import_output_dir = reactive.value(None)  # path to generated files
    import_log_text = reactive.value("")

    def _append_log(msg: str):
        import_log_text.set(import_log_text.get() + msg + "\n")

    @output
    @render.text
    def import_log():
        return import_log_text.get()

    _IMPORT_METER_OFFSETS = 2

    def _build_import_preview_layers(model, x_off, y_off):
        """Build deck.gl preview layers from parsed HecRasModel."""
        layers = []

        for reach in model.rivers:
            # River alignment (cyan path)
            path = [[float(p[0] - x_off), float(p[1] - y_off)] for p in reach.alignment]
            layers.append(path_layer(
                f"alignment-{reach.name}",
                [{"path": path}],
                getPath="@@=d.path",
                getColor=[0, 200, 255, 200],
                getWidth=4,
                widthMinPixels=2,
                widthMaxPixels=6,
                pickable=False,
                coordinateSystem=_IMPORT_METER_OFFSETS,
                coordinateOrigin=[0, 0],
            ))

            # Cross-section lines (yellow)
            xs_lines = []
            for xs in reach.cross_sections:
                if xs.coords.shape[0] >= 2:
                    xs_lines.append({
                        "sourcePosition": [float(xs.coords[0, 0] - x_off), float(xs.coords[0, 1] - y_off)],
                        "targetPosition": [float(xs.coords[-1, 0] - x_off), float(xs.coords[-1, 1] - y_off)],
                    })
            if xs_lines:
                layers.append(line_layer(
                    f"cross-sections-{reach.name}",
                    xs_lines,
                    getColor=[255, 220, 0, 180],
                    getWidth=2,
                    widthMinPixels=1,
                    widthMaxPixels=3,
                    pickable=False,
                    coordinateSystem=_IMPORT_METER_OFFSETS,
                    coordinateOrigin=[0, 0],
                ))

        # BC lines
        for i, bc in enumerate(model.boundaries):
            if bc.line_coords is not None and len(bc.line_coords) >= 2:
                color = [40, 120, 255, 220] if bc.location == "upstream" else [0, 200, 80, 220]
                layers.append(line_layer(
                    f"bc-{i}",
                    [{"sourcePosition": [float(bc.line_coords[0, 0] - x_off), float(bc.line_coords[0, 1] - y_off)],
                      "targetPosition": [float(bc.line_coords[-1, 0] - x_off), float(bc.line_coords[-1, 1] - y_off)]}],
                    getColor=color,
                    getWidth=4,
                    widthMinPixels=2,
                    widthMaxPixels=5,
                    pickable=False,
                    coordinateSystem=_IMPORT_METER_OFFSETS,
                    coordinateOrigin=[0, 0],
                ))

        # 2D area face points (scatter, subsampled)
        for area in model.areas_2d:
            step = max(1, len(area.face_points) // 2000)
            pts = [{"position": [float(p[0] - x_off), float(p[1] - y_off)]} for p in area.face_points[::step]]
            layers.append(scatterplot_layer(
                f"2d-{area.name}",
                pts,
                getPosition="@@=d.position",
                getColor=[0, 200, 255, 120],
                getRadius=3,
                radiusMinPixels=1,
                radiusMaxPixels=4,
                pickable=False,
                coordinateSystem=_IMPORT_METER_OFFSETS,
                coordinateOrigin=[0, 0],
            ))

        return layers

    @reactive.effect
    @reactive.event(input.import_preview)
    def handle_import_preview():
        """Parse HEC-RAS file and show summary."""
        import_log_text.set("")
        hdf_files = input.import_hdf()
        if not hdf_files:
            _append_log("Please upload a HEC-RAS geometry file (.hdf)")
            return

        hdf_path = hdf_files[0]["datapath"]
        _append_log(f"Parsing: {hdf_files[0]['name']}")

        try:
            from telemac_tools.hecras import parse_hecras
            model = parse_hecras(hdf_path)
            import_model.set(model)

            if model.rivers:
                for r in model.rivers:
                    _append_log(f"  Reach: {r.name}")
                    _append_log(f"    Cross-sections: {len(r.cross_sections)}")
                    if r.cross_sections:
                        stations = [xs.station for xs in r.cross_sections]
                        _append_log(f"    Station range: {min(stations):.0f} – {max(stations):.0f} m")
                    _append_log(f"    Alignment points: {r.alignment.shape[0]}")
            if model.areas_2d:
                for a in model.areas_2d:
                    _append_log(f"  2D Area: {a.name}")
                    _append_log(f"    Cells: {len(a.cells)}")
                    _append_log(f"    Face points: {a.face_points.shape[0]}")
            if model.boundaries:
                _append_log(f"  Boundary conditions: {len(model.boundaries)}")
                for bc in model.boundaries:
                    _append_log(f"    {bc.location}: {bc.bc_type}")

            # Compute center and zoom for preview map
            all_x, all_y = [], []
            for r in model.rivers:
                all_x.extend(r.alignment[:, 0].tolist())
                all_y.extend(r.alignment[:, 1].tolist())
            for a in model.areas_2d:
                all_x.extend(a.face_points[:, 0].tolist())
                all_y.extend(a.face_points[:, 1].tolist())
            if all_x:
                x_off = (min(all_x) + max(all_x)) / 2
                y_off = (min(all_y) + max(all_y)) / 2
                extent = max(max(all_x) - min(all_x), max(all_y) - min(all_y), 1.0)
                zoom = math.log2(600 * 360 / (256 * (extent / 111320))) if extent > 0 else 10

                preview_layers = _build_import_preview_layers(model, x_off, y_off)
                asyncio.ensure_future(import_map_widget.update(
                    session,
                    layers=preview_layers,
                    view_state={"longitude": 0, "latitude": 0, "zoom": zoom},
                ))

            _append_log("\nReady to convert. Click 'Convert' to generate TELEMAC files.")

        except Exception as e:
            _append_log(f"ERROR: {e}")
            import_model.set(None)

    @reactive.effect
    @reactive.event(input.import_convert)
    def handle_import_convert():
        """Run full conversion pipeline."""
        model = import_model.get()
        if model is None:
            _append_log("Please click Preview first to parse the HEC-RAS file.")
            return

        _append_log("\n--- Converting ---")

        hdf_files = input.import_hdf()
        hdf_path = hdf_files[0]["datapath"]
        hdf_name = hdf_files[0]["name"].rsplit(".", 2)[0]  # strip .g01.hdf

        dem_files = input.import_dem()
        dem_path = dem_files[0]["datapath"] if dem_files else None

        if model.rivers and not dem_path:
            _append_log("ERROR: DEM file required for 1D→2D conversion")
            return

        import tempfile
        out_dir = tempfile.mkdtemp(prefix="telemac_import_")

        try:
            from telemac_tools import hecras_to_telemac

            scheme = input.import_scheme()
            cas_overrides = {}
            if scheme == "finite_element":
                cas_overrides["EQUATIONS"] = "'SAINT-VENANT FE'"
                cas_overrides["FINITE VOLUME SCHEME"] = None

            hecras_to_telemac(
                hecras_path=hdf_path,
                dem_path=dem_path,
                output_dir=out_dir,
                name=hdf_name,
                floodplain_width=float(input.fp_width()),
                backend=input.import_mesher(),
                cas_overrides=cas_overrides if cas_overrides else None,
            )

            import_output_dir.set(out_dir)
            _append_log(f"Output directory: {out_dir}")
            _append_log(f"  {hdf_name}.slf — mesh + variables")
            _append_log(f"  {hdf_name}.cli — boundary conditions")
            _append_log(f"  {hdf_name}.cas — steering file")

            # Count mesh stats
            from data_manip.extraction.telemac_file import TelemacFile
            tf = TelemacFile(_os.path.join(out_dir, f"{hdf_name}.slf"))
            _append_log(f"\nMesh: {tf.npoin2} nodes, {tf.nelem2} elements")
            _append_log(f"Variables: {', '.join(tf.varnames)}")
            tf.close()

            _append_log("\nConversion complete. Use Download buttons below.")

        except Exception as e:
            _append_log(f"ERROR: {e}")
            import traceback
            _append_log(traceback.format_exc())

    @output
    @render.ui
    def import_status_ui():
        model = import_model.get()
        if model is None:
            return ui.span("Upload a file and click Preview", class_="small text-muted")
        n_rivers = len(model.rivers)
        n_areas = len(model.areas_2d)
        parts = []
        if n_rivers:
            total_xs = sum(len(r.cross_sections) for r in model.rivers)
            parts.append(f"{n_rivers} reach(es), {total_xs} XS")
        if n_areas:
            total_cells = sum(len(a.cells) for a in model.areas_2d)
            parts.append(f"{n_areas} 2D area(s), {total_cells} cells")
        return ui.span(" | ".join(parts), class_="small text-success")

    @output
    @render.ui
    def import_download_ui():
        out = import_output_dir.get()
        if out is None:
            return ui.span("No output yet", class_="small text-muted")
        return ui.div(
            ui.download_button("dl_slf", "Download .slf", class_="btn-sm btn-outline-primary me-1"),
            ui.download_button("dl_cli", "Download .cli", class_="btn-sm btn-outline-primary me-1"),
            ui.download_button("dl_cas", "Download .cas", class_="btn-sm btn-outline-primary"),
            class_="d-flex gap-1",
        )

    @render.download(filename=lambda: _import_filename(".slf"))
    def dl_slf():
        path = _import_file_path(".slf")
        if path:
            with open(path, "rb") as f:
                yield f.read()

    @render.download(filename=lambda: _import_filename(".cli"))
    def dl_cli():
        path = _import_file_path(".cli")
        if path:
            with open(path, "rb") as f:
                yield f.read()

    @render.download(filename=lambda: _import_filename(".cas"))
    def dl_cas():
        path = _import_file_path(".cas")
        if path:
            with open(path, "rb") as f:
                yield f.read()

    def _import_filename(ext: str) -> str:
        hdf_files = input.import_hdf()
        if hdf_files:
            return hdf_files[0]["name"].rsplit(".", 2)[0] + ext
        return "project" + ext

    def _import_file_path(ext: str) -> str | None:
        out = import_output_dir.get()
        if out is None:
            return None
        name = _import_filename(ext)
        path = _os.path.join(out, name)
        return path if _os.path.isfile(path) else None


app = App(app_ui, server)
