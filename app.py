# app.py — TELEMAC Viewer
import logging
import threading

from __version__ import __version__

import numpy as np
from shiny import App, reactive, render, ui
from server_import import import_map_widget
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

_logger = logging.getLogger(__name__)

from constants import (
    EXAMPLES,
    EXAMPLE_CHOICES,
    PALETTES,
    BASEMAP_STYLES,
    MAP_BG_DARK,
    MAP_TOOLTIP,
    cached_gradient_colors,
    format_time,
)
from layers import (
    build_mesh_layer,
    build_mesh_color_patch,
    build_velocity_layer,
    build_velocity_patch,
    build_contour_layer_fn,
    build_contour_patch,
    build_marker_layer,
    build_cross_section_layer,
    build_particle_layer,
    build_wireframe_layer,
    build_extrema_markers,
    build_measurement_layer,
    build_boundary_layer,
    build_polygon_layer,
)
from analysis import (
    find_extrema,
    compute_difference,
    compute_slope,
    compute_courant_number,
    get_var_values,
)
from telemac_defaults import is_bipolar
from app_dispatch import decide_dispatch

# ---------------------------------------------------------------------------
# Map widget
# ---------------------------------------------------------------------------

map_widget = MapWidget(
    "map",
    view_state={"longitude": 0, "latitude": 0, "zoom": 0},
    style=MAP_BG_DARK,
    cooperative_gestures=True,
    tooltip=MAP_TOOLTIP,
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
                    ui.tags.dd(
                        "Depth-averaged shallow water equations — river flows, dam breaks, flood propagation."
                    ),
                    ui.tags.dt("TELEMAC-3D"),
                    ui.tags.dd(
                        "Three-dimensional hydrodynamics — estuaries, stratified flows, thermal plumes."
                    ),
                    ui.tags.dt("TOMAWAC"),
                    ui.tags.dd(
                        "Spectral wave modelling — nearshore wave transformation, refraction, breaking."
                    ),
                    ui.tags.dt("ARTEMIS"),
                    ui.tags.dd(
                        "Harbour wave agitation — short-wave diffraction and reflection in port areas."
                    ),
                    ui.tags.dt("SISYPHE / GAIA"),
                    ui.tags.dd(
                        "Sediment transport and morphodynamics — bedload, suspended load, bed evolution."
                    ),
                    ui.tags.dt("WAQTEL"),
                    ui.tags.dd(
                        "Water quality — pollutant dispersion, dissolved oxygen, thermal modelling."
                    ),
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
                    ui.tags.li(
                        "Write a steering file (.cas) with parameters and boundary conditions"
                    ),
                    ui.tags.li(
                        "Run the simulation (e.g. telemac2d.py case.cas --ncsize=4)"
                    ),
                    ui.tags.li(
                        "Visualize results in this viewer or ParaView/BlueKenue"
                    ),
                ),
                ui.h6("Learn more"),
                ui.p(
                    ui.a(
                        "opentelemac.org",
                        href="https://www.opentelemac.org",
                        target="_blank",
                    ),
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
                    ui.tags.li(
                        ui.tags.b("Example case"),
                        " — Load a built-in example from TELEMAC-2D, 3D, TOMAWAC, ARTEMIS, SISYPHE, or GAIA.",
                    ),
                    ui.tags.li(
                        ui.tags.b("Upload .slf"),
                        " — Load your own SELAFIN result file.",
                    ),
                    ui.tags.li(
                        ui.tags.b("Dark map background"),
                        " — Toggle between light and dark map canvas.",
                    ),
                ),
                ui.h6("Visualization Panel"),
                ui.tags.ul(
                    ui.tags.li(
                        ui.tags.b("Variable"),
                        " — Select which result variable to display (e.g. WATER DEPTH, VELOCITY U).",
                    ),
                    ui.tags.li(
                        ui.tags.b("Color palette"),
                        " — Choose from Viridis, Plasma, Ocean, Thermal, or Chlorophyll color maps.",
                    ),
                    ui.tags.li(
                        ui.tags.b("Velocity vectors"),
                        " — Overlay arrow glyphs showing flow direction and magnitude.",
                    ),
                    ui.tags.li(
                        ui.tags.b("Contour lines"),
                        " — Show iso-value contour lines on the mesh.",
                    ),
                    ui.tags.li(
                        ui.tags.b("Mesh wireframe"),
                        " — Display the triangular mesh edges.",
                    ),
                    ui.tags.li(
                        ui.tags.b("Boundary nodes"),
                        " — Highlight nodes on the domain boundary.",
                    ),
                    ui.tags.li(
                        ui.tags.b("Min/Max locations"),
                        " — Mark the locations of extreme values on the mesh.",
                    ),
                    ui.tags.li(
                        ui.tags.b("Log scale"),
                        " — Apply logarithmic color scaling for variables with large ranges.",
                    ),
                    ui.tags.li(
                        ui.tags.b("Reverse palette"),
                        " — Invert the color ramp direction.",
                    ),
                    ui.tags.li(
                        ui.tags.b("Difference mode"),
                        " — Show the difference between the current timestep and a reference.",
                    ),
                    ui.tags.li(
                        ui.tags.b("Color range"),
                        " — Manually set min/max values for the color scale.",
                    ),
                    ui.tags.li(
                        ui.tags.b("Filter range"),
                        " — Gray out values outside a specified range.",
                    ),
                    ui.tags.li(
                        ui.tags.b("Cross-Section"),
                        " — Click two points to create a profile line; view elevation/value cross-section plots.",
                    ),
                    ui.tags.li(
                        ui.tags.b("Discharge"),
                        " — Compute volumetric flow rate across the cross-section line.",
                    ),
                    ui.tags.li(
                        ui.tags.b("Particle traces"),
                        " — Animate flow pathlines from a seed grid.",
                    ),
                    ui.tags.li(
                        ui.tags.b("Mesh quality"),
                        " — Color the mesh by element quality (aspect ratio).",
                    ),
                    ui.tags.li(
                        ui.tags.b("Slope/gradient"),
                        " — Visualize the spatial gradient magnitude of the selected variable.",
                    ),
                    ui.tags.li(
                        ui.tags.b("Courant number"),
                        " — Display the local CFL condition number.",
                    ),
                    ui.tags.li(
                        ui.tags.b("Element area"),
                        " — Color-code triangles by their area.",
                    ),
                    ui.tags.li(
                        ui.tags.b("Measure Distance"),
                        " — Click two points to measure Euclidean distance.",
                    ),
                    ui.tags.li(
                        ui.tags.b("Custom expression"),
                        " — Compute derived fields using math expressions (e.g. ",
                        ui.tags.code("VELOCITY_U**2 + VELOCITY_V**2"),
                        ").",
                    ),
                    ui.tags.li(
                        ui.tags.b("3D mode"),
                        " — Switch to 3D perspective view for multi-layer (3D) result files.",
                    ),
                ),
                ui.h6("Playback Panel"),
                ui.tags.ul(
                    ui.tags.li(
                        ui.tags.b("Time slider"),
                        " — Scrub through simulation timesteps.",
                    ),
                    ui.tags.li(
                        ui.tags.b("Play/Pause"),
                        " — Animate through timesteps automatically.",
                    ),
                    ui.tags.li(
                        ui.tags.b("Go to time"),
                        " — Jump to a specific simulation time in seconds.",
                    ),
                    ui.tags.li(
                        ui.tags.b("Speed"),
                        " — Control animation speed (seconds per frame).",
                    ),
                    ui.tags.li(
                        ui.tags.b("Loop"),
                        " — Restart animation from the beginning when it reaches the end.",
                    ),
                ),
                ui.h6("Statistics Panel"),
                ui.tags.ul(
                    ui.tags.li(
                        ui.tags.b("Stats"),
                        " — Current timestep statistics: min, max, mean, std dev.",
                    ),
                    ui.tags.li(
                        ui.tags.b("All Vars Time Series"),
                        " — Plot all variables over time at a clicked node.",
                    ),
                    ui.tags.li(
                        ui.tags.b("Value Histogram"),
                        " — Distribution histogram of the current variable.",
                    ),
                    ui.tags.li(
                        ui.tags.b("Compute Integral"),
                        " — Area-weighted integral of the variable over the domain.",
                    ),
                    ui.tags.li(
                        ui.tags.b("Temporal Stats"),
                        " — Compute min/max/mean over all timesteps.",
                    ),
                ),
                ui.h6("File Info Panel"),
                ui.tags.ul(
                    ui.tags.li(
                        "View metadata: mesh size, variable count, time range, precision, module type."
                    ),
                ),
                ui.h6("Run Simulation Panel"),
                ui.tags.ul(
                    ui.tags.li(
                        "Select a .cas steering file and run TELEMAC directly from the viewer."
                    ),
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
                        ui.tags.tr(
                            ui.tags.th("Key", style="width:140px"), ui.tags.th("Action")
                        ),
                    ),
                    ui.tags.tbody(
                        ui.tags.tr(
                            ui.tags.td(ui.tags.kbd("Space")),
                            ui.tags.td("Play / Pause animation"),
                        ),
                        ui.tags.tr(
                            ui.tags.td(ui.tags.kbd("\u2192 Right Arrow")),
                            ui.tags.td("Next timestep"),
                        ),
                        ui.tags.tr(
                            ui.tags.td(ui.tags.kbd("\u2190 Left Arrow")),
                            ui.tags.td("Previous timestep"),
                        ),
                        ui.tags.tr(
                            ui.tags.td(ui.tags.kbd("Page Down")),
                            ui.tags.td("Next variable"),
                        ),
                        ui.tags.tr(
                            ui.tags.td(ui.tags.kbd("Page Up")),
                            ui.tags.td("Previous variable"),
                        ),
                    ),
                    class_="table table-sm table-striped",
                    style="max-width:400px;",
                ),
                ui.h6("Map controls"),
                ui.tags.table(
                    ui.tags.thead(
                        ui.tags.tr(
                            ui.tags.th("Action", style="width:180px"), ui.tags.th("How")
                        ),
                    ),
                    ui.tags.tbody(
                        ui.tags.tr(ui.tags.td("Pan"), ui.tags.td("Click and drag")),
                        ui.tags.tr(
                            ui.tags.td("Zoom"),
                            ui.tags.td("Scroll wheel or +/- buttons"),
                        ),
                        ui.tags.tr(
                            ui.tags.td("Reset view"),
                            ui.tags.td("Reset View widget button"),
                        ),
                        ui.tags.tr(
                            ui.tags.td("Screenshot"),
                            ui.tags.td("Screenshot widget button"),
                        ),
                        ui.tags.tr(
                            ui.tags.td("Fullscreen"),
                            ui.tags.td("Fullscreen widget button"),
                        ),
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
                ui.span(
                    ui.output_text("stat_var_name", inline=True), class_="stat-val"
                ),
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
                ui.input_file(
                    "obs_upload", "Upload observations (.csv)", accept=[".csv"]
                ),
                ui.div(
                    ui.input_action_button(
                        "show_multivar",
                        "All Vars Time Series",
                        class_="btn-sm btn-outline-primary mb-1",
                    ),
                    ui.input_action_button(
                        "show_histogram",
                        "Value Histogram",
                        class_="btn-sm btn-outline-primary mb-1",
                    ),
                    ui.input_action_button(
                        "compute_integral",
                        "Compute Integral",
                        class_="btn-sm btn-outline-info mb-1",
                    ),
                    ui.input_action_button(
                        "compute_temporal",
                        "Temporal Stats",
                        class_="btn-sm btn-outline-info mb-1",
                    ),
                    ui.input_action_button(
                        "compute_volume",
                        "Volume over time",
                        class_="btn-sm btn-outline-info mb-1",
                    ),
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
                    ui.input_action_button(
                        "run_sim", "Run", class_="btn-sm btn-success me-2"
                    ),
                    ui.input_action_button(
                        "stop_sim", "Stop", class_="btn-sm btn-danger"
                    ),
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
                ui.input_select(
                    "import_source",
                    "Source format",
                    choices={
                        "hecras6": "HEC-RAS 6.x (.hdf)",
                    },
                ),
                ui.input_file(
                    "import_hdf",
                    "HEC-RAS geometry file (.hdf)",
                    accept=[".hdf", ".hdf5"],
                ),
                ui.input_file(
                    "import_dem", "DEM file (.tif)", accept=[".tif", ".tiff"]
                ),
                ui.tags.details(
                    ui.tags.summary("1D Options"),
                    ui.input_numeric(
                        "fp_width",
                        "Floodplain width (m)",
                        value=500,
                        min=50,
                        max=5000,
                        step=50,
                    ),
                    ui.input_numeric(
                        "channel_refine",
                        "Channel refinement (m²)",
                        value=10,
                        min=1,
                        max=100,
                        step=1,
                    ),
                    ui.input_numeric(
                        "floodplain_refine",
                        "Floodplain refinement (m²)",
                        value=200,
                        min=10,
                        max=1000,
                        step=10,
                    ),
                ),
                ui.input_select(
                    "import_mesher",
                    "Mesher",
                    choices={
                        "triangle": "Triangle (default)",
                    },
                ),
                ui.input_select(
                    "import_scheme",
                    "Numerical scheme",
                    choices={
                        "finite_volume": "Finite Volume (robust)",
                        "finite_element": "Finite Element (accurate)",
                    },
                ),
                ui.div(
                    ui.input_action_button(
                        "import_preview", "Preview", class_="btn-sm btn-primary me-2"
                    ),
                    ui.input_action_button(
                        "import_convert", "Convert", class_="btn-sm btn-success"
                    ),
                    class_="d-flex gap-2 mb-2",
                ),
                ui.output_ui("import_status_ui"),
                ui.tags.hr(),
                ui.tags.h6("Download Output"),
                ui.output_ui("import_download_ui"),
            ),
            ui.card(
                ui.card_header("Preview"),
                import_map_widget.ui(width="100%", height="400px"),
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
        ui.h6(
            f"TELEMAC Viewer v{__version__}",
            style="margin:0 0 8px; color: var(--coastal-text);",
        ),
        ui.input_select("example", "Example case", choices=EXAMPLE_CHOICES),
        ui.input_file("upload", "Or upload .slf file", accept=[".slf"]),
        ui.output_ui("clear_upload_ui"),
        ui.input_select(
            "basemap",
            "Background",
            choices={
                "dark": "Dark (ocean)",
                "light": "Light (blank)",
                "osm": "CartoDB Dark",
                "satellite": "Satellite (ESRI)",
            },
        ),
        ui.output_ui("var_select_ui"),
        ui.output_ui("time_slider_ui"),
        # ── Tier 2: Collapsed accordions ──
        ui.accordion(
            ui.accordion_panel(
                "Display",
                ui.input_select(
                    "palette", "Color palette", choices=list(PALETTES.keys())
                ),
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
                ui.input_select(
                    "diagnostic",
                    "Mesh diagnostic",
                    choices={
                        "none": "None",
                        "mesh_quality": "Mesh quality",
                        "slope": "Slope/gradient",
                        "courant": "Courant number",
                        "elem_area": "Element area",
                    },
                ),
                ui.input_action_button(
                    "draw_xsec",
                    "Draw Cross-Section",
                    class_="btn-sm btn-outline-primary w-100 mb-1",
                ),
                ui.output_ui("clear_xsec_ui"),
                ui.output_ui("discharge_ui"),
                ui.output_ui("rating_curve_ui"),
                ui.input_action_button(
                    "measure_btn",
                    "Measure Distance",
                    class_="btn-sm btn-outline-warning w-100 mb-1",
                ),
                ui.output_ui("measure_info_ui"),
                ui.input_action_button(
                    "draw_polygon",
                    "Draw Polygon",
                    class_="btn-sm btn-outline-success w-100 mb-1",
                ),
                ui.output_ui("polygon_stats_ui"),
                ui.div(
                    ui.input_text(
                        "expr_input",
                        "Custom expression",
                        placeholder="VELOCITY_U**2 + VELOCITY_V**2",
                    ),
                    ui.input_action_button(
                        "eval_expr", "Eval", class_="btn-sm btn-outline-secondary"
                    ),
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
                ui.input_action_button(
                    "record_btn",
                    "Record",
                    class_="btn-sm btn-outline-danger w-100 mb-1",
                ),
                ui.div(
                    ui.input_numeric(
                        "goto_time", "Go to time (s)", value=0, min=0, step=1
                    ),
                    ui.input_action_button(
                        "goto_btn", "Go", class_="btn-sm btn-outline-primary"
                    ),
                    class_="d-flex align-items-end gap-1 mb-2",
                ),
                ui.input_slider(
                    "speed",
                    "Speed (s/frame)",
                    min=0.1,
                    max=2.0,
                    value=0.5,
                    step=0.1,
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
    title=f"TELEMAC Viewer v{__version__}",
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
    last_structural_sig = reactive.value(None)
    analysis_mode = reactive.value("none")
    clicked_points = reactive.value([])  # list of (x_meters, y_meters)
    cross_section_points = reactive.value(None)  # [[x_m, y_m], ...] or None
    particle_paths = reactive.value(None)  # list of paths or None
    is_3d_mode = reactive.value(False)
    temporal_stats_cache = reactive.value(None)  # dict with min/max/mean arrays
    measure_points = reactive.value([])  # list of [x_mesh_m, y_mesh_m] (max 2)
    measure_mode = reactive.value(False)  # True when waiting for measurement clicks
    use_upload = reactive.value(False)  # True when uploaded file should be used
    obs_data = reactive.value(
        None
    )  # parsed observation CSV: (times, values, varname) or None
    compare_tf = reactive.value(None)  # secondary TelemacFile for comparison
    recording = reactive.value(False)  # animation recording state
    polygon_mode = reactive.value(False)  # polygon drawing mode
    polygon_stats_data = reactive.value(None)  # polygon zonal stats result
    polygon_geom = reactive.value(None)  # list of [x,y] polygon coords or None
    volume_cache = reactive.value(None)  # {"times": ndarray, "volumes": ndarray}
    expr_result = reactive.value(None)  # numpy array or None (custom expression result)
    integral_result = reactive.value(None)  # mesh integral result dict or None

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
        except (TypeError, AttributeError, KeyError):
            return "—"

    @output
    @render.text
    def stat_time():
        try:
            tf = tel_file()
            tidx = current_tidx()
            return format_time(tf.times[tidx])
        except (TypeError, AttributeError, KeyError):
            return "—"

    @output
    @render.text
    def stat_nodes():
        try:
            tf = tel_file()
            return f"{tf.npoin2:,}"
        except (TypeError, AttributeError, KeyError):
            return "—"

    @output
    @render.text
    def stat_range():
        try:
            vals = effective_values()
            return f"{vals.min():.3g} – {vals.max():.3g}"
        except (TypeError, AttributeError, KeyError):
            return "—"

    # -- Core reactive calcs (tel_file, mesh_geom, current_var, etc.) --
    from server_core import register_core_handlers

    _core = register_core_handlers(
        input,
        output,
        session,
        _tf_lock,
        particle_paths,
        cross_section_points,
        clicked_points,
        temporal_stats_cache,
        integral_result,
        expr_result,
        measure_points,
        measure_mode,
        analysis_mode,
        obs_data,
        compare_tf,
        volume_cache,
        polygon_stats_data,
        polygon_geom,
        use_upload,
        is_3d_mode,
    )
    tel_file = _core["tel_file"]
    mesh_geom = _core["mesh_geom"]
    current_var = _core["current_var"]
    current_tidx = _core["current_tidx"]
    current_values = _core["current_values"]
    mesh_quality_values = _core["mesh_quality_values"]
    elem_area_values = _core["elem_area_values"]
    boundary_nodes_cached = _core["boundary_nodes_cached"]
    cli_data = _core["cli_data"]
    current_crs = _core["current_crs"]
    _run_with_lock = _core["_run_with_lock"]

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
            ui.notification_show(
                "Courant number requires VELOCITY U/V variables",
                type="warning",
                duration=4,
                id="cfl_warn",
            )
            return current_values()
        elif diag == "elem_area":
            return elem_area_values()
        # Temporal stats display
        stats = temporal_stats_cache.get()
        try:
            td = input.temporal_display() or "none"
        except (TypeError, AttributeError, KeyError):
            td = "none"
        if stats is not None and td != "none":
            return stats[td]
        if input.diff_mode():
            tf = tel_file()
            var = current_var()
            tidx = current_tidx()
            try:
                ref = input.ref_tidx() if input.ref_tidx() is not None else 0
            except (TypeError, AttributeError, KeyError):
                ref = 0
            return compute_difference(tf, var, tidx, ref)
        return current_values()

    # -- Structural signature for map update dispatch --
    # Full update (map_widget.update) fires when this tuple changes between
    # ticks; otherwise we fall through to partial_update + set_widgets.
    # KEEP THIS IN SYNC with app_dispatch.make_sig() in tests/test_map_dispatch.py.
    @reactive.calc
    def _structural_sig() -> tuple:
        uploaded = input.upload()
        file_path = (
            uploaded[0]["datapath"]
            if uploaded and use_upload.get()
            else EXAMPLES.get(input.example(), "")
        )
        try:
            basemap = input.basemap() or "dark"
        except (TypeError, AttributeError, KeyError):
            basemap = "dark"
        try:
            compare_var = input.compare_var() or ""
        except (TypeError, AttributeError, KeyError):
            compare_var = ""
        overlay_counts = (
            len(clicked_points.get()),
            len(measure_points.get()),
            1 if polygon_geom.get() is not None else 0,
            1 if cross_section_points.get() is not None else 0,
        )
        return (
            file_path,
            bool(input.wireframe()),
            bool(input.boundary_nodes()),
            bool(input.vectors()),
            bool(input.contours()),
            bool(input.show_extrema()),
            bool(input.particles()),
            bool(input.diff_mode()),
            bool(is_3d_mode.get()),
            basemap,
            compare_var,
            overlay_counts,
        )

    # -- Cached static overlay layers --
    # These depend only on the file + their toggle; caching them here keeps
    # expensive topology work (compute_unique_edges, find_boundary_edges) off
    # the hot path during timestep scrubbing or palette changes. Shiny's
    # @reactive.calc invalidates only when a read dep changes, so the cached
    # dict is reused verbatim across ticks.

    @reactive.calc
    def wireframe_layer_cached():
        if not input.wireframe():
            return None
        geom = mesh_geom()
        return build_wireframe_layer(
            tel_file(), geom, origin=[geom.lon_off, geom.lat_off]
        )

    @reactive.calc
    def boundary_layers_cached():
        if not input.boundary_nodes():
            return []
        geom = mesh_geom()
        return build_boundary_layer(
            tel_file(),
            geom,
            boundary_nodes_cached(),
            bc_types=cli_data(),
            origin=[geom.lon_off, geom.lat_off],
        )

    # -- Analysis panel, charts, stats, CSV downloads, overlays --
    from server_analysis import register_analysis_handlers

    register_analysis_handlers(
        input,
        output,
        session,
        map_widget,
        tel_file,
        mesh_geom,
        current_var,
        current_tidx,
        current_values,
        effective_values,
        analysis_mode,
        clicked_points,
        cross_section_points,
        particle_paths,
        temporal_stats_cache,
        measure_points,
        measure_mode,
        obs_data,
        compare_tf,
        recording,
        polygon_mode,
        polygon_stats_data,
        polygon_geom,
        volume_cache,
        expr_result,
        integral_result,
        current_crs,
        use_upload,
        is_3d_mode,
        _run_with_lock,
    )

    # -- Simulation launcher --
    from server_simulation import register_simulation_handlers

    register_simulation_handlers(input, output, session, use_upload)

    # -- Playback controls, keyboard shortcuts, auto-advance --
    from server_playback import register_playback_handlers

    register_playback_handlers(input, output, session, playing, tel_file, current_var)

    # -- Map update helpers --

    def _resolve_display_var(var, tidx, diag):
        """Return the label string describing the currently displayed variable."""
        try:
            td = input.temporal_display() or "none"
        except (TypeError, AttributeError, KeyError):
            td = "none"
        if expr_result.get() is not None:
            return f"EXPR: {input.expr_input()}"
        if diag == "mesh_quality":
            return "MESH QUALITY"
        if diag == "slope":
            return f"SLOPE ({var})"
        if diag == "courant":
            return "COURANT NUMBER"
        if diag == "elem_area":
            return "ELEMENT AREA (m²)"
        if td != "none" and temporal_stats_cache.get() is not None:
            return f"{var} (temporal {td})"
        if input.diff_mode():
            try:
                ref = input.ref_tidx() if input.ref_tidx() is not None else 0
            except (TypeError, AttributeError, KeyError):
                ref = 0
            return f"Δ {var} (t{tidx}-t{ref})"
        return var

    def _compute_color_range(palette_id, values):
        """Return (crange, filt, use_log, reverse) for build_mesh_layer."""
        use_diverging = palette_id == "_diverging"
        try:
            custom_range = input.custom_range()
        except (TypeError, AttributeError, KeyError):
            custom_range = False
        crange = None
        if use_diverging and not custom_range:
            abs_max = max(abs(float(values.min())), abs(float(values.max())))
            if abs_max > 0:
                crange = (-abs_max, abs_max)
        elif custom_range:
            try:
                cmin = input.color_min()
                cmax = input.color_max()
            except (TypeError, AttributeError, KeyError):
                cmin, cmax = None, None
            if cmin is not None and cmax is not None:
                crange = (cmin, cmax)
        try:
            filt = input.filter_range()
        except (TypeError, AttributeError, KeyError):
            filt = None
        try:
            use_log = input.log_scale()
        except (TypeError, AttributeError, KeyError):
            use_log = False
        try:
            reverse = input.reverse_palette()
        except (TypeError, AttributeError, KeyError):
            reverse = False
        return crange, filt, use_log, reverse

    def _build_overlay_layers(tf, geom, tidx, values, origin):
        """Build all optional overlay layers (wireframe, vectors, contours, markers, etc.)."""
        layers = []

        wlyr = wireframe_layer_cached()
        if wlyr is not None:
            layers.append(wlyr)

        # Boundary edges (color-coded by hydrodynamic type)
        if input.boundary_nodes():
            layers.extend(boundary_layers_cached())
            if cli_data() is None:
                ui.notification_show(
                    "No .cli file found — boundary types shown as Wall (inferred).",
                    type="message",
                    duration=4,
                    id="cli_warn",
                )

        # Min/max location markers
        if input.show_extrema():
            extrema = find_extrema(tf, values)
            layers.extend(
                build_extrema_markers(extrema, geom.x_off, geom.y_off, origin=origin)
            )

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
        except (TypeError, AttributeError, KeyError):
            compare = ""
        compare_vals = None
        if compare.startswith("(2) ") and compare_tf.get() is not None:
            real_name = compare[4:]
            tf2 = compare_tf.get()
            safe_tidx = min(tidx, len(tf2.times) - 1)
            compare_vals = get_var_values(tf2, real_name, safe_tidx)
        elif compare:
            compare_vals = get_var_values(tf, compare, tidx)
        if compare_vals is not None:
            clyr2 = build_contour_layer_fn(
                tf,
                compare_vals,
                geom,
                n_contours=8,
                layer_id="compare-contours",
                contour_color=[0, 0, 180],
                origin=origin,
            )
            if clyr2 is not None:
                layers.append(clyr2)

        # Markers for clicked points
        pts = clicked_points.get()
        for i, pt in enumerate(pts):
            mx, my = float(pt[0] - geom.x_off), float(pt[1] - geom.y_off)
            layers.append(
                build_marker_layer(mx, my, layer_id=f"marker-{i}", origin=origin)
            )

        # Cross-section path
        xsec = cross_section_points.get()
        if xsec is not None:
            path_centered = [[x - geom.x_off, y - geom.y_off] for x, y in xsec]
            layers.append(build_cross_section_layer(path_centered, origin=origin))

        # Particle traces
        paths = particle_paths.get()
        if paths and input.particles():
            current_time = float(tf.times[tidx])
            trail = input.trail_length() if input.trail_length() is not None else 1.0
            layers.append(
                build_particle_layer(paths, current_time, trail, origin=origin)
            )

        # Measurement line (convert mesh-meter coords to centered for layer)
        mpts = measure_points.get()
        if mpts:
            mpts_centered = [[p[0] - geom.x_off, p[1] - geom.y_off] for p in mpts]
            layers.extend(build_measurement_layer(mpts_centered, origin=origin))

        # Polygon outline
        pgon = polygon_geom.get()
        if pgon is not None:
            pgon_centered = [[p[0] - geom.x_off, p[1] - geom.y_off] for p in pgon]
            layers.append(build_polygon_layer(pgon_centered, origin=origin))

        return layers

    def _build_legend(display_var, vmin, vmax, palette_id, reverse):
        """Build the layer_legend_widget for the map."""
        gradient_colors = cached_gradient_colors(palette_id, reverse=reverse)
        legend_entries = [
            {
                "layer_id": "mesh",
                "label": f"{display_var}  [{vmin:.3g} - {vmax:.3g}]",
                "colors": gradient_colors,
                "shape": "gradient",
            }
        ]
        # Add boundary type entries to legend
        if input.boundary_nodes():
            legend_entries.extend(
                [
                    {
                        "layer_id": "boundary-wall",
                        "label": "Wall",
                        "color": [160, 160, 170],
                        "shape": "line",
                    },
                    {
                        "layer_id": "boundary-free",
                        "label": "Free (Neumann)",
                        "color": [0, 200, 80],
                        "shape": "line",
                    },
                    {
                        "layer_id": "boundary-prescribed",
                        "label": "Prescribed (H/Q)",
                        "color": [40, 120, 255],
                        "shape": "line",
                    },
                ]
            )
        return layer_legend_widget(
            entries=legend_entries,
            placement="bottom-right",
            show_checkbox=False,
            title="Legend",
        )

    # -- Map update --

    @reactive.effect
    async def update_map():
        tf = tel_file()
        geom = mesh_geom()
        var = current_var()
        tidx = current_tidx()
        values = effective_values()
        palette_id = input.palette() if input.palette() in PALETTES else "Viridis"

        diag = input.diagnostic() if input.diagnostic() else "none"
        if is_bipolar(var) and diag == "none" and not input.diff_mode():
            palette_id = "_diverging"
        elif input.diff_mode() and diag == "none":
            palette_id = "_diverging"

        display_var = _resolve_display_var(var, tidx, diag)
        crange, filt, use_log, reverse = _compute_color_range(palette_id, values)
        origin = [geom.lon_off, geom.lat_off]

        curr_sig = _structural_sig()
        prev_sig = last_structural_sig.get()
        decision = decide_dispatch(prev_sig=prev_sig, curr_sig=curr_sig)

        if decision == "partial":
            # Fast path: mesh color patch + dynamic overlay patches + widgets.
            # sig is unchanged by definition on this branch — do not re-set
            # last_structural_sig. file_path is part of _structural_sig, so a
            # file change forces the full path, which is also where
            # last_file_path is maintained.
            patch, vmin, vmax, log_applied = build_mesh_color_patch(
                geom,
                values,
                palette_id,
                filter_range=filt,
                color_range_override=crange,
                log_scale=use_log,
                reverse_palette=reverse,
            )
            patches = [patch]
            if input.vectors():
                vpatch = build_velocity_patch(tf, tidx, geom, origin=origin)
                if vpatch is not None:
                    patches.append(vpatch)
            if input.contours():
                cpatch = build_contour_patch(tf, values, geom, origin=origin)
                if cpatch is not None:
                    patches.append(cpatch)
            if input.show_extrema():
                extrema = find_extrema(tf, values)
                patches.extend(
                    build_extrema_markers(
                        extrema, geom.x_off, geom.y_off, origin=origin
                    )
                )
            if use_log and not log_applied:
                ui.notification_show(
                    "Log scale requires positive values — using linear scale",
                    type="warning",
                    duration=3,
                    id="log_warn",
                )
            legend = _build_legend(display_var, vmin, vmax, palette_id, reverse)
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
            if is_3d_mode.get():
                widgets.append(gimbal_widget())
            try:
                await map_widget.partial_update(session, patches)
                await map_widget.set_widgets(session, widgets)
            except Exception:
                # Fast path failed — force the next tick to rebuild from
                # scratch rather than retrying the same partial_update forever.
                last_structural_sig.set(None)
                raise
            return

        # Full path: structural change — rebuild everything
        lyr, vmin, vmax, log_applied = build_mesh_layer(
            geom,
            values,
            palette_id,
            filter_range=filt,
            color_range_override=crange,
            log_scale=use_log,
            reverse_palette=reverse,
            origin=origin,
        )
        if use_log and not log_applied:
            ui.notification_show(
                "Log scale requires positive values — using linear scale",
                type="warning",
                duration=3,
                id="log_warn",
            )

        layers = [lyr] + _build_overlay_layers(tf, geom, tidx, values, origin)
        legend = _build_legend(display_var, vmin, vmax, palette_id, reverse)

        uploaded = input.upload()
        current_path = (
            uploaded[0]["datapath"]
            if uploaded and use_upload.get()
            else EXAMPLES.get(input.example(), "")
        )
        kwargs = {}
        if current_path != last_file_path.get():
            last_file_path.set(current_path)
            kwargs["view_state"] = {
                "longitude": geom.lon_off,
                "latitude": geom.lat_off,
                "zoom": geom.zoom,
            }

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

        try:
            basemap = input.basemap() or "dark"
        except (TypeError, AttributeError, KeyError):
            basemap = "dark"
        if basemap != "dark":
            kwargs["style"] = BASEMAP_STYLES.get(basemap, MAP_BG_DARK)

        if is_3d_mode.get():
            widgets.append(gimbal_widget())
            kwargs["views"] = [
                first_person_view(focalDistance=10, maxPitchAngle=80),
            ]
            kwargs["effects"] = [
                lighting_effect(
                    ambient_light(intensity=0.6),
                    directional_light(direction=[-1, -3, -1], intensity=0.8),
                    directional_light(direction=[1, 2, -0.5], intensity=0.3),
                )
            ]

        await map_widget.update(session, layers=layers, widgets=widgets, **kwargs)
        last_structural_sig.set(curr_sig)

    # ── Import tab server logic ─────────────────────────────────────────
    from server_import import register_import_handlers

    register_import_handlers(input, output, session)


app = App(app_ui, server)
