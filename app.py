import math
import sys
import os
import functools
import numpy as np

# Add shiny-deckgl and TELEMAC to path
sys.path.insert(0, "/home/razinka/.mamba/pkgs/shiny-deckgl-1.0.1-py_0/site-packages")
os.environ.setdefault("HOMETEL", "/home/razinka/telemac/telemac-v8p5r1")
os.environ.setdefault(
    "SYSTELCFG",
    os.path.join(os.environ["HOMETEL"], "configs/systel.local.cfg"),
)
os.environ.setdefault("USETELCFG", "gfortran.intelmpi")
sys.path.insert(0, os.path.join(os.environ["HOMETEL"], "scripts/python3"))

from shiny import App, reactive, render, ui
from shiny_deckgl import (
    MapWidget,
    head_includes,
    layer,
    line_layer,
    encode_binary_attribute,
    orthographic_view,
    color_range,
    PALETTE_VIRIDIS,
    PALETTE_PLASMA,
    PALETTE_OCEAN,
    PALETTE_THERMAL,
    PALETTE_CHLOROPHYLL,
    zoom_widget,
    fullscreen_widget,
    scale_widget,
    screenshot_widget,
    compass_widget,
    deck_legend_control,
    transition,
    contour_layer,
    data_filter_extension,
    reset_view_widget,
    loading_widget,
)
from data_manip.extraction.telemac_file import TelemacFile

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_E = os.environ["HOMETEL"]

# Flat lookup: display name -> file path
EXAMPLES = {}

# Grouped choices for the UI dropdown (optgroup support)
EXAMPLE_GROUPS = {
    "TELEMAC-2D": {
        "Gouttedo (raindrop)": os.path.join(_E, "examples/telemac2d/gouttedo/r2d_gouttedo_v1p0.slf"),
        "Ritter (dam break)": os.path.join(_E, "examples/telemac2d/dambreak/r2d_ritter-hllc.slf"),
        "Malpasset (dam break)": os.path.join(_E, "examples/telemac2d/malpasset/r2d_malpasset-hllc.slf"),
        "Bump (critical flow)": os.path.join(_E, "examples/telemac2d/bump/r2d_bump.slf"),
        "Vasque (basin flow)": os.path.join(_E, "examples/telemac2d/vasque/f2d_vasque.slf"),
        "Cavity (recirculation)": os.path.join(_E, "examples/telemac2d/cavity/f2d_cavity.slf"),
        "Estimation (parameter)": os.path.join(_E, "examples/telemac2d/estimation/f2d_estimation.slf"),
        "Pluie (rainfall-runoff)": os.path.join(_E, "examples/telemac2d/pluie/f2d_rain_CN.slf"),
        "Confluence (river junction)": os.path.join(_E, "examples/telemac2d/confluence/f2d_confluence.slf"),
        "Culm (river)": os.path.join(_E, "examples/telemac2d/culm/ini_culm.slf"),
        "Breach (dam failure)": os.path.join(_E, "examples/telemac2d/breach/r2d_breach.slf"),
    },
    "TELEMAC-3D": {
        "3D Canal": os.path.join(_E, "examples/telemac3d/canal/r3d_canal-t3d.slf"),
        "3D Pluie (rainfall)": os.path.join(_E, "examples/telemac3d/pluie/f3d_pluie.slf"),
        "3D V-shape": os.path.join(_E, "examples/telemac3d/V/f3d_V.slf"),
    },
    "GAIA (sediment)": {
        "Guenter (bedload)": os.path.join(_E, "examples/gaia/guenter-t2d/f2d_guenter.slf"),
        "Yen (multi-grain)": os.path.join(_E, "examples/gaia/yen-t2d/f2d_multi1.slf"),
        "Sliding (slope)": os.path.join(_E, "examples/gaia/sliding-t2d/f2d_slide1_slope1.slf"),
        "Bosse (bedform)": os.path.join(_E, "examples/gaia/bosse-t2d/f2d_bosse-t2d_fe.slf"),
    },
    "TOMAWAC (waves)": {
        "Wave-current interaction": os.path.join(_E, "examples/tomawac/opposing_current/fom_opposing_cur.slf"),
        "Coupled wind-waves": os.path.join(_E, "examples/tomawac/Coupling_Wind/fom_different.slf"),
        "3D wave coupling": os.path.join(_E, "examples/tomawac/3Dcoupling/fom_littoral_diff.slf"),
    },
    "ARTEMIS (coastal)": {
        "Beach waves": os.path.join(_E, "examples/artemis/beach/tom_plage.slf"),
        "Wave breaking (BJ78)": os.path.join(_E, "examples/artemis/bj78/famp_bj78_20per.slf"),
        "Westcoast (waves)": os.path.join(_E, "examples/artemis/westcoast/tom_westcoast.slf"),
    },
    "KHIONE (ice)": {
        "Frazil ice flume": os.path.join(_E, "examples/khione/flume_frazil-t2d/ini_longflume.slf"),
        "Ice clogging": os.path.join(_E, "examples/khione/clogging-t2d/ini_slowflume.slf"),
    },
}

# Build flat lookup and filter to files that actually exist
for _group, _items in EXAMPLE_GROUPS.items():
    for _name, _path in _items.items():
        if os.path.isfile(_path):
            EXAMPLES[_name] = _path

# Build grouped choices dict for input_select (only existing files)
EXAMPLE_CHOICES = {}
for _group, _items in EXAMPLE_GROUPS.items():
    valid = {k: k for k, v in _items.items() if os.path.isfile(v)}
    if valid:
        EXAMPLE_CHOICES[_group] = valid

PALETTES = {
    "Viridis": PALETTE_VIRIDIS,
    "Plasma": PALETTE_PLASMA,
    "Ocean": PALETTE_OCEAN,
    "Thermal": PALETTE_THERMAL,
    "Chlorophyll": PALETTE_CHLOROPHYLL,
}

# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=8)
def cached_palette_arr(palette_id):
    """Cache the 256-color palette array by palette identity."""
    palette = PALETTES[palette_id]
    return np.array(color_range(256, palette), dtype=np.uint8)


@functools.lru_cache(maxsize=8)
def cached_gradient_colors(palette_id):
    """Cache the 8-sample gradient for the legend."""
    palette = PALETTES[palette_id]
    full = color_range(256, palette)
    return [full[i] for i in range(0, 256, 32)]


def format_time(seconds):
    """Format simulation time for display."""
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes} min {secs:.0f} s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours} h {mins} min"


# ---------------------------------------------------------------------------
# Mesh geometry (static per file, cached)
# ---------------------------------------------------------------------------


def build_mesh_geometry(tf):
    """Build static mesh geometry arrays. Only depends on file, not timestep."""
    ikle = tf.ikle2
    x, y = tf.meshx, tf.meshy
    nelem = ikle.shape[0]
    i0, i1, i2 = ikle[:, 0], ikle[:, 1], ikle[:, 2]

    positions = np.empty((nelem, 3, 2), dtype=np.float64)
    positions[:, 0, 0] = x[i0]
    positions[:, 0, 1] = y[i0]
    positions[:, 1, 0] = x[i1]
    positions[:, 1, 1] = y[i1]
    positions[:, 2, 0] = x[i2]
    positions[:, 2, 1] = y[i2]

    start_indices = np.arange(0, nelem * 3 + 1, 3, dtype=np.int32).tolist()
    pos_binary = encode_binary_attribute(positions.reshape(-1, 2))

    # View extent
    cx = float((x.min() + x.max()) / 2)
    cy = float((y.min() + y.max()) / 2)
    extent = max(float(x.max() - x.min()), float(y.max() - y.min()), 1.0)
    zoom = math.log2(800 / extent) if extent > 0 else 0

    return {
        "nelem": nelem,
        "i0": i0, "i1": i1, "i2": i2,
        "start_indices": start_indices,
        "pos_binary": pos_binary,
        "cx": cx, "cy": cy, "zoom": zoom,
    }


# ---------------------------------------------------------------------------
# Layer builders
# ---------------------------------------------------------------------------


def build_mesh_layer(geom, values, palette_id, animate=False, filter_range=None):
    """Build SolidPolygonLayer from cached geometry + current values."""
    i0, i1, i2 = geom["i0"], geom["i1"], geom["i2"]
    nelem = geom["nelem"]

    tri_values = (values[i0] + values[i1] + values[i2]) / 3.0

    vmin, vmax = float(tri_values.min()), float(tri_values.max())
    if vmax == vmin:
        vmax = vmin + 1.0

    palette_arr = cached_palette_arr(palette_id)
    normalized = np.clip((tri_values - vmin) / (vmax - vmin), 0, 1)
    indices = (normalized * 255).astype(int)
    tri_colors = palette_arr[indices]
    fill_colors = np.repeat(tri_colors[:, np.newaxis, :], 3, axis=1)

    extra = {}
    if animate:
        extra["transitions"] = {
            "getFillColor": transition(200, easing="ease-in-out-cubic"),
        }

    if filter_range is not None:
        extra["extensions"] = [data_filter_extension(filter_size=1)]
        filter_vals = np.repeat(tri_values, 3).astype(np.float32)
        extra["getFilterValue"] = encode_binary_attribute(filter_vals.reshape(-1, 1))
        extra["filterRange"] = list(filter_range)

    lyr = layer(
        "SolidPolygonLayer",
        "mesh",
        data={"length": int(nelem), "startIndices": geom["start_indices"]},
        getPolygon=geom["pos_binary"],
        getFillColor=encode_binary_attribute(fill_colors.reshape(-1, 4)),
        _normalize=False,
        pickable=True,
        autoHighlight=True,
        highlightColor=[255, 255, 255, 60],
        **extra,
    )
    return lyr, vmin, vmax


def build_velocity_layer(tf, time_idx, geom):
    """Build velocity arrow layer from U/V components."""
    varnames = [v.strip() for v in tf.varnames]
    has_u = "VELOCITY U" in varnames
    has_v = "VELOCITY V" in varnames
    if not has_u or not has_v:
        return None

    u = tf.get_data_value("VELOCITY U", time_idx)
    v = tf.get_data_value("VELOCITY V", time_idx)
    x, y = tf.meshx, tf.meshy
    npoin = tf.npoin2

    mag = np.sqrt(u[:npoin]**2 + v[:npoin]**2)
    max_mag = float(mag.max())
    if max_mag < 1e-10:
        return None

    # Subsample for performance (max ~1500 arrows)
    step = max(1, npoin // 1500)
    idx = np.arange(0, npoin, step)

    # Scale arrows relative to mesh extent
    extent = max(float(x[:npoin].max() - x[:npoin].min()),
                 float(y[:npoin].max() - y[:npoin].min()), 1.0)
    arrow_scale = extent / (max_mag * 30)

    arrows = []
    for i in idx:
        if mag[i] < max_mag * 0.01:
            continue
        arrows.append({
            "sourcePosition": [float(x[i]), float(y[i])],
            "targetPosition": [float(x[i] + u[i] * arrow_scale),
                               float(y[i] + v[i] * arrow_scale)],
        })

    if not arrows:
        return None

    return line_layer(
        "velocity",
        arrows,
        getColor=[20, 20, 20, 160],
        getWidth=2,
        widthMinPixels=1,
        widthMaxPixels=3,
        pickable=False,
    )


def build_contour_layer_fn(tf, values, n_contours=6):
    """Build a ContourLayer from mesh node positions and values."""
    x, y = tf.meshx, tf.meshy
    npoin = tf.npoin2
    vmin, vmax = float(values[:npoin].min()), float(values[:npoin].max())
    if vmax == vmin:
        return None

    points = [{"position": [float(x[i]), float(y[i])], "weight": float(values[i])}
              for i in range(npoin)]

    step = (vmax - vmin) / (n_contours + 1)
    contours = [
        {"threshold": vmin + step * (i + 1), "color": [0, 0, 0], "strokeWidth": 1}
        for i in range(n_contours)
    ]

    dx = float(x[:npoin].max() - x[:npoin].min())
    dy = float(y[:npoin].max() - y[:npoin].min())
    cell_size = max(dx, dy) / 100

    return contour_layer(
        "contours",
        points,
        contours=contours,
        cellSize=cell_size,
        getPosition="@@d.position",
        getWeight="@@d.weight",
        pickable=False,
    )


# ---------------------------------------------------------------------------
# Map widget
# ---------------------------------------------------------------------------

map_widget = MapWidget(
    "map",
    view_state={"target": [0, 0, 0], "zoom": 0},
    style="",
    cooperative_gestures=True,
    tooltip={
        "html": "<b>TELEMAC Mesh</b><br/>Hover over the mesh to inspect",
        "style": {"backgroundColor": "#1a1a2e", "color": "#eee", "fontSize": "12px"},
    },
)

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.div(
            ui.h5("TELEMAC Viewer", style="display:inline"),
            ui.input_dark_mode(id="dark_mode", mode="light"),
            class_="d-flex justify-content-between align-items-center mb-2",
        ),
        ui.accordion(
            ui.accordion_panel(
                "Data",
                ui.input_select("example", "Example case", choices=EXAMPLE_CHOICES),
                ui.input_file("upload", "Or upload .slf file", accept=[".slf"]),
            ),
            ui.accordion_panel(
                "Visualization",
                ui.output_ui("var_select_ui"),
                ui.input_select("palette", "Color palette", choices=list(PALETTES.keys())),
                ui.input_switch("vectors", "Show velocity vectors", value=False),
                ui.input_switch("contours", "Show contour lines", value=False),
                ui.output_ui("filter_ui"),
            ),
            ui.accordion_panel(
                "Playback",
                ui.output_ui("time_slider_ui"),
                ui.div(
                    ui.input_action_button(
                        "play_btn", "▶ Play", class_="btn-sm btn-primary w-100"
                    ),
                    class_="mb-2",
                ),
                ui.input_slider(
                    "speed", "Speed (s/frame)",
                    min=0.1, max=2.0, value=0.5, step=0.1,
                ),
                ui.input_switch("loop", "Loop animation", value=True),
            ),
            ui.accordion_panel(
                "Statistics",
                ui.output_ui("stats_ui"),
                ui.output_ui("hover_info_ui"),
            ),
            id="sidebar_accordion",
            open=True,
            multiple=True,
        ),
        width="300px",
    ),
    ui.card(
        head_includes(),
        map_widget.ui(height="calc(100vh - 40px)"),
        full_screen=True,
    ),
    title="TELEMAC Result Viewer",
    fillable=True,
)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


def server(input, output, session):
    playing = reactive.value(False)
    last_file_path = reactive.value("")

    # -- Core reactive calcs (shared, computed once per change) --

    @reactive.calc
    def tel_file():
        uploaded = input.upload()
        if uploaded:
            path = uploaded[0]["datapath"]
        else:
            path = EXAMPLES[input.example()]
        return TelemacFile(path)

    @reactive.calc
    def mesh_geom():
        """Cached mesh geometry — only recomputes on file change."""
        tf = tel_file()
        return build_mesh_geometry(tf)

    @reactive.calc
    def current_var():
        """Current variable, validated against file."""
        tf = tel_file()
        var = input.variable() if input.variable() else None
        if var and var in tf.varnames:
            return var
        return tf.varnames[0]

    @reactive.calc
    def current_tidx():
        """Current time index, clamped to valid range."""
        tf = tel_file()
        tidx = input.time_idx() if input.time_idx() is not None else 0
        return min(tidx, len(tf.times) - 1)

    @reactive.calc
    def current_values():
        """Current variable data — single read shared by all consumers."""
        tf = tel_file()
        return tf.get_data_value(current_var(), current_tidx())

    # -- Dynamic UI outputs --

    @output
    @render.ui
    def var_select_ui():
        tf = tel_file()
        choices = {v: v for v in tf.varnames}
        selected = "WATER DEPTH" if "WATER DEPTH" in tf.varnames else tf.varnames[0]
        return ui.input_select("variable", "Variable", choices=choices, selected=selected)

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
    def filter_ui():
        vals = current_values()
        vmin, vmax = float(vals.min()), float(vals.max())
        step = round((vmax - vmin) / 100, 4) if vmax > vmin else 0.01
        return ui.input_slider(
            "filter_range", "Value filter",
            min=round(vmin, 4), max=round(vmax, 4),
            value=[round(vmin, 4), round(vmax, 4)],
            step=step,
        )

    # -- Playback --

    @reactive.effect
    @reactive.event(input.play_btn)
    def toggle_play():
        playing.set(not playing.get())
        if playing.get():
            ui.update_action_button("play_btn", label="⏸ Pause")
        else:
            ui.update_action_button("play_btn", label="▶ Play")

    @reactive.effect
    def auto_advance():
        if not playing.get():
            return
        speed = input.speed() if input.speed() is not None else 0.5
        reactive.invalidate_later(speed)
        tf = tel_file()
        n = len(tf.times)
        current = input.time_idx() if input.time_idx() is not None else 0
        next_idx = current + 1
        if next_idx >= n:
            if input.loop():
                next_idx = 0
            else:
                playing.set(False)
                ui.update_action_button("play_btn", label="▶ Play")
                return
        ui.update_slider("time_idx", value=next_idx)

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
            ui.value_box(
                "Time",
                format_time(t),
                f"Step {tidx + 1} of {len(tf.times)}",
            ),
            ui.value_box(
                "Min",
                f"{vals.min():.4f}",
                var,
            ),
            ui.value_box(
                "Max",
                f"{vals.max():.4f}",
                var,
            ),
            ui.value_box(
                "Mesh",
                f"{tf.nelem2:,} elements",
                f"{tf.npoin2:,} nodes",
            ),
            width=1,
        )

    # -- Hover info --

    @output
    @render.ui
    def hover_info_ui():
        hover = input.map_hover()
        if not hover or "coordinate" not in hover:
            return ui.p("Hover over the mesh to see values", class_="text-muted small")
        coord = hover["coordinate"]
        tf = tel_file()
        x, y = tf.meshx, tf.meshy
        npoin = tf.npoin2
        # Find nearest node
        px, py = float(coord[0]), float(coord[1])
        dists = (x[:npoin] - px)**2 + (y[:npoin] - py)**2
        nearest = int(np.argmin(dists))
        vals = current_values()
        var = current_var()
        val = float(vals[nearest])
        return ui.div(
            ui.strong(f"{var}: {val:.4f}"),
            ui.br(),
            ui.span(f"Node {nearest} ({x[nearest]:.1f}, {y[nearest]:.1f})",
                     class_="text-muted small"),
        )

    # -- Map update --

    @reactive.effect
    async def update_map():
        tf = tel_file()
        geom = mesh_geom()
        var = current_var()
        tidx = current_tidx()
        values = current_values()
        palette_id = input.palette() if input.palette() in PALETTES else "Viridis"

        filt = input.filter_range() if input.filter_range() is not None else None
        lyr, vmin, vmax = build_mesh_layer(
            geom, values, palette_id,
            animate=not playing.get(),
            filter_range=filt,
        )

        layers = [lyr]

        if input.vectors():
            vlyr = build_velocity_layer(tf, tidx, geom)
            if vlyr is not None:
                layers.append(vlyr)

        if input.contours():
            clyr = build_contour_layer_fn(tf, values)
            if clyr is not None:
                layers.append(clyr)

        gradient_colors = cached_gradient_colors(palette_id)
        legend = deck_legend_control(
            entries=[
                {
                    "layer_id": "mesh",
                    "label": f"{var}  [{vmin:.3g} – {vmax:.3g}]",
                    "colors": gradient_colors,
                    "shape": "gradient",
                }
            ],
            position="bottom-right",
            show_checkbox=False,
            title="Legend",
        )

        # Only send view_state on file change (P4: don't reset user's pan/zoom)
        uploaded = input.upload()
        current_path = uploaded[0]["datapath"] if uploaded else EXAMPLES.get(input.example(), "")
        kwargs = {}
        if current_path != last_file_path.get():
            last_file_path.set(current_path)
            kwargs["view_state"] = {"target": [geom["cx"], geom["cy"], 0], "zoom": geom["zoom"]}

        await map_widget.update(
            session,
            layers=layers,
            views=[orthographic_view()],
            widgets=[
                zoom_widget(),
                fullscreen_widget(),
                scale_widget(),
                screenshot_widget(),
                compass_widget(),
                reset_view_widget(),
                loading_widget(),
                legend,
            ],
            **kwargs,
        )


app = App(app_ui, server)
