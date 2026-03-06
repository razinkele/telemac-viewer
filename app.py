import math
import sys
import os
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
)
from data_manip.extraction.telemac_file import TelemacFile

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXAMPLES = {
    "Gouttedo (raindrop)": os.path.join(
        os.environ["HOMETEL"],
        "examples/telemac2d/gouttedo/r2d_gouttedo_v1p0.slf",
    ),
    "Ritter (dam break)": os.path.join(
        os.environ["HOMETEL"],
        "examples/telemac2d/dambreak/r2d_ritter-hllc.slf",
    ),
    "Malpasset (dam break)": os.path.join(
        os.environ["HOMETEL"],
        "examples/telemac2d/malpasset/r2d_malpasset-hllc.slf",
    ),
    "3D Canal": os.path.join(
        os.environ["HOMETEL"],
        "examples/telemac3d/canal/r3d_canal-t3d.slf",
    ),
}

PALETTES = {
    "Viridis": PALETTE_VIRIDIS,
    "Plasma": PALETTE_PLASMA,
    "Ocean": PALETTE_OCEAN,
    "Thermal": PALETTE_THERMAL,
    "Chlorophyll": PALETTE_CHLOROPHYLL,
}

# ---------------------------------------------------------------------------
# SELAFIN to deck.gl conversion
# ---------------------------------------------------------------------------


def build_mesh_layer(tf, var_name, time_idx, palette, layer_id="mesh", animate=False, filter_range=None):
    """Convert TelemacFile mesh + variable into a SolidPolygonLayer dict."""
    ikle = tf.ikle2  # (nelem, 3), 0-indexed
    x, y = tf.meshx, tf.meshy
    values = tf.get_data_value(var_name, time_idx)

    nelem = ikle.shape[0]
    i0, i1, i2 = ikle[:, 0], ikle[:, 1], ikle[:, 2]

    # Per-triangle vertex positions: (nelem, 3, 2)
    positions = np.empty((nelem, 3, 2), dtype=np.float64)
    positions[:, 0, 0] = x[i0]
    positions[:, 0, 1] = y[i0]
    positions[:, 1, 0] = x[i1]
    positions[:, 1, 1] = y[i1]
    positions[:, 2, 0] = x[i2]
    positions[:, 2, 1] = y[i2]

    # Per-triangle mean value
    tri_values = (values[i0] + values[i1] + values[i2]) / 3.0

    # Color mapping — vectorized
    vmin, vmax = float(tri_values.min()), float(tri_values.max())
    if vmax == vmin:
        vmax = vmin + 1.0
    palette_arr = np.array(color_range(256, palette), dtype=np.uint8)  # (256, 4)
    normalized = np.clip((tri_values - vmin) / (vmax - vmin), 0, 1)
    indices = (normalized * 255).astype(int)
    tri_colors = palette_arr[indices]  # (nelem, 4)
    fill_colors = np.repeat(tri_colors[:, np.newaxis, :], 3, axis=1)  # (nelem, 3, 4)

    # startIndices: every polygon is exactly 3 vertices
    start_indices = np.arange(0, nelem * 3 + 1, 3, dtype=np.int32).tolist()

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
        layer_id,
        data={"length": int(nelem), "startIndices": start_indices},
        getPolygon=encode_binary_attribute(positions.reshape(-1, 2)),
        getFillColor=encode_binary_attribute(fill_colors.reshape(-1, 4)),
        _normalize=False,
        pickable=True,
        autoHighlight=True,
        highlightColor=[255, 255, 255, 60],
        **extra,
    )
    return lyr, vmin, vmax


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


def build_contour_layer(tf, var_name, time_idx, n_contours=6):
    """Build a ContourLayer from mesh node positions and values."""
    x, y = tf.meshx, tf.meshy
    values = tf.get_data_value(var_name, time_idx)
    vmin, vmax = float(values.min()), float(values.max())
    if vmax == vmin:
        return None

    points = [{"position": [float(x[i]), float(y[i])], "weight": float(values[i])} for i in range(len(x))]

    step = (vmax - vmin) / (n_contours + 1)
    contours = [
        {"threshold": vmin + step * (i + 1), "color": [0, 0, 0], "strokeWidth": 1}
        for i in range(n_contours)
    ]

    dx = float(x.max() - x.min())
    dy = float(y.max() - y.min())
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
                ui.input_select("example", "Example case", choices=list(EXAMPLES.keys())),
                ui.input_file("upload", "Or upload .slf file", accept=[".slf"]),
            ),
            ui.accordion_panel(
                "Visualization",
                ui.output_ui("var_select_ui"),
                ui.input_select("palette", "Color palette", choices=list(PALETTES.keys())),
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
    last_good_file = reactive.value(None)

    @reactive.calc
    def tel_file():
        ui.notification_show("Loading file...", id="loading", duration=None)
        try:
            uploaded = input.upload()
            if uploaded:
                tf = TelemacFile(uploaded[0]["datapath"])
            else:
                tf = TelemacFile(EXAMPLES[input.example()])
            last_good_file.set(tf)
            ui.notification_remove("loading")
            return tf
        except Exception as e:
            ui.notification_remove("loading")
            ui.notification_show(
                f"Error loading file: {e}",
                type="error",
                duration=5,
            )
            if last_good_file.get() is not None:
                return last_good_file.get()
            return TelemacFile(EXAMPLES[list(EXAMPLES.keys())[0]])

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
        tf = tel_file()
        var = input.variable() if input.variable() else tf.varnames[0]
        tidx = input.time_idx() if input.time_idx() is not None else 0
        vals = tf.get_data_value(var, tidx)
        vmin, vmax = float(vals.min()), float(vals.max())
        step = round((vmax - vmin) / 100, 4) if vmax > vmin else 0.01
        return ui.input_slider(
            "filter_range", "Value filter",
            min=round(vmin, 4), max=round(vmax, 4),
            value=[round(vmin, 4), round(vmax, 4)],
            step=step,
        )

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

    @output
    @render.ui
    def stats_ui():
        tf = tel_file()
        var = input.variable() if input.variable() else tf.varnames[0]
        tidx = input.time_idx() if input.time_idx() is not None else 0
        vals = tf.get_data_value(var, tidx)
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

    @reactive.effect
    async def update_map():
        tf = tel_file()
        var = input.variable() if input.variable() else tf.varnames[0]
        tidx = input.time_idx() if input.time_idx() is not None else 0
        palette = PALETTES.get(input.palette(), PALETTE_VIRIDIS)

        filt = input.filter_range() if input.filter_range() is not None else None
        lyr, vmin, vmax = build_mesh_layer(tf, var, tidx, palette, animate=not playing.get(), filter_range=filt)

        layers = [lyr]
        if input.contours():
            clyr = build_contour_layer(tf, var, tidx)
            if clyr is not None:
                layers.append(clyr)

        cx = float((tf.meshx.min() + tf.meshx.max()) / 2)
        cy = float((tf.meshy.min() + tf.meshy.max()) / 2)
        dx = float(tf.meshx.max() - tf.meshx.min())
        dy = float(tf.meshy.max() - tf.meshy.min())
        extent = max(dx, dy, 1.0)
        zoom = math.log2(800 / extent) if extent > 0 else 0

        # Sample 8 colors from the palette for the gradient legend
        palette_arr = color_range(256, palette)
        gradient_colors = [palette_arr[i] for i in range(0, 256, 32)]
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

        await map_widget.update(
            session,
            layers=layers,
            views=[orthographic_view()],
            view_state={"target": [cx, cy, 0], "zoom": zoom},
            widgets=[
                zoom_widget(),
                fullscreen_widget(),
                scale_widget(),
                screenshot_widget(),
                compass_widget(),
                legend,
            ],
        )


app = App(app_ui, server)
