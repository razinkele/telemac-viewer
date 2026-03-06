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
    deck_legend_control,
    reset_view_widget,
    loading_widget,
    orbit_view,
    gimbal_widget,
    lighting_effect,
    ambient_light,
    directional_light,
)

from constants import (
    EXAMPLES, EXAMPLE_CHOICES, PALETTES, _M2D,
    cached_gradient_colors, format_time,
)
from geometry import build_mesh_geometry, build_mesh_geometry_3d
from layers import (
    build_mesh_layer,
    build_velocity_layer,
    build_contour_layer_fn,
    build_marker_layer,
    build_cross_section_layer,
    build_particle_layer,
)
from analysis import (
    coord_to_meters,
    time_series_at_point,
    cross_section_profile,
    compute_particle_paths,
    generate_seed_grid,
    get_available_derived,
    compute_derived,
    export_timeseries_csv,
    export_crosssection_csv,
)
from data_manip.extraction.telemac_file import TelemacFile

# ---------------------------------------------------------------------------
# Map widget
# ---------------------------------------------------------------------------

map_widget = MapWidget(
    "map",
    view_state={"longitude": 0, "latitude": 0, "zoom": 0},
    style="data:application/json;charset=utf-8,%7B%22version%22%3A8%2C%22sources%22%3A%7B%7D%2C%22layers%22%3A%5B%7B%22id%22%3A%22bg%22%2C%22type%22%3A%22background%22%2C%22paint%22%3A%7B%22background-color%22%3A%22%23f8f9fa%22%7D%7D%5D%7D",
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
                ui.input_action_button("draw_xsec", "Draw Cross-Section",
                                       class_="btn-sm btn-outline-primary w-100 mb-1"),
                ui.output_ui("clear_xsec_ui"),
                ui.input_switch("particles", "Show particle traces", value=False),
                ui.output_ui("particle_seed_ui"),
                ui.output_ui("toggle_3d_ui"),
            ),
            ui.accordion_panel(
                "Playback",
                ui.output_ui("time_slider_ui"),
                ui.div(
                    ui.input_action_button(
                        "play_btn", "Play", class_="btn-sm btn-primary w-100"
                    ),
                    class_="mb-2",
                ),
                ui.input_slider(
                    "speed", "Speed (s/frame)",
                    min=0.1, max=2.0, value=0.5, step=0.1,
                ),
                ui.input_switch("loop", "Loop animation", value=True),
                ui.output_ui("trail_length_ui"),
            ),
            ui.accordion_panel(
                "Statistics",
                ui.output_ui("stats_ui"),
                ui.output_ui("hover_info_ui"),
            ),
            ui.accordion_panel(
                "File Info",
                ui.output_ui("file_info_ui"),
            ),
            id="sidebar_accordion",
            open=True,
            multiple=True,
        ),
        width="300px",
    ),
    ui.card(
        head_includes(),
        ui.div(
            map_widget.ui(height="100%"),
            id="map-container",
            style="flex:1; min-height:0;",
        ),
        ui.output_ui("analysis_panel_ui"),
        full_screen=True,
        style="display:flex; flex-direction:column;",
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
    analysis_mode = reactive.value("none")
    clicked_points = reactive.value([])  # list of (x_meters, y_meters)
    cross_section_points = reactive.value(None)  # [[x_m, y_m], ...] or None
    particle_paths = reactive.value(None)  # list of paths or None
    is_3d_mode = reactive.value(False)

    # -- Core reactive calcs --

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
        tf = tel_file()
        if is_3d_mode.get() and tf.nplan > 1:
            z_scale = input.z_scale() if input.z_scale() is not None else 10
            try:
                z_name = tf.get_z_name()
                z_vals = tf.get_data_value(z_name, current_tidx())
            except Exception:
                z_vals = np.zeros(tf.npoin2, dtype=np.float32)
            return build_mesh_geometry_3d(tf, z_vals, z_scale)
        return build_mesh_geometry(tf)

    @reactive.calc
    def current_var():
        tf = tel_file()
        var = input.variable() if input.variable() else None
        if var and var in tf.varnames:
            return var
        return tf.varnames[0]

    @reactive.calc
    def current_tidx():
        tf = tel_file()
        tidx = input.time_idx() if input.time_idx() is not None else 0
        return min(tidx, len(tf.times) - 1)

    @reactive.calc
    def current_values():
        tf = tel_file()
        var = current_var()
        tidx = current_tidx()
        derived = get_available_derived(tf)
        if var in derived:
            return compute_derived(tf, var, tidx)
        return tf.get_data_value(var, tidx)

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
        except Exception:
            depth_range = 1.0
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
        title = "Time Series" if mode == "timeseries" else "Cross-Section"
        n_pts = len(clicked_points.get()) if mode == "timeseries" else 0
        subtitle = f" ({n_pts} point{'s' if n_pts != 1 else ''})" if n_pts else ""
        return ui.div(
            ui.div(
                ui.strong(title + subtitle),
                ui.download_button("download_csv", "CSV",
                                   class_="btn-sm btn-outline-success ms-2"),
                ui.input_action_button("close_analysis", "Close",
                                       class_="btn-sm btn-outline-secondary ms-2"),
                class_="d-flex align-items-center p-2 border-top",
            ),
            output_widget("analysis_chart"),
            style="height:250px; overflow:hidden;",
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
        x_m, y_m = coord_to_meters(coord[0], coord[1], geom["x_off"], geom["y_off"])
        pts = clicked_points.get().copy()
        pts.append((x_m, y_m))
        clicked_points.set(pts)
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
            if feat.get("geometry", {}).get("type") != "LineString":
                continue
            coords = feat["geometry"]["coordinates"]
            geom = mesh_geom()
            poly_m = []
            for c in coords:
                x_m = float(c[0]) * _M2D + geom["x_off"]
                y_m = float(c[1]) * _M2D + geom["y_off"]
                poly_m.append([x_m, y_m])

            if input.particles():
                # Seed line mode: distribute ~100 particles along drawn line
                n_seeds = 100
                seeds = []
                total_len = sum(
                    np.sqrt((poly_m[i+1][0]-poly_m[i][0])**2 + (poly_m[i+1][1]-poly_m[i][1])**2)
                    for i in range(len(poly_m)-1)
                )
                step_dist = total_len / n_seeds if total_len > 0 else 1
                dist = 0
                seg = 0
                for s in range(n_seeds):
                    target = s * step_dist
                    while seg < len(poly_m) - 1:
                        seg_len = np.sqrt(
                            (poly_m[seg+1][0]-poly_m[seg][0])**2 +
                            (poly_m[seg+1][1]-poly_m[seg][1])**2
                        )
                        if dist + seg_len >= target:
                            t = (target - dist) / seg_len if seg_len > 0 else 0
                            x = poly_m[seg][0] + t * (poly_m[seg+1][0] - poly_m[seg][0])
                            y = poly_m[seg][1] + t * (poly_m[seg+1][1] - poly_m[seg][1])
                            seeds.append([x, y])
                            break
                        dist += seg_len
                        seg += 1

                tf = tel_file()
                ui.notification_show("Computing particle paths from seed line...",
                                     duration=None, id="particle_notif")
                paths = compute_particle_paths(tf, seeds, geom["x_off"], geom["y_off"])
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
    def handle_particles_toggle():
        if input.particles():
            tf = tel_file()
            geom = mesh_geom()
            ui.notification_show("Computing particle trajectories...",
                                 duration=None, id="particle_notif")
            seeds = generate_seed_grid(tf, n_target=500)
            paths = compute_particle_paths(tf, seeds, geom["x_off"], geom["y_off"])
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
        current = input.time_idx() if input.time_idx() is not None else 0
        next_idx = current + 1
        if next_idx >= n:
            if input.loop():
                next_idx = 0
            else:
                playing.set(False)
                ui.update_action_button("play_btn", label="Play")
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
        x, y = tf.meshx, tf.meshy
        npoin = tf.npoin2
        px = float(coord[0]) * _M2D + geom["x_off"]
        py = float(coord[1]) * _M2D + geom["y_off"]
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

    # -- File info --

    @output
    @render.ui
    def file_info_ui():
        tf = tel_file()
        dt = tf.times[1] - tf.times[0] if len(tf.times) > 1 else 0
        precision = "Double" if getattr(tf, 'float_type', 'f') == 'd' else "Single"
        nplan = getattr(tf, 'nplan', 0)
        dim = "3D" if nplan > 1 else "2D"
        date_info = ""
        if hasattr(tf, 'datetime') and tf.datetime:
            date_info = str(tf.datetime[0]) if isinstance(tf.datetime, (list, tuple)) else str(tf.datetime)

        rows = [
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
            times_arr = np.array(tf.times)
            buf.write("Time (s)")
            all_values = []
            for i, pt in enumerate(pts):
                _, values = time_series_at_point(tf, var, pt[0], pt[1])
                all_values.append(values)
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
        else:
            yield ""

    # -- Map update --

    @reactive.effect
    async def update_map():
        tf = tel_file()
        geom = mesh_geom()
        var = current_var()
        tidx = current_tidx()
        values = current_values()
        palette_id = input.palette() if input.palette() in PALETTES else "Viridis"

        # Feature 1: pass filter_range to layer builder
        filt = input.filter_range() if input.filter_range() is not None else None
        lyr, vmin, vmax = build_mesh_layer(geom, values, palette_id, filter_range=filt)

        layers = [lyr]

        if input.vectors():
            vlyr = build_velocity_layer(tf, tidx, geom)
            if vlyr is not None:
                layers.append(vlyr)

        if input.contours():
            clyr = build_contour_layer_fn(tf, values, geom)
            if clyr is not None:
                layers.append(clyr)

        # Markers for clicked points
        pts = clicked_points.get()
        for i, pt in enumerate(pts):
            mx, my = float(pt[0] - geom["x_off"]), float(pt[1] - geom["y_off"])
            layers.append(build_marker_layer(mx, my, layer_id=f"marker-{i}"))

        # Cross-section path
        xsec = cross_section_points.get()
        if xsec is not None:
            path_centered = [[x - geom["x_off"], y - geom["y_off"]] for x, y in xsec]
            layers.append(build_cross_section_layer(path_centered))

        # Particle traces
        paths = particle_paths.get()
        if paths and input.particles():
            current_time = float(tf.times[tidx])
            trail = input.trail_length() if input.trail_length() is not None else 1.0
            layers.append(build_particle_layer(paths, current_time, trail))

        gradient_colors = cached_gradient_colors(palette_id)
        legend = deck_legend_control(
            entries=[{
                "layer_id": "mesh",
                "label": f"{var}  [{vmin:.3g} - {vmax:.3g}]",
                "colors": gradient_colors,
                "shape": "gradient",
            }],
            position="bottom-right",
            show_checkbox=False,
            title="Legend",
        )

        # Only update view_state on file change
        uploaded = input.upload()
        current_path = uploaded[0]["datapath"] if uploaded else EXAMPLES.get(input.example(), "")
        kwargs = {}
        if current_path != last_file_path.get():
            last_file_path.set(current_path)
            kwargs["view_state"] = {
                "longitude": 0,
                "latitude": 0,
                "zoom": geom["zoom"],
            }

        # Build widgets list
        widgets = [
            zoom_widget(),
            fullscreen_widget(),
            scale_widget(),
            screenshot_widget(),
            compass_widget(),
            reset_view_widget(),
            loading_widget(),
            legend,
        ]

        # 3D mode: add gimbal, orbit view, lighting
        if is_3d_mode.get():
            widgets.append(gimbal_widget())
            kwargs["views"] = [orbit_view(
                target=[0, 0, 0],
                rotationX=-30,
                rotationOrbit=-30,
                zoom=geom["zoom"],
            )]
            kwargs["effects"] = [lighting_effect(
                ambient_light(intensity=0.4),
                directional_light(direction=[-1, -3, -1], intensity=0.8),
            )]

        await map_widget.update(
            session,
            layers=layers,
            widgets=widgets,
            **kwargs,
        )


app = App(app_ui, server)
