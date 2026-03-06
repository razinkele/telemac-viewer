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
    build_wireframe_layer,
    build_extrema_markers,
    build_measurement_layer,
    build_boundary_layer,
)
import subprocess
import os as _os
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
    compute_mesh_quality,
    find_cas_files,
    detect_module,
    vertical_profile_at_point,
    compute_difference,
    compute_temporal_stats,
    find_extrema,
    find_boundary_nodes,
    compute_slope,
    export_all_variables_csv,
    compute_mesh_integral,
    evaluate_expression,
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
                ui.input_switch("dark_bg", "Dark map background", value=False),
            ),
            ui.accordion_panel(
                "Visualization",
                ui.output_ui("var_select_ui"),
                ui.input_select("palette", "Color palette", choices=list(PALETTES.keys())),
                ui.input_switch("vectors", "Show velocity vectors", value=False),
                ui.input_switch("contours", "Show contour lines", value=False),
                ui.input_switch("wireframe", "Show mesh wireframe", value=False),
                ui.input_switch("boundary_nodes", "Show boundary nodes", value=False),
                ui.input_switch("show_extrema", "Show min/max locations", value=False),
                ui.input_switch("log_scale", "Log scale coloring", value=False),
                ui.input_switch("reverse_palette", "Reverse palette", value=False),
                ui.input_switch("diff_mode", "Difference mode", value=False),
                ui.output_ui("ref_timestep_ui"),
                ui.output_ui("color_range_ui"),
                ui.output_ui("filter_ui"),
                ui.input_action_button("draw_xsec", "Draw Cross-Section",
                                       class_="btn-sm btn-outline-primary w-100 mb-1"),
                ui.output_ui("clear_xsec_ui"),
                ui.input_switch("particles", "Show particle traces", value=False),
                ui.output_ui("particle_seed_ui"),
                ui.input_switch("mesh_quality", "Show mesh quality", value=False),
                ui.input_switch("show_slope", "Show slope/gradient", value=False),
                ui.input_action_button("measure_btn", "Measure Distance",
                                       class_="btn-sm btn-outline-warning w-100 mb-1"),
                ui.output_ui("measure_info_ui"),
                ui.output_ui("compare_var_ui"),
                ui.div(
                    ui.input_text("expr_input", "Custom expression", placeholder="VELOCITY_U**2 + VELOCITY_V**2"),
                    ui.input_action_button("eval_expr", "Eval",
                                           class_="btn-sm btn-outline-secondary"),
                    class_="d-flex align-items-end gap-1 mb-1",
                ),
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
            ui.accordion_panel(
                "Statistics",
                ui.output_ui("stats_ui"),
                ui.output_ui("hover_info_ui"),
                ui.input_action_button("show_histogram", "Value Histogram",
                                       class_="btn-sm btn-outline-primary w-100 mb-1"),
                ui.input_action_button("compute_integral", "Compute Integral",
                                       class_="btn-sm btn-outline-info w-100 mb-1"),
                ui.output_ui("integral_ui"),
                ui.input_action_button("compute_temporal", "Compute Temporal Stats",
                                       class_="btn-sm btn-outline-info w-100 mb-1"),
                ui.output_ui("temporal_stats_ui"),
                ui.output_ui("node_inspector_ui"),
            ),
            ui.accordion_panel(
                "File Info",
                ui.output_ui("file_info_ui"),
            ),
            ui.accordion_panel(
                "Run Simulation",
                ui.output_ui("cas_select_ui"),
                ui.input_numeric("ncores", "CPU cores", value=4, min=0, max=28, step=1),
                ui.input_action_button("run_sim", "Run", class_="btn-sm btn-success w-100 mb-1"),
                ui.input_action_button("stop_sim", "Stop", class_="btn-sm btn-danger w-100 mb-1"),
                ui.output_ui("sim_status_ui"),
                ui.tags.pre(
                    ui.output_text_verbatim("sim_console"),
                    style="max-height:200px; overflow-y:auto; font-size:11px; background:#1a1a2e; color:#eee; padding:6px; margin-top:4px;",
                ),
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
        ui.output_ui("coord_readout_ui"),
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
        """),
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
    sim_process = reactive.value(None)  # subprocess.Popen or None
    sim_output = reactive.value("")  # accumulated console output
    sim_running = reactive.value(False)
    temporal_stats_cache = reactive.value(None)  # dict with min/max/mean arrays
    measure_points = reactive.value([])  # list of [x_centered, y_centered] (max 2)
    measure_mode = reactive.value(False)  # True when waiting for measurement clicks

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
        titles = {"timeseries": "Time Series", "crosssection": "Cross-Section",
                  "vertprofile": "Vertical Profile", "histogram": "Value Histogram"}
        title = titles.get(mode, mode)
        n_pts = len(clicked_points.get()) if mode == "timeseries" else 0
        subtitle = f" ({n_pts} point{'s' if n_pts != 1 else ''})" if n_pts else ""
        return ui.div(
            ui.div(
                ui.strong(title + subtitle),
                ui.download_button("download_csv", "CSV",
                                   class_="btn-sm btn-outline-success ms-2"),
                ui.download_button("download_all_vars", "All Vars",
                                   class_="btn-sm btn-outline-info ms-1"),
                ui.input_action_button("undo_point", "Undo",
                                       class_="btn-sm btn-outline-warning ms-1"),
                ui.input_action_button("close_analysis", "Close",
                                       class_="btn-sm btn-outline-secondary ms-1"),
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

        if mode == "vertprofile":
            pts = clicked_points.get()
            if not pts:
                return go.Figure()
            fig = go.Figure()
            for i, pt in enumerate(pts):
                elevations, values = vertical_profile_at_point(tf, var, tidx, pt[0], pt[1])
                if len(elevations) > 0:
                    fig.add_trace(go.Scatter(
                        x=values, y=elevations, mode="lines+markers",
                        name=f"Pt {i+1} ({pt[0]:.0f}, {pt[1]:.0f})",
                    ))
            fig.update_layout(
                xaxis_title=var, yaxis_title="Elevation (m)",
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

        # Measurement mode intercepts clicks
        if measure_mode.get():
            pts = measure_points.get().copy()
            # Store centered coordinates for layer rendering
            x_c, y_c = float(x_m - geom["x_off"]), float(y_m - geom["y_off"])
            pts.append([x_c, y_c])
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

    # -- Temporal statistics --

    @reactive.effect
    @reactive.event(input.compute_temporal)
    def handle_compute_temporal():
        tf = tel_file()
        var = current_var()
        ui.notification_show(f"Computing temporal stats for {var}...",
                             duration=None, id="temporal_notif")
        stats = compute_temporal_stats(tf, var)
        temporal_stats_cache.set(stats)
        ui.notification_remove("temporal_notif")
        ui.notification_show("Temporal statistics computed", duration=3)

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
        x, y = tf.meshx, tf.meshy
        npoin = tf.npoin2
        # Show info for the last clicked point
        px, py = pts[-1]
        dists = (x[:npoin] - px)**2 + (y[:npoin] - py)**2
        nearest = int(np.argmin(dists))
        rows = []
        for vname in tf.varnames:
            val = float(tf.get_data_value(vname, tidx)[nearest])
            rows.append(ui.div(
                ui.span(vname, class_="text-muted", style="width:140px;display:inline-block;font-size:11px;"),
                ui.strong(f"{val:.4f}", style="font-size:11px;"),
                class_="mb-0",
            ))
        return ui.div(
            ui.strong(f"Node {nearest}", class_="small"),
            ui.span(f" ({x[nearest]:.1f}, {y[nearest]:.1f})", class_="text-muted small"),
            ui.div(*rows, style="max-height:120px; overflow-y:auto; margin-top:4px;"),
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
        if not hover or "coordinate" not in hover:
            return ui.div(
                ui.span("Move cursor over mesh", class_="text-muted"),
                style="font-size:11px; padding:2px 8px; border-top:1px solid #ddd; background:#f8f9fa;",
            )
        coord = hover["coordinate"]
        geom = mesh_geom()
        x_m = float(coord[0]) * _M2D + geom["x_off"]
        y_m = float(coord[1]) * _M2D + geom["y_off"]
        return ui.div(
            ui.span(f"X: {x_m:.2f} m  Y: {y_m:.2f} m", style="font-family:monospace;"),
            style="font-size:11px; padding:2px 8px; border-top:1px solid #ddd; background:#f8f9fa;",
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

    # -- Comparison variable --

    @output
    @render.ui
    def compare_var_ui():
        tf = tel_file()
        choices = {"": "None (off)"}
        choices.update({v: v for v in tf.varnames})
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
        if input.mesh_quality():
            tf = tel_file()
            return compute_mesh_quality(tf)
        if input.show_slope():
            tf = tel_file()
            return compute_slope(tf, current_values())
        # Temporal stats display
        stats = temporal_stats_cache.get()
        td = input.temporal_display() if input.temporal_display() else "none"
        if stats is not None and td != "none":
            return stats[td]
        if input.diff_mode():
            tf = tel_file()
            var = current_var()
            tidx = current_tidx()
            ref = input.ref_tidx() if input.ref_tidx() is not None else 0
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
        cas_name = input.cas_file() if input.cas_file() else None
        if not cas_name or cas_name not in cas_files:
            ui.notification_show("No .cas file selected", type="warning", duration=3)
            return
        cas_path = cas_files[cas_name]
        module = detect_module(cas_path)
        ncores = input.ncores() if input.ncores() is not None else 4
        cas_dir = _os.path.dirname(cas_path)

        # Build command
        runner = f"{module}.py"
        cmd = [runner, cas_name, f"--ncsize={ncores}"]

        sim_output.set(f"$ cd {cas_dir}\n$ {' '.join(cmd)}\n\n")
        sim_running.set(True)

        try:
            # Source TELEMAC env and run
            env_script = "/home/razinka/telemac/telemac-v8p5r1/configs/pysource.local.sh"
            shell_cmd = f"source {env_script} && cd {cas_dir} && {' '.join(cmd)} 2>&1"
            proc = subprocess.Popen(
                ["bash", "-c", shell_cmd],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            sim_process.set(proc)

            # Read output incrementally
            buf = sim_output.get()
            for line in iter(proc.stdout.readline, ""):
                buf += line
                sim_output.set(buf)
                await reactive.flush()
            proc.wait()
            buf += f"\n--- Process exited with code {proc.returncode} ---\n"
            sim_output.set(buf)
        except Exception as e:
            sim_output.set(sim_output.get() + f"\nERROR: {e}\n")
        finally:
            sim_running.set(False)
            sim_process.set(None)

    @reactive.effect
    @reactive.event(input.stop_sim)
    def handle_stop_sim():
        proc = sim_process.get()
        if proc and proc.poll() is None:
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

        # Auto-switch to diverging palette for difference mode
        use_diverging = input.diff_mode() and not input.mesh_quality()
        if use_diverging:
            palette_id = "_diverging"

        # Display variable label
        td = input.temporal_display() if input.temporal_display() else "none"
        if expr_result.get() is not None:
            display_var = f"EXPR: {input.expr_input()}"
        elif input.mesh_quality():
            display_var = "MESH QUALITY"
        elif input.show_slope():
            display_var = f"SLOPE ({var})"
        elif td != "none" and temporal_stats_cache.get() is not None:
            display_var = f"{var} (temporal {td})"
        elif input.diff_mode():
            ref = input.ref_tidx() if input.ref_tidx() is not None else 0
            display_var = f"Δ {var} (t{tidx}-t{ref})"
        else:
            display_var = var

        # Custom color range
        crange = None
        if use_diverging and not input.custom_range():
            # Auto-symmetric range for difference mode
            abs_max = max(abs(float(values.min())), abs(float(values.max())))
            if abs_max > 0:
                crange = (-abs_max, abs_max)
        elif input.custom_range():
            cmin = input.color_min() if input.color_min() is not None else None
            cmax = input.color_max() if input.color_max() is not None else None
            if cmin is not None and cmax is not None:
                crange = (cmin, cmax)

        # Feature 1: pass filter_range to layer builder
        filt = input.filter_range() if input.filter_range() is not None else None
        use_log = input.log_scale() if input.log_scale() else False
        reverse = input.reverse_palette() if input.reverse_palette() else False
        lyr, vmin, vmax = build_mesh_layer(geom, values, palette_id,
                                           filter_range=filt,
                                           color_range_override=crange,
                                           log_scale=use_log,
                                           reverse_palette=reverse)

        layers = [lyr]

        if input.wireframe():
            layers.append(build_wireframe_layer(tf, geom))

        # Boundary nodes
        if input.boundary_nodes():
            bnodes = find_boundary_nodes(tf)
            layers.append(build_boundary_layer(tf, geom, bnodes))

        # Min/max location markers
        if input.show_extrema():
            extrema = find_extrema(tf, values)
            layers.extend(build_extrema_markers(extrema, geom["x_off"], geom["y_off"]))

        if input.vectors():
            vlyr = build_velocity_layer(tf, tidx, geom)
            if vlyr is not None:
                layers.append(vlyr)

        if input.contours():
            clyr = build_contour_layer_fn(tf, values, geom)
            if clyr is not None:
                layers.append(clyr)

        # Comparison variable contour overlay
        compare = input.compare_var() if input.compare_var() else ""
        if compare and compare in tf.varnames:
            compare_vals = tf.get_data_value(compare, tidx)
            clyr2 = build_contour_layer_fn(tf, compare_vals, geom, n_contours=8,
                                           layer_id="compare-contours",
                                           contour_color=[0, 0, 180])
            if clyr2 is not None:
                layers.append(clyr2)

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

        # Measurement line
        mpts = measure_points.get()
        if mpts:
            layers.extend(build_measurement_layer(mpts))

        gradient_colors = cached_gradient_colors(palette_id, reverse=reverse)
        legend = deck_legend_control(
            entries=[{
                "layer_id": "mesh",
                "label": f"{display_var}  [{vmin:.3g} - {vmax:.3g}]",
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

        # Map background
        if input.dark_bg():
            kwargs["style"] = "data:application/json;charset=utf-8,%7B%22version%22%3A8%2C%22sources%22%3A%7B%7D%2C%22layers%22%3A%5B%7B%22id%22%3A%22bg%22%2C%22type%22%3A%22background%22%2C%22paint%22%3A%7B%22background-color%22%3A%22%231a1a2e%22%7D%7D%5D%7D"

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
