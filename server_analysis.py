# server_analysis.py — Analysis panel, chart, stats, CSV downloads, and overlays
from __future__ import annotations

import glob
import io
import logging

import numpy as np
import plotly.graph_objects as go
from shiny import reactive, render, ui
from shinywidgets import output_widget, render_widget

from analysis import (
    nearest_node,
    time_series_at_point,
    cross_section_profile,
    compute_particle_paths,
    generate_seed_grid,
    distribute_seeds_along_line,
    export_crosssection_csv,
    find_extrema,
    find_boundary_nodes,
    find_boundary_edges,
    export_all_variables_csv,
    compute_mesh_integral,
    evaluate_expression,
    compute_discharge,
    compute_temporal_stats,
    compute_flood_envelope,
    compute_flood_arrival,
    compute_flood_duration,
    polygon_zonal_stats,
    vertical_profile_at_point,
    compute_volume_timeseries,
)
from constants import format_time
from crs import click_to_native, meters_to_wgs84
from telemac_defaults import detect_module_from_vars
from validation import (
    parse_observation_csv,
    compute_rmse,
    compute_nse,
    compute_volume_timeseries as _cv_vol,
    parse_liq_file,
)
from data_manip.extraction.telemac_file import TelemacFile
import os as _os
import asyncio

_logger = logging.getLogger(__name__)


def register_analysis_handlers(
    input, output, session, map_widget,
    # shared reactive calcs
    tel_file, mesh_geom, current_var, current_tidx, current_values, effective_values,
    # shared reactive values
    analysis_mode, clicked_points, cross_section_points, particle_paths,
    temporal_stats_cache, measure_points, measure_mode, obs_data, compare_tf,
    recording, polygon_mode, polygon_stats_data, volume_cache,
    expr_result, integral_result,
    current_crs,
    use_upload,
    is_3d_mode,
    # shared lock helper
    _run_with_lock,
):
    """Register all analysis panel, chart, stats, and CSV download handlers."""

    # -- Analysis panel UI --

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

    # -- Chart helpers --

    def _chart_timeseries(tf, var, tidx):
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

    def _chart_crosssection(tf, var, tidx):
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

    def _chart_vertprofile(tf, var, tidx):
        pts = clicked_points.get()
        if not pts:
            return go.Figure()
        fig = go.Figure()
        elev_label = "Elevation (m)"
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

    def _chart_histogram(tf, var):
        vals = effective_values()
        npoin = tf.npoin2
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=vals[:npoin], nbinsx=50, name=var))
        fig.update_layout(
            xaxis_title=var, yaxis_title="Count",
            margin=dict(l=50, r=20, t=10, b=40), height=220,
        )
        return fig

    def _chart_multivar(tf, tidx):
        pts = clicked_points.get()
        if not pts:
            return go.Figure()
        px_coord, py_coord = pts[-1]
        fig = go.Figure()
        for vname in tf.varnames:
            times, values = time_series_at_point(tf, vname, px_coord, py_coord)
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

    def _chart_rating(tf):
        xsec = cross_section_points.get()
        if xsec is None:
            return go.Figure()
        # Compute discharge and avg water level at cross-section for each timestep
        h_values = []
        q_values = []
        skipped_h = 0
        for t in range(len(tf.times)):
            result = compute_discharge(tf, t, xsec)
            if result["total_q"] is not None:
                # Average water level along the cross-section
                h_val = None
                try:
                    _, wl_vals = cross_section_profile(tf, "FREE SURFACE", t, xsec)
                    h_val = float(np.mean(wl_vals))
                except (KeyError, ValueError, IndexError):
                    try:
                        _, wd_vals = cross_section_profile(tf, "WATER DEPTH", t, xsec)
                        h_val = float(np.mean(wd_vals))
                    except (KeyError, ValueError, IndexError):
                        skipped_h += 1
                if h_val is not None:
                    h_values.append(h_val)
                    q_values.append(result["total_q"])
        if skipped_h > 0:
            _logger.warning("Rating curve: %d timesteps skipped (no water level data)", skipped_h)
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

    def _chart_volume():
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

    def _chart_boundary_ts():
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

    _CHART_DISPATCH = {
        "timeseries":  lambda tf, var, tidx: _chart_timeseries(tf, var, tidx),
        "crosssection": lambda tf, var, tidx: _chart_crosssection(tf, var, tidx),
        "vertprofile": lambda tf, var, tidx: _chart_vertprofile(tf, var, tidx),
        "histogram":   lambda tf, var, tidx: _chart_histogram(tf, var),
        "multivar":    lambda tf, var, tidx: _chart_multivar(tf, tidx),
        "rating":      lambda tf, var, tidx: _chart_rating(tf),
        "volume":      lambda tf, var, tidx: _chart_volume(),
        "boundary_ts": lambda tf, var, tidx: _chart_boundary_ts(),
    }

    @output
    @render_widget
    def analysis_chart():
        mode = analysis_mode.get()
        tf = tel_file()
        var = current_var()
        tidx = current_tidx()

        handler = _CHART_DISPATCH.get(mode)
        if handler is None:
            return go.Figure()
        return handler(tf, var, tidx)

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
                stats = polygon_zonal_stats(tel_file(), values, poly_m)
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
                x_off, y_off = geom.x_off, geom.y_off
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
            from telemac_defaults import find_velocity_pair
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
            try:
                seeds = generate_seed_grid(tf, n_target=500)
                x_off, y_off = geom.x_off, geom.y_off
                loop = asyncio.get_running_loop()
                paths = await loop.run_in_executor(
                    None, _run_with_lock, compute_particle_paths, tf, seeds, x_off, y_off)
                particle_paths.set(paths)
                ui.notification_show(f"Computed {len(paths)} particle paths", duration=3)
            except Exception as e:
                _logger.warning("Particle computation failed: %s", e)
                ui.notification_show(f"Particle tracing failed: {e}", type="warning", duration=6)
            finally:
                ui.notification_remove("particle_notif")
        else:
            particle_paths.set(None)

    @reactive.effect
    @reactive.event(input.draw_seed)
    async def start_seed_drawing():
        await map_widget.enable_draw(session, modes=["draw_line_string"])

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
        from constants import EXAMPLES
        uploaded = input.upload()
        if uploaded and use_upload.get():
            return None
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
            # Close previous comparison file to avoid FD leak
            old_tf = compare_tf.get()
            if old_tf is not None:
                try:
                    old_tf.close()
                except Exception:
                    _logger.warning("Failed to close previous compare file", exc_info=True)
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
        module = "TELEMAC-3D" if nplan > 1 else detect_module_from_vars(tf.varnames)
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
