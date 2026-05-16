# server_import.py — Import tab server handlers
from __future__ import annotations

import asyncio
import datetime
import errno
import logging
import math
import os as _os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from shiny import reactive, render, ui
from shiny_deckgl import (
    MapWidget,
    line_layer,
    scatterplot_layer,
    path_layer,
)
from auth.middleware import get_current_user_id_from_scope
from constants import MAP_BG_LIGHT
from layers import _COORD_METER_OFFSETS
from model_library import _sanitize_for_project_name, save_imported_to_library

_logger = logging.getLogger(__name__)

# Chunk size for streaming download handlers. The previous `yield f.read()`
# pattern slurped the entire output file into memory before yielding —
# fine for kilobyte CAS/CLI files, OOM-prone for multi-GB SLF outputs from
# large HEC-RAS imports.
_DOWNLOAD_CHUNK_SIZE = 64 * 1024


def _stream_file_chunks(path: str):
    """Yield a file's contents in fixed-size chunks for @render.download."""
    with open(path, "rb") as f:
        while chunk := f.read(_DOWNLOAD_CHUNK_SIZE):
            yield chunk


# shiny_deckgl.MapWidget is a native Shiny component (not an ipywidget),
# so it is embedded via its .ui() method directly in app.py — it must NOT
# be wrapped with shinywidgets.render_widget/output_widget.
import_map_widget = MapWidget(
    "import_map",
    view_state={"longitude": 0, "latitude": 0, "zoom": 0},
    style=MAP_BG_LIGHT,
)


def register_import_handlers(input, output, session, library_version=None):
    """Register all server handlers for the Import tab.

    All state is local to this function — nothing is shared with the
    main server, because the import tab is fully self-contained.

    ``library_version`` is the app-level reactive that drives the My-Models
    library view. The HEC-RAS convert handler bumps it after a successful
    auto-save so the user's library refreshes without a manual click. If
    None (back-compat for tests that don't pass it), auto-save still runs
    but the UI bump is skipped.
    """

    import_model = reactive.value(None)  # parsed HecRasModel
    import_output_dir = reactive.value(None)  # path to generated files
    import_log_text = reactive.value("")

    # -- helpers --

    def _append_log(msg: str):
        import_log_text.set(import_log_text.get() + msg + "\n")

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

    # -- outputs --

    @output
    @render.text
    def import_log():
        return import_log_text.get()

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
        buttons = [
            ui.download_button(
                "dl_slf", "Download .slf", class_="btn-sm btn-outline-primary me-1"
            ),
            ui.download_button(
                "dl_cli", "Download .cli", class_="btn-sm btn-outline-primary me-1"
            ),
            ui.download_button(
                "dl_cas", "Download .cas", class_="btn-sm btn-outline-primary me-1"
            ),
        ]
        liq_path = _import_file_path(".liq")
        if liq_path:
            buttons.append(
                ui.download_button(
                    "dl_liq", "Download .liq", class_="btn-sm btn-outline-success"
                ),
            )
        return ui.div(*buttons, class_="d-flex gap-1")

    @render.download(filename=lambda: _import_filename(".slf"))
    def dl_slf():
        path = _import_file_path(".slf")
        if path:
            yield from _stream_file_chunks(path)

    @render.download(filename=lambda: _import_filename(".cli"))
    def dl_cli():
        path = _import_file_path(".cli")
        if path:
            yield from _stream_file_chunks(path)

    @render.download(filename=lambda: _import_filename(".cas"))
    def dl_cas():
        path = _import_file_path(".cas")
        if path:
            yield from _stream_file_chunks(path)

    @render.download(filename=lambda: _import_filename(".liq"))
    def dl_liq():
        path = _import_file_path(".liq")
        if path:
            yield from _stream_file_chunks(path)

    # -- layer builder --

    def _build_import_preview_layers(model, x_off, y_off):
        """Build deck.gl preview layers from parsed HecRasModel."""
        layers = []

        for reach in model.rivers:
            # River alignment (cyan path) — skip NaN/Inf coordinates
            path = []
            for p in reach.alignment:
                px, py = float(p[0] - x_off), float(p[1] - y_off)
                if np.isfinite(px) and np.isfinite(py):
                    path.append([px, py])
            layers.append(
                path_layer(
                    f"alignment-{reach.name}",
                    [{"path": path}],
                    getPath="@@=d.path",
                    getColor=[0, 200, 255, 200],
                    getWidth=4,
                    widthMinPixels=2,
                    widthMaxPixels=6,
                    pickable=False,
                    coordinateSystem=_COORD_METER_OFFSETS,
                    coordinateOrigin=[0, 0],
                )
            )

            # Cross-section lines (yellow)
            xs_lines = []
            for xs in reach.cross_sections:
                if xs.coords.shape[0] >= 2:
                    xs_lines.append(
                        {
                            "sourcePosition": [
                                float(xs.coords[0, 0] - x_off),
                                float(xs.coords[0, 1] - y_off),
                            ],
                            "targetPosition": [
                                float(xs.coords[-1, 0] - x_off),
                                float(xs.coords[-1, 1] - y_off),
                            ],
                        }
                    )
            if xs_lines:
                layers.append(
                    line_layer(
                        f"cross-sections-{reach.name}",
                        xs_lines,
                        getColor=[255, 220, 0, 180],
                        getWidth=2,
                        widthMinPixels=1,
                        widthMaxPixels=3,
                        pickable=False,
                        coordinateSystem=_COORD_METER_OFFSETS,
                        coordinateOrigin=[0, 0],
                    )
                )

        # BC lines
        for i, bc in enumerate(model.boundaries):
            if bc.line_coords is not None and len(bc.line_coords) >= 2:
                color = (
                    [40, 120, 255, 220]
                    if bc.location == "upstream"
                    else [0, 200, 80, 220]
                )
                layers.append(
                    line_layer(
                        f"bc-{i}",
                        [
                            {
                                "sourcePosition": [
                                    float(bc.line_coords[0, 0] - x_off),
                                    float(bc.line_coords[0, 1] - y_off),
                                ],
                                "targetPosition": [
                                    float(bc.line_coords[-1, 0] - x_off),
                                    float(bc.line_coords[-1, 1] - y_off),
                                ],
                            }
                        ],
                        getColor=color,
                        getWidth=4,
                        widthMinPixels=2,
                        widthMaxPixels=5,
                        pickable=False,
                        coordinateSystem=_COORD_METER_OFFSETS,
                        coordinateOrigin=[0, 0],
                    )
                )

        # 2D area face points (scatter, subsampled)
        for area in model.areas_2d:
            step = max(1, len(area.face_points) // 2000)
            pts = [
                {"position": [float(p[0] - x_off), float(p[1] - y_off)]}
                for p in area.face_points[::step]
            ]
            layers.append(
                scatterplot_layer(
                    f"2d-{area.name}",
                    pts,
                    getPosition="@@=d.position",
                    getColor=[0, 200, 255, 120],
                    getRadius=3,
                    radiusMinPixels=1,
                    radiusMaxPixels=4,
                    pickable=False,
                    coordinateSystem=_COORD_METER_OFFSETS,
                    coordinateOrigin=[0, 0],
                )
            )

        return layers

    # -- effects --

    @reactive.effect
    @reactive.event(input.import_preview)
    async def handle_import_preview():
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
                        _append_log(
                            f"    Station range: {min(stations):.0f} – {max(stations):.0f} m"
                        )
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
                zoom = (
                    math.log2(600 * 360 / (256 * (extent / 111320)))
                    if extent > 0
                    else 10
                )

                preview_layers = _build_import_preview_layers(model, x_off, y_off)
                await import_map_widget.update(
                    session,
                    layers=preview_layers,
                    view_state={"longitude": 0, "latitude": 0, "zoom": zoom},
                )

            _append_log(
                "\nReady to convert. Click 'Convert' to generate TELEMAC files."
            )

        except (OSError, ValueError, KeyError, RuntimeError) as e:
            _logger.exception("HEC-RAS preview failed for %s", hdf_path)
            _append_log(f"ERROR: {e}")
            import_model.set(None)

    _import_out_dir = reactive.value(None)

    @reactive.effect
    @reactive.event(input.import_convert)
    async def handle_import_convert():
        """Run full conversion pipeline, then auto-save to the user's library.

        Async so the auto-save (potentially ~30 s on multi-GB SLFs) can run
        on a thread via ``asyncio.to_thread`` while ``ui.Progress`` keeps the
        WebSocket heartbeat alive.
        """
        model = import_model.get()
        if model is None:
            _append_log("Please click Preview first to parse the HEC-RAS file.")
            return

        _append_log("\n--- Converting ---")

        hdf_files = input.import_hdf()
        if not hdf_files:
            _append_log("ERROR: No HEC-RAS file uploaded")
            return
        hdf_path = hdf_files[0]["datapath"]
        # Sanitise uploaded filename to prevent path traversal
        hdf_name = _os.path.basename(hdf_files[0]["name"]).rsplit(".", 2)[0]

        dem_files = input.import_dem()
        dem_path = dem_files[0]["datapath"] if dem_files else None

        if model.rivers and not dem_path:
            _append_log("ERROR: DEM file required for 1D→2D conversion")
            return

        # Clean up previous temp dir (hardened with Path.resolve.is_relative_to
        # against `..` / symlink escapes that a naive startswith would admit).
        old_dir = _import_out_dir.get()
        if old_dir and _os.path.isdir(old_dir):
            try:
                old_resolved = Path(old_dir).resolve(strict=True)
                tmp_resolved = Path(tempfile.gettempdir()).resolve(strict=True)
                if old_resolved.is_relative_to(tmp_resolved):
                    shutil.rmtree(old_resolved, ignore_errors=True)
                else:
                    _logger.warning(
                        "refusing to rmtree _import_out_dir=%r — resolved %r is not "
                        "under tempfile.gettempdir()=%r; clearing reactive only",
                        old_dir,
                        str(old_resolved),
                        str(tmp_resolved),
                    )
            except (OSError, ValueError) as e:
                _logger.warning(
                    "could not resolve old _import_out_dir=%r: %s", old_dir, e
                )

        out_dir = tempfile.mkdtemp(prefix="telemac_import_")
        _import_out_dir.set(out_dir)

        try:
            from telemac_tools import hecras_to_telemac

            scheme = input.import_scheme()
            cas_overrides = {}
            if scheme == "finite_element":
                cas_overrides["EQUATIONS"] = "'SAINT-VENANT FE'"
                cas_overrides["FINITE VOLUME SCHEME"] = None

            channel_refine = (
                float(input.channel_refine()) if input.channel_refine() else 10.0
            )
            fp_refine = (
                float(input.floodplain_refine()) if input.floodplain_refine() else None
            )
            channel_spacing = max(1.0, channel_refine**0.5)

            hecras_to_telemac(
                hecras_path=hdf_path,
                dem_path=dem_path,
                output_dir=out_dir,
                name=hdf_name,
                floodplain_width=float(input.fp_width()),
                channel_spacing=channel_spacing,
                floodplain_area=fp_refine,
                backend=input.import_mesher(),
                cas_overrides=cas_overrides if cas_overrides else None,
            )

            _append_log(f"Output directory: {out_dir}")
            _append_log(f"  {hdf_name}.slf — mesh + variables")
            _append_log(f"  {hdf_name}.cli — boundary conditions")
            _append_log(f"  {hdf_name}.cas — steering file")
            if _os.path.isfile(_os.path.join(out_dir, f"{hdf_name}.liq")):
                _append_log(f"  {hdf_name}.liq — liquid boundary time series")

            # Count mesh stats
            from data_manip.extraction.telemac_file import TelemacFile

            tf = TelemacFile(_os.path.join(out_dir, f"{hdf_name}.slf"))
            try:
                _append_log(f"\nMesh: {tf.npoin2} nodes, {tf.nelem2} elements")
                _append_log(f"Variables: {', '.join(tf.varnames)}")
            finally:
                tf.close()

            _append_log("\nConversion complete.")

            # --- Auto-save to user library -------------------------------
            user_id = (
                get_current_user_id_from_scope(session.http_conn.scope)
                if session.http_conn
                else None
            )
            if user_id is None:
                _logger.warning(
                    "HEC-RAS import without user_id — skipping auto-save (defensive)"
                )
                _append_log("Warning: not logged in; outputs in /tmp only.")
                import_output_dir.set(out_dir)
            else:
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                auto_name = _sanitize_for_project_name(f"hecras_{hdf_name}_{timestamp}")
                try:
                    with ui.Progress(min=0, max=1) as p:
                        p.set(
                            0.0,
                            message="Saving to your library…",
                            detail=f"{auto_name} (this may take ~30 s for large meshes)",
                        )
                        saved = await asyncio.to_thread(
                            save_imported_to_library,
                            user_id,
                            Path(out_dir),
                            auto_name,
                        )
                        p.set(1.0, message="Saved.")
                    _append_log(f"Auto-saved to library: My models → {auto_name}")
                    if library_version is not None:
                        library_version.set(library_version.get() + 1)
                    # Diverge: _import_out_dir = cleanup target (NEVER library path);
                    # import_output_dir = download source (can be library path).
                    _import_out_dir.set(None)
                    import_output_dir.set(str(saved))
                except FileExistsError:
                    _logger.warning(
                        "auto-save name collision uid=%d name=%s — keeping outputs in /tmp",
                        user_id,
                        auto_name,
                    )
                    _append_log(
                        f"WARNING: A library entry named {auto_name} already exists. "
                        "Use Download buttons to save manually."
                    )
                    import_output_dir.set(out_dir)
                    ui.notification_show(
                        "Library name collision — auto-save skipped; "
                        "outputs available via Download.",
                        type="warning",
                        duration=5,
                    )
                except OSError as e:
                    errno_name = errno.errorcode.get(getattr(e, "errno", None), "?")
                    _logger.exception(
                        "auto-save OSError uid=%d errno=%s(%s) name=%s",
                        user_id,
                        e.errno,
                        errno_name,
                        auto_name,
                    )
                    if e.errno == errno.ENOSPC:
                        msg = (
                            "Auto-save failed: disk full. "
                            "Use Download buttons to save manually."
                        )
                    elif e.errno in (errno.EACCES, errno.EPERM):
                        msg = (
                            "Auto-save failed: permission denied on library dir. "
                            "Contact admin."
                        )
                    else:
                        msg = (
                            f"Auto-save failed ({errno_name}). "
                            "Use Download buttons to save manually."
                        )
                    _append_log(f"WARNING: {msg}")
                    import_output_dir.set(out_dir)
                    ui.notification_show(msg, type="warning", duration=8)

        except (OSError, ValueError, KeyError, RuntimeError) as e:
            _logger.exception("HEC-RAS conversion failed")
            import traceback

            _append_log(f"ERROR: {e}")
            _append_log(traceback.format_exc())
            ui.notification_show(
                f"Import conversion failed: {e}", type="error", duration=8
            )
            shutil.rmtree(out_dir, ignore_errors=True)
            _import_out_dir.set(None)
            import_output_dir.set(None)
