# server_core.py — Core reactive calculations: tel_file, mesh_geom, current_var, etc.
from __future__ import annotations

import glob
import logging
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyproj
from shiny import reactive, render, ui

from analysis import (
    get_available_derived,
    compute_derived,
    get_var_values,
    compute_mesh_quality,
    compute_element_area,
    find_boundary_nodes,
    find_cas_files,
    compute_slope,
    compute_courant_number,
    compute_difference,
    read_cli_file,
    extract_layer_2d,
)
from constants import EXAMPLES, PALETTES
from crs import crs_from_epsg, detect_crs_from_cas, guess_crs_from_coords
from geometry import build_mesh_geometry
from telemac_defaults import suggest_palette
from data_manip.extraction.telemac_file import TelemacFile
import os as _os

_logger = logging.getLogger(__name__)


def _safe_close(tf, context: str) -> None:
    """Close a TelemacFile, logging any failure without re-raising."""
    if tf is None:
        return
    try:
        tf.close()
    except Exception:
        _logger.warning("Failed to close %s TelemacFile", context, exc_info=True)


@dataclass(frozen=True)
class CrsResolution:
    """Outcome of resolving CRS from user + file inputs.

    source values:
      - "manual":   user typed a valid EPSG code in the textbox
      - "invalid":  user typed something that failed to parse; error is set
      - "disabled": auto-detect is off and no manual code was given
      - "cas":      .cas file GEOGRAPHIC SYSTEM keyword matched; detected_epsg is set
      - "coords":   coordinate heuristic matched; detected_epsg is set
      - "none":     no detection method succeeded
    """

    crs: object | None  # CRS instance (from crs.py) or None
    source: str
    error: str | None = None  # human-readable, when source=="invalid"
    detected_epsg: int | None = None  # populated for "cas" and "coords" only


def _resolve_crs_from_inputs(
    *,
    epsg_text: str,
    auto_crs_enabled: bool,
    cas_candidates: tuple[str, ...],
    mesh_xy: tuple[np.ndarray, np.ndarray] | None,
) -> CrsResolution:
    """Pure CRS-resolution decision for server_core.resolve_crs.

    Precedence: manual EPSG > auto-detect disabled > .cas files > coordinate
    heuristic > None.

    Parameters
    ----------
    epsg_text
        Raw user input from the EPSG textbox; whitespace-stripped internally.
    auto_crs_enabled
        Whether the auto-detect checkbox is enabled.
    cas_candidates
        Paths to .cas files to scan for GEOGRAPHIC SYSTEM. Empty for uploads.
    mesh_xy
        (x_array, y_array) from the loaded mesh for coordinate-heuristic
        fallback, or None to skip that path.
    """
    stripped = epsg_text.strip()
    if stripped:
        try:
            code = int(stripped)
            return CrsResolution(crs=crs_from_epsg(code), source="manual")
        except (ValueError, pyproj.exceptions.CRSError) as exc:
            return CrsResolution(
                crs=None,
                source="invalid",
                error=f"Invalid EPSG code '{stripped}': {exc}",
            )
    if not auto_crs_enabled:
        return CrsResolution(crs=None, source="disabled")
    for cas_path in cas_candidates:
        detected = detect_crs_from_cas(cas_path)
        if detected is not None:
            return CrsResolution(
                crs=detected,
                source="cas",
                detected_epsg=detected.epsg,
            )
    if mesh_xy is not None:
        x, y = mesh_xy
        detected = guess_crs_from_coords(x, y)
        if detected is not None:
            return CrsResolution(
                crs=detected,
                source="coords",
                detected_epsg=detected.epsg,
            )
    return CrsResolution(crs=None, source="none")


def _find_uploaded_by_ext(uploaded: list | None, ext: str) -> str | None:
    """Return the datapath of the first uploaded file whose name ends with ext.

    Matching is case-insensitive and uses the original filename the user
    uploaded (``item["name"]``), not the temp datapath (which Shiny sets
    to a randomised filename).
    """
    if not uploaded:
        return None
    ext_lower = ext.lower()
    for item in uploaded:
        name = (item.get("name") or "").lower()
        if name.endswith(ext_lower):
            return item.get("datapath")
    return None


def _pick_file_path(
    *,
    uploaded: list | None,
    use_upload: bool,
    library_selection: tuple[str, str] | None = None,
    lib_root: Path | None = None,
    example_key: str,
    examples: dict[str, str],
) -> str:
    """Return the .slf path to open: upload, library, or example (in priority order).

    `library_selection` and `lib_root` default to `None` so existing call sites
    can migrate one-by-one. When both are set, the library branch is consulted.

    Raises FileNotFoundError if `library_selection` names a project that is no
    longer in `lib_root` (folder deleted or renamed under the running app).
    Caller is expected to clear the selection and re-evaluate.
    """
    if uploaded and use_upload:
        slf_path = _find_uploaded_by_ext(uploaded, ".slf")
        if slf_path is not None:
            return slf_path
        return uploaded[0]["datapath"]
    if (library_selection is None) != (lib_root is None):
        raise TypeError(
            "_pick_file_path: library_selection and lib_root must both be set or both be None"
        )
    if library_selection is not None and lib_root is not None:
        from model_library import scan_library, resolve_project

        project_name, slf_name = library_selection
        # scan_library() applies the path-safety guard and is cheap relative
        # to the reactive cadence (called on selection/upload changes, not on
        # every render). Direct resolution would skip the safety guard.
        for entry in scan_library(lib_root):
            if entry.name == project_name:
                return str(resolve_project(entry, slf_name).slf)
        raise FileNotFoundError(
            f"library project {project_name!r} no longer exists in {lib_root}"
        )
    return examples.get(example_key, "")


def register_core_handlers(
    input,
    output,
    session,
    _tf_lock,
    # shared reactive values that core resets on file change
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
    # shared upload state
    use_upload,
    is_3d_mode,
    library_selection=None,  # NEW — optional until Task 10
):
    """Register core reactive calcs and return them for use by other modules.

    Returns
    -------
    dict with keys:
        tel_file, mesh_geom, current_var, current_tidx, current_values,
        mesh_quality_values, elem_area_values, boundary_nodes_cached,
        cli_data, current_crs,
        effective_values (defined by caller after core is set up)
    """

    def _run_with_lock(fn, *args):
        """Run fn(*args) while holding the TelemacFile lock."""
        with _tf_lock:
            return fn(*args)

    from model_library import library_root, find_companion

    # When library_selection is None (pre-Task-10 transitional state),
    # use a sentinel reactive value that always reads None.
    if library_selection is None:
        library_selection = reactive.value(None)

    _warned_session: set[str] = set()

    def _warn_once_session(key: str, message: str) -> None:
        if key not in _warned_session:
            _warned_session.add(key)
            print(f"[viewer] {message}", file=sys.stderr)

    # -- Upload management --

    @reactive.effect
    @reactive.event(input.upload)
    def handle_upload_change():
        if input.upload():
            use_upload.set(True)
            library_selection.set(None)
            ui.update_select("library_project", selected="")

    @output
    @render.ui
    def clear_upload_ui():
        if not use_upload.get():
            return ui.div()
        return ui.input_action_button(
            "clear_upload",
            "Clear upload",
            class_="btn-sm btn-outline-danger w-100 mb-1",
        )

    @output
    @render.ui
    def compare_upload_ui():
        if input.diff_mode():
            return ui.input_file(
                "compare_upload", "Compare file (.slf)", accept=[".slf"]
            )
        return ui.div()

    @reactive.effect
    @reactive.event(input.clear_upload)
    def handle_clear_upload():
        use_upload.set(False)

    # -- Core file reactive --

    _prev_tel_file = [None]  # plain list to avoid reactive dependency

    @reactive.calc
    def tel_file():
        try:
            path = _pick_file_path(
                uploaded=input.upload(),
                use_upload=use_upload.get(),
                library_selection=library_selection.get(),
                lib_root=library_root(),
                example_key=input.example(),
                examples=EXAMPLES,
            )
        except FileNotFoundError as exc:
            # Library project disappeared under us. Clear the selection (so future
            # ticks see no library), then retry on this same tick with library
            # disabled so we don't fall through to TelemacFile("") and show the
            # user a confusing error toast — they get the upload or example load
            # silently instead.
            _warn_once_session(f"stale-library:{exc}", str(exc))
            library_selection.set(None)
            path = _pick_file_path(
                uploaded=input.upload(),
                use_upload=use_upload.get(),
                library_selection=None,
                lib_root=None,
                example_key=input.example(),
                examples=EXAMPLES,
            )
        try:
            tf = TelemacFile(path)
        except Exception as e:
            ui.notification_show(
                f"Failed to open file: {e}",
                type="error",
                duration=8,
                id="file_open_err",
            )
            raise
        old = _prev_tel_file[0]
        _prev_tel_file[0] = tf
        if old is not None:
            with _tf_lock:
                _safe_close(old, "previous tel_file")
        return tf

    @reactive.effect
    def _reset_state_on_new_file():
        """Reset all derived state when a new file is loaded."""
        tel_file()  # take dependency on file changes
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
        _safe_close(compare_tf.get(), "compare")
        compare_tf.set(None)
        volume_cache.set(None)
        polygon_stats_data.set(None)
        polygon_geom.set(None)
        if use_upload.get():
            uploaded = input.upload() or []
            companions = (
                (".cas", "CRS detection"),
                (".cli", "boundary coloring"),
                (".liq", "hydrograph overlays"),
            )
            active = [
                f"{ext} {label}"
                for ext, label in companions
                if _find_uploaded_by_ext(uploaded, ext) is not None
            ]
            missing = [
                f"{ext} (for {label})"
                for ext, label in companions
                if _find_uploaded_by_ext(uploaded, ext) is None
            ]
            if active and missing:
                msg = (
                    "Uploaded file: " + ", ".join(active) + " active. "
                    "Add the missing companions to enable the rest: "
                    + ", ".join(missing)
                    + "."
                )
            elif active:
                msg = (
                    "Uploaded file: " + ", ".join(active) + " active — "
                    "all companion features available."
                )
            else:
                msg = (
                    "Uploaded file: companion features unavailable. "
                    "Drop .cas / .cli / .liq files alongside the .slf to "
                    "enable CRS auto-detect, boundary coloring, and "
                    "hydrograph overlays."
                )
            ui.notification_show(
                msg,
                type="message",
                duration=6,
                id="upload_notice",
            )

    # -- CRS reactive --

    current_crs = reactive.value(None)
    # Tracks how the CRS was set: "manual" / "cas" / "coords" / "none" /
    # "disabled" / "invalid". Drives the source-suffix on crs_status_ui.
    current_crs_source = reactive.value("none")

    @reactive.effect
    def resolve_crs():
        """Auto-detect or manually set CRS when file or EPSG input changes."""
        epsg_text = input.epsg_input() if input.epsg_input() else ""

        # Assemble inputs for the pure helper
        try:
            auto_crs_enabled = bool(input.auto_crs())
        except (TypeError, AttributeError, KeyError):
            auto_crs_enabled = True
        uploaded = input.upload()
        if library_selection.get() is not None:
            cas_path = find_companion(library_selection.get(), library_root(), ".cas")
            cas_candidates: tuple[str, ...] = (str(cas_path),) if cas_path else ()
        elif uploaded and use_upload.get():
            # If the user uploaded a companion .cas alongside the .slf,
            # scan it for the GEOGRAPHIC SYSTEM keyword. Previously uploads
            # always got an empty tuple, disabling .cas-based CRS detection.
            uploaded_cas = _find_uploaded_by_ext(uploaded, ".cas")
            cas_candidates = (uploaded_cas,) if uploaded_cas else ()
        else:
            slf_path = EXAMPLES.get(input.example(), "")
            cas_candidates = (
                tuple(find_cas_files(slf_path).values()) if slf_path else ()
            )
        # Consult the mesh heuristic only when it might actually be used:
        # the manual-EPSG and auto-disabled paths terminate before the
        # mesh_xy is inspected, so gate the tel_file() read to avoid
        # creating an avoidable reactive dep on every EPSG keystroke.
        mesh_xy = None
        stripped = epsg_text.strip()
        if not stripped and auto_crs_enabled:
            try:
                tf = tel_file()
                mesh_xy = (tf.meshx, tf.meshy)
            except (AttributeError, OSError, KeyError, ValueError, RuntimeError):
                # File not loaded yet / malformed / transient read error —
                # skip the coord heuristic, let the helper return source="none".
                mesh_xy = None

        outcome = _resolve_crs_from_inputs(
            epsg_text=epsg_text,
            auto_crs_enabled=auto_crs_enabled,
            cas_candidates=cas_candidates,
            mesh_xy=mesh_xy,
        )

        current_crs.set(outcome.crs)
        current_crs_source.set(outcome.source)
        if outcome.source == "invalid":
            ui.notification_show(
                outcome.error,
                type="warning",
                duration=6,
                id="epsg_warn",
            )
        if outcome.detected_epsg is not None:
            with reactive.isolate():
                ui.update_text("epsg_input", value=str(outcome.detected_epsg))

    _CRS_SOURCE_LABELS = {
        "manual": "manual EPSG",
        "cas": "auto-detected from .cas",
        "coords": "auto-detected from coordinates",
        "disabled": "manual entry only",
        "invalid": "invalid input",
        "none": "no CRS",
    }

    @output
    @render.ui
    def crs_status_ui():
        crs = current_crs.get()
        source = current_crs_source.get()
        source_label = _CRS_SOURCE_LABELS.get(source, source)
        if crs:
            return ui.span(
                f"EPSG:{crs.epsg} — {crs.name} ({source_label})",
                class_="small text-success",
            )
        return ui.span(
            f"No CRS — basemap alignment disabled ({source_label})",
            class_="small text-muted",
        )

    @output
    @render.ui
    def crs_offset_ui():
        if input.epsg_input() and input.epsg_input().strip():
            return ui.div(
                ui.input_numeric("crs_x_offset", "X offset (m)", value=0, step=1000),
                ui.input_numeric("crs_y_offset", "Y offset (m)", value=0, step=1000),
            )
        return ui.div()

    # -- Mesh geometry --

    @reactive.calc
    def mesh_geom():
        tf = tel_file()
        crs = current_crs.get()
        try:
            x_offset = float(input.crs_x_offset() or 0)
            y_offset = float(input.crs_y_offset() or 0)
        except (TypeError, AttributeError, KeyError, ValueError):
            x_offset, y_offset = 0, 0
        origin_offset = (x_offset, y_offset)
        if is_3d_mode.get() and tf.nplan > 1:
            try:
                z_scale = input.z_scale() if input.z_scale() is not None else 10
            except (TypeError, AttributeError, KeyError):
                z_scale = 10
            try:
                z_name = tf.get_z_name()
                z_vals = tf.get_data_value(z_name, current_tidx())
            except (KeyError, ValueError, AttributeError, IndexError) as e:
                ui.notification_show(
                    f"Cannot read Z elevation ({e}). Switched back to 2D view.",
                    type="warning",
                    duration=8,
                    id="z_warn",
                )
                is_3d_mode.set(False)
                return build_mesh_geometry(tf, crs=crs, origin_offset=origin_offset)
            return build_mesh_geometry(
                tf,
                crs=crs,
                z_values=z_vals,
                z_scale=z_scale,
                origin_offset=origin_offset,
            )
        return build_mesh_geometry(tf, crs=crs, origin_offset=origin_offset)

    # -- Variable and timestep calcs --

    @reactive.calc
    def current_var():
        tf = tel_file()
        try:
            var = input.variable()
        except (TypeError, AttributeError, KeyError):
            var = None
        if var and (var in tf.varnames or var in get_available_derived(tf)):
            return var
        return tf.varnames[0]

    @reactive.calc
    def current_tidx():
        tf = tel_file()
        try:
            tidx = input.time_idx()
        except (TypeError, AttributeError, KeyError):
            tidx = 0
        if tidx is None:
            tidx = 0
        return min(tidx, len(tf.times) - 1)

    @reactive.calc
    def current_values():
        tf = tel_file()
        var = current_var()
        tidx = current_tidx()
        vals = get_var_values(tf, var, tidx)
        # 3D layer extraction
        try:
            layer = input.layer_select()
            if layer and layer != "all":
                nplan = getattr(tf, "nplan", 0)
                if nplan > 1:
                    return extract_layer_2d(vals, tf.npoin2, int(layer))
        except (TypeError, AttributeError, KeyError):
            pass  # Widget not rendered yet
        except (ValueError, IndexError) as exc:
            _logger.warning("Layer extraction failed for layer %s: %s", layer, exc)
            ui.notification_show(
                f"Could not extract layer {layer} — showing full data",
                type="warning",
                duration=5,
                id="var_extract_warn",
            )
            # Return truncated 2D slice to avoid shape mismatch downstream
            return vals[: tf.npoin2] if len(vals) > tf.npoin2 else vals
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
        """Try to read .cli file: library project first, upload second,
        else from the example file's directory.
        """
        if library_selection.get() is not None:
            cli_path = find_companion(library_selection.get(), library_root(), ".cli")
            return read_cli_file(str(cli_path)) if cli_path else None
        uploaded = input.upload()
        if uploaded and use_upload.get():
            uploaded_cli = _find_uploaded_by_ext(uploaded, ".cli")
            return read_cli_file(uploaded_cli) if uploaded_cli else None
        # Example-file path lookup (companion .cli sits next to the .slf).
        path = EXAMPLES.get(input.example(), "")
        cli_files = glob.glob(_os.path.join(_os.path.dirname(path), "*.cli"))
        return read_cli_file(cli_files[0]) if cli_files else None

    # -- Dynamic UI outputs --

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
        return ui.input_select(
            "variable", "Variable", choices=choices, selected=selected
        )

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
            "ref_tidx",
            "Reference time step",
            min=0,
            max=n - 1,
            value=0,
            step=1,
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
        if np.isnan(vmin) or np.isinf(vmin) or np.isinf(vmax) or vmax <= vmin:
            vmin, vmax = 0.0, 1.0
        step = round((vmax - vmin) / 100, 4) if vmax > vmin else 0.01
        return ui.input_slider(
            "filter_range",
            "Value filter",
            min=round(vmin, 4),
            max=round(vmax, 4),
            value=[round(vmin, 4), round(vmax, 4)],
            step=step,
        )

    @output
    @render.ui
    def clear_xsec_ui():
        if cross_section_points.get() is None:
            return ui.div()
        return ui.input_action_button(
            "clear_xsec",
            "Clear Cross-Section",
            class_="btn-sm btn-outline-danger w-100 mb-1",
        )

    @output
    @render.ui
    def particle_seed_ui():
        if not input.particles():
            return ui.div()
        return ui.input_action_button(
            "draw_seed", "Draw Seed Line", class_="btn-sm btn-outline-info w-100 mb-1"
        )

    @output
    @render.ui
    def trail_length_ui():
        if not input.particles():
            return ui.div()
        tf = tel_file()
        max_trail = float(tf.times[-1] - tf.times[0]) if len(tf.times) > 1 else 1.0
        default_trail = max_trail * 0.2
        return ui.input_slider(
            "trail_length",
            "Trail length (s)",
            min=0,
            max=round(max_trail, 1),
            value=round(default_trail, 1),
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
            ui.input_select(
                "layer_select",
                "Display layer",
                choices={"all": "All planes (3D)"}
                | {
                    str(k): f"Layer {k}"
                    + (
                        " (bottom)"
                        if k == 0
                        else " (surface)"
                        if k == tf.nplan - 1
                        else ""
                    )
                    for k in range(tf.nplan)
                },
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
                f"Cannot determine depth range: {e}",
                type="warning",
                duration=5,
                id="zscale_warn",
            )
            depth_range = 0.0
        default_scale = (
            min(50, int(geom.extent_m / depth_range)) if depth_range > 0 else 10
        )
        return ui.input_slider(
            "z_scale",
            "Z Scale",
            min=1,
            max=100,
            value=default_scale,
            step=1,
        )

    # -- 3D mode sync --

    @reactive.effect
    @reactive.event(input.view_3d)
    def sync_3d_mode():
        is_3d_mode.set(input.view_3d())

    return {
        "tel_file": tel_file,
        "mesh_geom": mesh_geom,
        "current_var": current_var,
        "current_tidx": current_tidx,
        "current_values": current_values,
        "mesh_quality_values": mesh_quality_values,
        "elem_area_values": elem_area_values,
        "boundary_nodes_cached": boundary_nodes_cached,
        "cli_data": cli_data,
        "current_crs": current_crs,
        "_run_with_lock": _run_with_lock,
    }
