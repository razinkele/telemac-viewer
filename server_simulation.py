# server_simulation.py — Simulation launcher server handlers
from __future__ import annotations

import asyncio
import logging
import os as _os
import shlex

from shiny import reactive, render, ui

from analysis import find_cas_files, detect_module_from_path
from constants import EXAMPLES

_logger = logging.getLogger(__name__)


def register_simulation_handlers(input, output, session, use_upload=None):
    """Register server handlers for the Run Simulation tab.

    All simulation state is local to this function.
    """

    sim_process = reactive.value(None)   # asyncio.Process or None
    sim_output = reactive.value("")      # accumulated console output
    sim_running = reactive.value(False)

    def _is_upload_mode():
        if use_upload is not None:
            return use_upload.get()
        return bool(input.upload())

    @output
    @render.ui
    def cas_select_ui():
        if _is_upload_mode():
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
        uploaded = _is_upload_mode()
        if uploaded:
            ui.notification_show("Cannot run simulation on uploaded files", type="warning", duration=3)
            return
        path = EXAMPLES.get(input.example(), "")
        cas_files = find_cas_files(path)
        try:
            cas_name = input.cas_file() if input.cas_file() else None
        except (TypeError, AttributeError, KeyError):
            cas_name = None
        if not cas_name or cas_name not in cas_files:
            ui.notification_show("No .cas file selected", type="warning", duration=3)
            return
        cas_path = cas_files[cas_name]
        module = detect_module_from_path(cas_path)
        ncores = input.ncores() if input.ncores() is not None else 4
        cas_dir = _os.path.dirname(cas_path)

        # Build command
        runner = f"{module}.py"
        cmd_display = f"{runner} {cas_name} --ncsize={ncores}"

        sim_output.set(f"$ cd {cas_dir}\n$ {cmd_display}\n\n")
        sim_running.set(True)

        try:
            # Source TELEMAC env and run (quote all paths for shell safety)
            hometel = _os.environ.get("HOMETEL", "")
            if not hometel or not _os.path.isdir(hometel):
                sim_output.set(sim_output.get() + "ERROR: HOMETEL not set or directory not found.\n")
                sim_running.set(False)
                return
            env_script = _os.path.join(hometel, "configs/pysource.local.sh")
            if not _os.path.isfile(env_script):
                sim_output.set(sim_output.get() +
                               f"ERROR: pysource.local.sh not found at {env_script}\n")
                sim_running.set(False)
                return
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
            _READLINE_TIMEOUT = 120  # seconds of inactivity before timeout
            lines_buf = [sim_output.get()]
            line_count = 0
            try:
                while True:
                    line = await asyncio.wait_for(
                        proc.stdout.readline(), timeout=_READLINE_TIMEOUT)
                    if not line:
                        break
                    lines_buf.append(line.decode())
                    line_count += 1
                    if len(lines_buf) > 500:
                        lines_buf = lines_buf[-500:]
                    if line_count % 10 == 0:
                        sim_output.set("".join(lines_buf))
                        await reactive.flush()
            except asyncio.TimeoutError:
                lines_buf.append(
                    f"\n--- No output for {_READLINE_TIMEOUT}s, killing process ---\n")
                proc.kill()
            await proc.wait()
            lines_buf.append(f"\n--- Process exited with code {proc.returncode} ---\n")
            sim_output.set("".join(lines_buf))
        except Exception as e:
            sim_output.set(sim_output.get() + f"\nERROR: {e}\n")
            # Kill process on any exception to avoid orphans
            if proc is not None and proc.returncode is None:
                proc.kill()
                await proc.wait()
        finally:
            sim_running.set(False)
            sim_process.set(None)

    @reactive.effect
    @reactive.event(input.stop_sim)
    async def handle_stop_sim():
        proc = sim_process.get()
        if proc and proc.returncode is None:
            proc.terminate()
            sim_output.set(sim_output.get() + "\n--- Terminating... ---\n")
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                sim_output.set(sim_output.get() + "--- Force killed ---\n")
            sim_output.set(sim_output.get() + "--- Terminated by user ---\n")
            sim_running.set(False)
            sim_process.set(None)
        else:
            ui.notification_show("No simulation running", type="info", duration=2)
