# server_playback.py — Playback and keyboard shortcut server handlers
from __future__ import annotations

import logging

import numpy as np
from shiny import reactive, ui

from analysis import get_available_derived

_logger = logging.getLogger(__name__)


def register_playback_handlers(input, output, session, playing, tel_file, current_var):
    """Register server handlers for playback controls and keyboard shortcuts.

    Parameters
    ----------
    playing:
        Shared reactive.value(bool) — True while animation is running.
    tel_file:
        Shared reactive.calc returning the current TelemacFile.
    current_var:
        Shared reactive.calc returning the currently selected variable name.
    """

    # -- Play / Pause --

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
        speed = max(0.1, min(10.0, float(speed)))
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
