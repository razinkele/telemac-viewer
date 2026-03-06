# constants.py
import math
import os
import functools
import numpy as np

os.environ.setdefault("HOMETEL", "/home/razinka/telemac/telemac-v8p5r1")
os.environ.setdefault(
    "SYSTELCFG",
    os.path.join(os.environ["HOMETEL"], "configs/systel.local.cfg"),
)
os.environ.setdefault("USETELCFG", "gfortran.intelmpi")

import sys
sys.path.insert(0, os.path.join(os.environ["HOMETEL"], "scripts/python3"))

from shiny_deckgl import (
    color_range,
    PALETTE_VIRIDIS,
    PALETTE_PLASMA,
    PALETTE_OCEAN,
    PALETTE_THERMAL,
    PALETTE_CHLOROPHYLL,
)

_E = os.environ["HOMETEL"]

EXAMPLES = {}

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

for _group, _items in EXAMPLE_GROUPS.items():
    for _name, _path in _items.items():
        if os.path.isfile(_path):
            EXAMPLES[_name] = _path

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

_M2D = 111320.0  # meters per degree at equator


@functools.lru_cache(maxsize=8)
def cached_palette_arr(palette_id):
    palette = PALETTES[palette_id]
    return np.array(color_range(256, palette), dtype=np.uint8)


@functools.lru_cache(maxsize=8)
def cached_gradient_colors(palette_id):
    palette = PALETTES[palette_id]
    full = color_range(256, palette)
    return [full[i] for i in range(0, 256, 32)]


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes} min {secs:.0f} s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours} h {mins} min"
