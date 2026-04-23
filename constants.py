# constants.py
from __future__ import annotations
import os
import functools
import logging
import numpy as np

_HOMETEL = os.environ.get("HOMETEL", "/home/razinka/telemac/telemac-v8p5r1")
os.environ.setdefault("HOMETEL", _HOMETEL)
os.environ.setdefault(
    "SYSTELCFG",
    os.path.join(_HOMETEL, "configs/systel.local.cfg"),
)
os.environ.setdefault("USETELCFG", "gfortran.intelmpi")

import sys

sys.path.insert(0, os.path.join(_HOMETEL, "scripts/python3"))

from urllib.parse import quote

from shiny_deckgl import (
    color_range,
    PALETTE_VIRIDIS,
    PALETTE_PLASMA,
    PALETTE_OCEAN,
    PALETTE_THERMAL,
    PALETTE_CHLOROPHYLL,
    PALETTE_BLUES,
    PALETTE_GREENS,
    PALETTE_REDS,
    PALETTE_YELLOW_RED,
    PALETTE_BLUE_WHITE,
)


def solid_bg_style(hex_color: str) -> str:
    """Return a MapLibre data-URI style that renders a flat ``hex_color`` background.

    Used instead of a tiled basemap when we only want the deck.gl layers
    to sit on a solid backdrop (the TELEMAC app's default look).
    """
    spec = (
        '{"version":8,"sources":{},'
        '"layers":[{"id":"bg","type":"background",'
        f'"paint":{{"background-color":"{hex_color}"}}}}]}}'
    )
    return "data:application/json;charset=utf-8," + quote(spec)


# Match the app's theme (see --ocean-bg / --coastal-bg CSS vars below).
MAP_BG_DARK = solid_bg_style("#0f1923")
MAP_BG_LIGHT = solid_bg_style("#f0f4f8")

# Full basemap set exposed as a selector in the sidebar.
BASEMAP_STYLES = {
    "dark": MAP_BG_DARK,
    "light": MAP_BG_LIGHT,
    "osm": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    "satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}",
}

MAP_TOOLTIP = {
    "html": "<b>{layerType}</b><br/>{info}",
    "style": {
        "backgroundColor": "rgba(15, 25, 35, 0.92)",
        "color": "#c8dce8",
        "fontSize": "12px",
        "borderRadius": "6px",
        "border": "1px solid rgba(13, 115, 119, 0.5)",
        "padding": "6px 10px",
        "boxShadow": "0 2px 8px rgba(0,0,0,0.3)",
    },
}

_E = os.environ["HOMETEL"]

EXAMPLES = {}

EXAMPLE_GROUPS = {
    "TELEMAC-2D": {
        "Gouttedo (raindrop)": os.path.join(
            _E, "examples/telemac2d/gouttedo/r2d_gouttedo_v1p0.slf"
        ),
        "Ritter (dam break)": os.path.join(
            _E, "examples/telemac2d/dambreak/r2d_ritter-hllc.slf"
        ),
        "Malpasset (dam break)": os.path.join(
            _E, "examples/telemac2d/malpasset/r2d_malpasset-hllc.slf"
        ),
        "Bump (critical flow)": os.path.join(
            _E, "examples/telemac2d/bump/r2d_bump.slf"
        ),
        "Vasque (basin flow)": os.path.join(
            _E, "examples/telemac2d/vasque/f2d_vasque.slf"
        ),
        "Cavity (recirculation)": os.path.join(
            _E, "examples/telemac2d/cavity/f2d_cavity.slf"
        ),
        "Estimation (parameter)": os.path.join(
            _E, "examples/telemac2d/estimation/f2d_estimation.slf"
        ),
        "Pluie (rainfall-runoff)": os.path.join(
            _E, "examples/telemac2d/pluie/f2d_rain_CN.slf"
        ),
        "Confluence (river junction)": os.path.join(
            _E, "examples/telemac2d/confluence/f2d_confluence.slf"
        ),
        "Culm (river)": os.path.join(_E, "examples/telemac2d/culm/ini_culm.slf"),
        "Breach (dam failure)": os.path.join(
            _E, "examples/telemac2d/breach/r2d_breach.slf"
        ),
    },
    "TELEMAC-3D": {
        "3D Canal": os.path.join(_E, "examples/telemac3d/canal/r3d_canal-t3d.slf"),
        "3D Pluie (rainfall)": os.path.join(
            _E, "examples/telemac3d/pluie/f3d_pluie.slf"
        ),
        "3D V-shape": os.path.join(_E, "examples/telemac3d/V/f3d_V.slf"),
    },
    "GAIA (sediment)": {
        "Guenter (bedload)": os.path.join(
            _E, "examples/gaia/guenter-t2d/f2d_guenter.slf"
        ),
        "Yen (multi-grain)": os.path.join(_E, "examples/gaia/yen-t2d/f2d_multi1.slf"),
        "Sliding (slope)": os.path.join(
            _E, "examples/gaia/sliding-t2d/f2d_slide1_slope1.slf"
        ),
        "Bosse (bedform)": os.path.join(
            _E, "examples/gaia/bosse-t2d/f2d_bosse-t2d_fe.slf"
        ),
    },
    "TOMAWAC (waves)": {
        "Wave-current interaction": os.path.join(
            _E, "examples/tomawac/opposing_current/fom_opposing_cur.slf"
        ),
        "Coupled wind-waves": os.path.join(
            _E, "examples/tomawac/Coupling_Wind/fom_different.slf"
        ),
        "3D wave coupling": os.path.join(
            _E, "examples/tomawac/3Dcoupling/fom_littoral_diff.slf"
        ),
    },
    "ARTEMIS (coastal)": {
        "Beach waves": os.path.join(_E, "examples/artemis/beach/tom_plage.slf"),
        "Wave breaking (BJ78)": os.path.join(
            _E, "examples/artemis/bj78/famp_bj78_20per.slf"
        ),
        "Westcoast (waves)": os.path.join(
            _E, "examples/artemis/westcoast/tom_westcoast.slf"
        ),
    },
    "KHIONE (ice)": {
        "Frazil ice flume": os.path.join(
            _E, "examples/khione/flume_frazil-t2d/ini_longflume.slf"
        ),
        "Ice clogging": os.path.join(
            _E, "examples/khione/clogging-t2d/ini_slowflume.slf"
        ),
    },
    "Case Studies": {
        "Curonian Lagoon (24h)": "/home/razinka/telemac/Curonian/case/r2d_curonian.slf",
        "Curonian Lagoon (geometry)": "/home/razinka/telemac/Curonian/case/curonian_geo.slf",
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
    "Blues": PALETTE_BLUES,
    "Greens": PALETTE_GREENS,
    "Reds": PALETTE_REDS,
    "Yellow-Red": PALETTE_YELLOW_RED,
    "Blue-White": PALETTE_BLUE_WHITE,
}

# Diverging palette for difference mode (blue → white → red)
PALETTE_DIVERGING = [
    [49, 54, 149, 255],
    [69, 117, 180, 255],
    [116, 173, 209, 255],
    [171, 217, 233, 255],
    [224, 243, 248, 255],
    [255, 255, 255, 255],
    [254, 224, 144, 255],
    [253, 174, 97, 255],
    [244, 109, 67, 255],
    [215, 48, 39, 255],
    [165, 0, 38, 255],
]

# Meters per degree of longitude at the equator. Used to compute map zoom
# level from mesh extent. TELEMAC meshes use metric CRS, not geographic —
# this constant is only for the deck.gl zoom calculation in geometry.py.
_M2D = 111320.0


@functools.lru_cache(maxsize=16)
def cached_palette_arr(palette_id: str, reverse: bool = False) -> np.ndarray:
    if palette_id == "_diverging":
        arr = np.array(color_range(256, PALETTE_DIVERGING), dtype=np.uint8)
    else:
        palette = PALETTES.get(palette_id)
        if palette is None:
            _logger = logging.getLogger(__name__)
            _logger.warning(
                "Unknown palette '%s', falling back to 'Viridis'", palette_id
            )
            palette = PALETTES["Viridis"]
        arr = np.array(color_range(256, palette), dtype=np.uint8)
    if reverse:
        arr = arr[::-1].copy()
    return arr


@functools.lru_cache(maxsize=16)
def cached_gradient_colors(palette_id: str, reverse: bool = False) -> list[list[int]]:
    if palette_id == "_diverging":
        full = color_range(256, PALETTE_DIVERGING)
    else:
        palette = PALETTES[palette_id]
        full = color_range(256, palette)
    colors = [full[i] for i in range(0, 256, 32)]
    if reverse:
        colors = colors[::-1]
    return colors


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes} min {secs:.0f} s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours} h {mins} min"
