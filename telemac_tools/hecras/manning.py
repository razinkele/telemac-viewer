"""Manning's n extraction from HEC-RAS HDF5."""
from __future__ import annotations
from telemac_tools.hecras.parser_1d import parse_hecras_1d


def extract_mannings_1d(path: str) -> list[list[float]]:
    """Extract Manning's n values from all cross-sections.
    Returns list of [left, channel, right] per cross-section.
    """
    model = parse_hecras_1d(path)
    result = []
    for reach in model.rivers:
        for xs in reach.cross_sections:
            result.append(xs.mannings_n)
    return result
