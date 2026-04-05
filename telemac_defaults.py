"""TELEMAC variable semantics — palette defaults, module detection, velocity resolution."""
from __future__ import annotations

# Variable name patterns -> recommended palette
# Keys are substrings matched case-insensitively against variable names
VARIABLE_PALETTE_MAP: list[tuple[str, str]] = [
    # Longer/more specific patterns first to avoid false matches
    ("BED EVOLUTION", "_diverging"),
    ("FOND EVOLUTION", "_diverging"),
    ("EVOLUTION DU FOND", "_diverging"),
    ("EVOLUTION", "_diverging"),
    ("WATER DEPTH", "Ocean"),
    ("HAUTEUR D EAU", "Ocean"),
    ("FREE SURFACE", "Ocean"),
    ("SURFACE LIBRE", "Ocean"),
    ("BOTTOM", "Plasma"),
    ("FOND", "Plasma"),
    ("TEMPERATURE", "Thermal"),
    ("HM0", "Plasma"),
    ("HAUTEUR SIGNIFICATIVE", "Plasma"),
    ("WAVE HEIGHT", "Plasma"),
    ("FRAZIL", "Ocean"),
    ("ICE", "Ocean"),
]

# Variable names that indicate bipolar data (auto-symmetrize color range)
BIPOLAR_VARIABLES: set[str] = {
    "EVOLUTION", "BED EVOLUTION", "EVOLUTION DU FOND", "FOND EVOLUTION",
}

# Module detection from variable names (first match wins)
MODULE_SIGNATURES: dict[str, list[str]] = {
    "TOMAWAC": ["HM0", "DMOY", "SPD", "TPD", "TM01", "WAVE HEIGHT"],
    "ARTEMIS": ["PHAS", "QB", "WAVE HEIGHT D"],
    "GAIA": ["EVOLUTION", "QSBLX", "QSBLY", "BEDLOAD", "TOB", "D50"],
    "KHIONE": ["FRAZIL", "SOLID ICE", "ICE COVER"],
    "WAQTEL": ["O2D", "BOD", "PHYTO", "NITRATE"],
    "TELEMAC-3D": [],
    "TELEMAC-2D": [],
}

# Velocity variable pairs by module
VELOCITY_PAIRS: list[tuple[str, str]] = [
    ("VELOCITY U", "VELOCITY V"),
    ("UX", "UY"),
    ("VX", "VY"),
    ("QSBLX", "QSBLY"),
]


def suggest_palette(varname: str) -> str | None:
    """Return suggested palette name for a variable, or None for default."""
    upper = varname.strip().upper()
    for pattern, palette in VARIABLE_PALETTE_MAP:
        if pattern.upper() in upper:
            return palette
    return None


def is_bipolar(varname: str) -> bool:
    """Check if a variable is bipolar (erosion/deposition, positive/negative)."""
    upper = varname.strip().upper()
    return any(bp.upper() in upper for bp in BIPOLAR_VARIABLES)


def detect_module_from_vars(varnames: list[str]) -> str:
    """Detect TELEMAC module from variable names."""
    joined = " ".join(v.strip().upper() for v in varnames)
    for module, signatures in MODULE_SIGNATURES.items():
        if any(sig.upper() in joined for sig in signatures):
            return module
    return "TELEMAC-2D"


def find_velocity_pair(varnames: list[str]) -> tuple[str, str] | None:
    """Find the first available velocity U/V pair from variable names.

    Returns (u_name, v_name) using the exact names from the file,
    or None if no velocity pair is found.
    """
    stripped = [v.strip() for v in varnames]
    for u_pattern, v_pattern in VELOCITY_PAIRS:
        if u_pattern in stripped and v_pattern in stripped:
            return u_pattern, v_pattern
    return None
