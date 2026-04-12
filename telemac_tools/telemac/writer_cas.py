"""Write a TELEMAC-2D steering (.cas) file."""
from __future__ import annotations


_TEMPLATE = """\
/----------------------------------------------------------------------
/ TELEMAC-2D STEERING FILE
/----------------------------------------------------------------------
/
TITLE = '{title}'
/
/ FILES
/----------------------------------------------------------------------
GEOMETRY FILE = {name}.slf
BOUNDARY CONDITIONS FILE = {name}.cli
RESULTS FILE = {name}_res.slf
/
/ EQUATIONS AND NUMERICAL SCHEME
/----------------------------------------------------------------------
EQUATIONS = 'SAINT-VENANT FV'
FINITE VOLUME SCHEME = 5
VARIABLE TIME-STEP = YES
DESIRED COURANT NUMBER = 0.8
/
/ FRICTION
/----------------------------------------------------------------------
LAW OF BOTTOM FRICTION = 4
/
/ INITIAL CONDITIONS
/----------------------------------------------------------------------
INITIAL CONDITIONS = 'CONSTANT DEPTH'
INITIAL DEPTH = 0.1
/
/ TIDAL FLATS
/----------------------------------------------------------------------
TIDAL FLATS = YES
CONTINUITY CORRECTION = YES
TREATMENT OF NEGATIVE DEPTHS = 2
/
/ OUTPUT
/----------------------------------------------------------------------
DURATION = {duration}
GRAPHIC PRINTOUT PERIOD = 60
VARIABLES FOR GRAPHIC PRINTOUTS = 'U,V,H,S,B'
"""


def write_cas(
    path: str,
    *,
    name: str = "project",
    title: str | None = None,
    duration: float = 3600.0,
    timestep: float = 1.0,
    overrides: dict[str, object] | None = None,
) -> None:
    """Write a TELEMAC-2D .cas steering file.

    Parameters
    ----------
    path : str
        Output file path.
    name : str
        Base name for geometry/boundary/results files.
    title : str
        Simulation title (defaults to *name*).
    duration : float
        Total simulation time in seconds.
    timestep : float
        Time step in seconds.
    overrides : dict
        Extra keywords to append (or override template values).
    """
    if title is None:
        title = name

    text = _TEMPLATE.format(
        title=title,
        name=name,
        duration=duration,
    )

    if overrides:
        text += "/\n/ USER OVERRIDES\n/----------------------------------------------------------------------\n"
        for key, value in overrides.items():
            if value is None:
                continue
            if isinstance(value, str) and not value.startswith("'"):
                text += f"{key} = '{value}'\n"
            else:
                text += f"{key} = {value}\n"

    with open(path, "w") as f:
        f.write(text)
