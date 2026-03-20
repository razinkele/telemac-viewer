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
/ GENERAL PARAMETERS
/----------------------------------------------------------------------
TIME STEP = {timestep}
NUMBER OF TIME STEPS = {nsteps}
GRAPHIC PRINTOUT PERIOD = {print_period}
LISTING PRINTOUT PERIOD = {print_period}
/
/ EQUATIONS
/----------------------------------------------------------------------
VARIABLES FOR GRAPHIC PRINTOUTS = U,V,H,S,B
/
/ INITIAL CONDITIONS
/----------------------------------------------------------------------
INITIAL CONDITIONS = 'CONSTANT DEPTH'
INITIAL DEPTH = 0.1
/
/ NUMERICAL PARAMETERS
/----------------------------------------------------------------------
SOLVER = 1
SOLVER ACCURACY = 1.E-4
MAXIMUM NUMBER OF ITERATIONS FOR SOLVER = 200
DISCRETIZATION IN SPACE = 11 ; 11
/
/ FRICTION
/----------------------------------------------------------------------
LAW OF BOTTOM FRICTION = 4
FRICTION COEFFICIENT = 0.035
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

    nsteps = int(duration / timestep)
    print_period = max(1, nsteps // 100)

    text = _TEMPLATE.format(
        title=title,
        name=name,
        timestep=timestep,
        nsteps=nsteps,
        print_period=print_period,
    )

    if overrides:
        text += "/\n/ USER OVERRIDES\n/----------------------------------------------------------------------\n"
        for key, value in overrides.items():
            text += f"{key} = {value}\n"

    with open(path, "w") as f:
        f.write(text)
