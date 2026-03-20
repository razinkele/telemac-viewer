"""TELEMAC file writers (.slf, .cli, .cas)."""
from telemac_tools.telemac.writer_slf import write_slf
from telemac_tools.telemac.writer_cli import write_cli
from telemac_tools.telemac.writer_cas import write_cas


def write_telemac(mesh, domain, output_dir, name="project",
                  duration=3600.0, cas_overrides=None):
    """Write a complete set of TELEMAC input files (.slf, .cli, .cas).

    Parameters
    ----------
    mesh : Mesh2D
        Triangle mesh with elevation and Manning's n.
    domain : TelemacDomain
        Domain with BC segments.
    output_dir : str
        Directory to write files into (created if needed).
    name : str
        Base name for all files.
    duration : float
        Simulation duration in seconds.
    cas_overrides : dict or None
        Extra .cas keywords.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    write_slf(mesh, os.path.join(output_dir, f"{name}.slf"))
    write_cli(mesh, domain, os.path.join(output_dir, f"{name}.cli"))
    write_cas(os.path.join(output_dir, f"{name}.cas"), name=name,
              duration=duration, overrides=cas_overrides)
