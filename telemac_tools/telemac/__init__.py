"""TELEMAC file writers (.slf, .cli, .cas, .liq)."""
from telemac_tools.telemac.writer_slf import write_slf
from telemac_tools.telemac.writer_cli import write_cli
from telemac_tools.telemac.writer_cas import write_cas
from telemac_tools.telemac.writer_liq import write_liq


def write_telemac(mesh, domain, output_dir, name="project",
                  duration=3600.0, cas_overrides=None,
                  boundaries=None):
    """Write a complete set of TELEMAC input files (.slf, .cli, .cas, .liq).

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
    boundaries : list[BoundaryCondition] or None
        Boundary conditions with optional timeseries for .liq generation.
    """
    import os
    from telemac_tools.telemac.writer_cli import _find_boundary_nodes
    os.makedirs(output_dir, exist_ok=True)
    bnodes = _find_boundary_nodes(mesh.elements, mesh.nodes.shape[0])
    write_slf(mesh, os.path.join(output_dir, f"{name}.slf"),
              boundary_nodes=bnodes)
    write_cli(mesh, domain, os.path.join(output_dir, f"{name}.cli"))

    # Write .liq if boundary timeseries data available
    has_liq = False
    if boundaries:
        has_liq = write_liq(boundaries, os.path.join(output_dir, f"{name}.liq"))

    # Add .liq reference to .cas if written
    overrides = dict(cas_overrides) if cas_overrides else {}
    if has_liq:
        overrides["LIQUID BOUNDARIES FILE"] = f"{name}.liq"
    write_cas(os.path.join(output_dir, f"{name}.cas"), name=name,
              duration=duration, overrides=overrides or None)
