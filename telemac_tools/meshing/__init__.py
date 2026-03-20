"""Mesh generation backends."""
from telemac_tools.model import TelemacDomain, Mesh2D


def generate_mesh(domain: TelemacDomain, backend: str = "triangle",
                  min_angle: float = 30.0, max_area: float | None = None) -> Mesh2D:
    """Generate a triangular mesh for the given TELEMAC domain.

    Parameters
    ----------
    domain : TelemacDomain
        Domain definition including boundary polygon and optional channel constraints.
    backend : str
        Meshing backend to use (currently only ``"triangle"``).
    min_angle : float
        Minimum angle constraint in degrees (default 30).
    max_area : float | None
        Maximum triangle area.  If *None*, a default is computed from the domain extent.

    Returns
    -------
    Mesh2D
    """
    if backend == "triangle":
        from telemac_tools.meshing.triangle_mesh import TriangleBackend
        return TriangleBackend().generate(domain, min_angle, max_area)
    elif backend == "gmsh":
        from telemac_tools.meshing.gmsh_mesh import GmshBackend
        return GmshBackend().generate(domain, min_angle, max_area)
    else:
        raise ValueError(f"Unknown backend: {backend}")
