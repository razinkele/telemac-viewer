"""HEC-RAS → TELEMAC import tools."""
from __future__ import annotations


def hecras_to_telemac(
    hecras_path: str,
    dem_path: str | None = None,
    output_dir: str = ".",
    name: str = "project",
    floodplain_width: float = 500.0,
    channel_spacing: float = 10.0,
    floodplain_area: float | None = None,
    backend: str = "triangle",
    duration: float = 3600.0,
    cas_overrides: dict | None = None,
) -> None:
    """Convert HEC-RAS geometry to TELEMAC simulation files.

    Produces {name}.slf, {name}.cli, {name}.cas in output_dir.

    Parameters
    ----------
    channel_spacing : point spacing along channel constraint lines (meters).
    floodplain_area : max triangle area in floodplain (m²). None = auto.
    """
    from telemac_tools.hecras import parse_hecras
    from telemac_tools.domain import build_domain_1d, build_domain_2d
    from telemac_tools.domain.builder import assign_bc_nodes
    from telemac_tools.meshing import generate_mesh
    from telemac_tools.telemac import write_telemac
    from telemac_tools.hecras.parser_2d import triangulate_2d_area

    model = parse_hecras(hecras_path)

    if model.areas_2d:
        mesh = triangulate_2d_area(model.areas_2d[0])
        domain = build_domain_2d(model)
    elif model.rivers:
        if dem_path is None:
            raise ValueError("DEM path required for 1D→2D conversion")
        domain = build_domain_1d(model, dem_path, floodplain_width=floodplain_width,
                                channel_spacing=channel_spacing)
        mesh = generate_mesh(domain, backend=backend, max_area=floodplain_area)
        # Extract boundary nodes for BC assignment
        from telemac_tools.telemac.writer_cli import _find_boundary_nodes
        import numpy as np
        bnd_indices = np.array(
            _find_boundary_nodes(mesh.elements, mesh.nodes.shape[0])
        )
        bnd_coords = mesh.nodes[bnd_indices]
        assign_bc_nodes(bnd_indices, bnd_coords, domain)
    else:
        raise ValueError("No 1D or 2D geometry found in HEC-RAS file")

    write_telemac(mesh, domain, output_dir, name=name,
                  duration=duration, cas_overrides=cas_overrides)
