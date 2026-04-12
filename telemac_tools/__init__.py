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

    Produces {name}.slf, {name}.cli, {name}.cas (and {name}.liq if BC
    timeseries are available) in output_dir.

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

    # Try to extract BC timeseries from companion .u##.hdf file
    _enrich_bc_timeseries(model, hecras_path)

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
                  duration=duration, cas_overrides=cas_overrides,
                  boundaries=model.boundaries)


def _enrich_bc_timeseries(model, hecras_path: str) -> None:
    """Try to populate BC timeseries from the companion unsteady flow HDF5.

    HEC-RAS geometry files are typically .g01.hdf; the unsteady flow file
    is .p01.hdf or .u01.hdf in the same directory. We try several patterns.
    """
    import os
    import glob as _glob

    base_dir = os.path.dirname(hecras_path)
    stem = os.path.basename(hecras_path).split(".")[0]  # "project"

    # Look for unsteady flow files (.u##.hdf or .p##.hdf)
    candidates = (
        _glob.glob(os.path.join(base_dir, f"{stem}.u*.hdf")) +
        _glob.glob(os.path.join(base_dir, f"{stem}.p*.hdf"))
    )
    if not candidates:
        return

    try:
        from telemac_tools.hecras.parser_bc import parse_bc_timeseries
        parsed_bcs = parse_bc_timeseries(candidates[0])
    except (OSError, ImportError, ValueError, KeyError) as exc:
        import logging
        logging.getLogger(__name__).warning(
            "Could not parse BC timeseries from %s: %s", candidates[0], exc)
        return

    if not parsed_bcs:
        return

    # Match parsed BCs to model boundaries by location
    for model_bc in model.boundaries:
        if model_bc.timeseries is not None:
            continue
        for parsed_bc in parsed_bcs:
            if (parsed_bc.timeseries is not None and
                    parsed_bc.location.lower() == model_bc.location.lower()):
                model_bc.timeseries = parsed_bc.timeseries
                model_bc.bc_type = parsed_bc.bc_type
                break
