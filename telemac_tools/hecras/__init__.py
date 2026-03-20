"""HEC-RAS HDF5 parsers."""
from telemac_tools.hecras.parser_1d import parse_hecras_1d
from telemac_tools.hecras.parser_2d import parse_hecras_2d, triangulate_2d_area
from telemac_tools.model import HecRasModel, HecRasParseError
import h5py


def parse_hecras(path: str) -> HecRasModel:
    """Auto-detect 1D/2D content and parse accordingly."""
    with h5py.File(path, "r") as f:
        # Only consider 1D if Cross Sections has an "Attributes" dataset
        # (some files have the group without usable attribute data)
        has_1d = (
            "Geometry" in f
            and "Cross Sections" in f["Geometry"]
            and "Attributes" in f["Geometry/Cross Sections"]
        )
        has_2d = "Geometry" in f and "2D Flow Areas" in f["Geometry"]

    if not has_1d and not has_2d:
        raise HecRasParseError(f"No 1D or 2D geometry found in {path}")

    model = HecRasModel()
    if has_1d:
        m1d = parse_hecras_1d(path)
        model.rivers = m1d.rivers
        model.boundaries = m1d.boundaries
    if has_2d:
        m2d = parse_hecras_2d(path)
        model.areas_2d = m2d.areas_2d

    return model
