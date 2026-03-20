"""Parse BC time series from HEC-RAS unsteady flow files (.u##.hdf)."""
from __future__ import annotations
import h5py
import numpy as np
from telemac_tools.model import BoundaryCondition


def parse_bc_timeseries(path: str) -> list[BoundaryCondition]:
    """Parse boundary condition time series from HEC-RAS unsteady flow HDF5.

    Args:
        path: Path to .u##.hdf file.

    Returns:
        List of BoundaryCondition with timeseries populated.
    """
    bcs = []
    with h5py.File(path, "r") as f:
        bc_grp = f.get("Unsteady Flow Data/Boundary Conditions")
        if bc_grp is None:
            return []

        for name in bc_grp:
            if not isinstance(bc_grp[name], h5py.Group):
                continue
            grp = bc_grp[name]
            ts = None
            bc_type = "unknown"

            if "Flow Hydrograph" in grp:
                fh = grp["Flow Hydrograph"]
                if "Flow" in fh and "Time" in fh:
                    ts = {
                        "time": fh["Time"][:] * 3600,  # hours -> seconds
                        "values": fh["Flow"][:],
                        "unit": "m3/s",
                    }
                    bc_type = "flow"
            elif "Stage Hydrograph" in grp:
                sh = grp["Stage Hydrograph"]
                if "Stage" in sh and "Time" in sh:
                    ts = {
                        "time": sh["Time"][:] * 3600,
                        "values": sh["Stage"][:],
                        "unit": "m",
                    }
                    bc_type = "stage"

            location = "upstream" if "upstream" in name.lower() else "downstream"
            bcs.append(BoundaryCondition(
                bc_type=bc_type,
                location=location,
                timeseries=ts,
            ))

    return bcs
