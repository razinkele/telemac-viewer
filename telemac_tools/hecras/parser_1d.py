"""Parse 1D HEC-RAS geometry from HDF5 (.g01.hdf) files."""
from __future__ import annotations

import h5py
import numpy as np

from telemac_tools.model import (
    BCType,
    BoundaryCondition,
    CrossSection,
    HecRasModel,
    HecRasParseError,
    Reach,
)


def _interp_stations_to_world(
    station_vals: np.ndarray,
    elevations: np.ndarray,
    polyline: np.ndarray,
) -> np.ndarray:
    """Interpolate station-elevation points onto a world-coordinate polyline.

    Parameters
    ----------
    station_vals : (N,) local station distances along the cross-section
    elevations : (N,) elevation at each station point
    polyline : (M, 2) world x,y coordinates of the cross-section line

    Returns
    -------
    coords : (N, 3) world x, y, z
    """
    # Cumulative arc-length along the polyline
    diffs = np.diff(polyline, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cum_length = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = cum_length[-1]

    if total_length == 0:
        # Degenerate polyline — all points at same location
        x = np.full(len(station_vals), polyline[0, 0])
        y = np.full(len(station_vals), polyline[0, 1])
        return np.column_stack([x, y, elevations])

    # Normalize station values to [0, total_length] range
    sta_min = station_vals[0]
    sta_max = station_vals[-1]
    sta_range = sta_max - sta_min
    if sta_range == 0:
        fracs = np.zeros(len(station_vals))
    else:
        fracs = (station_vals - sta_min) / sta_range

    distances = fracs * total_length

    # Interpolate x and y separately along the polyline
    x_interp = np.interp(distances, cum_length, polyline[:, 0])
    y_interp = np.interp(distances, cum_length, polyline[:, 1])

    return np.column_stack([x_interp, y_interp, elevations])


def _decode(val: bytes | str) -> str:
    """Decode byte string if needed and strip whitespace."""
    if isinstance(val, bytes):
        return val.decode().strip()
    return str(val).strip()


def parse_hecras_1d(path: str) -> HecRasModel:
    """Parse 1D geometry from a HEC-RAS HDF5 file.

    Parameters
    ----------
    path : str
        Path to the .g01.hdf file.

    Returns
    -------
    HecRasModel with rivers and boundaries populated.

    Raises
    ------
    HecRasParseError
        If the file lacks a "Geometry" group.
    """
    with h5py.File(path, "r") as f:
        if "Geometry" not in f:
            raise HecRasParseError(
                f"No 'Geometry' group found in {path}"
            )
        geo = f["Geometry"]

        # --- River Centerlines ---
        reaches: dict[str, Reach] = {}
        if "River Centerlines" in geo:
            rc = geo["River Centerlines"]
            rc_attrs = rc["Attributes"][:]
            rc_poly_info = rc["Polyline Info"][:]
            rc_poly_pts = rc["Polyline Points"][:]

            # Detect field names: real files use "River Name"/"Reach Name",
            # synthetic fixtures use "River"/"Reach"
            field_names = rc_attrs.dtype.names
            river_field = "River Name" if "River Name" in field_names else "River"
            reach_field = "Reach Name" if "Reach Name" in field_names else "Reach"

            for i, row in enumerate(rc_attrs):
                river = _decode(row[river_field])
                reach = _decode(row[reach_field])
                name = f"{river}/{reach}"
                offset, count = int(rc_poly_info[i, 0]), int(rc_poly_info[i, 1])
                alignment = rc_poly_pts[offset : offset + count].copy()
                reaches[name] = Reach(name=name, alignment=alignment)

        # --- Cross Sections ---
        if "Cross Sections" in geo and "Attributes" in geo["Cross Sections"]:
            xs_grp = geo["Cross Sections"]
            xs_attrs = xs_grp["Attributes"][:]
            se_info = xs_grp["Station Elevation Info"][:]
            se_values = xs_grp["Station Elevation Values"][:]
            poly_info = xs_grp["Polyline Info"][:]
            poly_pts = xs_grp["Polyline Points"][:]

            # Bank stations (optional — missing in some real files)
            has_banks = "Bank Stations" in xs_grp
            if has_banks:
                bank_stations_ds = xs_grp["Bank Stations"][:]

            # Manning's n (optional)
            has_manning = "Manning's n Info" in xs_grp
            if has_manning:
                mann_info = xs_grp["Manning's n Info"][:]
                mann_values = xs_grp["Manning's n Values"][:]

            # Detect field names: real files may use "RS" instead of
            # "Station", and field names vary across versions
            attr_fields = xs_attrs.dtype.names
            river_field = "River Name" if "River Name" in attr_fields else "River"
            reach_field = "Reach Name" if "Reach Name" in attr_fields else "Reach"
            # Station field: try "RS" (real files), then "Station" (synthetic)
            if "RS" in attr_fields:
                station_field = "RS"
            elif "Station" in attr_fields:
                station_field = "Station"
            else:
                station_field = None

            for i, row in enumerate(xs_attrs):
                river = _decode(row[river_field])
                reach = _decode(row[reach_field])

                # Parse station value
                if station_field is not None:
                    raw_station = row[station_field]
                    station = float(_decode(raw_station))
                else:
                    station = float(i)

                name = f"{river}/{reach}"

                # Station-elevation
                se_off, se_cnt = int(se_info[i, 0]), int(se_info[i, 1])
                se_data = se_values[se_off : se_off + se_cnt]
                sta_vals = se_data[:, 0]
                elev_vals = se_data[:, 1]

                # Polyline
                p_off, p_cnt = int(poly_info[i, 0]), int(poly_info[i, 1])
                polyline = poly_pts[p_off : p_off + p_cnt]

                # Interpolate to world coordinates
                coords = _interp_stations_to_world(sta_vals, elev_vals, polyline)

                # Manning's n
                if has_manning:
                    m_off, m_cnt = int(mann_info[i, 0]), int(mann_info[i, 1])
                    mannings_n = mann_values[m_off : m_off + m_cnt].tolist()
                else:
                    mannings_n = [0.035, 0.035, 0.035]

                # Bank stations (use Left Bank / Right Bank from attrs
                # if dedicated dataset is missing)
                if has_banks:
                    banks = bank_stations_ds[i]
                    bank_st = (float(banks[0]), float(banks[1]))
                elif "Left Bank" in attr_fields and "Right Bank" in attr_fields:
                    bank_st = (float(row["Left Bank"]), float(row["Right Bank"]))
                else:
                    # Fallback: first and last station values
                    bank_st = (float(sta_vals[0]), float(sta_vals[-1]))

                # Bank world coordinates: interpolate bank station positions
                bank_fracs = []
                sta_min, sta_max = sta_vals[0], sta_vals[-1]
                sta_range = sta_max - sta_min
                diffs = np.diff(polyline, axis=0)
                seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
                cum_length = np.concatenate([[0.0], np.cumsum(seg_lengths)])
                total_length = cum_length[-1]

                for bs in bank_st:
                    if sta_range > 0:
                        frac = (bs - sta_min) / sta_range
                    else:
                        frac = 0.0
                    dist = frac * total_length
                    bx = np.interp(dist, cum_length, polyline[:, 0])
                    by = np.interp(dist, cum_length, polyline[:, 1])
                    bank_fracs.append([bx, by])
                bank_coords = np.array(bank_fracs)

                xs = CrossSection(
                    station=station,
                    coords=coords,
                    mannings_n=mannings_n,
                    bank_stations=bank_st,
                    bank_coords=bank_coords,
                )

                if name in reaches:
                    reaches[name].cross_sections.append(xs)
                else:
                    # Create reach without alignment if centerline missing
                    reaches[name] = Reach(
                        name=name,
                        alignment=np.empty((0, 2)),
                        cross_sections=[xs],
                    )

        # --- Boundary Condition Lines ---
        boundaries: list[BoundaryCondition] = []
        if "Boundary Condition Lines" in geo:
            bc_grp = geo["Boundary Condition Lines"]
            bc_attrs = bc_grp["Attributes"][:]
            bc_poly_info = bc_grp["Polyline Info"][:]
            bc_poly_pts = bc_grp["Polyline Points"][:]

            for i, row in enumerate(bc_attrs):
                location = _decode(row["Name"])
                decoded_type = _decode(row["Type"])
                try:
                    bc_type = BCType(decoded_type)
                except ValueError:
                    bc_type = BCType.UNKNOWN
                p_off, p_cnt = int(bc_poly_info[i, 0]), int(bc_poly_info[i, 1])
                line_coords = bc_poly_pts[p_off : p_off + p_cnt].copy()

                boundaries.append(
                    BoundaryCondition(
                        bc_type=bc_type,
                        location=location,
                        line_coords=line_coords,
                    )
                )

    return HecRasModel(
        rivers=list(reaches.values()),
        boundaries=boundaries,
    )
