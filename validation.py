# validation.py — Observation data I/O and validation metrics
"""Functions for parsing observation CSV files and computing validation statistics."""

import csv
import logging
import numpy as np
from numpy import ndarray


def parse_observation_csv(file_path: str) -> tuple[ndarray, ndarray, str]:
    """Read a 2-column CSV file with time (seconds) and observed values.

    The first row must be a header. The second column name becomes the
    returned ``varname``.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    times : ndarray
        1-D array of time values (seconds).
    values : ndarray
        1-D array of observed values.
    varname : str
        Name taken from the second column header.
    """
    times = []
    values = []
    varname = ""
    with open(file_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if len(header) < 2:
            raise ValueError("CSV must have at least 2 columns (time, value)")
        varname = header[1].strip()
        for row in reader:
            if len(row) < 2:
                continue
            times.append(float(row[0]))
            values.append(float(row[1]))
    if not times:
        raise ValueError("CSV contains no data rows")
    return np.array(times, dtype=np.float64), np.array(values, dtype=np.float64), varname


def compute_rmse(model: ndarray, observed: ndarray) -> float:
    """Root Mean Square Error between model and observed arrays.

    Parameters
    ----------
    model : ndarray
        Model predictions (same length as *observed*).
    observed : ndarray
        Observed values.

    Returns
    -------
    float
        RMSE value.
    """
    model = np.asarray(model, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    if model.shape != observed.shape:
        raise ValueError(f"Array shapes must match: {model.shape} vs {observed.shape}")
    return float(np.sqrt(np.mean((model - observed) ** 2)))


def compute_nse(model: ndarray, observed: ndarray) -> float:
    """Nash-Sutcliffe Efficiency.

    NSE = 1 - sum((model - observed)^2) / sum((observed - mean(observed))^2)

    A value of 1.0 indicates a perfect match.

    Parameters
    ----------
    model : ndarray
        Model predictions (same length as *observed*).
    observed : ndarray
        Observed values.

    Returns
    -------
    float
        NSE value.  Can be negative for very poor models.
    """
    model = np.asarray(model, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    if model.shape != observed.shape:
        raise ValueError(f"Array shapes must match: {model.shape} vs {observed.shape}")
    obs_mean = np.mean(observed)
    ss_res = np.sum((model - observed) ** 2)
    ss_tot = np.sum((observed - obs_mean) ** 2)
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else float("-inf")
    return float(1.0 - ss_res / ss_tot)


def compute_volume_timeseries(tf, compute_integral_fn):
    """Compute total water volume at each timestep.

    Uses compute_integral_fn(tf, values, threshold) for area-weighted integration.
    Returns (times: ndarray, volumes: ndarray).
    """
    npoin = tf.npoin2
    varnames = [v.strip() for v in tf.varnames]
    depth_var = "WATER DEPTH" if "WATER DEPTH" in varnames else tf.varnames[0]
    times = []
    volumes = []
    for t in range(len(tf.times)):
        vals = tf.get_data_value(depth_var, t)[:npoin]
        result = compute_integral_fn(tf, vals, threshold=0.001)
        times.append(float(tf.times[t]))
        volumes.append(result["integral"])
    return np.array(times), np.array(volumes)


_logger = logging.getLogger(__name__)


def parse_liq_file(liq_path):
    """Parse TELEMAC .liq liquid boundary file.

    Returns dict mapping column name (e.g. 'Q(1)', 'SL(2)') to:
      {"times": ndarray, "values": ndarray, "unit": str}
    Returns None if file cannot be parsed.
    """
    try:
        with open(liq_path) as f:
            lines = [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]
    except OSError:
        return None
    if len(lines) < 3:
        return None
    try:
        headers = lines[0].split()
        units = lines[1].split()
        data_lines = lines[2:]
        ncols = len(headers)
        data = np.array([[float(x) for x in line.split()[:ncols]] for line in data_lines])
        if data.size == 0:
            return None
        times = data[:, 0]
        result = {}
        for i in range(1, ncols):
            col_name = headers[i] if i < len(headers) else f"Col{i}"
            unit = units[i] if i < len(units) else ""
            result[col_name] = {
                "times": times,
                "values": data[:, i],
                "unit": unit,
            }
        return result
    except (ValueError, IndexError, KeyError) as exc:
        _logger.warning("Failed to parse .liq file '%s': %s", liq_path, exc)
        return None
