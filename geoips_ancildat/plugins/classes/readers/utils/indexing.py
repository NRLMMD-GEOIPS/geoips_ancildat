# # # This source code is subject to the license referenced at
# # # https://github.com/NRLMMD-GEOIPS.

"""Find where each latitude and longitude falls in an ancillary data file.

The ancillary files are flat binary grids. Row 1 is the north pole and each
row steps south. Column 1 is 180 W and each column steps east. These functions
match the index math in the Fortran ancildat package (`get_inds` in
`ancildat.f90`), so the readers return the same values as the Fortran.

These Fortran quirks are kept on purpose, because changing them would change
the outputs:

- Grid steps are the Fortran float32 values, stored as float64.
- Halves round away from zero, like Fortran `nint`.
- The row math adds a full grid step before rounding. The column math adds
  half a step.
- A longitude of -999 is not treated as fill. It is shifted by 180 like any
  other longitude (to -819), which still gives an invalid column.
- A column past the end of a row reads from the start of the next row.
- Points south of the last row (for example latitude -90) fall outside the
  file and raise an error. The Fortran crashes on these points.

One difference from the Fortran:

- A NaN or infinite latitude or longitude is invalid, so the output is -999.
  The Fortran reads row 1 or column 1 of the file instead.
"""

import numpy as np

# interface = None indicates to the GeoIPS interfaces that this is not a valid
# plugin, and this module will not be added to the GeoIPS plugin registry.
# This allows including python modules within the geoips/plugins directory
# that provide helper or utility functions to the geoips plugins, but are
# not full GeoIPS plugins on their own.
interface = None

FILL_VALUE = -999


def fortran_grid_step(value, divisor=1):
    """Return a grid step the way the Fortran stores it.

    Parameters
    ----------
    value : float
        Grid step, or the top of a fraction such as 1 in 1/240.
    divisor : float, optional
        Bottom of the fraction. Default is 1.

    Returns
    -------
    float
        The grid step in float32, stored as float64.

    Notes
    -----
    The Fortran writes grid steps such as `res = 1.0/240.0`. The division
    happens in float32 and the result is stored as float64. Using the exact
    value instead gives different indices for some points.
    """
    return float(np.float32(value) / np.float32(divisor))


def fortran_nint(values):
    """Round to the nearest integer, with halves rounded away from zero.

    Parameters
    ----------
    values : numpy.ndarray
        Finite float values.

    Returns
    -------
    numpy.ndarray
        int64 array of the same shape.

    Notes
    -----
    numpy `rint` rounds 2.5 to 2. Fortran `nint` rounds it to 3.
    """
    truncated = np.trunc(values)
    # This subtraction is exact, so the test for .5 is exact too.
    is_half = np.abs(values - truncated) == 0.5
    rounded = np.where(is_half, truncated + np.sign(values), np.rint(values))
    return rounded.astype(np.int64)


def get_line_and_sample(lons, lats, lon_res, lat_res):
    """Return the 1-based file row and column for each point.

    Parameters
    ----------
    lons, lats : array_like
        Longitudes and latitudes in degrees, with matching shapes.
    lon_res, lat_res : float
        Grid steps in degrees, from `fortran_grid_step`.

    Returns
    -------
    line, sample : numpy.ndarray
        int64 arrays of the input shape. Fill inputs give values of -999 or
        less, which `get_flat_indices` marks invalid.
    """
    lons = np.asarray(lons, dtype=np.float64)
    lats = np.asarray(lats, dtype=np.float64)
    lat_is_finite = np.isfinite(lats)
    lon_is_finite = np.isfinite(lons)
    # Put 0 in place of NaN and inf so the rounding does not warn. These
    # points are set to fill at the end.
    lats = np.where(lat_is_finite, lats, 0.0)
    lons = np.where(lon_is_finite, lons, 0.0)

    # Degrees south of the north pole, and degrees east of 180 W.
    lat_position = np.where(lats <= FILL_VALUE, FILL_VALUE, 90.0 - lats)
    lon_position = lons + 180.0

    line = fortran_nint((lat_position + lat_res) / lat_res) + 1
    sample = fortran_nint((lon_position + lon_res / 2.0) / lon_res) + 1
    line[~lat_is_finite] = FILL_VALUE
    sample[~lon_is_finite] = FILL_VALUE
    return line, sample


def get_flat_indices(line, sample, num_lines, num_samples):
    """Return the 0-based record number of each valid point, and the valid mask.

    Parameters
    ----------
    line, sample : numpy.ndarray
        1-based rows and columns from `get_line_and_sample`.
    num_lines, num_samples : int
        Number of rows and columns in the file.

    Returns
    -------
    flat_indices : numpy.ndarray
        1-D int64 array with one record number per valid point, in the
        order of the input.
    valid : numpy.ndarray
        Boolean mask of the input shape. True where both row and column are
        greater than -999.

    Raises
    ------
    IndexError
        If a valid point falls outside the file.
    """
    valid = (line > FILL_VALUE) & (sample > FILL_VALUE)
    valid_lines = line[valid]
    valid_samples = sample[valid]
    # Rows are stored one after another.
    flat_indices = (valid_lines - 1) * num_samples + (valid_samples - 1)

    outside = (flat_indices < 0) | (flat_indices >= num_lines * num_samples)
    if outside.any():
        first = np.flatnonzero(outside)[0]
        raise IndexError(
            f"{np.count_nonzero(outside)} points fall outside the "
            f"{num_lines} x {num_samples} file. The first is at line "
            f"{valid_lines[first]}, sample {valid_samples[first]}."
        )
    return flat_indices, valid
