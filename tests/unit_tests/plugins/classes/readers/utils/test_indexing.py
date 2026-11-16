# # # This source code is subject to the license referenced at
# # # https://github.com/NRLMMD-GEOIPS.

"""pytest test-suite for `geoips_ancildat.plugins.classes.readers.utils.indexing`.

These tests pin down the Fortran index math, quirks included, so the readers
keep returning the same values as the Fortran ancildat package. Expected rows
and columns for real grids were checked against Fortran output.

Sections:

1. Grid steps
2. Rounding
3. Rows and columns of known points
4. Invalid inputs
5. Record numbers and the edges of the file
"""

import warnings

import numpy as np
import pytest

from geoips_ancildat.plugins.classes.readers.utils.indexing import (
    fortran_grid_step,
    fortran_nint,
    get_flat_indices,
    get_line_and_sample,
)

CITY_LIGHTS_RES = fortran_grid_step(1, 240)
CITY_LIGHTS_LINES = 43200
CITY_LIGHTS_SAMPLES = 86400


def city_lights_indices(lat, lon):
    """Return the record numbers and valid mask for one city_lights point."""
    line, sample = get_line_and_sample(
        [[lon]], [[lat]], CITY_LIGHTS_RES, CITY_LIGHTS_RES
    )
    return get_flat_indices(line, sample, CITY_LIGHTS_LINES, CITY_LIGHTS_SAMPLES)


# 1. Grid steps ---------------------------------------------------------------
# The Fortran works out each grid step in float32. Using the exact value instead
# would move some points to a neighbouring cell.


@pytest.mark.parametrize(
    "args, expected",
    [
        pytest.param((1, 240), 0.004166666883975267, id="city_lights 1/240"),
        pytest.param((2, 3), 0.6666666865348816, id="surface_temperature 2/3"),
        pytest.param((0.05,), 0.05000000074505806, id="surface_emissivity 0.05"),
    ],
)
def test_fortran_grid_step_matches_fortran(args, expected):
    """Grid steps match the float32 values the Fortran uses."""
    assert fortran_grid_step(*args) == expected


# 2. Rounding -----------------------------------------------------------------
# Fortran rounds halves away from zero. numpy rounds them to the nearest even
# number, which would put points on an exact half (such as longitude -180) in
# the wrong column.


@pytest.mark.parametrize(
    "value, expected",
    [
        pytest.param(0.5, 1, id="0.5 rounds up (numpy gives 0)"),
        pytest.param(-2.5, -3, id="-2.5 rounds down (numpy gives -2)"),
        pytest.param(np.nextafter(0.5, 0), 0, id="just below 0.5 rounds down"),
        pytest.param(2.6, 3, id="2.6 rounds to nearest"),
    ],
)
def test_fortran_nint(value, expected):
    """Values round like Fortran `nint` and come back as int64."""
    result = fortran_nint(np.array([value]))
    assert result[0] == expected
    assert result.dtype == np.int64


# 3. Rows and columns of known points ----------------------------------------
# Rows and columns are 1-based. Row 1 is the north pole and column 1 is 180 W.


@pytest.mark.parametrize(
    "lat, lon, lon_res, lat_res, expected_line, expected_sample",
    [
        pytest.param(
            90.0, -180.0, CITY_LIGHTS_RES, CITY_LIGHTS_RES, 2, 2, id="north pole"
        ),
        pytest.param(
            0.0, -180.0, CITY_LIGHTS_RES, CITY_LIGHTS_RES, 21602, 2, id="lon -180"
        ),
        pytest.param(
            0.0, 180.0, CITY_LIGHTS_RES, CITY_LIGHTS_RES, 21602, 86401, id="lon 180"
        ),
        pytest.param(
            -89.5,
            0.0,
            fortran_grid_step(2, 3),
            0.5,
            361,
            271,
            id="last row of surface_temperature",
        ),
    ],
)
def test_get_line_and_sample_known_points(
    lat, lon, lon_res, lat_res, expected_line, expected_sample
):
    """Rows and columns match the Fortran for known points."""
    line, sample = get_line_and_sample([[lon]], [[lat]], lon_res, lat_res)
    assert line[0, 0] == expected_line
    assert sample[0, 0] == expected_sample


# 4. Invalid inputs -----------------------------------------------------------
# -999 is the Fortran fill value. NaN and infinite values are also invalid in
# this package (the Fortran reads row 1 or column 1 for NaN instead).


@pytest.mark.parametrize(
    "lat, lon",
    [
        pytest.param(-999.0, 0.0, id="lat -999"),
        pytest.param(0.0, -999.0, id="lon -999"),
        pytest.param(np.nan, 0.0, id="lat NaN"),
        pytest.param(0.0, np.nan, id="lon NaN"),
        pytest.param(np.inf, 0.0, id="lat inf"),
        pytest.param(0.0, -np.inf, id="lon -inf"),
    ],
)
def test_invalid_inputs(lat, lon):
    """Fill, NaN and infinite inputs are invalid and give no numpy warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        flat_indices, valid = city_lights_indices(lat, lon)
    assert not valid[0, 0]
    assert flat_indices.size == 0


# 5. Record numbers and the edges of the file --------------------------------
# These use a small 4 x 10 grid. Record numbers start at 0 and count along
# each row, then down to the next row.


@pytest.mark.parametrize(
    "line, sample, expected_record",
    [
        pytest.param(2, 3, 12, id="row 2, column 3"),
        pytest.param(1, 12, 11, id="column past the row end reads the next row"),
        pytest.param(4, 10, 39, id="last record in the file"),
    ],
)
def test_record_numbers(line, sample, expected_record):
    """Rows and columns turn into the right record number."""
    flat_indices, _ = get_flat_indices(
        np.array([line]), np.array([sample]), num_lines=4, num_samples=10
    )
    assert flat_indices[0] == expected_record


@pytest.mark.parametrize(
    "line, sample",
    [
        pytest.param(5, 1, id="below the last row"),
        pytest.param(1, 0, id="before the first record"),
        pytest.param(4, 11, id="past the last record"),
    ],
)
def test_point_outside_file_raises(line, sample):
    """A valid point that lands outside the file raises an error."""
    with pytest.raises(IndexError, match="outside the 4 x 10 file"):
        get_flat_indices(
            np.array([line]), np.array([sample]), num_lines=4, num_samples=10
        )


def test_south_pole_is_outside_city_lights_file():
    """Latitude -90 falls south of the last row, as in the Fortran."""
    with pytest.raises(IndexError):
        city_lights_indices(-90.0, 0.0)


def test_valid_mask_keeps_input_shape():
    """The valid mask has the input shape, and invalid points get no record."""
    lats = np.array([[10.0, 20.0, 30.0], [-999.0, 40.0, 50.0]])
    lons = np.zeros_like(lats)
    line, sample = get_line_and_sample(lons, lats, CITY_LIGHTS_RES, CITY_LIGHTS_RES)
    flat_indices, valid = get_flat_indices(
        line, sample, CITY_LIGHTS_LINES, CITY_LIGHTS_SAMPLES
    )
    assert valid.shape == (2, 3)
    assert not valid[1, 0]
    assert flat_indices.size == 5
