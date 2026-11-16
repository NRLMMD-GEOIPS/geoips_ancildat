# # # This source code is subject to the license referenced at
# # # https://github.com/NRLMMD-GEOIPS.

"""pytest test-suite for `geoips_ancildat.plugins.classes.readers.utils.indexing`."""

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
    """Return flat indices and valid mask for one city_lights point."""
    line, sample = get_line_and_sample(
        [[lon]], [[lat]], CITY_LIGHTS_RES, CITY_LIGHTS_RES
    )
    return get_flat_indices(line, sample, CITY_LIGHTS_LINES, CITY_LIGHTS_SAMPLES)


@pytest.mark.parametrize(
    "args, expected",
    [
        ((1, 240), 0.004166666883975267),
        ((1, 120), 0.008333333767950535),
        ((2, 3), 0.6666666865348816),
        ((0.05,), 0.05000000074505806),
    ],
)
def test_fortran_grid_step_matches_fortran(args, expected):
    """Grid steps match the float32 values the Fortran uses."""
    assert fortran_grid_step(*args) == expected


def test_fortran_nint_rounds_halves_away_from_zero():
    """Halves round away from zero, not to the nearest even number."""
    values = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    expected = np.array([-3, -2, -1, 1, 2, 3])
    np.testing.assert_array_equal(fortran_nint(values), expected)


def test_fortran_nint_rounds_other_values_to_nearest():
    """Values that are not halves round to the nearest integer."""
    just_below_half = np.nextafter(0.5, 0)
    values = np.array([-2.6, -0.4, just_below_half, 2.4, 2.6])
    result = fortran_nint(values)
    np.testing.assert_array_equal(result, [-3, 0, 0, 2, 3])
    assert result.dtype == np.int64


@pytest.mark.parametrize(
    "lat, lon, lon_res, lat_res, expected_line, expected_sample",
    [
        # Longitude -180 lands on an exact half, which rounds up.
        (90.0, -180.0, CITY_LIGHTS_RES, CITY_LIGHTS_RES, 2, 2),
        (0.0, -180.0, CITY_LIGHTS_RES, CITY_LIGHTS_RES, 21602, 2),
        (0.0, 180.0, CITY_LIGHTS_RES, CITY_LIGHTS_RES, 21602, 86401),
        # Last row of the surface temperature file.
        (-89.5, 0.0, fortran_grid_step(2, 3), 0.5, 361, 271),
    ],
)
def test_get_line_and_sample_known_points(
    lat, lon, lon_res, lat_res, expected_line, expected_sample
):
    """Rows and columns match the Fortran for known points."""
    line, sample = get_line_and_sample([[lon]], [[lat]], lon_res, lat_res)
    assert line[0, 0] == expected_line
    assert sample[0, 0] == expected_sample


def test_fill_longitude_follows_fortran():
    """A -999 longitude is shifted like any other longitude."""
    _, sample = get_line_and_sample([[-999.0]], [[0.0]], CITY_LIGHTS_RES, 1.0)
    assert sample[0, 0] == -196558


@pytest.mark.parametrize(
    "lat, lon",
    [
        (-999.0, 0.0),
        (-1e30, 0.0),
        (0.0, -999.0),
        (np.nan, 0.0),
        (0.0, np.nan),
        (np.inf, 0.0),
        (-np.inf, 0.0),
        (0.0, np.inf),
        (0.0, -np.inf),
    ],
)
def test_fill_and_non_finite_inputs_are_invalid(lat, lon):
    """Fill, NaN and infinite inputs are invalid and do not warn."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        flat_indices, valid = city_lights_indices(lat, lon)
    assert not valid[0, 0]
    assert flat_indices.size == 0


def test_flat_index_steps_through_rows():
    """Record numbers count across each row, then down to the next."""
    flat_indices, _ = get_flat_indices(
        np.array([2]), np.array([3]), num_lines=4, num_samples=10
    )
    assert flat_indices[0] == 12


def test_sample_past_row_end_wraps_to_next_row():
    """A column past the end of a row reads from the next row."""
    flat_indices, _ = get_flat_indices(
        np.array([1]), np.array([12]), num_lines=4, num_samples=10
    )
    assert flat_indices[0] == 11


def test_last_record_is_inside_file():
    """The last row and column give the last record in the file."""
    flat_indices, _ = get_flat_indices(
        np.array([4]), np.array([10]), num_lines=4, num_samples=10
    )
    assert flat_indices[0] == 39


@pytest.mark.parametrize("line, sample", [(5, 1), (1, 0), (4, 11)])
def test_point_outside_file_raises(line, sample):
    """A valid point outside the file raises an error."""
    with pytest.raises(IndexError, match="outside the 4 x 10 file"):
        get_flat_indices(
            np.array([line]), np.array([sample]), num_lines=4, num_samples=10
        )


def test_south_pole_is_outside_city_lights_file():
    """Latitude -90 falls south of the last row and raises an error."""
    with pytest.raises(IndexError):
        city_lights_indices(-90.0, 0.0)


def test_valid_mask_keeps_input_shape():
    """The valid mask has the input shape and skips fill points."""
    lats = np.array([[10.0, 20.0, 30.0], [-999.0, 40.0, 50.0]])
    lons = np.zeros_like(lats)
    line, sample = get_line_and_sample(lons, lats, CITY_LIGHTS_RES, CITY_LIGHTS_RES)
    flat_indices, valid = get_flat_indices(
        line, sample, CITY_LIGHTS_LINES, CITY_LIGHTS_SAMPLES
    )
    assert valid.shape == (2, 3)
    assert not valid[1, 0]
    assert flat_indices.size == 5
