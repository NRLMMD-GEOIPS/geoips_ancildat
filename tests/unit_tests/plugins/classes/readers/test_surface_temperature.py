# # # This source code is subject to the license referenced at
# # # https://github.com/NRLMMD-GEOIPS.

"""pytest test-suite for the surface_temperature reader.

The reader picks one MERRA file by the day of year and hour of the data time,
then reads it like the other readers. Its golden cases are:

- global: 12 x 24 points over the whole globe, at four times: a summer
  afternoon, the first file of the year, a leap year day after February 29,
  and the last file of the year.
- fine and edges: the same points as for the static readers, at the summer
  afternoon time.
"""

import logging
from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from geoips_ancildat.plugins.classes.readers.surface_temperature import (
    SurfaceTemperatureReaderPlugin,
    to_utc_datetime,
)

FILE_SUFFIX = "_ts_MERRA.tdf_540_samples_361_lines.bin"

# Golden case -> (time given to the Fortran, file it should read).
GOLDEN_CASES = {
    # Minutes are ignored, so 13:45 reads hour 13.
    "global_20210715_13": ("2021-07-15T13:45:00Z", "196_13"),
    "global_20200101_00": ("2020-01-01T00:00:00Z", "001_00"),
    # Leap year: March 1 is day 61, not 60.
    "global_20200301_06": ("2020-03-01T06:00:00Z", "061_06"),
    # 23:59:59 still reads hour 23.
    "global_20211231_23": ("2021-12-31T23:59:59Z", "365_23"),
    "fine": ("2021-07-15T13:45:00Z", "196_13"),
    "edges": ("2021-07-15T13:45:00Z", "196_13"),
}


# 1. Matches the Fortran, and reads the expected file.
@pytest.mark.parametrize("case_name", GOLDEN_CASES)
def test_matches_fortran(case_name, load_golden, require_file):
    """Every point equals the Fortran output, with the same dtype."""
    start_datetime, day_and_hour = GOLDEN_CASES[case_name]
    reader = SurfaceTemperatureReaderPlugin()
    require_file(
        reader.get_filename(relative_path=f"merra/{day_and_hour}{FILE_SUFFIX}"),
        "MERRA data file",
    )
    xarray_obj, expected = load_golden("surface_temperature", case_name)

    # Pass the time as a string, like debra does today.
    result = reader.call(
        fnames=None, xarray_obj=xarray_obj, start_datetime=start_datetime
    )
    output = result["surface_temperature"]["surface_temperature"].values

    assert result["METADATA"].attrs["source_file_names"] == [
        f"{day_and_hour}{FILE_SUFFIX}"
    ]
    assert output.dtype == expected.dtype
    np.testing.assert_array_equal(output, expected)


# 2. File names that no golden case covers.
@pytest.mark.parametrize(
    "start_datetime, day_and_hour",
    [
        # December 31 of a leap year is day 366. There is no such file.
        pytest.param("2020-12-31T00:00:00Z", "366_00", id="leap_year_day_366"),
        # A time with a time zone is turned into UTC first.
        pytest.param("2021-07-15T09:45:00-04:00", "196_13", id="time_zone_offset"),
    ],
)
def test_file_name_from_data_time(start_datetime, day_and_hour):
    """The file name comes from the UTC day of year and hour."""
    reader = SurfaceTemperatureReaderPlugin()
    relative_path = reader.get_relative_path(to_utc_datetime(start_datetime))
    assert relative_path == f"merra/{day_and_hour}{FILE_SUFFIX}"


# 3. Where the data time comes from.
def test_data_time_from_xarray_obj():
    """With no start_datetime, the xarray_obj start time is used."""
    xarray_obj = xr.Dataset(attrs={"start_datetime": datetime(2021, 7, 15, 13, 45)})
    data_datetime = SurfaceTemperatureReaderPlugin().get_data_datetime(
        xarray_obj=xarray_obj
    )
    assert data_datetime == datetime(2021, 7, 15, 13, 45)


def test_start_datetime_wins_over_xarray_obj(caplog):
    """The start_datetime argument is used, and the difference is logged."""
    xarray_obj = xr.Dataset(attrs={"start_datetime": datetime(2021, 7, 15, 13, 45)})
    with caplog.at_level(logging.INFO):
        data_datetime = SurfaceTemperatureReaderPlugin().get_data_datetime(
            "2020-01-01T00:00:00Z", xarray_obj
        )
    assert data_datetime == datetime(2020, 1, 1)
    assert "instead of the xarray_obj start_datetime" in caplog.text


def test_no_data_time_raises():
    """With no time from either input, the reader raises."""
    with pytest.raises(ValueError, match="needs start_datetime"):
        SurfaceTemperatureReaderPlugin().get_data_datetime(xarray_obj=xr.Dataset())
