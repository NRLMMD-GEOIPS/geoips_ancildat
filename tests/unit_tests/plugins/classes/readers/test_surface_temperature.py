# # # This source code is subject to the license referenced at
# # # https://github.com/NRLMMD-GEOIPS.

"""pytest test-suite for the surface_temperature reader.

The reader picks one MERRA file by the day of year and hour of the data time,
then reads it like the other readers. These tests need no data files. The
comparison with the Fortran is in the integration tests.
"""

import logging
from datetime import datetime

import pytest
import xarray as xr

from geoips_ancildat.plugins.classes.readers.surface_temperature import (
    SurfaceTemperatureReaderPlugin,
    to_utc_datetime,
)

FILE_SUFFIX = "_ts_MERRA.tdf_540_samples_361_lines.bin"


# 1. File names that the Fortran comparison cases do not cover.
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


# 2. Where the data time comes from.
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
