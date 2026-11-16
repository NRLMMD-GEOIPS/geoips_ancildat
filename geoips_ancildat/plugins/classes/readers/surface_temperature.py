# # # This source code is subject to the license referenced at
# # # https://github.com/NRLMMD-GEOIPS.

"""Reader for the MERRA surface temperature ancillary data files."""

import logging
from datetime import datetime, timezone

import numpy as np

from geoips_ancildat.plugins.classes.readers.utils.flat_binary import (
    FlatBinaryReaderPlugin,
)
from geoips_ancildat.plugins.classes.readers.utils.indexing import fortran_grid_step

LOG = logging.getLogger(__name__)


def to_utc_datetime(value):
    """Return a time as a UTC datetime with no time zone attached.

    Parameters
    ----------
    value : datetime.datetime or str
        Time as a datetime, or as an ISO string such as
        `"2021-07-15T13:45:00Z"`. A time with no time zone is taken as UTC.

    Returns
    -------
    datetime.datetime
        The same time in UTC, with no time zone attached.
    """
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


class SurfaceTemperatureReaderPlugin(FlatBinaryReaderPlugin):
    """Read surface temperature for one time at the requested points.

    Values are ten-year mean surface temperatures from MERRA on a 2/3 x 1/2
    degree (longitude x latitude) grid. There is one file for each hour of
    each day of the year. The files hold Celsius and the reader returns
    Kelvin, adding 273.15 like the Fortran.

    The file is picked by the day of year and the hour of the data time, like
    the Fortran. Minutes and seconds are ignored. The day of year counts
    February 29, so in a leap year every later day reads the file of the next
    day, and December 31 asks for day 366, which has no file.
    """

    name = "surface_temperature_ancildat"
    source_names = [name]
    # Filled in with the day of year and hour of the data time.
    relative_path = (
        "merra/{day_of_year:03d}_{hour:02d}_ts_MERRA.tdf_540_samples_361_lines.bin"
    )
    num_lines = 361
    num_samples = 540
    lon_res = fortran_grid_step(2, 3)
    lat_res = fortran_grid_step(0.5)
    file_dtype = "<f4"
    output_dtype = np.float64
    variable_name = "surface_temperature"

    def call(
        self,
        fnames,
        metadata_only=False,
        chans=None,
        area_def=None,
        self_register=False,
        xarray_obj=None,
        data_file=None,
        start_datetime=None,
    ):
        """Read surface temperature at the requested points.

        See `FlatBinaryReaderPlugin.read_points` for the other arguments.
        GeoIPS requires `fnames` and `chans`, but both are ignored: the reader
        finds its own data file, which holds only one variable. Pass
        `fnames=None` when calling the reader yourself.

        Parameters
        ----------
        start_datetime : datetime.datetime or str, optional
            Time of the data. If not given, the `start_datetime` attribute of
            `xarray_obj` is used. A time is needed even when `data_file` is
            given, since it is also the METADATA time.
        """
        data_datetime = self.get_data_datetime(start_datetime, xarray_obj)
        return self.read_points(
            fnames=fnames,
            data_file=data_file,
            metadata_only=metadata_only,
            area_def=area_def,
            xarray_obj=xarray_obj,
            self_register=self_register,
            relative_path=self.get_relative_path(data_datetime),
            data_datetime=data_datetime,
        )

    def get_data_datetime(self, start_datetime=None, xarray_obj=None):
        """Return the time of the data, in UTC.

        The `start_datetime` argument wins over the `xarray_obj` attribute. If
        both are given and they differ, the difference is logged.

        Parameters
        ----------
        start_datetime : datetime.datetime or str, optional
            Requested time of the data.
        xarray_obj : xarray.Dataset, optional
            Main dataset, for its `start_datetime` attribute.

        Returns
        -------
        datetime.datetime
            Time of the data in UTC, with no time zone attached.

        Raises
        ------
        ValueError
            If neither input gives a time.
        """
        dataset_datetime = None
        if xarray_obj is not None:
            dataset_datetime = xarray_obj.attrs.get("start_datetime")
        if start_datetime is None:
            if dataset_datetime is None:
                raise ValueError(
                    f"{self.name} needs start_datetime, or xarray_obj with a "
                    "start_datetime attribute."
                )
            return to_utc_datetime(dataset_datetime)

        data_datetime = to_utc_datetime(start_datetime)
        if (
            dataset_datetime is not None
            and to_utc_datetime(dataset_datetime) != data_datetime
        ):
            LOG.info(
                "Using start_datetime %s instead of the xarray_obj start_datetime "
                "%s.",
                data_datetime,
                dataset_datetime,
            )
        return data_datetime

    def get_relative_path(self, data_datetime):
        """Return the data file for the day of year and hour of a time.

        Parameters
        ----------
        data_datetime : datetime.datetime
            Time of the data, in UTC.

        Returns
        -------
        str
            Path of the data file inside the ancillary data folder.
        """
        return self.relative_path.format(
            day_of_year=data_datetime.timetuple().tm_yday, hour=data_datetime.hour
        )

    def convert_values(self, values):
        """Turn Celsius into Kelvin.

        The sum is done in float32, like the Fortran, then stored as float64.
        A float64 sum gives slightly different values.
        """
        return (values + np.float32(273.15)).astype(np.float64)


PLUGIN_CLASS = SurfaceTemperatureReaderPlugin
