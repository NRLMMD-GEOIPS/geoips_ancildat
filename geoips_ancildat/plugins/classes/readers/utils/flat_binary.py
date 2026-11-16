# # # This source code is subject to the license referenced at
# # # https://github.com/NRLMMD-GEOIPS.

"""Shared base class for readers of flat binary ancillary files.

Each ancillary file is one grid of values with no header. A reader sets a few
class attributes that describe its file, and writes a short `call` method that
passes its arguments to `read_points`. GeoIPS needs every reader class to
define its own `call`, so the shared steps live in `read_points`.
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from geoips.config import config
from geoips.interfaces.class_based.readers import BaseReaderPlugin
from geoips_ancildat.plugins.classes.readers.utils.indexing import (
    FILL_VALUE,
    get_flat_indices,
    get_line_and_sample,
)

# interface = None indicates to the GeoIPS interfaces that this is not a valid
# plugin, and this module will not be added to the GeoIPS plugin registry.
# This allows including python modules within the geoips/plugins directory
# that provide helper or utility functions to the geoips plugins, but are
# not full GeoIPS plugins on their own.
interface = None

LOG = logging.getLogger(__name__)


def read_exactly(file_descriptor, array, byte_offset, fname):
    """Fill an array from an open file, starting at a byte offset.

    Keep reading until the array is full, because one read can return fewer
    bytes than asked for.

    Parameters
    ----------
    file_descriptor : int
        Open file, from `os.open`.
    array : numpy.ndarray
        1-D array to fill.
    byte_offset : int
        Position in the file of the first byte to read.
    fname : pathlib.Path
        File name, used in the error message.

    Raises
    ------
    EOFError
        If the file ends before the array is full.
    """
    byte_view = array.view(np.uint8)
    bytes_read = 0
    while bytes_read < byte_view.size:
        num_bytes_read = os.preadv(
            file_descriptor, [byte_view[bytes_read:]], byte_offset + bytes_read
        )
        if num_bytes_read == 0:
            raise EOFError(f"{fname} ended early while reading at byte {byte_offset}.")
        bytes_read += num_bytes_read


class FlatBinaryReaderPlugin(BaseReaderPlugin, abstract=True):
    """Base class for readers of flat binary ancillary files.

    Each reader sets these. The base class does not.

    Attributes
    ----------
    name : str
        Plugin name, also used as the `source_name` in METADATA.
    source_names : list of str
        `[name]`.
    relative_path : str
        Path of the data file inside the GeoIPS ancillary data folder
        (`config.output_paths.ancildat`, set with `GEOIPS_ANCILDAT`).
    num_lines, num_samples : int
        Number of rows and columns in the file.
    lon_res, lat_res : float
        Grid steps in degrees, from `fortran_grid_step`.
    file_dtype : str
        Type of the values in the file, such as `"<f4"`.
    output_dtype : numpy.dtype
        Type of the returned values. It must be able to hold -999.
    variable_name : str
        Name of the returned variable, and of its key in the returned dict.

    Notes
    -----
    There is no cache yet. To add one, override `compute_indices` or
    `read_values` in a subclass and call `super()` on a cache miss. An index
    cache key must include the grid size and step. A value cache key must
    include the file path.
    """

    interface = "readers"
    family = "standard"

    def read_points(
        self,
        *,
        fnames=None,
        data_file=None,
        metadata_only=False,
        area_def=None,
        xarray_obj=None,
        self_register=False,
        relative_path=None,
        data_datetime=None,
    ):
        """Read values at the requested points.

        Parameters
        ----------
        fnames : list of str, optional
            Ignored. GeoIPS passes the satellite files here, but these
            readers always find their own data file.
        data_file : str or pathlib.Path, optional
            Data file to read instead of the default one.
        metadata_only : bool, optional
            If True, return only the METADATA dataset.
        area_def : pyresample.AreaDefinition, optional
            Read values at the points of this area.
        xarray_obj : xarray.Dataset, optional
            Read values at its `latitude` and `longitude` points, if no
            `area_def` is given. Its `start_datetime` and `end_datetime`
            attributes are used as the METADATA times.
        self_register : bool, optional
            Not supported. Must be False.
        relative_path : str, optional
            Data file inside the ancillary data folder, for readers that pick
            the file per call. Defaults to the class `relative_path`.
        data_datetime : datetime.datetime, optional
            Requested time of the data. If given, used as the METADATA start
            and end time.

        Returns
        -------
        dict of xarray.Dataset
            `{variable_name: data, "METADATA": metadata}`. The data dataset
            holds the values, plus `latitude` and `longitude` as given.

        Raises
        ------
        ValueError
            If `self_register` is set.
        """
        if self_register:
            raise ValueError(f"{self.name} does not support self_register.")
        if fnames:
            LOG.info(
                "%s ignores fnames and reads its own data file. Use data_file to "
                "pick a different file.",
                self.name,
            )

        fname = self.get_filename(data_file, relative_path)
        metadata = self.build_metadata(fname, xarray_obj, data_datetime)
        if metadata_only:
            return {"METADATA": metadata}

        lons, lats, dims = self.get_lons_lats(area_def, xarray_obj)
        flat_indices, valid = self.compute_indices(lons, lats)
        values = self.read_values(fname, flat_indices)

        output = np.full(valid.shape, FILL_VALUE, dtype=self.output_dtype)
        output[valid] = self.convert_values(values)
        dataset = xr.Dataset(
            {
                self.variable_name: (dims, output),
                "latitude": (dims, lats),
                "longitude": (dims, lons),
            },
            attrs=dict(metadata.attrs),
        )
        return {self.variable_name: dataset, "METADATA": metadata}

    def get_filename(self, data_file=None, relative_path=None):
        """Return the data file to read.

        Parameters
        ----------
        data_file : str or pathlib.Path, optional
            Data file to use. If not given, use the default file.
        relative_path : str, optional
            Default file inside the ancillary data folder. Defaults to the
            class `relative_path`.

        Returns
        -------
        pathlib.Path
            Path of the data file.
        """
        if data_file:
            return Path(data_file)
        ancildat_dir = Path(config.output_paths.ancildat)
        return ancildat_dir / (relative_path or self.relative_path)

    def build_metadata(self, fname, xarray_obj=None, data_datetime=None):
        """Return the METADATA dataset.

        The start and end times come from `data_datetime` if given, else from
        `xarray_obj`, else from the last-modified time of the data file.

        Parameters
        ----------
        fname : pathlib.Path
            Data file.
        xarray_obj : xarray.Dataset, optional
            Main dataset, for its `start_datetime` and `end_datetime`.
        data_datetime : datetime.datetime, optional
            Requested time of the data.

        Returns
        -------
        xarray.Dataset
            Dataset with no variables and the METADATA attributes.
        """
        if data_datetime is not None:
            start_datetime = end_datetime = data_datetime
        elif xarray_obj is not None and "start_datetime" in xarray_obj.attrs:
            start_datetime = xarray_obj.attrs["start_datetime"]
            end_datetime = xarray_obj.attrs.get("end_datetime", start_datetime)
        else:
            modified_time = datetime.fromtimestamp(
                os.path.getmtime(fname), tz=timezone.utc
            )
            start_datetime = end_datetime = modified_time.replace(tzinfo=None)
            LOG.info(
                "No data time given. Using the last-modified time of %s as the "
                "start and end time.",
                fname,
            )
        return xr.Dataset(
            attrs={
                "source_name": self.name,
                "start_datetime": start_datetime,
                "end_datetime": end_datetime,
                "source_file_names": [fname.name],
            }
        )

    def get_lons_lats(self, area_def=None, xarray_obj=None):
        """Return the longitudes, latitudes and dimension names of the points.

        Parameters
        ----------
        area_def : pyresample.AreaDefinition, optional
            Area to read. Used first if given.
        xarray_obj : xarray.Dataset, optional
            Dataset with `latitude` and `longitude`.

        Returns
        -------
        lons, lats : numpy.ndarray
            Longitudes and latitudes of the points.
        dims : tuple of str
            `("dim_0", "dim_1")` for an area, else the dimension names of the
            `xarray_obj` latitude.

        Raises
        ------
        ValueError
            If neither input gives any points.
        """
        if area_def is not None:
            if xarray_obj is not None:
                LOG.info(
                    "Both area_def and xarray_obj were given. Using the area_def "
                    "latitudes and longitudes."
                )
            lons, lats = area_def.get_lonlats()
            return np.asarray(lons), np.asarray(lats), ("dim_0", "dim_1")
        if (
            xarray_obj is not None
            and "latitude" in xarray_obj
            and "longitude" in xarray_obj
        ):
            latitude = xarray_obj["latitude"]
            return xarray_obj["longitude"].values, latitude.values, latitude.dims
        raise ValueError(
            f"{self.name} needs area_def, or xarray_obj with latitude and longitude."
        )

    def compute_indices(self, lons, lats):
        """Return the record number of each valid point, and the valid mask.

        Parameters
        ----------
        lons, lats : numpy.ndarray
            Longitudes and latitudes of the points.

        Returns
        -------
        flat_indices : numpy.ndarray
            1-D record numbers of the valid points.
        valid : numpy.ndarray
            Boolean mask of the input shape.
        """
        line, sample = get_line_and_sample(lons, lats, self.lon_res, self.lat_res)
        return get_flat_indices(line, sample, self.num_lines, self.num_samples)

    def read_values(self, fname, flat_indices):
        """Read the file values at the given record numbers.

        Parameters
        ----------
        fname : pathlib.Path
            Data file.
        flat_indices : numpy.ndarray
            1-D record numbers, from `compute_indices`.

        Returns
        -------
        numpy.ndarray
            1-D array of `file_dtype` values, in the order of `flat_indices`.

        Raises
        ------
        ValueError
            If the file size does not match the grid size.

        Notes
        -----
        Each needed row is read once, from the first to the last needed
        column of all the points. Points that cross 180 degrees, or columns
        that wrap to the next row, make this the whole row.
        """
        file_dtype = np.dtype(self.file_dtype)
        if flat_indices.size == 0:
            return np.empty(0, dtype=file_dtype)

        rows = flat_indices // self.num_samples
        columns = flat_indices % self.num_samples
        unique_rows, row_positions = np.unique(rows, return_inverse=True)
        first_column = int(columns.min())
        num_columns = int(columns.max()) - first_column + 1
        row_block = np.empty((unique_rows.size, num_columns), dtype=file_dtype)

        file_descriptor = os.open(fname, os.O_RDONLY)
        try:
            expected_size = self.num_lines * self.num_samples * file_dtype.itemsize
            file_size = os.fstat(file_descriptor).st_size
            if file_size != expected_size:
                raise ValueError(
                    f"{fname} is {file_size} bytes, but a {self.num_lines} x "
                    f"{self.num_samples} grid of {file_dtype} is {expected_size} "
                    "bytes."
                )
            for row_position, row in enumerate(unique_rows):
                byte_offset = (
                    int(row) * self.num_samples + first_column
                ) * file_dtype.itemsize
                read_exactly(
                    file_descriptor, row_block[row_position], byte_offset, fname
                )
        finally:
            os.close(file_descriptor)
        return row_block[row_positions, columns - first_column]

    def convert_values(self, values):
        """Return the values to store in the output.

        The base class returns them unchanged. Readers that change units or
        types override this. It is only called for valid points.

        Parameters
        ----------
        values : numpy.ndarray
            1-D file values of the valid points.

        Returns
        -------
        numpy.ndarray
            Values to store.
        """
        return values
