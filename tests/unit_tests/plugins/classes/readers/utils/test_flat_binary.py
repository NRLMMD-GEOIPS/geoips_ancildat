# # # This source code is subject to the license referenced at
# # # https://github.com/NRLMMD-GEOIPS.

"""pytest test-suite for `geoips_ancildat.plugins.classes.readers.utils.flat_binary`.

The tests use a tiny test reader and a 4 x 8 test grid, written to a temporary
folder. The grid map below shows which record each point reads.

Sections:

1. Reading values
2. Where the points come from
3. Which data file is read
4. Reading the file safely
5. METADATA
6. Calls from GeoIPS
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from geoips.config import config as geoips_config

from geoips_ancildat.plugins.classes.readers.utils import flat_binary
from geoips_ancildat.plugins.classes.readers.utils.flat_binary import (
    FlatBinaryReaderPlugin,
)

# Test grid map ---------------------------------------------------------------
# 4 rows x 8 columns with 45 degree steps. Each value equals its own record
# number (0 to 31), so the expected value of a point is its record number.
#
#   lat 90 -> row 2 (records 8-15)     lon -180 -> column 2
#   lat 45 -> row 3 (records 16-23)    lon -135 -> column 3
#   lat  0 -> row 4 (records 24-31)    lon    0 -> column 6
#   lat -45 -> row 5, outside the file lon   90 -> column 8
#                                      lon  180 -> column 10, wraps to next row
#
#   record = (row - 1) * 8 + (column - 1)
#
# Row 1 and column 1 are never reached. That is the Fortran row and column math,
# which is kept on purpose. This grid is too coarse for -999 inputs (they round
# to a valid row), so these tests use NaN and inf as invalid points. -999 is
# tested in test_indexing.py at the real grid steps.


class TinyReader(FlatBinaryReaderPlugin):
    """Reader for the 4 x 8 test grid."""

    name = "tiny_ancildat"
    source_names = [name]
    relative_path = "tiny.bin"
    num_lines = 4
    num_samples = 8
    lon_res = 45.0
    lat_res = 45.0
    file_dtype = "<f4"
    output_dtype = np.float64
    variable_name = "tiny"

    def call(
        self,
        fnames=None,
        metadata_only=False,
        chans=None,
        area_def=None,
        self_register=False,
        xarray_obj=None,
        data_file=None,
    ):
        """Read the test grid at the requested points."""
        return self.read_points(
            fnames=fnames,
            data_file=data_file,
            metadata_only=metadata_only,
            area_def=area_def,
            xarray_obj=xarray_obj,
            self_register=self_register,
        )


class PlusOneReader(TinyReader):
    """Test reader that adds 1 to every value it reads."""

    name = "plus_one_ancildat"

    def call(self, xarray_obj=None):
        """Read the test grid at the requested points."""
        return self.read_points(xarray_obj=xarray_obj)

    def convert_values(self, values):
        """Add 1 to each value."""
        return values + 1


class FakeArea:
    """Stand-in for a pyresample area with fixed points."""

    def __init__(self, lats, lons):
        """Store the points."""
        self.lats = np.array(lats, dtype=float)
        self.lons = np.array(lons, dtype=float)

    def get_lonlats(self):
        """Return the longitudes and latitudes."""
        return self.lons, self.lats


def points_dataset(lats, lons, dims=("y", "x"), **attrs):
    """Return a dataset with latitude and longitude, like a main reader output."""
    return xr.Dataset(
        {
            "latitude": (dims, np.array(lats, dtype=float)),
            "longitude": (dims, np.array(lons, dtype=float)),
        },
        attrs=attrs,
    )


def read_one_point(lat, lon, **kwargs):
    """Read one point with the test reader and return its value."""
    xarray_obj = points_dataset([[lat]], [[lon]])
    return TinyReader().call(xarray_obj=xarray_obj, **kwargs)["tiny"]["tiny"].item()


@pytest.fixture
def ancildat_dir(tmp_path, monkeypatch):
    """Write the test grid and use its folder as the ancillary data folder."""
    np.arange(32, dtype="<f4").tofile(tmp_path / "tiny.bin")
    # The GeoIPS config is frozen, so swap in a stand-in with only this path.
    test_config = SimpleNamespace(output_paths=SimpleNamespace(ancildat=tmp_path))
    monkeypatch.setattr(flat_binary, "config", test_config)
    return tmp_path


# 1. Reading values -----------------------------------------------------------


def test_reads_values_at_points(ancildat_dir):
    """Values come from the right records and keep the input layout."""
    # Points cover rows 2-4, are out of order, repeat one point, and include
    # lon 180, which wraps to the start of the next row (record 17).
    lats = [[0.0, 90.0, 45.0], [90.0, 45.0, 90.0]]
    lons = [[0.0, -180.0, -135.0], [-180.0, 90.0, 180.0]]
    start = datetime(2021, 7, 15, 13)
    xarray_obj = points_dataset(lats, lons, start_datetime=start, end_datetime=start)

    result = TinyReader().call(xarray_obj=xarray_obj)

    assert set(result) == {"tiny", "METADATA"}
    dataset = result["tiny"]
    np.testing.assert_array_equal(dataset["tiny"], [[29, 9, 18], [9, 23, 17]])
    assert dataset["tiny"].dtype == np.float64
    # Dims are copied from the input latitude, so the output lines up with it.
    assert dataset["tiny"].dims == ("y", "x")
    np.testing.assert_array_equal(dataset["latitude"], lats)
    np.testing.assert_array_equal(dataset["longitude"], lons)
    assert dataset.attrs == result["METADATA"].attrs
    assert result["METADATA"].attrs["source_name"] == "tiny_ancildat"


def test_invalid_points_get_fill(ancildat_dir):
    """Points that are not finite give -999, and the file is not needed."""
    lats = [[np.nan, -np.inf], [np.inf, 0.0]]
    lons = [[0.0, 0.0], [0.0, np.nan]]
    xarray_obj = points_dataset(lats, lons, start_datetime=datetime(2020, 1, 1))
    # No point is valid, so a missing data file is never opened.
    result = TinyReader().call(
        xarray_obj=xarray_obj, data_file=ancildat_dir / "missing.bin"
    )
    np.testing.assert_array_equal(result["tiny"]["tiny"], -999)


def test_convert_values_skips_invalid_points(ancildat_dir):
    """Converted values stay -999 where the point is invalid."""
    # Matters for surface_temperature, which adds 273.15 only to valid points.
    xarray_obj = points_dataset([[90.0, np.nan]], [[-180.0, 0.0]])
    result = PlusOneReader().call(xarray_obj=xarray_obj)
    np.testing.assert_array_equal(result["tiny"]["tiny"], [[10, -999]])


# 2. Where the points come from -----------------------------------------------


def test_area_def_points_win_over_xarray_obj(ancildat_dir, caplog):
    """Area points are used when both are given, with default dims and a log."""
    xarray_obj = points_dataset([[0.0]], [[0.0]])
    with caplog.at_level(logging.INFO):
        result = TinyReader().call(
            area_def=FakeArea([[90.0]], [[-180.0]]), xarray_obj=xarray_obj
        )
    assert result["tiny"]["tiny"].item() == 9
    assert result["tiny"]["tiny"].dims == ("dim_0", "dim_1")
    assert "Using the area_def latitudes and longitudes" in caplog.text


def test_missing_points_raise(ancildat_dir):
    """Without an area or an xarray_obj there are no points to read."""
    with pytest.raises(ValueError, match="needs area_def"):
        TinyReader().call()


# 3. Which data file is read --------------------------------------------------


def test_default_file_is_in_ancildat_folder():
    """The default file is inside the folder from the real GeoIPS config."""
    expected = Path(geoips_config.output_paths.ancildat) / "tiny.bin"
    assert TinyReader().get_filename() == expected


def test_data_file_is_used_and_fnames_are_ignored(ancildat_dir, caplog):
    """`data_file` picks the file. `fnames` (satellite files) is ignored."""
    other_file = ancildat_dir / "other.bin"
    np.arange(100, 132, dtype="<f4").tofile(other_file)
    with caplog.at_level(logging.INFO):
        value = read_one_point(
            90.0, -180.0, fnames=["satellite.nc"], data_file=other_file
        )
    assert value == 109
    assert "ignores fnames" in caplog.text


# 4. Reading the file safely --------------------------------------------------


def test_short_reads_are_retried(ancildat_dir, monkeypatch):
    """Reads that return only a few bytes are repeated until done."""
    real_preadv = os.preadv

    def preadv_three_bytes(file_descriptor, buffers, offset):
        """Read at most 3 bytes per call."""
        return real_preadv(file_descriptor, [buffers[0][:3]], offset)

    monkeypatch.setattr(flat_binary.os, "preadv", preadv_three_bytes)
    xarray_obj = points_dataset([[0.0, 0.0]], [[-180.0, 90.0]])
    result = TinyReader().call(xarray_obj=xarray_obj)
    np.testing.assert_array_equal(result["tiny"]["tiny"], [[25, 31]])


def test_file_ending_early_raises(ancildat_dir, monkeypatch):
    """A read that returns no bytes raises an error."""
    monkeypatch.setattr(flat_binary.os, "preadv", lambda *args: 0)
    with pytest.raises(EOFError, match="ended early"):
        read_one_point(90.0, -180.0)


def test_wrong_file_size_raises(ancildat_dir):
    """A file that does not match the grid size raises an error."""
    np.arange(31, dtype="<f4").tofile(ancildat_dir / "tiny.bin")
    with pytest.raises(ValueError, match="is 124 bytes"):
        read_one_point(90.0, -180.0)


# 5. METADATA -----------------------------------------------------------------
# Start and end times come from `data_datetime`, else from `xarray_obj`, else
# from the data file's last-modified time.


def test_metadata_only_uses_xarray_obj_times(ancildat_dir):
    """Only METADATA is returned, its times come from xarray_obj, no file read."""
    start = datetime(2021, 7, 15, 13)
    xarray_obj = points_dataset([[90.0]], [[-180.0]], start_datetime=start)
    result = TinyReader().call(
        metadata_only=True,
        xarray_obj=xarray_obj,
        data_file=ancildat_dir / "missing.bin",
    )
    assert set(result) == {"METADATA"}
    assert result["METADATA"].attrs["start_datetime"] == start
    assert result["METADATA"].attrs["end_datetime"] == start


def test_metadata_times_from_data_datetime(ancildat_dir):
    """A requested data time wins over the xarray_obj times."""
    requested = datetime(2020, 3, 1, 6)
    xarray_obj = points_dataset([[0.0]], [[0.0]], start_datetime=datetime(2021, 1, 1))
    metadata = TinyReader().build_metadata(
        ancildat_dir / "tiny.bin", xarray_obj, data_datetime=requested
    )
    assert metadata.attrs["start_datetime"] == requested
    assert metadata.attrs["end_datetime"] == requested


def test_metadata_times_from_file_modified_time(ancildat_dir, caplog):
    """With no time given, the file's last-modified time is used and logged."""
    modified = datetime(2020, 1, 2, 3, 4, 5, tzinfo=timezone.utc).timestamp()
    os.utime(ancildat_dir / "tiny.bin", (modified, modified))
    with caplog.at_level(logging.INFO):
        metadata = TinyReader().build_metadata(ancildat_dir / "tiny.bin")
    assert metadata.attrs["start_datetime"] == datetime(2020, 1, 2, 3, 4, 5)
    assert metadata.attrs["end_datetime"] == datetime(2020, 1, 2, 3, 4, 5)
    assert "last-modified time" in caplog.text


# 6. Calls from GeoIPS --------------------------------------------------------


def test_workflow_style_call_returns_data_tree(ancildat_dir):
    """Called like a workflow reader step, the satellite files are ignored."""
    # A workflow passes its satellite files as `filenames` to every reader step.
    xarray_obj = points_dataset(
        [[90.0]], [[-180.0]], start_datetime=datetime(2020, 1, 1)
    )
    result = TinyReader()(
        filenames=["satellite.nc"], _obp_initiated=True, xarray_obj=xarray_obj
    )
    assert isinstance(result, xr.DataTree)
    assert result["tiny"]["tiny"].item() == 9


def test_self_register_raises(ancildat_dir):
    """`self_register` is not supported."""
    with pytest.raises(ValueError, match="self_register"):
        TinyReader().call(self_register="LOW")
