# # # This source code is subject to the license referenced at
# # # https://github.com/NRLMMD-GEOIPS.

"""pytest test-suite for `geoips_ancildat.plugins.classes.readers.utils.flat_binary`."""

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


class TinyReader(FlatBinaryReaderPlugin):
    """Reader for a 4 x 8 test grid with 45 degree steps."""

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
    """Test reader that adds 1 to every value."""

    name = "plus_one_ancildat"

    def call(self, xarray_obj=None):
        """Read the test grid at the requested points."""
        return self.read_points(xarray_obj=xarray_obj)

    def convert_values(self, values):
        """Add 1 to each value."""
        return values + 1


class TinyIntReader(TinyReader):
    """Test reader for an integer grid."""

    name = "tiny_int_ancildat"
    relative_path = "tiny_int.bin"
    file_dtype = "<i4"
    output_dtype = np.int64

    def call(self, xarray_obj=None):
        """Read the test grid at the requested points."""
        return self.read_points(xarray_obj=xarray_obj)


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


@pytest.fixture
def ancildat_dir(tmp_path, monkeypatch):
    """Write test grids holding their own record numbers, and point to them."""
    np.arange(32, dtype="<f4").tofile(tmp_path / "tiny.bin")
    np.arange(32, dtype="<i4").tofile(tmp_path / "tiny_int.bin")
    # The GeoIPS config is frozen, so swap in a stand-in with only this path.
    test_config = SimpleNamespace(output_paths=SimpleNamespace(ancildat=tmp_path))
    monkeypatch.setattr(flat_binary, "config", test_config)
    return tmp_path


def test_reads_values_at_points(ancildat_dir):
    """Values come from the right records, in the input order."""
    # Rows of the file are 0-7, 8-15, 16-23, 24-31. Points cover three rows,
    # are out of order, and repeat one point.
    lats = [[0.0, 90.0, 45.0], [90.0, 45.0, 45.0]]
    lons = [[0.0, -180.0, -135.0], [-180.0, 90.0, -135.0]]
    start = datetime(2021, 7, 15, 13)
    xarray_obj = points_dataset(lats, lons, start_datetime=start, end_datetime=start)

    result = TinyReader().call(xarray_obj=xarray_obj)

    assert set(result) == {"tiny", "METADATA"}
    dataset = result["tiny"]
    np.testing.assert_array_equal(dataset["tiny"], [[29, 9, 18], [9, 23, 18]])
    assert dataset["tiny"].dtype == np.float64
    assert dataset["tiny"].dims == ("y", "x")
    np.testing.assert_array_equal(dataset["latitude"], lats)
    np.testing.assert_array_equal(dataset["longitude"], lons)
    assert dataset.attrs == result["METADATA"].attrs
    assert result["METADATA"].attrs["source_name"] == "tiny_ancildat"
    assert result["METADATA"].attrs["source_file_names"] == ["tiny.bin"]


def test_area_def_points_use_default_dims(ancildat_dir):
    """Points from an area get the dims `dim_0` and `dim_1`."""
    result = TinyReader().call(area_def=FakeArea([[90.0]], [[-180.0]]))
    assert result["tiny"]["tiny"].dims == ("dim_0", "dim_1")
    assert result["tiny"]["tiny"].item() == 9


def test_area_def_wins_over_xarray_obj(ancildat_dir, caplog):
    """When both are given, the area points are used and this is logged."""
    xarray_obj = points_dataset([[0.0]], [[0.0]])
    with caplog.at_level(logging.INFO):
        result = TinyReader().call(
            area_def=FakeArea([[90.0]], [[-180.0]]), xarray_obj=xarray_obj
        )
    assert result["tiny"]["tiny"].item() == 9
    assert "Using the area_def latitudes and longitudes" in caplog.text


def test_invalid_points_get_fill(ancildat_dir):
    """Points that are not finite give -999, even when no point is valid."""
    # -999 inputs are tested in test_indexing.py at the real grid steps. This
    # 45 degree test grid is too coarse for them: -999 rounds to line -20.
    lats = [[np.nan, -np.inf], [np.inf, 0.0]]
    lons = [[0.0, 0.0], [0.0, np.nan]]
    xarray_obj = points_dataset(lats, lons)
    result = TinyReader().call(xarray_obj=xarray_obj)
    np.testing.assert_array_equal(result["tiny"]["tiny"], -999)


def test_no_valid_points_does_not_open_file(ancildat_dir):
    """With no valid points the data file is not needed."""
    xarray_obj = points_dataset(
        [[np.nan]], [[0.0]], start_datetime=datetime(2020, 1, 1)
    )
    result = TinyReader().call(
        xarray_obj=xarray_obj, data_file=ancildat_dir / "missing.bin"
    )
    assert result["tiny"]["tiny"].item() == -999


def test_convert_values_skips_invalid_points(ancildat_dir):
    """Converted values stay -999 where the point is invalid."""
    xarray_obj = points_dataset([[90.0, np.nan]], [[-180.0, 0.0]])
    result = PlusOneReader().call(xarray_obj=xarray_obj)
    np.testing.assert_array_equal(result["tiny"]["tiny"], [[10, -999]])


def test_integer_grid_uses_integer_fill(ancildat_dir):
    """An integer grid returns integers with -999 fill."""
    xarray_obj = points_dataset([[90.0, np.nan]], [[-180.0, 0.0]])
    result = TinyIntReader().call(xarray_obj=xarray_obj)
    np.testing.assert_array_equal(result["tiny"]["tiny"], [[9, -999]])
    assert result["tiny"]["tiny"].dtype == np.int64


def test_column_past_row_end_reads_next_row(ancildat_dir):
    """Longitude 180 on file row 2 reads from row 3 (record 17)."""
    xarray_obj = points_dataset([[90.0]], [[180.0]])
    assert TinyReader().call(xarray_obj=xarray_obj)["tiny"]["tiny"].item() == 17


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
        TinyReader().call(xarray_obj=points_dataset([[90.0]], [[-180.0]]))


def test_wrong_file_size_raises(ancildat_dir):
    """A file that does not match the grid size raises an error."""
    np.arange(31, dtype="<f4").tofile(ancildat_dir / "tiny.bin")
    with pytest.raises(ValueError, match="is 124 bytes"):
        TinyReader().call(xarray_obj=points_dataset([[90.0]], [[-180.0]]))


def test_point_outside_file_raises(ancildat_dir):
    """A point south of the last row raises an error."""
    with pytest.raises(IndexError):
        TinyReader().call(xarray_obj=points_dataset([[-45.0]], [[0.0]]))


def test_metadata_only_does_not_read_data(ancildat_dir):
    """Only METADATA is returned, and the data file is not needed."""
    start = datetime(2021, 7, 15, 13)
    xarray_obj = points_dataset([[90.0]], [[-180.0]], start_datetime=start)
    result = TinyReader().call(
        metadata_only=True,
        xarray_obj=xarray_obj,
        data_file=ancildat_dir / "missing.bin",
    )
    assert set(result) == {"METADATA"}
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


def test_fnames_are_ignored(ancildat_dir, caplog):
    """Satellite files passed as fnames do not replace the data file."""
    xarray_obj = points_dataset([[90.0]], [[-180.0]])
    with caplog.at_level(logging.INFO):
        result = TinyReader().call(fnames=["satellite.nc"], xarray_obj=xarray_obj)
    assert result["tiny"]["tiny"].item() == 9
    assert "ignores fnames" in caplog.text


def test_data_file_replaces_default_file(ancildat_dir):
    """`data_file` picks a different data file."""
    other_file = ancildat_dir / "other.bin"
    np.arange(100, 132, dtype="<f4").tofile(other_file)
    xarray_obj = points_dataset([[90.0]], [[-180.0]])
    result = TinyReader().call(xarray_obj=xarray_obj, data_file=other_file)
    assert result["tiny"]["tiny"].item() == 109
    assert result["METADATA"].attrs["source_file_names"] == ["other.bin"]


def test_workflow_style_call_returns_data_tree(ancildat_dir):
    """Called like a workflow reader step, the satellite files are ignored."""
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


def test_missing_points_raise(ancildat_dir):
    """Without an area or an xarray_obj there are no points to read."""
    with pytest.raises(ValueError, match="needs area_def"):
        TinyReader().call()


def test_default_file_is_in_ancildat_folder():
    """The default file is inside the folder from the GeoIPS config."""
    expected = Path(geoips_config.output_paths.ancildat) / "tiny.bin"
    assert TinyReader().get_filename() == expected
