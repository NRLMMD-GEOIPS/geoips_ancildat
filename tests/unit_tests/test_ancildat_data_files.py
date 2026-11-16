# # # This source code is subject to the license referenced at
# # # https://github.com/NRLMMD-GEOIPS.

"""Check that the ancillary data files are installed.

This is the only check that the data is present. The readers do not check
when they run. Run with `--no-fail-on-missing-data` to skip instead of fail.
"""

import numpy as np
import pytest

from geoips_ancildat.plugins.classes.readers.city_lights import (
    CityLightsReaderPlugin,
)
from geoips_ancildat.plugins.classes.readers.elevation import ElevationReaderPlugin
from geoips_ancildat.plugins.classes.readers.land_sea_mask import (
    LandSeaMaskReaderPlugin,
)
from geoips_ancildat.plugins.classes.readers.surface_temperature import (
    SurfaceTemperatureReaderPlugin,
)


@pytest.mark.parametrize(
    "reader_class",
    [CityLightsReaderPlugin, LandSeaMaskReaderPlugin, ElevationReaderPlugin],
    ids=lambda reader_class: reader_class.variable_name,
)
def test_data_file_is_installed(reader_class, require_file):
    """The data file is present, can be read, and is the expected size."""
    reader = reader_class()
    path = reader.get_filename()
    require_file(path, f"{reader.name} data file")

    with open(path, "rb") as data_file:
        assert data_file.read(1), f"Cannot read {path}"
    item_size = np.dtype(reader.file_dtype).itemsize
    expected_size = reader.num_lines * reader.num_samples * item_size
    assert path.stat().st_size == expected_size


def test_surface_temperature_files_are_installed(require_file):
    """There is a MERRA file of the expected size for every hour of the year."""
    reader = SurfaceTemperatureReaderPlugin()
    paths = [
        reader.get_filename(
            relative_path=reader.relative_path.format(day_of_year=day, hour=hour)
        )
        for day in range(1, 366)
        for hour in range(24)
    ]
    require_file(paths[0], "MERRA data files")

    expected_size = reader.num_lines * reader.num_samples * 4
    # Check every file first, so a failure lists all the bad ones at once.
    bad_files = [
        path.name
        for path in paths
        if not path.is_file() or path.stat().st_size != expected_size
    ]
    assert not bad_files, f"Missing or wrong size: {bad_files}"
