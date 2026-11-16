# # # This source code is subject to the license referenced at
# # # https://github.com/NRLMMD-GEOIPS.

"""pytest test-suite that compares each reader with the Fortran ancildat.

Each reader runs on the points of the cases in `data/fortran_comparison` and
must give exactly the Fortran output. These are integration tests because
they read the installed ancillary data files.

The grids are:

- southeast_asia: 24 x 24 points made of four 12 x 12 patches: Lake Toba and
  Medan (Sumatra), Jakarta and West Java, Mount Kinabalu (Borneo), and Manila,
  Laguna de Bay and Taal (Luzon). Has sea, coast, lakes, volcanoes, mountains
  up to 2035 m, and lit cities.
- fine: 16 x 16 points closer together than the grid step, on the Manila Bay
  shore, to catch any point landing in the wrong cell.
- edges: 4 x 8 special points such as the poles, the dateline, the last row
  of the file, and NaN and -999 inputs.
- global: 12 x 24 points over the whole globe (surface temperature only).

The edges grid also covers no-data values stored in the files, which must come
back unchanged: city_lights -999 near the poles (positions 0-6, 18 and 19) and
elevation -32768 near the north pole (positions 0, 18 and 19).
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

pytestmark = [pytest.mark.integration, pytest.mark.full]

STATIC_READER_CLASSES = [
    CityLightsReaderPlugin,
    LandSeaMaskReaderPlugin,
    ElevationReaderPlugin,
]
STATIC_GRIDS = ["southeast_asia", "fine", "edges"]

TEMPERATURE_FILE_SUFFIX = "_ts_MERRA.tdf_540_samples_361_lines.bin"
# Case -> (time given to the Fortran, file it should read).
TEMPERATURE_CASES = {
    # A summer afternoon. Minutes are ignored, so 13:45 reads hour 13.
    "global_20210715_13": ("2021-07-15T13:45:00Z", "196_13"),
    # The first file of the year.
    "global_20200101_00": ("2020-01-01T00:00:00Z", "001_00"),
    # Leap year: March 1 is day 61, not 60.
    "global_20200301_06": ("2020-03-01T06:00:00Z", "061_06"),
    # The last file of the year. 23:59:59 still reads hour 23.
    "global_20211231_23": ("2021-12-31T23:59:59Z", "365_23"),
    "fine": ("2021-07-15T13:45:00Z", "196_13"),
    "edges": ("2021-07-15T13:45:00Z", "196_13"),
}


# 1. Static readers: city_lights, land_sea_mask and elevation.
@pytest.mark.parametrize(
    "reader_class, grid",
    [
        pytest.param(reader_class, grid, id=f"{reader_class.variable_name}-{grid}")
        for reader_class in STATIC_READER_CLASSES
        for grid in STATIC_GRIDS
    ],
)
def test_static_reader_matches_fortran(reader_class, grid, load_golden, require_file):
    """Every point equals the Fortran output, with the same dtype."""
    reader = reader_class()
    require_file(reader.get_filename(), f"{reader.name} data file")
    xarray_obj, expected = load_golden(reader.variable_name, grid)

    result = reader.call(fnames=None, xarray_obj=xarray_obj)
    output = result[reader.variable_name][reader.variable_name].values

    assert output.dtype == expected.dtype
    np.testing.assert_array_equal(output, expected)


# 2. surface_temperature, which must also read the expected file.
@pytest.mark.parametrize("case_name", TEMPERATURE_CASES)
def test_surface_temperature_matches_fortran(case_name, load_golden, require_file):
    """Every point equals the Fortran output, with the same dtype."""
    start_datetime, day_and_hour = TEMPERATURE_CASES[case_name]
    file_name = f"{day_and_hour}{TEMPERATURE_FILE_SUFFIX}"
    reader = SurfaceTemperatureReaderPlugin()
    require_file(
        reader.get_filename(relative_path=f"merra/{file_name}"), "MERRA data file"
    )
    xarray_obj, expected = load_golden("surface_temperature", case_name)

    # Pass the time as a string, like debra does today.
    result = reader.call(
        fnames=None, xarray_obj=xarray_obj, start_datetime=start_datetime
    )
    output = result["surface_temperature"]["surface_temperature"].values

    assert result["METADATA"].attrs["source_file_names"] == [file_name]
    assert output.dtype == expected.dtype
    np.testing.assert_array_equal(output, expected)
