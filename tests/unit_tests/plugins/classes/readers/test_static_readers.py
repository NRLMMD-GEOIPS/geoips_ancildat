# # # This source code is subject to the license referenced at
# # # https://github.com/NRLMMD-GEOIPS.

"""pytest test-suite for the city_lights, land_sea_mask and elevation readers.

These are the static readers: their data does not change with time, so each
reads one fixed file. Each reader runs on the points of the Fortran golden
cases and must give exactly the Fortran output.

The golden grids are:

- southeast_asia: 24 x 24 points made of four 12 x 12 patches: Lake Toba and
  Medan (Sumatra), Jakarta and West Java, Mount Kinabalu (Borneo), and Manila,
  Laguna de Bay and Taal (Luzon). Has sea, coast, lakes, volcanoes, mountains
  up to 2035 m, and lit cities.
- fine: 16 x 16 points closer together than the grid step, on the Manila Bay
  shore, to catch any point landing in the wrong cell.
- edges: 4 x 8 special points such as the poles, the dateline, the last row
  of the file, and NaN and -999 inputs.

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

READER_CLASSES = [
    CityLightsReaderPlugin,
    LandSeaMaskReaderPlugin,
    ElevationReaderPlugin,
]
GRIDS = ["southeast_asia", "fine", "edges"]


@pytest.mark.parametrize(
    "reader_class, grid",
    [
        pytest.param(reader_class, grid, id=f"{reader_class.variable_name}-{grid}")
        for reader_class in READER_CLASSES
        for grid in GRIDS
    ],
)
def test_matches_fortran(reader_class, grid, load_golden, require_file):
    """Every point equals the Fortran output, with the same dtype."""
    reader = reader_class()
    require_file(reader.get_filename(), f"{reader.name} data file")
    xarray_obj, expected = load_golden(reader.variable_name, grid)

    result = reader.call(fnames=None, xarray_obj=xarray_obj)
    output = result[reader.variable_name][reader.variable_name].values

    assert output.dtype == expected.dtype
    np.testing.assert_array_equal(output, expected)
