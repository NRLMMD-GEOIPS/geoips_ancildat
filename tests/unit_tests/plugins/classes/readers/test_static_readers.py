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
import xarray as xr
from geoips.interfaces import readers

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

# Positions in the 4 x 8 edges grid, counted along each row from 0. Position 16
# has a NaN latitude and 17 a NaN longitude. This package gives -999 there,
# while the Fortran reads row 1 or column 1, so we expect -999 instead.
NAN_POSITIONS = [16, 17]


@pytest.mark.parametrize(
    "reader_class, grid",
    [
        pytest.param(reader_class, grid, id=f"{reader_class.variable_name}-{grid}")
        for reader_class in READER_CLASSES
        for grid in GRIDS
    ],
)
def test_matches_fortran(reader_class, grid, golden_dir, require_file):
    """Every point equals the Fortran output, with the same dtype."""
    reader = reader_class()
    require_file(reader.get_filename(), f"{reader.name} data file")
    golden_path = golden_dir / f"{reader.variable_name}__{grid}.npz"
    require_file(golden_path, "Fortran golden file")
    with np.load(golden_path) as golden:
        lats, lons, expected = golden["lats"], golden["lons"], golden["output"]
    if grid == "edges":
        expected.flat[NAN_POSITIONS] = -999

    xarray_obj = xr.Dataset(
        {"latitude": (("y", "x"), lats), "longitude": (("y", "x"), lons)}
    )
    result = reader.call(fnames=None, xarray_obj=xarray_obj)
    output = result[reader.variable_name][reader.variable_name].values

    assert output.dtype == expected.dtype
    np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize(
    "reader_class",
    READER_CLASSES,
    ids=lambda reader_class: reader_class.variable_name,
)
def test_reader_loads_from_geoips(reader_class):
    """The reader loads from the GeoIPS registry and passes its checks."""
    # GeoIPS checks the call arguments when it loads a plugin. For example,
    # `fnames` must have no default value. Needs `geoips config create-registries`.
    plugin = readers.get_plugin(reader_class.name)
    assert plugin.source_names == [reader_class.name]
