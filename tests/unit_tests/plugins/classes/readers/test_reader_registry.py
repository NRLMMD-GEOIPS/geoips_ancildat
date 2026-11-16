# # # This source code is subject to the license referenced at
# # # https://github.com/NRLMMD-GEOIPS.

"""pytest test-suite that checks every reader loads from the GeoIPS registry."""

import pytest
from geoips.interfaces import readers

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
    [
        CityLightsReaderPlugin,
        LandSeaMaskReaderPlugin,
        ElevationReaderPlugin,
        SurfaceTemperatureReaderPlugin,
    ],
    ids=lambda reader_class: reader_class.variable_name,
)
def test_reader_loads_from_geoips(reader_class):
    """The reader loads from the GeoIPS registry and passes its checks."""
    # GeoIPS checks the call arguments when it loads a plugin. For example,
    # `fnames` must have no default value. Needs `geoips config create-registries`.
    plugin = readers.get_plugin(reader_class.name)
    assert plugin.source_names == [reader_class.name]
