# # # This source code is subject to the license referenced at
# # # https://github.com/NRLMMD-GEOIPS.

"""Reader for the city lights ancillary data file."""

import numpy as np

from geoips_ancildat.plugins.classes.readers.utils.flat_binary import (
    FlatBinaryReaderPlugin,
)
from geoips_ancildat.plugins.classes.readers.utils.indexing import fortran_grid_step


class CityLightsReaderPlugin(FlatBinaryReaderPlugin):
    """Read night-time city lights at the requested points.

    Values are the 2015 yearly average radiance from the VIIRS Day/Night Band,
    in nanoWatts/cm2/sr, on a 15 arc second grid. Each value is cut to a whole
    number, like the Fortran. The source data covers 75 N to 65 S, so the file
    holds -999 north and south of that.
    """

    name = "city_lights_ancildat"
    source_names = [name]
    relative_path = "city_lights/city_lights.bin"
    num_lines = 43200
    num_samples = 86400
    lon_res = fortran_grid_step(1, 240)
    lat_res = fortran_grid_step(1, 240)
    file_dtype = "<f4"
    output_dtype = np.float64
    variable_name = "city_lights"

    def call(
        self,
        fnames,
        metadata_only=False,
        chans=None,
        area_def=None,
        self_register=False,
        xarray_obj=None,
        data_file=None,
    ):
        """Read city lights at the requested points.

        See `FlatBinaryReaderPlugin.read_points` for the arguments. GeoIPS
        requires `fnames` and `chans`, but both are ignored: the reader finds
        its own data file, which holds only one variable. Pass `fnames=None`
        when calling the reader yourself.
        """
        return self.read_points(
            fnames=fnames,
            data_file=data_file,
            metadata_only=metadata_only,
            area_def=area_def,
            xarray_obj=xarray_obj,
            self_register=self_register,
        )

    def convert_values(self, values):
        """Drop the fraction from each value, like the Fortran `int()`."""
        # Going through int64 also turns -0.0 into 0.0. np.trunc would keep -0.0.
        return values.astype(np.int64).astype(np.float64)


PLUGIN_CLASS = CityLightsReaderPlugin
