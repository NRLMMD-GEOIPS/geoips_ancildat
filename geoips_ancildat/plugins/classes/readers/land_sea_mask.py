# # # This source code is subject to the license referenced at
# # # https://github.com/NRLMMD-GEOIPS.

"""Reader for the land/sea mask ancillary data file."""

import numpy as np

from geoips_ancildat.plugins.classes.readers.utils.flat_binary import (
    FlatBinaryReaderPlugin,
)
from geoips_ancildat.plugins.classes.readers.utils.indexing import fortran_grid_step


class LandSeaMaskReaderPlugin(FlatBinaryReaderPlugin):
    """Read the land/sea mask at the requested points.

    The mask comes from a 2001 MODIS land/water mask on a 30 arc second grid.
    Its values are:

    - 0: shallow ocean
    - 1: land
    - 2: coastline
    - 3: shallow inland water
    - 4: ephemeral water
    - 5: deep inland water
    - 6: moderate ocean
    - 7: deep ocean
    """

    name = "land_sea_mask_ancildat"
    source_names = [name]
    relative_path = "lw_mask/lw_geo_2001001_v03m.bin"
    num_lines = 21600
    num_samples = 43200
    lon_res = fortran_grid_step(1, 120)
    lat_res = fortran_grid_step(1, 120)
    file_dtype = "|u1"
    output_dtype = np.int64
    variable_name = "land_sea_mask"

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
        """Read the land/sea mask at the requested points.

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


PLUGIN_CLASS = LandSeaMaskReaderPlugin
