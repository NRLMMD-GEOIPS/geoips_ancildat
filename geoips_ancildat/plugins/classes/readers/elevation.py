# # # This source code is subject to the license referenced at
# # # https://github.com/NRLMMD-GEOIPS.

"""Reader for the elevation ancillary data file."""

import numpy as np

from geoips_ancildat.plugins.classes.readers.utils.flat_binary import (
    FlatBinaryReaderPlugin,
)
from geoips_ancildat.plugins.classes.readers.utils.indexing import fortran_grid_step


class ElevationReaderPlugin(FlatBinaryReaderPlugin):
    """Read surface elevation at the requested points.

    Values are mean elevation in meters from the GMTED2010 dataset on a 30 arc
    second grid. The file holds -32768 where it has no data, such as near the
    north pole.
    """

    name = "elevation_ancildat"
    source_names = [name]
    relative_path = "elevation/elevation.bin"
    num_lines = 21600
    num_samples = 43200
    lon_res = fortran_grid_step(1, 120)
    lat_res = fortran_grid_step(1, 120)
    file_dtype = "<i4"
    output_dtype = np.int64
    variable_name = "elevation"

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
        """Read elevation at the requested points.

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


PLUGIN_CLASS = ElevationReaderPlugin
