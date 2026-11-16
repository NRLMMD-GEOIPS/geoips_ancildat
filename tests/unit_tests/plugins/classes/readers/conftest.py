# # # This source code is subject to the license referenced at
# # # https://github.com/NRLMMD-GEOIPS.

"""Shared fixtures for the reader tests."""

import numpy as np
import pytest
import xarray as xr

# Positions in the 4 x 8 edges grid, counted along each row from 0. Position 16
# has a NaN latitude and 17 a NaN longitude. This package gives -999 there,
# while the Fortran reads row 1 or column 1, so we expect -999 instead.
NAN_POSITIONS = [16, 17]


@pytest.fixture
def load_golden(golden_dir, require_file):
    """Return a loader for one Fortran golden case.

    Returns
    -------
    function
        `load(dataset_name, case_name)` returns `(xarray_obj, expected)`:
        a dataset with the case's `latitude` and `longitude`, and the
        expected reader output. Edges cases expect -999 at `NAN_POSITIONS`.
    """

    def load(dataset_name, case_name):
        """Load one golden case."""
        golden_path = golden_dir / f"{dataset_name}__{case_name}.npz"
        require_file(golden_path, "Fortran golden file")
        with np.load(golden_path) as golden:
            lats, lons, expected = golden["lats"], golden["lons"], golden["output"]
        if case_name.startswith("edges"):
            expected.flat[NAN_POSITIONS] = -999
        xarray_obj = xr.Dataset(
            {"latitude": (("y", "x"), lats), "longitude": (("y", "x"), lons)}
        )
        return xarray_obj, expected

    return load
