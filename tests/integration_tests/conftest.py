# # # This source code is subject to the license referenced at
# # # https://github.com/NRLMMD-GEOIPS.

"""Shared fixtures for the integration tests."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

# Outputs of the Fortran ancildat library, one `<dataset>__<case>.npz` file per
# case (arrays `lats`, `lons`, `output` and `code`). `manifest.json` lists how
# each case was made.
FORTRAN_COMPARISON_DIR = Path(__file__).parent / "data" / "fortran_comparison"

# Positions in the 4 x 8 edges grid, counted along each row from 0. Position 16
# has a NaN latitude and 17 a NaN longitude. This package gives -999 there,
# while the Fortran reads row 1 or column 1, so we expect -999 instead.
NAN_POSITIONS = [16, 17]


@pytest.fixture
def load_golden():
    """Return a loader for one Fortran comparison case.

    Returns
    -------
    function
        `load(dataset_name, case_name)` returns `(xarray_obj, expected)`:
        a dataset with the case's `latitude` and `longitude`, and the
        expected reader output. Edges cases expect -999 at `NAN_POSITIONS`.
    """

    def load(dataset_name, case_name):
        """Load one comparison case."""
        golden_path = FORTRAN_COMPARISON_DIR / f"{dataset_name}__{case_name}.npz"
        with np.load(golden_path) as golden:
            lats, lons, expected = golden["lats"], golden["lons"], golden["output"]
        if case_name.startswith("edges"):
            expected.flat[NAN_POSITIONS] = -999
        xarray_obj = xr.Dataset(
            {"latitude": (("y", "x"), lats), "longitude": (("y", "x"), lons)}
        )
        return xarray_obj, expected

    return load
