# # # This source code is subject to the license referenced at
# # # https://github.com/NRLMMD-GEOIPS.

"""Pytest configuration for test data validation.

Provides command-line options and fixtures to control test behavior
when required test data files are missing.
"""

import os
from pathlib import Path

import pytest


def pytest_addoption(parser):
    """Add custom command-line options to pytest.

    Parameters
    ----------
    parser : pytest.Parser
        Pytest command-line argument parser.

    Notes
    -----
    Adds --no-fail-on-missing-data flag to skip tests instead of
    failing when test data is unavailable.
    """
    try:
        parser.addoption(
            "--no-fail-on-missing-data",
            action="store_true",
            default=False,
            help="Fail tests if test data is missing",
        )
    except ValueError as resp:
        if "option names {'--no-fail-on-missing-data'} already added" in str(resp):
            pass


@pytest.fixture
def fail_on_missing_data(request):
    """Determine whether to fail tests on missing data.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Pytest fixture request object.

    Returns
    -------
    bool
        True if tests should fail on missing data, False to skip instead.
    """
    return not request.config.getoption("--no-fail-on-missing-data")


def fail_or_skip(message, fail_on_missing_data):
    """Fail the test, or skip it when run with `--no-fail-on-missing-data`."""
    if fail_on_missing_data:
        pytest.fail(message)
    pytest.skip(message)


@pytest.fixture
def require_file(fail_on_missing_data):
    """Return a check that a needed data file is installed.

    Returns
    -------
    function
        `check(path, description)` fails or skips the test if `path` is not
        a file.
    """

    def check(path, description):
        """Fail or skip if the file is missing."""
        if not Path(path).is_file():
            fail_or_skip(
                f"Missing {description}: {path}. See the README for how to "
                "install the ancillary data.",
                fail_on_missing_data,
            )

    return check


@pytest.fixture
def golden_dir(fail_on_missing_data):
    """Return the folder that holds the Fortran golden outputs.

    The goldens stay outside the repository until the team decides where to
    keep them. Point `GEOIPS_ANCILDAT_GOLDEN_DIR` at the folder for now. Once
    the location is settled, change this fixture to use it.

    Returns
    -------
    pathlib.Path
        Folder with one `<dataset>__<grid>.npz` file per golden case.
    """
    folder = os.environ.get("GEOIPS_ANCILDAT_GOLDEN_DIR")
    if not folder or not Path(folder).is_dir():
        fail_or_skip(
            "Set GEOIPS_ANCILDAT_GOLDEN_DIR to the folder of Fortran golden "
            "outputs.",
            fail_on_missing_data,
        )
    return Path(folder)
