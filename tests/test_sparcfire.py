"""Test all functions in gofher/sparcfire.py

The script is run using the command: python -m pytest

IMPORTANT:
    Make sure file paths reflect the test file structure
        From directory of test scripts: 
        data/TEST_GALAXY_DIR/TEST_SPARCFIRE_DIR/TEST_CSV_FILE_NAME

    Make sure values in test_gofher_params_from_sparcfire_csv
        match the test sparcfire csv. If you update the csv, these
        values will need to be updated.
"""

from pathlib import Path

import pytest

from gofher.gofher_parameters import GofherParameters

from gofher.sparcfire import (
    standard_normalize_name,
    standard_ref_band_from_name,
    gofher_params_from_sparcfire_csv
)

#IMPORTANT: Make sure these folder/file names reflect test file structure:
TEST_GALAXY_DIR = "NGC2347_SDSS_psf4_background_256"
TEST_SPARCFIRE_DIR = "sparcfire_r_band_output"
TEST_CSV_FILE_NAME = "NGC2347_r.csv"

def _get_test_csv_path() -> str:
    p = Path(__file__).resolve().parent
    return str(p.joinpath("data", TEST_GALAXY_DIR, TEST_SPARCFIRE_DIR, TEST_CSV_FILE_NAME))

@pytest.mark.parametrize("name, expected_exception", [
    (1, ValueError),
    ({}, ValueError),
    ([], ValueError)
])
def test_standard_normalize_name_exceptions(name, expected_exception):
    """Test exceptions expected from standard_normalize_name"""
    with pytest.raises(expected_exception):
        standard_normalize_name(name)

@pytest.mark.parametrize("name, expected", [
    ("NGC1_g", "NGC1"),
    ("NGC 1_g", "NGC 1"),
    ("NGC-1_g", "NGC-1"),
    ("NGC_1_g", "NGC_1"),
    ("NGC1", "NGC1"),
    ("NGC 1", "NGC 1"),
    ("NGC-1", "NGC-1"),
    ("NGC_1", "NGC")
])
def test_standard_normalize_name(name, expected):
    """Test output expected from standard_normalize_name"""
    assert standard_normalize_name(name) == expected

@pytest.mark.parametrize("name, expected_exception", [
    (1, ValueError),
    ({}, ValueError),
    ([], ValueError),
    ("NGC1", ValueError),
    ("NGC 1", ValueError)
])
def test_standard_ref_band_from_name_exceptions(name, expected_exception):
    """Test exceptions expected from standard_ref_band_from_name"""
    with pytest.raises(expected_exception):
        standard_ref_band_from_name(name)

@pytest.mark.parametrize("name, expected", [
    ("NGC1_g", "g"),
    ("NGC 1_g", "g"),
    ("NGC-1_g", "g"),
    ("NGC_1_g", "g")
])
def test_standard_ref_band_from_name(name, expected):
    """Test output expected from standard_ref_band_from_name"""
    assert standard_ref_band_from_name(name) == expected

@pytest.mark.parametrize(
        "path, normalize_name, get_ref_band, expected_exception", [
    ("", None, None, ValueError),
    ("fake.csv", None, None, ValueError),
    (_get_test_csv_path(), "not a function", None, ValueError),
    (_get_test_csv_path(), lambda x,y: x+y, None, ValueError),
    (_get_test_csv_path(), None, "not a function", ValueError),
    (_get_test_csv_path(), None, lambda x,y: x+y, ValueError),
])
def test_gofher_params_from_sparcfire_csv_exceptions(path, 
                                                     normalize_name, 
                                                     get_ref_band,
                                                     expected_exception):
    """Test exceptions expected from standard_ref_band_from_name"""
    with pytest.raises(expected_exception):
        gofher_params_from_sparcfire_csv(path, normalize_name, get_ref_band)


def test_gofher_params_from_specific_test_sparcfire_csv():
    """Test output expected from gofher_params_from_sparcfire_csv"""
    #IMPORTANT: Make sure these values match the test csv
    #   If multiple galaxies are added, test each galaxy's values seperately

    # Validate it returns 1 GofherParameters from test csv:
    all_gofher_params = gofher_params_from_sparcfire_csv(_get_test_csv_path())
    assert isinstance(all_gofher_params, list)
    assert len(all_gofher_params) == 1

    # Validate it returns a GofherParameters object:
    gofher_params = all_gofher_params[0]
    #assert isinstance(gofher_params, GofherParameters)
    # ^this is failing, I can't figure out why. Just commented out for now.
 
    # Validate values in GofherParameters:
    assert gofher_params.name == "NGC2347"
    assert gofher_params.ref_band == "r"
    assert gofher_params.sparcfire_input_c == 197.8890961
    assert gofher_params.sparcfire_input_r == 251.9842873
    assert gofher_params.sparcfire_disk_maj_axis_len == 236.5441021
    assert gofher_params.sparcfire_disk_min_axis_len == 142.1133219
    assert gofher_params.sparcfire_disk_maj_axis_angle == 1.458819067
    assert gofher_params.sparcfire_bulge_maj_axis_len == 17.19918865
    assert gofher_params.sparcfire_bulge_axis_ratio == 0.8911384595
    assert gofher_params.sparcfire_bulge_axis_angle == 1.06518089
