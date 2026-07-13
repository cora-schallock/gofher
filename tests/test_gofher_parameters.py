"""Test all functions in gofher/gofher_parameters.py

The script is run using the commend: python -m pytest
"""


import pytest
import numpy as np
from pathlib import Path

from gofher.gofher_parameters import (
    GofherParameters,
    read_gofher_parameters_from_csv
)

#IMPORTANT: Make sure these folder/file names reflect test file structure:
TEST_GALAXY_DIR = "NGC2347_SDSS_psf4_background_256"
TEST_OUTPUT_DIR = "gofher_output"
TEST_CSV_FILE_NAME = "NGC2347_gofher_parameters.csv"

def _get_test_gofher_params_csv_path() -> str:
    p = Path(__file__).resolve().parent
    return str(p.joinpath("data", TEST_GALAXY_DIR, TEST_OUTPUT_DIR, TEST_CSV_FILE_NAME))

@pytest.mark.parametrize(
    "csv_path, expected_exception",
    [
        (1, ValueError),
        ({}, ValueError),
        ([], ValueError)
    ]
)
def test_read_gofher_parameters_exceptions(csv_path, expected_exception):
    """Tests exceptions expected from read_gofher_parameters_from_csv"""
    with pytest.raises(expected_exception):
        read_gofher_parameters_from_csv(csv_path)


def test_read_gofher_parameters():
    """Test read_gofher_parameters_from_csv
    
    Residual Tolerance:
        residule < 1e-6
    """
    RESIDUAL_TOLERANCE = 1e-6

    # Read in GofherParameters test CSV:
    gofher_params_csv_path = _get_test_gofher_params_csv_path()
    gofher_params = read_gofher_parameters_from_csv(gofher_params_csv_path)
    assert isinstance(gofher_params, GofherParameters)

    # Validate values:
    assert gofher_params.name == "NGC2347"
    assert gofher_params.ref_band == "r"
    assert gofher_params.shape == (419,419)

    h_residual = np.abs(gofher_params.h) - 196.3890961
    assert h_residual < RESIDUAL_TOLERANCE

    k_residual = np.abs(gofher_params.k - 250.4842873)
    assert k_residual < RESIDUAL_TOLERANCE

    a_residual = np.abs(gofher_params.a - 36.01770850625)
    assert  a_residual < RESIDUAL_TOLERANCE

    b_residual = np.abs(gofher_params.b - 21.639077692519116)
    assert b_residual < RESIDUAL_TOLERANCE

    theta_residual = np.abs(gofher_params.theta - -1.458819067)
    assert theta_residual < RESIDUAL_TOLERANCE

    input_c_residual = np.abs(
        gofher_params.sparcfire_input_c - 197.8890961) 
    assert input_c_residual < RESIDUAL_TOLERANCE

    input_r_residual = np.abs(
        gofher_params.sparcfire_input_r - 251.9842873)
    assert input_r_residual < RESIDUAL_TOLERANCE

    sparcfire_disk_maj_axis_len_residual = np.abs(
        gofher_params.sparcfire_disk_maj_axis_len - 236.5441021) 
    assert sparcfire_disk_maj_axis_len_residual < RESIDUAL_TOLERANCE

    sparcfire_disk_min_axis_len_residual = np.abs(
        gofher_params.sparcfire_disk_min_axis_len - 142.1133219)
    assert sparcfire_disk_min_axis_len_residual < RESIDUAL_TOLERANCE
    
    sparcfire_disk_maj_axis_angle_residual = np.abs(
        gofher_params.sparcfire_disk_maj_axis_angle - 1.458819067)
    assert sparcfire_disk_maj_axis_angle_residual < RESIDUAL_TOLERANCE

    sparcfire_bulge_maj_axis_len_residual = np.abs(
        gofher_params.sparcfire_bulge_maj_axis_len - 17.19918865)
    assert sparcfire_bulge_maj_axis_len_residual < RESIDUAL_TOLERANCE

    sparcfire_bulge_axis_ratio_residual = np.abs(
        gofher_params.sparcfire_bulge_axis_ratio - 0.8911384595)
    assert sparcfire_bulge_axis_ratio_residual < RESIDUAL_TOLERANCE
    
    sparcfire_bulge_axis_angle_residual = np.abs(
        gofher_params.sparcfire_bulge_axis_angle - 1.06518089) 
    assert sparcfire_bulge_axis_angle_residual < RESIDUAL_TOLERANCE

#TODO: test remainning gofher_parameter methods