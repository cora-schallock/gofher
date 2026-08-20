"""Test all functions in gofher/gofher_parameters.py

The script is run using the commend: python -m pytest
"""

from pathlib import Path

import pytest
import numpy as np


from gofher.gofher_parameters import (
    GofherParameters,
    read_gofher_parameters_from_csv
)

from gofher.utils import (
    is_2d_bool_array
)

RESIDUAL_TOLERANCE = 1e-6

#IMPORTANT: Make sure these folder/file names reflect test file structure:
TEST_GALAXY_DIR = "NGC2347_SDSS_psf4_background_256"
TEST_OUTPUT_DIR = "gofher_output"
TEST_CSV_FILE_NAME = "NGC2347_gofher_parameters.csv"

def _get_test_gofher_params_csv_path() -> str:
    p = Path(__file__).resolve().parent
    return str(p.joinpath("data", TEST_GALAXY_DIR, TEST_OUTPUT_DIR, TEST_CSV_FILE_NAME))

def validate_gofher_param_specific_test(gofher_params: GofherParameters):
    """Test the specific Gofher parameters expected from the specific
        test csv

        Important: All values are from original SpArcFiRe CSV
        #   so if test_csv_changes this needs to be update!
    """

    # Validate test values:
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

@pytest.mark.parametrize(
    "c, r, ang, dmaj, dmin, bmaj, f, expected_exception",
    [
        ("", 50, 0.0, 10.0, 5.0, 2.0, 0.5, TypeError), #c wrong type
        (50, [], 0.0, 10.0, 5.0, 2.0, 0.5, TypeError), #r wrong type
        (50, 50, {}, 10.0, 5.0, 2.0, 0.5, TypeError), #disk angle wrong type
        (50, 50, 0.0, "", 5.0, 2.0, 0.5, TypeError), #disk maj. wrong type
        (50, 50, 0.0, 10.0, [], 2.0, 0.5, TypeError), #disk min. wrong type
        (50, 50, 0.0, 10.0, 5.0, {}, 0.5, TypeError), #bulge maj. wrong type
        (50, 50, 0.0, 10.0, 5.0, 200.0, 0.5, ValueError), #bulge maj. too big
        (50, 50, 0.0, 10.0, 5.0, 2.0, "", TypeError), #bulge_disk_f wrong type
        (50, 50, 0.0, 10.0, 5.0, 2.0, -0.25, ValueError), #bulge_disk_f too small,
        (50, 50, 0.0, 10.0, 5.0, 2.0, 1.5, ValueError), #bulge_disk_f too large
    ]
)
def test_calculate_from_sparcfire_exceptions(c, r, ang, 
                                             dmaj, dmin, 
                                             bmaj, f, 
                                             expected_exception):
    """Test exceptions from GofherParameters.calculate_from_sparcfire()"""
    # Create a gofher parameters object and set the values provided:
    # Note: bulge_disk_f is used when calling calculate_from_sparcfire()
    #   This is done to allow for multiple runs without rerunning code
    #   to red in gofher parameters.
    gofher_params = GofherParameters()
    gofher_params.sparcfire_input_c = c
    gofher_params.sparcfire_input_r = r
    gofher_params.sparcfire_disk_maj_axis_angle = ang
    gofher_params.sparcfire_disk_maj_axis_len = dmaj
    gofher_params.sparcfire_disk_min_axis_len = dmin
    gofher_params.sparcfire_bulge_maj_axis_len = bmaj
    
    with pytest.raises(expected_exception):
        gofher_params.calculate_from_sparcfire(f)

def test_calculate_from_sparcfire_bulge_disk_f_scale():
    """Test bulge_disk_f from GofherParameters.calculate_from_sparcfire()"""

    # Initalize gofher_params with some arbitray allowed values:
    gofher_params = GofherParameters()
    gofher_params.sparcfire_input_c = 50.0
    gofher_params.sparcfire_input_r = 50.0
    gofher_params.sparcfire_disk_maj_axis_angle = 0.0

    # For this test we use:
    #   * disk_axis_ratio = 1/4
    #   * bulge_maj/disk_maj = 1/5
    #   * we are going to step by 0.25 from 0.0 to 1.0 (inclusive), 
    #   *   so numbers are 'nice'
    gofher_params.sparcfire_disk_maj_axis_len = 200.0
    gofher_params.sparcfire_disk_min_axis_len = 50.0
    gofher_params.sparcfire_bulge_maj_axis_len = 40.0

    # Validate f = 0.0, 0.25, 0.5, 0.75, 1.0
    for i in range(0,5):
        f = 0.0 + i/4 #Programmer note: written like this for assert statement below
        gofher_params.calculate_from_sparcfire(f)

        # Important: Note - sparcfire maj/min are halved for a/b
        expected_a = gofher_params.sparcfire_bulge_maj_axis_len * (1+i) * 0.5
        expected_b = gofher_params.a * 0.25

        assert gofher_params.a == expected_a
        assert gofher_params.b == expected_b

def test_calculate_from_sparcfire():
    """Test calculate from sparcifre"""
    # Initalize a gofher_param with specific values:
    gofher_params = GofherParameters()
    gofher_params.sparcfire_input_c = 101.5
    gofher_params.sparcfire_input_r = 102.5
    gofher_params.sparcfire_disk_maj_axis_angle = -np.pi/4

    # for this test:
    #   * disk_maj = 105.0
    #   * disk_min = 21.0
    #   * bulge_maj = 5.0
    #   Therefore: disk_axis_ratio = 1/5
    #       and differnce in disk_maj and bulge_maj = 100
    gofher_params.sparcfire_disk_maj_axis_len = 210.0
    gofher_params.sparcfire_disk_min_axis_len = 42.0
    gofher_params.sparcfire_bulge_maj_axis_len = 10.0

    # We these specific test numbers, if we use bulge_disk_f = 0.5
    #   Then a = 5.0 + (100.0/2) = 55.0
    #        b = a * 1/5 = 55.0/5.0 = 11.0
    gofher_params.calculate_from_sparcfire(0.5)

    # Verify values are what we expect:
    # Impoartant: Note - Sparcfire maj/min are halved
    assert np.abs(gofher_params.h - 100.0) < RESIDUAL_TOLERANCE
    assert np.abs(gofher_params.k - 101.0) < RESIDUAL_TOLERANCE
    assert np.abs(gofher_params.theta - np.pi/4) < RESIDUAL_TOLERANCE
    assert np.abs(gofher_params.a - 55.0) < RESIDUAL_TOLERANCE
    assert np.abs(gofher_params.b - 11.0) < RESIDUAL_TOLERANCE

@pytest.mark.parametrize(
    "r, shape, h, k, a, b, theta, expected_excpetion",
    [
        ("",(100,100), 50, 50, 20, 12, 0.0, ValueError), #wrong type r
        (0.0,(100,100), 50, 50, 20, 12, 0.0, ValueError), #r too small
        (1.0, "", 50, 50, 20, 12, 0.0, ValueError), #wrong type shape
        (0.0,(100, 100, 2), 50, 50, 20, 12, 0.0, ValueError), #not 2D shape
        (0.0,(100, 100), "", 50, 20, 12, 0.0, ValueError), #h wrong type
        (0.0,(100, 100), -5, 50, 20, 12, 0.0, ValueError), #h too small
        (0.0,(100, 100), 50, "", 20, 12, 0.0, ValueError), #k wrong type
        (0.0,(100, 100), 50, -5, 20, 12, 0.0, ValueError), #k too small
        (0.0,(100, 100), 50, 50, "", 12, 0.0, ValueError), #a wrong type
        (0.0,(100, 100), 50, 50, 0, 12, 0.0, ValueError), #a too small
        (0.0,(100, 100), 50, 50, 20, "", 0.0, ValueError), #b wrong type
        (0.0,(100, 100), 50, 50, 20, 0, 0.0, ValueError), #b too small
        (0.0,(100, 100), 50, 50, 20, 12, "", ValueError) #theta wrong type
    ]
)
def test_create_ellipse_mask_exceptions(r, shape, h, k, a, b, theta, expected_excpetion):
    """Test exceptions of GofherParameters.create_ellipse_mask()"""

    # Create a gofher parameter and set the values:
    gofher_params = GofherParameters()
    gofher_params.shape = shape
    gofher_params.h = h
    gofher_params.k = k
    gofher_params.a = a
    gofher_params.b = b
    gofher_params.theta = theta

    # Validate that exception occurs:
    with pytest.raises(expected_excpetion):
        gofher_params.create_ellipse_mask(r)

def test_create_ellipse_mask():
    """Test GofherParameters.create_ellipse_mask()"""
    
    # Create a gofher parameter and set the values:
    gofher_params = GofherParameters()
    gofher_params.shape = (100,100)
    gofher_params.h = 50
    gofher_params.k = 50
    gofher_params.a = 20
    gofher_params.b = 18
    gofher_params.theta = np.pi/6

    ellipse_mask = gofher_params.create_ellipse_mask()
    assert is_2d_bool_array(ellipse_mask)
    assert ellipse_mask.shape == (100,100)

@pytest.mark.parametrize(
    "shape, h, k, theta, expected_excpetion",
    [
        ("", 50, 50, 0.0, ValueError), #wrong type shape
        ((100,100,2), 50, 50, 0.0, ValueError), #shape not 2D
        ((100,100), "", 50, 0.0, ValueError), #wrong type h
        ((100,100), 50, "", 0.0, ValueError), #wrong type k
        ((100,100), 50, 50, "", ValueError) #wrong type theta
    ]
)
def test_create_bisection_masks_exceptions(shape, h, k, theta, expected_excpetion):
    """Test exceptions of GofherParameters.create_bisection_mask()"""

    # Create a gofher parameter and set the values:
    gofher_params = GofherParameters()
    gofher_params.shape = shape
    gofher_params.h = h
    gofher_params.k = k
    gofher_params.theta = theta

    # Validate that exception occurs:
    with pytest.raises(expected_excpetion):
        gofher_params.create_bisection_masks()

def test_create_bisection_masks():
    """Test GofherParamaters.create_bisection_masks()"""

    # Create a gofher parameter and set the values:
    gofher_params = GofherParameters()
    gofher_params.shape = (100,100)
    gofher_params.h = 50
    gofher_params.k = 50
    gofher_params.theta = np.pi/6
    
    bisection_masks = gofher_params.create_bisection_masks()
    assert len(bisection_masks) == 2

    for mask in bisection_masks:
        assert is_2d_bool_array(mask)
        assert mask.shape == (100,100)

@pytest.mark.parametrize(
    "sweep, shape, h, k, theta, expected_excpetion",
    [
        ("", (100,100), 50, 50, 0.0, ValueError), #wrong type sweep
        (-np.pi/4, (100,100), 50, 50, 0.0, ValueError), #sweep too small
        (np.pi, (100,100), 50, 50, 0.0, ValueError), #sweep too big
        (np.pi/4, "", 50, 50, 0.0, ValueError), #wrong type shape
        (np.pi/4, (100,100,2), 50, 50, 0.0, ValueError), #shape not 2D
        (np.pi/4, (100,100), "", 50, 0.0, ValueError), #wrong type h
        (np.pi/4, (100,100), 50, "", 0.0, ValueError), #wrong type k
        (np.pi/4, (100,100), 50, 50, "", ValueError) #wrong type theta
    ]
)
def create_near_major_and_minor_axis_mask_exceptions(sweep, shape, 
                                                     h, k, theta, 
                                                     expected_exception):

    """Test exceptions from GofherParameter.create_major_axis_mask() and 
    GofherParameter.create_minor_axis_mask()
    
    Programmer Note: These two functions have same restrictions of values 
    and should raise same exceptions, hence why tested together
    """
    
    # Create a gofher parameter and add values:
    gofher_param = GofherParameters()
    gofher_param.shape = shape
    gofher_param.h = h
    gofher_param.k = k
    gofher_param.theta = theta

    # Verify major axis raises exception:
    with pytest.raises(expected_exception):
        gofher_param.create_near_major_axis_mask(sweep)

    # Verify minor axis raises exception:
    with pytest.raises(expected_exception):
        gofher_param.create_near_minor_axis_mask(sweep)

def create_near_major_and_minor_axis_mask():
    """Test exceptions from GofherParameter.create_major_axis_mask() and 
        GofherParameter.create_minor_axis_mask()
        
        Programmer Note: These two functions have same restrictions of values 
        and should raise same exceptions, hence why tested together"""

    # Create a gofher parameter and add values:
    gofher_param = GofherParameters()
    gofher_param.shape = (100,100)
    gofher_param.h = 49.5
    gofher_param.k = 49.5
    gofher_param.theta = 0.0

    # Create the near major axis mask and verify it is 2D boolean mask
    #   of correct shape.
    maj_mask = gofher_param.create_near_major_axis_mask(np.pi/4)
    assert is_2d_bool_array(maj_mask)
    assert maj_mask.shape == gofher_param.shape

    # Create the near major axis mask and verify it is 2D boolean mask
    #   of correct shape.
    min_mask = gofher_param.create_near_major_axis_mask(np.pi/4)
    assert is_2d_bool_array(min_mask)
    assert min_mask.shape == gofher_param.shape

@pytest.mark.parametrize(
    "csv_path, expected_exception",
    [
        (1, TypeError),
        ({}, TypeError),
        ("not_a_csv.txt", ValueError)
    ]
)
def test_output_to_csv_exceptions(csv_path, expected_exception):
    """Tests exceptions expected from gofher_parameters.output_to_csv()"""
    gofher_params = GofherParameters()
    with pytest.raises(expected_exception):
        gofher_params.output_to_csv(csv_path)

def test_output_to_csv():
    """Test gofher_parameters.output_to_csv()"""
    # Specify a file path for the test case:
    test_csv_path = "test_output_to_csv.csv"

    # Read in the gofher parameters csv:
    gofher_params_csv_path = _get_test_gofher_params_csv_path()
    gofher_params = read_gofher_parameters_from_csv(gofher_params_csv_path)
    assert isinstance(gofher_params, GofherParameters)

    # Now output it to a new csv:
    gofher_params.output_to_csv(test_csv_path)
    assert Path.is_file(test_csv_path)

    # Read in the new csv:
    gofher_params_test = read_gofher_parameters_from_csv(test_csv_path)
    assert isinstance(gofher_params, GofherParameters)

    # Validate the values in new csv
    #   This is assuring values in csv are same as new csv
    validate_gofher_param_specific_test(gofher_params_test)

    # Cleanup the test file path:
    Path.unlink(test_csv_path)


@pytest.mark.parametrize(
    "csv_path, expected_exception",
    [
        (1, TypeError),
        ("not_a_csv.txt", ValueError),
        ("missing_csv.csv", ValueError)
    ]
)
def test_read_gofher_parameters_exceptions(csv_path, expected_exception):
    """Tests exceptions expected from read_gofher_parameters_from_csv"""
    with pytest.raises(expected_exception):
        read_gofher_parameters_from_csv(csv_path)


def test_read_gofher_parameters():
    """Test read_gofher_parameters_from_csv"""

    # Read in GofherParameters test CSV:
    gofher_params_csv_path = _get_test_gofher_params_csv_path()
    gofher_params = read_gofher_parameters_from_csv(gofher_params_csv_path)
    assert isinstance(gofher_params, GofherParameters)

    # Validate gofher params:
    validate_gofher_param_specific_test(gofher_params)
