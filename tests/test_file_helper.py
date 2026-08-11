"""Test all functions in gofher/fits.py

The script is run using the commend: python -m pytest
"""

from pathlib import Path
import uuid

import pytest
import numpy as np


from file_helper import (
    read_fits,
    write_array_file,
    read_array_file,
    assure_folder_exists
)

#IMPORTANT: Make sure these folder/file names reflect test file structure:
TEST_GALAXY_DIR = "NGC2347_SDSS_psf4_background_256"
TEST_SPARCFIRE_DIR = "fits"
TEST_FITS_FILE_NAME = "NGC2347_g.fits"

def _get_test_fits_path() -> str:
    """Helper function to get path to test fits file"""
    p = Path(__file__).resolve().parent
    return str(p.joinpath("data", TEST_GALAXY_DIR, TEST_SPARCFIRE_DIR, TEST_FITS_FILE_NAME))

def _generate_random_array(shape =  (50,50)) -> np.ndarray:
    """Helper function to generate a random numpy array of size shape"""
    rng = np.random.default_rng()
    return rng.random(shape)

@pytest.mark.parametrize("path, expected_exception", [
    ([], ValueError),
    ("fake_file.fits", FileNotFoundError),
    ("tests/test_utils.py", ValueError)
])
def test_read_fits_exceptions(path, expected_exception):
    """Test exceptions expected from read_fits"""
    with pytest.raises(expected_exception):
        read_fits(path)

def test_read_fits():
    """Test output expected from read_fits

    Residual Tolerance:
        residule < 1e-6
    """
    RESIDUAL_TOLERANCE = 1e-6
    #IMPORTANT: Make sure these values match the test fits
    #   If multiple galaxies are added, test each galaxy's values seperately

    # Validate if returns one np.ndarray:
    the_fits = read_fits(_get_test_fits_path())
    assert isinstance(the_fits,np.ndarray)
    assert the_fits.shape == (419,419)

    # Validate range of values we expect from test fits:
    #   Important: these values are found from external fits viewer
    #       and must be updated if a new test case is used.
    expected_mean = 0.017264254644141455
    expected_max = 1.0
    expected_min = 0.0

    assert np.abs(np.mean(the_fits)- expected_mean) < RESIDUAL_TOLERANCE
    assert np.abs(np.max(the_fits) - expected_max) < RESIDUAL_TOLERANCE
    assert np.abs(np.min(the_fits) - expected_min) < RESIDUAL_TOLERANCE

@pytest.mark.parametrize("array, path, expected_exception", [
    ([], "example.npy", ValueError),
    (_generate_random_array(), "not_npy.txt", ValueError)
])
def test_write_array_file_exceptions(array, path, expected_exception):
    """Test exceptions expected from write_array_file"""
    with pytest.raises(expected_exception):
        write_array_file(array,path)

def test_write_array_file():
    """Test output expected from write_array_file"""
    
    # Generate a random array and path:
    random_array = _generate_random_array()
    random_file_path = f"{uuid.uuid4()}.npy"

    # Write the array:
    write_array_file(random_array,random_file_path)

    # Validate the file exists:
    assert Path.exists(random_file_path)

    # Cleanup:
    Path.unlink(random_file_path)

@pytest.mark.parametrize("path, expected_exception", [
    ([], ValueError),
    ("fake_file.npy", FileNotFoundError),
    ("tests/test_utils.py", ValueError)
])
def test_read_array_file_exceptions(path, expected_exception):
    """Test exceptions expected from read_array_file"""
    with pytest.raises(expected_exception):
        read_array_file(path)

def test_read_array_file():
    """Test read array file
    
    Residual Tolerance:
        residule < 1e-6
    """
    RESIDUAL_TOLERANCE = 1e-6

    # Generate a random numpy array:
    expected_arr = _generate_random_array()

    # Generate a random path:
    arr_path = f"{uuid.uuid4()}.npy"

    # Write random array to file:
    write_array_file(expected_arr, arr_path)

    # Read written array and validate it is the same:
    arr = read_array_file(arr_path)
    assert arr.shape == expected_arr.shape
    assert np.abs(np.mean(arr)-np.mean(expected_arr)) < RESIDUAL_TOLERANCE
    assert np.abs(np.min(arr)-np.min(expected_arr)) < RESIDUAL_TOLERANCE
    assert np.abs(np.max(arr)-np.max(expected_arr)) < RESIDUAL_TOLERANCE

    # Cleanup:
    Path.unlink(arr_path)

@pytest.mark.parametrize("path, expected_exception", [
    ([], ValueError),
    ("", ValueError)
])
def test_assure_folder_exists_exceptions(path, expected_exception):
    """Test exceptions expected from assure_folder_exists"""
    with pytest.raises(expected_exception):
        assure_folder_exists(path)

def test_assure_folder_exists():
    """Test functionality from assure_folder_exists"""

    # Generate a random folder name:
    random_folder_name = f"{uuid.uuid4()}"

    # Call function and validate directory exists:
    assure_folder_exists(random_folder_name)

    assert Path(random_folder_name).is_dir()

    # Cleanup:
    Path(random_folder_name).rmdir()
