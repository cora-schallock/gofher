"""Test all functions in gofher/galaxy.py

The script is run using the commend: python -m pytest
"""
from pathlib import Path

import pytest

from galaxy import Galaxy
from gofher_parameters import GofherParameters
from sparcfire import gofher_params_from_sparcfire_csv

#IMPORTANT: Make sure these folder/file names reflect test file structure:
TEST_GALAXY_DIR = "NGC2347_SDSS_psf4_background_256"
TEST_SPARCFIRE_DIR = "sparcfire_r_band_output"
TEST_FITS_DIR = "fits"
TEST_CSV_FILE_NAME = "NGC2347_r.csv"

def _get_test_csv_path() -> str:
    p = Path(__file__).resolve().parent
    return str(p.joinpath("data", TEST_GALAXY_DIR, TEST_SPARCFIRE_DIR, TEST_CSV_FILE_NAME))

def _get_test_fits_path(band) -> str:
    """Helper function to get path to test fits file"""
    fits_file_name = f"NGC2347_{band}.fits"

    p = Path(__file__).resolve().parent
    return str(p.joinpath("data", TEST_GALAXY_DIR, TEST_SPARCFIRE_DIR, fits_file_name))

def _get_test_galaxy_param() -> GofherParameters:
    # Read in the gofher params from the provided SpArcFiRe CSV:
    gofher_params = gofher_params_from_sparcfire_csv(_get_test_csv_path())
    
    # For setting up this test, we need to verify the gofher_params 
    # is a list containning one gofher. If there is an issue with this,
    # double check gofher_params_from_sparcfire_csv()
    # Note this should already be checked in test_sparcifre.py 
    assert isinstance(gofher_params,list)
    assert len(gofher_params) == 1
    assert isinstance(gofher_params[0], GofherParameters)

    return gofher_params[0]

@pytest.mark.parametrize("gofher_params, expected_exception", [
    ("", TypeError)
])
def test_galaxy_initalization_excpetions(gofher_params, expected_exception):
    """Test exceptions expected from galaxy __init__"""
    with pytest.raises(expected_exception):
        Galaxy(gofher_params)

def test_galaxy_initalization():
    """Test galaxy initalziation
    
    Note to programmer - This test may seem silly, but it is verifying that 
    this doesn't cause an exception. It also is a way to validate the test 
    SpArcFiRe csv is being read correctly
    """

    # Get test gofher param:
    gofher_param = _get_test_galaxy_param()

    # Construct the galaxy:
    Galaxy(gofher_param)

@pytest.mark.parametrize("band, fits_path, expected_exception", [
    ([],_get_test_fits_path("g"), TypeError),
    ("",_get_test_fits_path("g"), ValueError),
    ("g-",_get_test_fits_path("g"), ValueError),
    ("g_",_get_test_fits_path("g"), ValueError),
    ("g",{}, TypeError),
    ("g","not_a_fits.pdf", ValueError),
    ("g","fake_file_path.fits", ValueError)
])
def test_construct_galaxy_band_from_fits_excpetions(band,fits_path,expected_exception):
    """Test exceptions from construct_galaxy_band_from_fits()"""

    # Get test gofher param:
    gofher_param = _get_test_galaxy_param()

    # Construct the galaxy:
    the_galaxy = Galaxy(gofher_param)

    # Validate the correct excpetion is raised:
    with pytest.raises(expected_exception):
        the_galaxy.construct_galaxy_band_from_fits(band,fits_path)

def test_construct_galaxy_band_from_fits():
    """Test galaxy.construct_galaxy_band_from_fits()"""
    
    # Get test gofher param:
    gofher_param = _get_test_galaxy_param()
    
    # Construct the galaxy:
    the_galaxy = Galaxy(gofher_param)
    bands = ['g','r','i','z','u']
    for band in bands:
        the_galaxy.construct_galaxy_band_from_fits(band,_get_test_fits_path(band))

@pytest.mark.parametrize("bands, f, fail_silently, expected_exception", [
    ([], 1.0, True, TypeError),
    (["a",{}], 1.0, True, ValueError),
    (["g","r"], dict(), True, TypeError),
    (["g","r"], -1.0, True, TypeError),
    (["g","r"], 1.5, True, TypeError),
    (["g","r"], 0.5, "", TypeError)
])
def test_run_exceptions(bands, f, fail_silently, expected_exception):
    """Verify the exceptions raised in Galaxy.run()"""
    # Get test gofher param:
    gofher_param = _get_test_galaxy_param()
    
    # Construct the galaxy:
    the_galaxy = Galaxy(gofher_param)

    # Add g and r band fits:
    the_galaxy.construct_galaxy_band_from_fits("g",_get_test_fits_path("g"))
    the_galaxy.construct_galaxy_band_from_fits("r",_get_test_fits_path("r"))

    # Validate the correct excpetion is raised:
    with pytest.raises(expected_exception):
        the_galaxy.run(bands,f,fail_silently)

def test_run_silent_fail():
    """Test fail_silently_on_missing_band parameter in Galaxy.run()"""
    # The parameters for the test:
    bands = ["g","r"]
    f = 1.0

    # Get test gofher param:
    gofher_param = _get_test_galaxy_param()
        
    # Construct the galaxy:
    the_galaxy = Galaxy(gofher_param)

    # No bands are present yet, so it causes a RuntimeError:
    with pytest.raises(RuntimeError):
        the_galaxy.run(bands,f,False)

    # Add the g band:
    the_galaxy.construct_galaxy_band_from_fits("g",_get_test_fits_path("g"))

    # r band still missing, so it casues a RuntimeError:
    with pytest.raises(RuntimeError):
        the_galaxy.run(bands,f,False)

    # Add the r band:
    the_galaxy.construct_galaxy_band_from_fits("r",_get_test_fits_path("r"))

    # Now run should work without casuing an exception
    the_galaxy.run(bands,f,False)

def test_run():
    """Test Galaxy.run()"""
    # The parameters for the test:
    bands = ['g','r','i','z','u']
    f = 0.5
    
    # Get test gofher param:
    gofher_param = _get_test_galaxy_param()
        
    # Construct the galaxy and bands:
    the_galaxy = Galaxy(gofher_param)
    for band in bands:
        the_galaxy.construct_galaxy_band_from_fits(band,_get_test_fits_path(band))

    # Run
    the_galaxy.run(bands,f,False)

    # TODO - verify the correct waveband pairs exist
    # idk what else to check?
    