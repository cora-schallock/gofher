"""Test all functions in gofher/galaxy.py

The script is run using the commend: python -m pytest
"""
from pathlib import Path
import numpy as np

import pytest

from gofher.galaxy import Galaxy
from gofher.gofher_parameters import GofherParameters
from gofher.sparcfire import gofher_params_from_sparcfire_csv

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
    return str(p.joinpath("data", TEST_GALAXY_DIR, TEST_FITS_DIR, fits_file_name))

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

@pytest.mark.parametrize("band, expected_exception", [
    ([], TypeError)
])
def test_has_band_exception(band, expected_exception):
    """Test exceptions raised by Galaxy.has_band()"""

    # Get test gofher param:
    gofher_param = _get_test_galaxy_param()
    
    # Construct the galaxy:
    the_galaxy = Galaxy(gofher_param)

    # Check exception is raised:
    with pytest.raises(expected_exception):
        the_galaxy.has_band(band)

def test_has_band():
    """Test Galaxy.has_band()"""

    # Specify test band and fits:
    test_band = "g"
    test_fits = _get_test_fits_path(test_band)

    # Get test gofher param:
    gofher_param = _get_test_galaxy_param()
        
    # Construct the galaxy:
    the_galaxy = Galaxy(gofher_param)

    # Verify the galaxy doesn't currently have band:
    assert not the_galaxy.has_band(test_band)

    # Add the test band to the galaxy:
    the_galaxy.construct_galaxy_band_from_fits(test_band, test_fits)

    # Verify the galaxy now has band:
    assert the_galaxy.has_band(test_band)

@pytest.mark.parametrize("band, expected_exception", [
    ([], TypeError),
    ("missingkey", KeyError)
])
def test_get_band_exception(band, expected_exception):
    """Test exceptions raised by Galaxy.get_band()"""

    # Get test gofher param:
    gofher_param = _get_test_galaxy_param()
    
    # Construct the galaxy:
    the_galaxy = Galaxy(gofher_param)

    # Check exception is raised:
    with pytest.raises(expected_exception):
        the_galaxy.get_band(band)

def test_get_band():
    """Test Galaxy.get_band()"""

    # Specify test band and fits:
    test_band = "g"
    test_fits = _get_test_fits_path(test_band)

    # Get test gofher param:
    gofher_param = _get_test_galaxy_param()
        
    # Construct the galaxy:
    the_galaxy = Galaxy(gofher_param)

    # Add the test band to the galaxy:
    the_band = the_galaxy.construct_galaxy_band_from_fits(test_band, test_fits)

    assert the_galaxy.get_band(test_band) == the_band

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

@pytest.mark.parametrize("bands, f, area, fail_silently, expected_exception", [
    ([], 1.0, True, None, ValueError),
    (["a",{}], 1.0, None, True, TypeError),
    (["g","r"], dict(), None, True, TypeError),
    (["g","r"], -1.0, None, True, ValueError),
    (["g","r"], 1.5, None, True, ValueError),
    (["g","r"], 0.5, "", True, TypeError),
    (["g","r"], 0.5, np.ones((10,10),bool), True, ValueError),
    (["g","r"], 0.5, None, "", TypeError)
])
def test_run_exceptions(bands, f, area, fail_silently, expected_exception):
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
        the_galaxy.run(bluer_to_redder_bands=bands,
                       sparcfire_bulge_disk_f=f,
                       area_to_consider=area,
                       fail_silently_on_missing_band=fail_silently)

def test_run_area_to_consider_exception():
    """Verify the exceptions raised in Galaxy.run() if area_to_consider has no Pixels"""
    # Get test gofher param:
    gofher_param = _get_test_galaxy_param()
        
    # Construct the galaxy:
    the_galaxy = Galaxy(gofher_param)
    
    # Add g and r band fits:
    the_band = the_galaxy.construct_galaxy_band_from_fits("g",_get_test_fits_path("g"))
    the_galaxy.construct_galaxy_band_from_fits("r",_get_test_fits_path("r"))

    # Get shape of fits
    shape = the_band.data.shape

    # If area_to_consider has all Falses, it raises a ValueError exception:
    no_area = np.zeros(shape,bool)
    with pytest.raises(ValueError):
        the_galaxy.run(bluer_to_redder_bands=["g","r"],
                       sparcfire_bulge_disk_f=0.5,
                       area_to_consider=no_area,
                       fail_silently_on_missing_band=False)

    # If area_to_consider is passed in with at least one True
    #   but shares no intersecting True pixels ellipse mask/ valid mask
    #   it reaises a RunTime exception:
    no_area[0,0] = True
    with pytest.raises(RuntimeError):
        the_galaxy.run(bluer_to_redder_bands=["g","r"],
                       sparcfire_bulge_disk_f=0.5,
                       area_to_consider=no_area,
                       fail_silently_on_missing_band=False)

    # Now we will in all True to area_to_consider and it should work:
    area = np.ones(shape,bool)
    the_galaxy.run(bluer_to_redder_bands=["g","r"],
                   sparcfire_bulge_disk_f=0.5,
                   area_to_consider=area,
                   fail_silently_on_missing_band=False)

def test_run_no_pos_neg_area_exception():
    """Verify an exception is raised in pos mask or neg mask has no area"""

    sparcfire_bulge_disk = 0.5

    # Get test gofher param:
    gofher_param = _get_test_galaxy_param()
           
    # Construct the galaxy:
    the_galaxy = Galaxy(gofher_param)
    the_galaxy.construct_galaxy_band_from_fits("g",_get_test_fits_path("g"))
    the_galaxy.construct_galaxy_band_from_fits("r",_get_test_fits_path("r"))

    gofher_param.calculate_from_sparcfire(sparcfire_bulge_disk)
    pos_mask, neg_mask = gofher_param.create_bisection_masks()

    # This should work:
    the_mask = np.logical_or(pos_mask, neg_mask)
    the_galaxy.run(bluer_to_redder_bands=["g","r"],
                   sparcfire_bulge_disk_f=sparcfire_bulge_disk,
                   area_to_consider=the_mask)

    # When area_to_consider shares no intersecting True pixels with pos side
    #   (i.e. in this toy example we use neg_mask) it raises a Runtime Exception
    with pytest.raises(RuntimeError):
        the_galaxy.run(bluer_to_redder_bands=["g","r"],
                       sparcfire_bulge_disk_f=sparcfire_bulge_disk,
                       area_to_consider=neg_mask)

    # When area_to_consider shares no intersecting True pixels with neg side
    #   (i.e. in this toy example we use pos_mask) it raises a Runtime Exception
    with pytest.raises(RuntimeError):
        the_galaxy.run(bluer_to_redder_bands=["g","r"],
                       sparcfire_bulge_disk_f=sparcfire_bulge_disk,
                       area_to_consider=pos_mask)

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
        the_galaxy.run(bluer_to_redder_bands=bands,
                       sparcfire_bulge_disk_f=f,
                       area_to_consider=None,
                       fail_silently_on_missing_band=False)

    # Add the g band:
    the_galaxy.construct_galaxy_band_from_fits("g",_get_test_fits_path("g"))

    # r band still missing, so it casues a RuntimeError:
    with pytest.raises(RuntimeError):
        the_galaxy.run(bluer_to_redder_bands=bands,
                       sparcfire_bulge_disk_f=f,
                       area_to_consider=None,
                       fail_silently_on_missing_band=False)

    # Add the r band:
    the_galaxy.construct_galaxy_band_from_fits("r",_get_test_fits_path("r"))

    # Now run should work without casuing an exception
    the_galaxy.run(bluer_to_redder_bands=bands,
                   sparcfire_bulge_disk_f=f,
                   area_to_consider=None,
                   fail_silently_on_missing_band=False)

def test_run():
    """Test Galaxy.run()"""
    # The parameters for the test:
    bands = ['u','g','r','i','z']
    f = 0.5
    
    # Get test gofher param:
    gofher_param = _get_test_galaxy_param()
        
    # Construct the galaxy and bands:
    the_galaxy = Galaxy(gofher_param)
    for band in bands:
        the_galaxy.construct_galaxy_band_from_fits(band,_get_test_fits_path(band))

    # Run
    the_galaxy.run(bluer_to_redder_bands=bands,
                   sparcfire_bulge_disk_f=f)
    
#TODO: make_lupton_rgb, plot_figure, save_normalizations, csv_dict, output_to_csv