"""Test all functions in gofher/galaxy_band_pair.py

The script is run using the commend: python -m pytest
"""
from pathlib import Path

import pytest
import numpy as np

from file_helper import read_fits
from galaxy_band import GalaxyBand
from galaxy_band_pair import GalaxyBandPair

#IMPORTANT: Make sure these folder/file names reflect test file structure:
TEST_GALAXY_DIR = "NGC2347_SDSS_psf4_background_256"
TEST_SPARCFIRE_DIR = "fits"
TEST_FITS_FILE_NAME = "NGC2347_{}.fits"

# Residual tolerance for tests expecting a specific numerical value:
RESIDUAL_TOLERANCE = 1e-6

def _get_fits_path(band:str):
    p = Path(__file__).resolve().parent
    file_name = TEST_FITS_FILE_NAME.format(band)
    return str(p.joinpath("data", TEST_GALAXY_DIR, TEST_SPARCFIRE_DIR, file_name))

def _generate_random_galaxy_band(band="r",size=(50,50)):
    rng = np.random.default_rng()
    return GalaxyBand(band, rng.random(size=size))

@pytest.mark.parametrize("blue, red, expected_exception", [
    ([],_generate_random_galaxy_band(), TypeError),
    (_generate_random_galaxy_band,"", TypeError),
    (_generate_random_galaxy_band("g"),
     _generate_random_galaxy_band("g"), ValueError),
    (_generate_random_galaxy_band(size=(10,10)), 
     _generate_random_galaxy_band(size=(50,50)), ValueError)
])
def test_galaxy_band_pair_excpetions(blue,red,expected_exception):
    """Test exceptions expected from galaxy band pair __init__"""
    with pytest.raises(expected_exception):
        GalaxyBandPair(blue, red)
     
def test_galaxy_band_pair_run_normalization_exception():
    """Test to verify runtime exception occurs if you have not yet applied normalization"""

    # Create two galaxy bands and create a waveband pair:
    test_size = (10,10)
    g = _generate_random_galaxy_band("g",test_size)
    r = _generate_random_galaxy_band("r",test_size)
    g_minus_r = GalaxyBandPair(g,r)

    # Neither has normalization so runtime exception should occur:
    with pytest.raises(RuntimeError):
        g_minus_r.calculate_diff_image()

    # Apply normalziation to one galaxy band, so runtime exception should still occur:
    area_to_norm = np.ones(test_size,bool)
    g.apply_normalization(area_to_norm)

    with pytest.raises(RuntimeError):
        g_minus_r.calculate_diff_image()

    # Apply normalization to second galaxy band so should run without exception
    r.apply_normalization(area_to_norm)

    g_minus_r.calculate_diff_image()

def test_galaxy_band_pair_calculate_diff_image():
    """Test galaxy_band_pair.calculate_diff_image() by running an example

    The example:
    * Takes the g-r band for NGC2347 (psf=4, background=256)
    * area_to_norm only including rows 0-199 (inclusive)
        all others masked out, calculates diff image.

    Important: If specific example is changed, must update:
      the_max, the_min, the_sum, the_mean, the_std
    These can be found by manual making the example and using numpy
    to get the new values.
    i.e. 
        bluer_band = read_fits(...)
        redder_band = read_fits(...)
    
        # then normalize each band ()

        new_example = bluer_band-redder_band
        the_max = np.max(new_example)
    
    Residual Tolerance:
        residule < 1e-6
    """

    # Expected quantities derived from specifc example:
    # Important: Update these if example is updated!!
    the_max = 0.037522398780925226
    the_min = -0.13923398530790754
    the_sum = -475.35498
    the_mean = -0.0027076341808162426
    the_std = 0.005804397360304486

    # Read in the fits of the bluer (g) and redder (r) wave band:
    g_data = read_fits(_get_fits_path("g"))
    r_data = read_fits(_get_fits_path("r"))

    # Create GalaxyBands and make a GalaxyBandPair
    g_band = GalaxyBand("g",g_data)
    r_band = GalaxyBand("r",r_data)
    g_minus_r = GalaxyBandPair(g_band,r_band)

    # Create area_to_norm boolean mask
    # For this example, only include 0-199 rows (inclusive)
    area_to_norm = np.zeros(g_data.shape,bool)
    area_to_norm[:200,:] = True

    # Normalize wavebands:
    g_band.apply_normalization(area_to_norm)
    r_band.apply_normalization(area_to_norm)

    # Calculate diff image:
    diff_image = g_minus_r.calculate_diff_image()

    # Calculate the residual between expected values of (the_min, the_max, etc.)
    # and the diff image.
    # Format note: These are written on seperate lines to make the pytest
    #   failure strins more readable to the programmer.
    max_residual = np.abs(np.max(diff_image)-the_max)
    min_residual = np.abs(np.min(diff_image)-the_min)
    sum_residual = np.abs(np.sum(diff_image)-the_sum)
    mean_residual = np.abs(np.mean(diff_image)-the_mean)
    std_residual = np.abs(np.std(diff_image)-the_std)

    # Validate diff_image values with expected values:
    assert max_residual < RESIDUAL_TOLERANCE
    assert min_residual < RESIDUAL_TOLERANCE
    assert sum_residual < RESIDUAL_TOLERANCE
    assert mean_residual < RESIDUAL_TOLERANCE
    assert std_residual < RESIDUAL_TOLERANCE
