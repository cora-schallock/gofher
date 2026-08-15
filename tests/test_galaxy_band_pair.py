"""Test all functions in gofher/galaxy_band_pair.py

The script is run using the commend: python -m pytest
"""
from pathlib import Path

import pytest
import numpy as np

from galaxy_band import GalaxyBand
from galaxy_band_pair import GalaxyBandPair

#IMPORTANT: Make sure these folder/file names reflect test file structure:
TEST_GALAXY_DIR = "NGC2347_SDSS_psf4_background_256"
TEST_SPARCFIRE_DIR = "fits"
TEST_FITS_FILE_NAME = "NGC2347_{}.fits"

def _get_fits_path():
    p = Path(__file__).resolve().parent
    return str(p.joinpath("data", TEST_GALAXY_DIR, TEST_SPARCFIRE_DIR, TEST_FITS_FILE_NAME))

def _generate_random_galaxy_band(band="r",size=(50,50)):
    rng = np.random.default_rng()
    return GalaxyBand(band, rng.random(size=size))

@pytest.mark.parametrize("blue, red, expected_exception", [
    ([],_generate_random_galaxy_band(), TypeError),
    (_generate_random_galaxy_band,"", ValueError),
    (_generate_random_galaxy_band("g"),_generate_random_galaxy_band("g"), ValueError),
    (_generate_random_galaxy_band(size=(10,10)), _generate_random_galaxy_band(size=(50,50)), ValueError)
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
         g_minus_r.run()

    # Apply normalziation to one galaxy band, so runtime exception should still occur:
    area_to_norm = np.ones(test_size,bool)
    g.apply_normalization(area_to_norm)

    with pytest.raises(RuntimeError):
        g_minus_r.run()

    # Apply normalization to second galaxy band so should run without exception
    r.apply_normalization(area_to_norm)

    g_minus_r.run()

def test_galaxy_band_pair_run():
     raise NotImplementedError

    


    
     