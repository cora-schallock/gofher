"""Test all functions in gofher/galaxy_band.py

The script is run using the commend: python -m pytest
"""

from itertools import product

import pytest
import numpy as np

from galaxy_band import GalaxyBand
from utils import is_2d_bool_array, is_2d_same_shape_arrays, is_float_int_array

# Residual tolerance for tests expecting a specific numerical value:
RESIDUAL_TOLERANCE = 1e-6

def _generate_random_array(shape =  (50,50)) -> np.ndarray:
    """Helper function to generate a random numpy array of size shape"""
    rng = np.random.default_rng()
    return rng.random(shape)

def _generate_all_binary_mask_permutations(shape: tuple[int]):
    """Generate all binary masks that can mask a 2d array with provided shape"""
    masks = []
    n = shape[0] * shape[1]
    for p in product([False, True], repeat=n):
        masks.append(np.array(p).reshape(shape))
    return masks

def _calculate_expected_normalization(arr: np.ndarray, mask: np.ndarray):
    expected = np.zeros(arr.shape,np.float32)
    if int(np.sum(mask)) != 0:
        the_max = np.max(arr[mask])
        the_min = np.min(arr[mask])
        if the_min != the_max:
            scale = the_max-the_min
            expected[mask] = (arr[mask]-the_min)/scale
    return expected

@pytest.mark.parametrize("band, data, expected_exception", [
    ([],_generate_random_array(), TypeError),
    ("",_generate_random_array(), ValueError),
    ("g",[], TypeError),
    ("g",_generate_random_array((50,50,2)), TypeError),
    ("g",_generate_random_array((50,0)), ValueError)
])
def test_galaxy_band_expections(band, data, expected_exception):
    """Test exceptions expected from galaxy_band constructor"""
    with pytest.raises(expected_exception):
        GalaxyBand(band, data)

def test_galaxy_band():
    """Test output expected from galaxy_band constructor

    Residual Tolerance:
        residule < 1e-6
    """

    band = "r"
    data = np.array([[4.0,0.0],[1.0,2.0]])
    
    the_band = GalaxyBand(band,data)

    assert the_band.band == band
    assert the_band.data.shape == data.shape
    assert np.sum(the_band.data-data) < RESIDUAL_TOLERANCE

def test_get_shape():
    """Test output expected from galaxy_band.get_shape()"""

    band = "g"
    data = _generate_random_array()
    gb = GalaxyBand(band,data)
    assert gb.get_shape() == data.shape

def test_has_normalization():
    """Test output expected from galaxy_band.has_normalization()"""
    band = "i"
    data = _generate_random_array()
    gb = GalaxyBand(band,data)

    assert not gb.has_normalization()
    gb.apply_normalization()

    assert gb.has_normalization()

def test_get_normalization_excpetion():
    """Test exception from galaxy_band.get_normalization()"""

    band = "i"
    data = np.array([[4.0,0.0],[1.0,2.0]])
    gb = GalaxyBand(band,data)

    with pytest.raises(ValueError):
        gb.get_normalization()

def test_get_normalization():
    """Test output from galaxy_band.get_normalization()
    
    Residual Tolerance:
        sum of residule < 1e-6
    """

    band = "i"
    data = np.array([[4.0,0.0],[1.0,2.0]])
    gb = GalaxyBand(band,data)
    
    mask = np.ones(data.shape,bool)
    expected = np.array([[1.0,0.0],[0.25,0.5]])
    gb.apply_normalization(mask)

    normed = gb.get_normalization()
    assert np.sum(np.abs(normed-expected)) < RESIDUAL_TOLERANCE

def test_get_valid_pixel_mask():
    """Test get valid pixel mask"""

    # Create a sample data and get the expected mask:
    data = np.array([[4.0,np.nan],
                     [1.0,np.inf]])
    expected = np.array([[True,False],
                         [True, False]])

    # Get the valid pixel mask:
    g_band = GalaxyBand("g",data)
    valid_pixel_mask = g_band.get_valid_pixel_mask()

    # Assert that it is bool 2D np.ndarray and matches expected
    assert is_2d_bool_array(valid_pixel_mask)
    assert is_2d_same_shape_arrays(valid_pixel_mask,data)
    assert not np.any(np.logical_xor(valid_pixel_mask,expected))

@pytest.mark.parametrize("data, area_to_norm, expected_exception", [
    (np.array([[4.0,0.0],[1.0,2.0]]),
     np.array([[True,True]]), ValueError), #shapes different size
    (np.array([[4.0,0.0],[1.0,2.0]]),
     np.array([[True,True],[True,0.25]]), TypeError), #invalid mask value
    (np.array([[4.0,0.0],[1.0,np.nan]]),
     np.array([[False,False],[False,True]]), ValueError), #includes Nan value,
    (np.array([[np.inf,0.0],[1.0,2.0]]),
     np.array([[True,False],[False,True]]), ValueError) #includes INF value
])
def test_apply_normalization_expections(data, area_to_norm, expected_exception):
    """Test exceptions expected from galaxy_band constructor"""
    band = "y"
    gb = GalaxyBand(band,data)
    with pytest.raises(expected_exception):
        gb.apply_normalization(area_to_norm)

def test_apply_normalization():
    """Test output from Galaxyband.apply_normalization()
    
    Generates all permutations of binary mask and calculates
    expected normalization. This only tests functions without
    NAN or INF values.

    Calculates total difference from expected, and sum can
    not exceed residual tolerence.

    Resdisual tolerence: 1e-6 
    """

    band = "i"
    test_array = np.array([[4.0,0.0],[1.0,2.0]])
    gb = GalaxyBand(band,test_array)
    masks = _generate_all_binary_mask_permutations(test_array.shape)

    for mask in masks:
        expected = _calculate_expected_normalization(test_array,mask)
        normalization = gb.apply_normalization(mask)

        assert is_float_int_array(normalization)
        assert is_2d_same_shape_arrays(normalization,expected)
        assert np.sum(np.abs(normalization-expected)) < RESIDUAL_TOLERANCE
