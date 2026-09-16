import matplotlib.pyplot as plt
import numpy as np
import pytest


from gofher.galaxy_band_pair import GalaxyBandPair
from gofher.galaxy_band import GalaxyBand
from gofher.gofher_parameters import GofherParameters
from gofher.plot import plot_diff_histogram, plot_diff_image, plot_mask

def _genrate_axs():
    _, (ax_left, _) = plt.subplots(1, 2)
    return ax_left

def _generate_random_galaxy_band_pair(blue_band="g", red_band="r",size=(50,50)):
    rng = np.random.default_rng()
    bluer = GalaxyBand(blue_band, rng.random(size=size))
    redder = GalaxyBand(red_band, rng.random(size=size))
    return GalaxyBandPair(bluer,redder)

def _generate_area_to_norm_mask(size=(50,50),dtype=bool):
    return np.ones(size,dtype=dtype)

def _generate_gofher_params():
    the_params = GofherParameters()
    the_params.a = 10
    the_params.b = 5
    the_params.h = 25
    the_params.k = 25
    the_params.shape = 0
    return the_params

@pytest.mark.parametrize("ax, ref_band_data, gp, area_to_norm, pixel_bounds, c, expected_exception", [
    ("",
     _generate_area_to_norm_mask(dtype=float),
     _generate_gofher_params(),
     _generate_area_to_norm_mask(),
     None,
     1,
     TypeError), #Case 1: ax wrong type
    (_genrate_axs(),
     _generate_area_to_norm_mask(size=(50,50,2),dtype=float),
     _generate_gofher_params(),
     _generate_area_to_norm_mask(),
     None,
     1,
     TypeError), #Case 2: ref_band_data not 2D array
    (_genrate_axs(),
     _generate_area_to_norm_mask(dtype=float),
     "",
     _generate_area_to_norm_mask(),
     None,
     1,
     TypeError), #Case 3: gofher_params wrong type
    (_genrate_axs(),
     _generate_area_to_norm_mask(dtype=float),
     _generate_gofher_params(),
     "",
     None,
     1,
     TypeError), #Case 4: area_to_norm is wrong type
    (_genrate_axs(),
     _generate_area_to_norm_mask(size=(50,50),dtype=float),
     _generate_gofher_params(),
     _generate_area_to_norm_mask(size=(51,51),dtype=float),
     None,
     1,
     ValueError), #Case 5: area_to_norm is wrong shape
    (_genrate_axs(),
     _generate_area_to_norm_mask(dtype=float),
     _generate_gofher_params(),
     _generate_area_to_norm_mask(dtype=float),
     {},
     1,
     TypeError), #Case 6: pixel_bounds wrong type
    (_genrate_axs(),
     _generate_area_to_norm_mask(dtype=float),
     _generate_gofher_params(),
     _generate_area_to_norm_mask(dtype=float),
     (20,30),
     1,
     ValueError), #Case 7: pixel_bounds has too few elements
    (_genrate_axs(),
     _generate_area_to_norm_mask(dtype=float),
     _generate_gofher_params(),
     _generate_area_to_norm_mask(dtype=float),
     (20.5,30,40,50),
     1,
     ValueError), #Case 7: pixel_bounds has non int element
    (_genrate_axs(),
      _generate_area_to_norm_mask(dtype=float),
      _generate_gofher_params(),
      _generate_area_to_norm_mask(dtype=float),
      (200,30,40,50),
      1,
      ValueError), #Case 8: pixel_bounds xmin greater than xmax
    (_genrate_axs(),
     _generate_area_to_norm_mask(dtype=float),
     _generate_gofher_params(),
     _generate_area_to_norm_mask(dtype=float),
     (20,30,400,50),
     1,
     ValueError), #Case 9: pixel_bounds ymin greater than ymax
    (_genrate_axs(),
     _generate_area_to_norm_mask(dtype=float),
     _generate_gofher_params(),
     _generate_area_to_norm_mask(dtype=float),
     (20,30,40,50),
     "",
     ValueError), #Case 9: c is wrong type
    (_genrate_axs(),
     _generate_area_to_norm_mask(dtype=float),
     _generate_gofher_params(),
     _generate_area_to_norm_mask(dtype=float),
     (20,30,40,50),
     0,
     ValueError), #Case 10: c is too small
])
def test_plot_mask_exceptions(ax, ref_band_data, gp, area_to_norm, pixel_bounds, c, expected_exception):
    """Test exceptions from plot_mask"""
    with pytest.raises(expected_exception):
        plot_mask(ax,
                  ref_band_data,
                  gp,
                  area_to_norm,
                  pixel_bounds,
                  c)

@pytest.mark.parametrize("ax, bp, vmin, vmax, expected", [
    ("",_generate_random_galaxy_band_pair(),None,TypeError),
    (_genrate_axs(),[],None,TypeError),
    (_genrate_axs(),_generate_random_galaxy_band_pair(),"",TypeError),
    (_genrate_axs(),_generate_random_galaxy_band_pair(),(1.0),ValueError),
    (_genrate_axs(),_generate_random_galaxy_band_pair(),(1.0,"a"),TypeError),
     (_genrate_axs(),_generate_random_galaxy_band_pair(),(1.0,0.0),ValueError)
])
def test_plot_diff_histogram_exceptions(ax, bp, vrange, expected):
    """Test plot_diff_histogram exceptions"""
    with pytest.raises(expected):
        plot_diff_histogram(ax,bp,vrange)

@pytest.mark.parametrize("ax, bp, area_to_norm, pb, expected", [
    ("",_generate_random_galaxy_band_pair(),_generate_area_to_norm_mask(), None, TypeError),
    (_genrate_axs(),"",_generate_area_to_norm_mask(),None,TypeError),
    (_genrate_axs(),_generate_random_galaxy_band_pair(),"",None,TypeError),
    (_genrate_axs(),_generate_random_galaxy_band_pair(),_generate_area_to_norm_mask(dtype=float),None,ValueError),
    (_genrate_axs(),_generate_random_galaxy_band_pair(size=(30,30)),_generate_area_to_norm_mask(size=(50,50)),None,ValueError),
    (_genrate_axs(),_generate_random_galaxy_band_pair(),_generate_area_to_norm_mask(),{},TypeError),
    (_genrate_axs(),_generate_random_galaxy_band_pair(),_generate_area_to_norm_mask(),[10],ValueError),
    (_genrate_axs(),_generate_random_galaxy_band_pair(),_generate_area_to_norm_mask(),[10.5,11.5,20.5,21.5],TypeError),
    (_genrate_axs(),_generate_random_galaxy_band_pair(),_generate_area_to_norm_mask(),[10,11,9,20],TypeError),
    (_genrate_axs(),_generate_random_galaxy_band_pair(),_generate_area_to_norm_mask(),[10,11,20,10],TypeError)
])
def test_plot_diff_image_exceptions(ax, bp, area_to_norm, pb, expected):
    """Test plot_diff_histogram exceptions"""
    with pytest.raises(expected):
        plot_diff_image(ax,bp,area_to_norm, pb)