import matplotlib.pyplot as plt
from galaxy_band_pair import GalaxyBandPair
from galaxy_band import GalaxyBand
from plot import plot_diff_histogram, plot_diff_image

import numpy as np

import pytest

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
    (_genrate_axs(),_generate_random_galaxy_band_pair((30,30)),_generate_area_to_norm_mask((50,50)),None,ValueError),
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

#TODO: plot_mask