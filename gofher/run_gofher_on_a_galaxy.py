import itertools
import os

from galaxy import galaxy
from gofher import run_gofher_on_galaxy
from sdss import visualize_sdss, SDSS_BANDS_IN_ORDER, SDSS_REF_BANDS_IN_ORDER

def run_sdss(name, fits_path, save_vis_path='', dark_side_label=''):
    """run gofher on a single sdss galaxy"""
    the_gal = galaxy(name,dark_side_label)

    for band in SDSS_BANDS_IN_ORDER:
        the_gal.construct_band(band,fits_path(name,band))

    for ref_band in SDSS_REF_BANDS_IN_ORDER:
        if the_gal.has_valid_band(ref_band):
            the_gal.ref_band = ref_band
            break

    if the_gal.ref_band == "":
        print("error: no valid ref band")
        return
    
    the_band_pairs = list(itertools.combinations(SDSS_BANDS_IN_ORDER, 2))

    the_gal = run_gofher_on_galaxy(the_gal,the_band_pairs)
    if save_vis_path != '':
        visualize_sdss(the_gal,save_vis_path)
    return the_gal

