import itertools
import os

from galaxy import galaxy
from gofher import run_gofher_on_galaxy
from sparcfire import get_ref_band_and_gofher_params
from sdss import visualize_sdss, SDSS_BANDS_IN_ORDER, SDSS_REF_BANDS_IN_ORDER
from panstarrs import visualize_panstarrs, visualize_panstarrs_diff, PANSTARRS_BANDS_IN_ORDER, PANSTARRS_REF_BANDS_IN_ORDER

#test:
import matplotlib.pyplot as plt

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


def run_panstarrs(name,fits_path,save_vis_path='', dark_side_label='', color_image_path=''):
    """run gofher on a single sdss galaxy"""
    the_gal = galaxy(name,dark_side_label)

    for band in PANSTARRS_BANDS_IN_ORDER:
        the_gal.construct_band(band,fits_path(name,band))

    for ref_band in PANSTARRS_REF_BANDS_IN_ORDER:
        if the_gal.has_valid_band(ref_band):
            the_gal.ref_band = ref_band
            break

    if the_gal.ref_band == "":
        print("error: no valid ref band")
        return
    
    the_band_pairs = list(itertools.combinations(PANSTARRS_BANDS_IN_ORDER, 2))

    the_gal = run_gofher_on_galaxy(the_gal,the_band_pairs)
    if save_vis_path != '':
        visualize_panstarrs(the_gal,save_vis_path,color_image_path)

    ##pa = "C:\\Users\\school\\Desktop\\diff_output\\{}_diff.png".format(the_gal.name)
    ##visualize_panstarrs_diff(the_gal,pa,color_image_path)
    
    return the_gal

def get_fits_path(name,band):
    """the file path of where existing fits files can be found"""
    return os.path.join('C:\\Users\\school\\Downloads\\',name,"{}_{}.fits".format(name,band))

#if __name__ == "__main__":
#    run_panstarrs('NGC891',get_fits_path,'C:\\Users\\school\\Downloads\\NGC891\\NGC891_vis.png','','C:\\Users\\school\\Downloads\\NGC891\\NGC891_color.jpg')