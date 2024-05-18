import copy
import scipy
import itertools
import numpy as np
from skimage import measure
from scipy.stats import expon

from galaxy import galaxy
from sep_helper import run_sep
from gofher_parameters import gofher_parameters
from sersic import get_gopher_params_from_sersic_fit



def calculate_dist(cm,center):
    """Distance norm"""
    return np.linalg.norm(np.array(cm)-np.array(center))

def find_mask_spot_closest_to_center(the_mask,approx_center):
    """Locate spots in mask, and mask out all that are not center most"""
    shape=the_mask.shape
    all_labels = measure.label(the_mask) #https://scipy-lectures.org/packages/scikit-image/auto_examples/plot_labels.html
    blobs_labels = measure.label(all_labels, background=0)
    
    unique = np.unique(blobs_labels.flatten()) #https://stackoverflow.com/a/28663910/13544635
    center_of_masses = scipy.ndimage.center_of_mass(np.ones(shape),blobs_labels, index=unique)
    dist_to_center = list(map(lambda i: np.inf if i == 0 else calculate_dist(center_of_masses[i],approx_center),range(len(center_of_masses))))

    return blobs_labels==unique[np.argmin(dist_to_center)]

def run_default_gofher_ellipse_mask_fitting(the_gal,the_band_pairs):
    """Figure out which side is closer given diffenetial extinction reddening location"""
    #Step 1: Setup inital necessary variables:
    inital_gofher_parameters = gofher_parameters()
    data = copy.deepcopy(the_gal[the_gal.ref_band].data)
    shape = the_gal.get_shape()

    #Step 2: Run Sep on ref_band and find inital_gofher_parameters
    (cm_x, cm_y) = (shape[1]*0.5, shape[0]*0.5)
    the_el_sep, mu_bkg = run_sep(data, cm_x, cm_y)
    inital_gofher_parameters.load_from_sep_object(the_el_sep)

    #Step 3: Create an ellipse mask using the inital_gofher_parameters
    #el_mask = create_ellipse_mask_from_gofher_params(inital_gofher_parameters,shape,r=1.0)
    el_mask = inital_gofher_parameters.create_ellipse_mask(shape,r=1.0)

    #Step 4: Using the inital_gofher_parameters fit two pdfs pdf_in (probability inside ellipse) and pdf_out (probability outside ellipse)
    inside_ellipse = data[np.logical_and(el_mask,the_gal[the_gal.ref_band].valid_pixel_mask)].flatten()
    loc, scale = expon.fit(inside_ellipse) #https://stackoverflow.com/questions/25085200/scipy-stats-expon-fit-with-no-location-parameter
    pdf_in = expon.pdf(data, loc=loc, scale=scale)

    outside_ellipse = data[np.logical_and(np.logical_not(el_mask),the_gal[the_gal.ref_band].valid_pixel_mask)].flatten()
    loc, scale = expon.fit(outside_ellipse) #https://stackoverflow.com/questions/25085200/scipy-stats-expon-fit-with-no-location-parameter
    pdf_out = expon.pdf(data, loc=loc, scale=scale)

    #Step 5: Create a mask that only includes pixels in which there is a higher probability that it in inside the inital ellipse compared to the probability it is outside the ellipse
    the_mask = pdf_out < pdf_in

    center_mask = find_mask_spot_closest_to_center(the_mask,(cm_x, cm_y))
    bright_spot_mask = np.logical_and(the_mask,np.logical_not(center_mask))

    the_mask = np.logical_and(center_mask,the_gal[the_gal.ref_band].valid_pixel_mask)

    #Step 6: Try fitting sersic:
    try:
        the_sersic_mask = np.logical_and(the_mask,the_gal[the_gal.ref_band].valid_pixel_mask)
        the_gal.gofher_params = get_gopher_params_from_sersic_fit(inital_gofher_parameters,data,the_sersic_mask)
        the_gal.gofher_params.a = the_el_sep['a']
        the_gal.gofher_params.b = the_el_sep['b']
    except:
        print("Error fitting sersic, using inital_gofher_parameters from sep")
        the_gal.gofher_params = inital_gofher_parameters
        the_gal.encountered_sersic_fit_error = True

    return the_gal

def run_gofher(name,fits_path_function,blue_to_red_bands_in_order,ref_bands_in_order,paper_label=None):
    """run gofher on a single sdss galaxy"""
    the_gal = galaxy(name,paper_label)

    for band in blue_to_red_bands_in_order:
        the_gal.construct_band(band,fits_path_function(name,band))

    for ref_band in ref_bands_in_order:
        if the_gal.has_valid_band(ref_band):
            the_gal.ref_band = ref_band
            break
    
    if the_gal.ref_band == "":
        raise ValueError("run_gofher")
    
    the_band_pairs = list(itertools.combinations(blue_to_red_bands_in_order, 2))

    #Use Default Ellipse Mask Fitting Procdure:
    the_gal = run_default_gofher_ellipse_mask_fitting(the_gal,the_band_pairs)

    #run gofher on the galaxy
    the_gal.run_gofher(the_band_pairs)

    return the_gal