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
    """L2 Distance Norm"""
    return np.linalg.norm(np.array(cm)-np.array(center))

def find_mask_spot_closest_to_center(the_mask,approx_center):
    """Uses connected regions to locate center most cluster

        Args:
            the_mask: the mask to locate closest cluster
            approx_center: location of center
        Returns:
            mask of cluster closest to center
    """
    shape=the_mask.shape
    all_labels = measure.label(the_mask) #https://scipy-lectures.org/packages/scikit-image/auto_examples/plot_labels.html
    blobs_labels = measure.label(all_labels, background=0)
    
    unique = np.unique(blobs_labels.flatten()) #https://stackoverflow.com/a/28663910/13544635
    center_of_masses = scipy.ndimage.center_of_mass(np.ones(shape),blobs_labels, index=unique)
    dist_to_center = list(map(lambda i: np.inf if i == 0 else calculate_dist(center_of_masses[i],approx_center),range(len(center_of_masses))))

    return blobs_labels==unique[np.argmin(dist_to_center)]

def run_default_gofher_parameters_fitting(the_gal):
    """Finds gofher_parameters for the galaxy that will be used to generate the ellipse masks and bisection masks

        Args:
            shape: The shape of the ellipse mask
            r: how much to scale semi-major/semi-minor axis length of ellipse
                semi-major=r*a semi-minor=r*b
        Returns:
            the galaxy with gofher_parameters found using default fitting process
    """

    #See: Schallock et. al 2024 - Algorithm 1 Ellipse Mask Fitting

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
    """Runs gofher of a single galaxy

        Args:
            name: the name of the galaxy
            fits_path_function: a function that takes in galaxy_name and waveband and returns the file paths of the fits files
            blue_to_red_bands_in_order: a list of the name of wavebands in order of Bluest to Reddest wavebands
            ref_bands_in_order: a list of the name of wavebands in order of priority [highest, 2nd_highest, ... 2nd_lowest, lowest]
                of prefernce waveband is used as reference band
            paper_label: the baseline near side label gofher is comparing its answer to
                Important: If no baseline label, leave as none
        Returns:
            the galaxy
    """
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

    #Use default gofher_parameter fitting:
    the_gal = run_default_gofher_parameters_fitting(the_gal)

    #run gofher on the galaxy
    the_gal.run_gofher(the_band_pairs)

    return the_gal