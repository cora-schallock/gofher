import scipy
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from scipy.stats import expon
import copy

from sep_helper import run_sep
from mask import create_ellipse_mask_from_gofher_params

from run_sersic import fit_sersic

class gofher_parameters:
    """Contains gofher ellipse parameters"""
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.a = 0.0
        self.b = 0.0
        self.theta = 0.0

    def load_from_sep_object(self,sep_object):
        self.x = sep_object['x']
        self.y = sep_object['y']
        self.a = sep_object['a']
        self.b = sep_object['b']
        self.theta = sep_object['theta']

def calculate_dist(cm,center):
    """Distance norm"""
    return np.linalg.norm(np.array(cm)-np.array(center))

def normalize_array(data,to_diff_mask):
    """data - fits file (after read fits)
    to_diff_mask - a single boolean mask indicating where to create diff (1's = included, 0's = ignored)"""
    normalized = np.zeros(data.shape)
    the_min = np.min(data[to_diff_mask]); the_max = np.max(data[to_diff_mask])
    normalized[to_diff_mask] = (data[to_diff_mask]- the_min)/(the_max-the_min)
    return normalized

def create_diff_image(first_data,base_data,to_diff_mask):
    """first_data - fits data of bluer band
    base_data - fits data of redder band
    to_diff_mask - a single boolean mask indicating where to create diff (1's = included, 0's = ignored)"""
    first_norm = normalize_array(first_data,to_diff_mask)
    base_norm = normalize_array(base_data,to_diff_mask)
    return first_norm-base_norm

def find_mask_spot_closest_to_center(the_mask,approx_center):
    """Locate spots in mask, and mask out all that are not center most"""
    shape=the_mask.shape
    all_labels = measure.label(the_mask) #https://scipy-lectures.org/packages/scikit-image/auto_examples/plot_labels.html
    blobs_labels = measure.label(all_labels, background=0)
    
    unique = np.unique(blobs_labels.flatten()) #https://stackoverflow.com/a/28663910/13544635
    center_of_masses = scipy.ndimage.center_of_mass(np.ones(shape),blobs_labels, index=unique)
    dist_to_center = list(map(lambda i: np.inf if i == 0 else calculate_dist(center_of_masses[i],approx_center),range(len(center_of_masses))))

    return blobs_labels==unique[np.argmin(dist_to_center)]

def get_gopher_params_from_sersic_fit(inital_gofher_parameters,data,mask_to_fit,sigma=None):
    """Fit a sersic to galaxy and extract ellipse params from it"""
    sersic_model = fit_sersic(data, inital_gofher_parameters.b*0.5, 
               inital_gofher_parameters.x,
               inital_gofher_parameters.y,
               inital_gofher_parameters.a,
               inital_gofher_parameters.b,
               inital_gofher_parameters.theta,
               mask_to_fit,
               sigma_clip_for_fitting=sigma)
    
    sersic_output_gofher_params = gofher_parameters()
    sersic_output_gofher_params.x = getattr(sersic_model,'x_0').value
    sersic_output_gofher_params.y = getattr(sersic_model,'y_0').value
    sersic_output_gofher_params.theta = getattr(sersic_model,'theta').value
    return sersic_output_gofher_params

def run_gofher_on_galaxy(the_gal,the_band_pairs):
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
    el_mask = create_ellipse_mask_from_gofher_params(inital_gofher_parameters,shape,r=1.0)

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
    #if True:
    try:
        the_sersic_mask = np.logical_and(the_mask,the_gal[the_gal.ref_band].valid_pixel_mask)
        the_gal.gofher_params = get_gopher_params_from_sersic_fit(inital_gofher_parameters,data,the_sersic_mask)
        the_gal.gofher_params.a = the_el_sep['a']
        the_gal.gofher_params.b = the_el_sep['b']
    except:
        print("Error fitting sersic, using inital_gofher_parameters from sep")
        the_gal.gofher_params = inital_gofher_parameters
        the_gal.encountered_sersic_fit_error = True

    #Step 7: Now run gofher!
    the_gal.run_gofher(the_band_pairs)

    return the_gal

def run_gofher_on_galaxy_with_fixed_gofher_parameters(the_gal,the_band_pairs,fixed_gofher_params):
    the_gal.gofher_params = fixed_gofher_params
    #print(the_band_pairs)
    the_gal.run_gofher(the_band_pairs)

    return the_gal

def run_gofher_on_galaxy_with_fixed_center_only(the_gal,the_band_pairs,fixed_gofher_params):
    """Figure out which side is closer given diffenetial extinction reddening location"""
    #Step 1: Setup inital necessary variables:
    inital_gofher_parameters = gofher_parameters()
    data = copy.deepcopy(the_gal[the_gal.ref_band].data)
    shape = the_gal.get_shape()

    #Step 2: Run Sep on ref_band and find inital_gofher_parameters
    (cm_x, cm_y) = (shape[1]*0.5, shape[0]*0.5)
    the_el_sep, mu_bkg = run_sep(data, cm_x, cm_y)
    inital_gofher_parameters.load_from_sep_object(the_el_sep)

    #Step 3: Manually Fix center:
    inital_gofher_parameters.x = fixed_gofher_params.x
    inital_gofher_parameters.y = fixed_gofher_params.y
    the_gal.gofher_params = inital_gofher_parameters

    #Step 4: Now run gofher!
    the_gal.run_gofher(the_band_pairs)

    return the_gal

def run_gofher_on_galaxy_with_sparcfire_center_inital_guess(the_gal,the_band_pairs,fixed_gofher_params):
    #Step 1: Setup inital necessary variables:
    inital_gofher_parameters = gofher_parameters()
    data = copy.deepcopy(the_gal[the_gal.ref_band].data)
    shape = the_gal.get_shape()

    #Step 2: Run Sep on ref_band and find inital_gofher_parameters
    (cm_x, cm_y) = (shape[1]*0.5, shape[0]*0.5)
    the_el_sep, mu_bkg = run_sep(data, cm_x, cm_y)
    inital_gofher_parameters.load_from_sep_object(the_el_sep)

    #Step 3: Manually Fix center:
    inital_gofher_parameters.x = fixed_gofher_params.x
    inital_gofher_parameters.y = fixed_gofher_params.y
    
    #Step 4: Create an ellipse mask using the inital_gofher_parameters
    el_mask = create_ellipse_mask_from_gofher_params(inital_gofher_parameters,shape,r=1.0)

    #Step 5: Using the inital_gofher_parameters fit two pdfs pdf_in (probability inside ellipse) and pdf_out (probability outside ellipse)
    inside_ellipse = data[np.logical_and(el_mask,the_gal[the_gal.ref_band].valid_pixel_mask)].flatten()
    loc, scale = expon.fit(inside_ellipse) #https://stackoverflow.com/questions/25085200/scipy-stats-expon-fit-with-no-location-parameter
    pdf_in = expon.pdf(data, loc=loc, scale=scale)

    outside_ellipse = data[np.logical_and(np.logical_not(el_mask),the_gal[the_gal.ref_band].valid_pixel_mask)].flatten()
    loc, scale = expon.fit(outside_ellipse) #https://stackoverflow.com/questions/25085200/scipy-stats-expon-fit-with-no-location-parameter
    pdf_out = expon.pdf(data, loc=loc, scale=scale)

    #Step 6: Create a mask that only includes pixels in which there is a higher probability that it in inside the inital ellipse compared to the probability it is outside the ellipse
    the_mask = pdf_out < pdf_in

    center_mask = find_mask_spot_closest_to_center(the_mask,(cm_x, cm_y))
    bright_spot_mask = np.logical_and(the_mask,np.logical_not(center_mask))

    the_mask = np.logical_and(center_mask,the_gal[the_gal.ref_band].valid_pixel_mask)

    #Step 7: Try fitting sersic:
    #if True:
    try:
        the_sersic_mask = np.logical_and(the_mask,the_gal[the_gal.ref_band].valid_pixel_mask)
        the_gal.gofher_params = get_gopher_params_from_sersic_fit(inital_gofher_parameters,data,the_sersic_mask)
        the_gal.gofher_params.a = the_el_sep['a']
        the_gal.gofher_params.b = the_el_sep['b']
    except:
        print("Error fitting sersic, using inital_gofher_parameters from sep")
        the_gal.gofher_params = inital_gofher_parameters
        the_gal.encountered_sersic_fit_error = True

    #Step 8: Now run gofher!
    the_gal.run_gofher(the_band_pairs)

    return the_gal
