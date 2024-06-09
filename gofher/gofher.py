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



def calculate_dist(coord0: tuple, coord1: tuple)-> float:
    """L2 Distance Norm between coord0 and coord1 coordinates
        Args:
            coord0: the first coordinate
            coord1: the second coordinate
        Returns:
            the euclidian distance between cm and center
    """
    return np.linalg.norm(np.array(coord0)-np.array(coord1))

def find_mask_spot_closest_to_center(the_mask: np.ndarray, approx_center: tuple) -> np.ndarray:
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

def run_default_gofher_parameters_fitting(the_gal: galaxy) -> galaxy:
    """Finds gofher_parameters for the galaxy that will be used to generate the ellipse masks and bisection masks

        Args:
            the_gal: the galaxy
        Returns:
            the galaxy with gofher_parameters found using default fitting process
    """

    #See: Schallock & Hayes 2024 - Algorithm 1 Ellipse Mask Fitting

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

def run_fixed_center_gofher_parameter_fitting(the_gal: galaxy, inital_gofher_params: gofher_parameters) -> galaxy:
    """Using a fixed center from inital_gofher_params (x,y), fits a/b/theta from sep

        Args:
            the_gal: the galaxy
            inital_gofher_params: the gofher_parameters provided that specify the fixed (x,y) center
            IMPORTANT: This (x,y) center from inital_gofher_params is used in the heuristic that is used to select the primary sep object
        Returns:
            the galaxy with gofher_parameters found using fixed center
    """

    #Step 1: Setup inital necessary variables:
    fixed_center_gofher_parameters = gofher_parameters()
    data = copy.deepcopy(the_gal[the_gal.ref_band].data)

    #Step 2: Run Sep on ref_band and find inital_gofher_parameters
    (cm_x, cm_y) = (inital_gofher_params.x, inital_gofher_params.y)
    the_el_sep, mu_bkg = run_sep(data, cm_x, cm_y)
    fixed_center_gofher_parameters.load_from_sep_object(the_el_sep)

    #Step 3: Manually Fix center to (x,y) from inital_gofher_params:
    fixed_center_gofher_parameters.x = inital_gofher_params.x
    fixed_center_gofher_parameters.y = inital_gofher_params.y

    #Step 4: Set galaxy's gofher_parameters to those found fixed_center_gofher_parameters
    the_gal.gofher_params = fixed_center_gofher_parameters

    return the_gal

def run_inital_gofher_parameters_fitting(the_gal: galaxy, inital_gofher_parameters: gofher_parameters) -> galaxy:
    """Using a inital ellipse specified by inital_gofher_parameters, fits ellipse by comparing pdf>pdf_out and doing a bounded sersic fit

        Args:
            the_gal: the galaxy
            inital_gofher_params: the gofher_parameters provided that specify the fixed (x,y) center
            IMPORTANT: This (x,y) center from inital_gofher_params is used in the heuristic that is used to select the center most spot in pdf_in > pdf_out
        Returns:
            the galaxy with gofher_parameters found using inital ellipse from inital_gofher_parameters
    """

    #Step 1: Create an ellipse mask using the inital_gofher_parameters
    data = copy.deepcopy(the_gal[the_gal.ref_band].data)
    shape = the_gal.get_shape()
    el_mask = inital_gofher_parameters.create_ellipse_mask(shape,r=1.0)
    (cm_x, cm_y) = (inital_gofher_parameters.x,inital_gofher_parameters.y)

    #Step 2: Using the inital_gofher_parameters fit two pdfs pdf_in (probability inside ellipse) and pdf_out (probability outside ellipse)
    inside_ellipse = data[np.logical_and(el_mask,the_gal[the_gal.ref_band].valid_pixel_mask)].flatten()
    loc, scale = expon.fit(inside_ellipse) #https://stackoverflow.com/questions/25085200/scipy-stats-expon-fit-with-no-location-parameter
    pdf_in = expon.pdf(data, loc=loc, scale=scale)

    outside_ellipse = data[np.logical_and(np.logical_not(el_mask),the_gal[the_gal.ref_band].valid_pixel_mask)].flatten()
    loc, scale = expon.fit(outside_ellipse) #https://stackoverflow.com/questions/25085200/scipy-stats-expon-fit-with-no-location-parameter
    pdf_out = expon.pdf(data, loc=loc, scale=scale)

    #Step 3: Create a mask that only includes pixels in which there is a higher probability that it in inside the inital ellipse compared to the probability it is outside the ellipse
    the_mask = pdf_out < pdf_in

    center_mask = find_mask_spot_closest_to_center(the_mask,(cm_x, cm_y))
    bright_spot_mask = np.logical_and(the_mask,np.logical_not(center_mask))

    the_mask = np.logical_and(center_mask,the_gal[the_gal.ref_band].valid_pixel_mask)

    #Step 4: Try fitting sersic:
    try:
        the_sersic_mask = np.logical_and(the_mask,the_gal[the_gal.ref_band].valid_pixel_mask)
        the_gal.gofher_params = get_gopher_params_from_sersic_fit(inital_gofher_parameters,data,the_sersic_mask)
        the_gal.gofher_params.a = inital_gofher_parameters.a
        the_gal.gofher_params.b = inital_gofher_parameters.b
    except:
        print("Error fitting sersic, using provided inital_gofher_parameters")
        the_gal.gofher_params = inital_gofher_parameters
        the_gal.encountered_sersic_fit_error = True

    return the_gal

def run_gofher(name,fits_path_function,blue_to_red_bands_in_order,ref_bands_in_order,paper_label=None):
    """Runs gofher of a single galaxy - using default_gofher_parameters_fitting

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
    #Create galaxy:
    the_gal = galaxy(name,paper_label)

    #Constuct individual galaxy bands:
    for band in blue_to_red_bands_in_order:
        the_gal.construct_band(band,fits_path_function(name,band))

    #Find reference band for galaxy:
    for ref_band in ref_bands_in_order:
        if the_gal.has_valid_band(ref_band):
            the_gal.ref_band = ref_band
            break
    
    if the_gal.ref_band == "":
        raise ValueError("run_gofher")
    
    #Generate the_band_pairs:
    the_band_pairs = list(itertools.combinations(blue_to_red_bands_in_order, 2))

    #Use default gofher_parameter fitting:
    the_gal = run_default_gofher_parameters_fitting(the_gal)

    #Run gofher on the galaxy:
    the_gal.run_gofher(the_band_pairs)

    return the_gal

def run_gofher_with_parameters(name,fits_path_function,blue_to_red_bands_in_order,ref_band,inital_gofher_params,paper_label="",mode="fixed"):
    """Runs gofher of a single galaxy - using fixed parameteres (such as SpArcFiRe derived parameters)

        Args:
            name: the name of the galaxy
            fits_path_function: a function that takes in galaxy_name and waveband and returns the file paths of the fits files
            blue_to_red_bands_in_order: a list of the name of wavebands in order of Bluest to Reddest wavebands
            ref_band: the specific ref band to us2
            inital_gofher_params: the inital gofher parameters
                Note: This can be derived from a different fitting process such as SpArcFiRe
            paper_label: the baseline near side label gofher is comparing its answer to
                Important: If no baseline label, leave as none
            mode: specifies the run mode of gofher fitting
                inital: inital_gofher_params used as inital ellipse and further fitting is done - see: run_inital_gofher_parameters_fitting()
                fixed-center: inital_gofher_params center is fixed, but uses a and b found from sep - see: run_fixed_center_gofher_parameter_fitting()
                fixed: inital_gofher_params used as is by gofher
        Returns:
            the galaxy
    """

    #Check provided mode is valid:
    modes = ["inital","fixed-center","fixed"]
    
    if mode.strip().lower() in modes:
        mode = mode.strip().lower()
    else:
        print('Warning: run_gofher_with_parameters() mode {} is invalid, must be either ["inital","fixed-center","fixed"], using mode "fixed"')

    #Create galaxy:
    the_gal = galaxy(name,paper_label)
    
    #Constuct individual galaxy bands:
    for band in blue_to_red_bands_in_order:
        the_gal.construct_band(band,fits_path_function(name,band))

    #Use provided ref_band:
    the_gal.ref_band = ref_band
    
    #Generate the_band_pairs:
    the_band_pairs = list(itertools.combinations(blue_to_red_bands_in_order, 2))

    if mode == "inital":
        #Use inital_gofher_params as inital guess for gofher fitting:
        the_gal = run_inital_gofher_parameters_fitting(the_gal, inital_gofher_params)
    elif mode == "fixed-center":
        #Use fixed center gofher_parameter fitting:
        the_gal = run_fixed_center_gofher_parameter_fitting(the_gal, inital_gofher_params)
    else:
        #Use provided inital_gofher_params:
        the_gal.gofher_params = inital_gofher_params

    #Run gofher on the galaxy:
    the_gal.run_gofher(the_band_pairs)

    return the_gal