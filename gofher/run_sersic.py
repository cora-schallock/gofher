
import numpy as np
import warnings
from astropy.modeling import models, fitting
from astropy.utils.exceptions import AstropyUserWarning
from photutils.isophote import EllipseGeometry, Ellipse
import matplotlib.pyplot as plt
import scipy
from skimage import color, morphology

from bright_spot import create_bright_spot_mask, combine_spot_and_star_mask, find_mask_spot_closest_to_center
#from run_fit import run_cutom_segmentation #fit_ellipse_to_mask,

from fits import view_fits

def evaluate_sersic_model(sersic_model, shape):
    x,y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    return sersic_model(x,y)

def inital_ellipticity(a,b):
    #The ellipticity: e = 1 − b/a #https://iopscience.iop.org/article/10.3847/0004-637X/824/2/112
    return 1 - (b/a)

def inital_amplitude(data, r_eff, x, y, ellip, theta):
    #https://petrofit.readthedocs.io/en/latest/fitting.html#AstroPy-S%C3%A9rsic-Model
    #return get_amplitude_at_r(r_eff, data, x, y, e, theta)
    g = EllipseGeometry(x, y, 1., ellip, theta)

    # Create Ellipse model
    ellipse = Ellipse(data, geometry=g)

    # Fit isophote at r_eff
    iso = ellipse.fit_isophote(r_eff)

    # Get flux at r_eff
    amplitude = iso.intens
    
    return amplitude

def fit_sersic(data, r_eff, x, y, a, b, theta, to_fit_sersic_mask, inital_n=4):
    #Step 1) Find inital params for sersic:
    ellip = inital_ellipticity(a,b)
    amplitude = inital_amplitude(data, r_eff, x, y, ellip, theta)

    #Step 2) Set inital sersic with inintal params (and bounds):
    """
    sersic_model_init = models.Sersic2D(
        amplitude=amplitude,
        r_eff=r_eff,
        n=inital_n,
        x_0=x,
        y_0=y,
        ellip=ellip,
        theta=theta,

        bounds = {
            'amplitude': (amplitude*0.5, amplitude*2),
            'r_eff': (r_eff*0.5, r_eff*2),
            'n': (0.5, 10),
            'ellip': (0, 1),
            'theta': (theta-theta_buffer, theta+theta_buffer),
            'x_0': (x - center_buffer/2, x + center_buffer/2),
            'y_0': (y - center_buffer/2, y + center_buffer/2)})
    """
    sersic_model_init = models.Sersic2D(
        amplitude=amplitude,
        r_eff=r_eff,
        n=inital_n,
        x_0=x,
        y_0=y,
        ellip=ellip,
        theta=theta)

    #Step 3) Fit model to grid:
    y_arange, x_arange = np.where(to_fit_sersic_mask)
    z = data[(y_arange, x_arange)]
    fitter = fitting.LevMarLSQFitter()
    
    with warnings.catch_warnings(record=True) as w:  #https://stackoverflow.com/a/5645133
        sersic_model_final = fitter(sersic_model_init, x_arange, y_arange, z, maxiter=500)

    return sersic_model_final