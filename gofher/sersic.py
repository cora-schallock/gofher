import scipy
import warnings
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, morphology
from astropy.modeling import models, fitting
from astropy.utils.exceptions import AstropyUserWarning
from photutils.isophote import EllipseGeometry, Ellipse

from gofher_parameters import gofher_parameters

def get_gopher_params_from_sersic_fit(inital_gofher_parameters,data,mask_to_fit,sigma=None):
    """Fine tune gofher parameters using a sersic fitting
    Uses (x,y) and theta from sersic
    (a,b) from inital_gofher_parameters
    
    Args:
        inital_gofher_parameters: the inital gofher parameters describing region sersic should be fit on
        data: data to fit sersic on
        mask_to_fit: masks data to sersic fit on
        sigma: for sigma clipping prior to sersic fitting
    """

    #See: Schallock et. al 2024 - Algorithm 1 Ellipse Mask Fitting

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

def evaluate_sersic_model(sersic_model, shape):
    """Evaluates sersic model and outputs it to np.array with given shape
    
    Args:
        sersic_model: sersic model to evaluate
        shape: size of sersic model
    """
    x,y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    return sersic_model(x,y)

def inital_ellipticity(a,b) -> float:
    """calculate inital ellipticity e=1-(a/b)
    
    Args:
        a: semi-major axis length
        b = semi-minor axis length
    """
    #The ellipticity: e = 1 − b/a #https://iopscience.iop.org/article/10.3847/0004-637X/824/2/112
    return 1 - (b/a)

def inital_amplitude(data, r_eff, x, y, ellip, theta):
    """calculate inital amplitude of data
    See: https://petrofit.readthedocs.io/en/latest/fitting.html#AstroPy-S%C3%A9rsic-Model

    Args:
        data: data to evaluate on
        r_eff: effective radius
        x: x cordinate of center
        y: y cordinate of cneter
        ellip: ellipticity
        theta: angle
    """

    g = EllipseGeometry(x, y, 1., ellip, theta)

    # Create Ellipse model
    ellipse = Ellipse(data, geometry=g)

    # Fit isophote at r_eff
    iso = ellipse.fit_isophote(r_eff)

    # Get flux at r_eff
    amplitude = iso.intens
    
    return amplitude

def fit_sersic(data, r_eff, x, y, a, b, theta, to_fit_sersic_mask, inital_n=4, sigma_clip_for_fitting=None,center_buffer=6,theta_buffer=np.pi/16):
    """Performs a bounded sersic fitting

    Args:
        data: data to fit on
        r_eff: effective radius
        x: x cordinate of center
        y: y cordinate of center
        a: semi-major axis length
        b: semi-minor axis length
        theta: angle of disk
        to_fit_sersic_mask: mask of data to fit sersic on
        inital_n: inital sersic index
        sigma_clip_for_fitting: specifies is uses sigma_clipping for evaluation
            Note: if none no sigma_clipping is performed
        center_buffer: the center bounds of the bounded sersic fitting
             x range: (x - center_buffer/2, x + center_buffer/2)
            y range: (y - center_buffer/2, y + center_buffer/2)})
        theta_buffer: the theta bounds of the bounded sersic fitting
            theta range: (theta-theta_buffer, theta+theta_buffer)

    Returns:
        sersic_model_final
    """
    #Step 1) Find inital params for sersic:
    ellip = inital_ellipticity(a,b)
    amplitude = inital_amplitude(data, r_eff, x, y, ellip, theta)

    #Step 2) Set inital sersic with inintal params (and bounds):
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


    #Step 3) Fit model to grid:
    if sigma_clip_for_fitting != None:
        mean = np.mean(data[to_fit_sersic_mask])
        sig = np.std(data[to_fit_sersic_mask])
        in_range_mask = (mean-sigma_clip_for_fitting*sig<data)&(data<mean+sigma_clip_for_fitting*sig)
        to_fit_sersic_mask = np.logical_and(to_fit_sersic_mask,in_range_mask)

    y_arange, x_arange = np.where(to_fit_sersic_mask)
    z = data[(y_arange, x_arange)]
    fitter = fitting.LevMarLSQFitter()
    
    with warnings.catch_warnings(record=True) as w:  #https://stackoverflow.com/a/5645133
        sersic_model_final = fitter(sersic_model_init, x_arange, y_arange, z, maxiter=500)

    return sersic_model_final