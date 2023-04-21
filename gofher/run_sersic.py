
import numpy as np
import warnings
from astropy.modeling import models, fitting
from astropy.utils.exceptions import AstropyUserWarning
from photutils.isophote import EllipseGeometry, Ellipse
import matplotlib.pyplot as plt
import scipy
from skimage import color, morphology

from bright_spot import create_bright_spot_mask, combine_spot_and_star_mask, find_mask_spot_closest_to_center
from run_fit import run_cutom_segmentation #fit_ellipse_to_mask,

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

def fit_sersic(data, r_eff, x, y, a, b, theta, to_fit_sersic_mask, inital_n=4,center_buffer=10,theta_buffer=np.pi/6):
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
    y_arange, x_arange = np.where(to_fit_sersic_mask)
    z = data[(y_arange, x_arange)]
    fitter = fitting.LevMarLSQFitter()
    
    with warnings.catch_warnings(record=True) as w:  #https://stackoverflow.com/a/5645133
        sersic_model_final = fitter(sersic_model_init, x_arange, y_arange, z, maxiter=500)

    return sersic_model_final


def the_min(x):
    return np.min(x)

def run_custom_sersic(data,debug=False):
    shape=data.shape
    data, foreground_mask, background_mask, bulge_mask, star_mask = run_cutom_segmentation(data, False, debug)
    spot_mask = create_bright_spot_mask(data)
    to_mask = combine_spot_and_star_mask(spot_mask,bulge_mask,star_mask)
    
    if debug:
        print("run_fit - 1 spots")
        view_fits(data)
        view_fits(spot_mask)
        view_fits(to_mask)
    
    foreground_approx_mask_ns = np.logical_and(foreground_mask,np.logical_not(to_mask))
    background_approx_mask_ns = np.logical_and(background_mask,np.logical_not(to_mask))
    
    area_oi = np.logical_and(foreground_mask,np.logical_not(spot_mask))
    bulge_mean = np.mean(data[area_oi])+6*np.std(data[area_oi])

    foreground_ns_mean = np.mean(data[foreground_approx_mask_ns])
    background_ns_mean = np.mean(data[background_approx_mask_ns])
    
    c=(bulge_mean-foreground_ns_mean)*0.5 + foreground_ns_mean #bulge foreground cutoff
    d=(foreground_ns_mean-background_ns_mean)*0.5 + background_ns_mean #forground background cutoff
    
    cat_mask = np.ones(data.shape)*5
    cat_mask[data<=c] = 4
    cat_mask[data<=foreground_ns_mean] = 3
    cat_mask[data<=d] = 2
    cat_mask[data<=background_ns_mean] = 1
    
    cat_mask = scipy.ndimage.generic_filter(cat_mask,the_min,footprint=np.ones((5,5)))
    cat_mask[to_mask]=0
    
    if debug:
        print("run_fit - 2 categories")
        colored_by_label = color.label2rgb(cat_mask)
        fig, ax = plt.subplots() #https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D
        im = ax.imshow(colored_by_label, interpolation='nearest', origin='lower')
        plt.show()
    
    updated_foreground_mask = np.zeros(data.shape)
    updated_foreground_mask[cat_mask>=3] = 1.0
    
    if debug:
        print("run_fit - 3 updated_forground")
        view_fits(updated_foreground_mask)

    
    approx_center = (shape[1]/2,shape[0]/2)
    updated_foreground_mask = find_mask_spot_closest_to_center(updated_foreground_mask.astype(bool),approx_center)

    bulge_threshold = np.mean(data[updated_foreground_mask])+2.0*np.std(data[updated_foreground_mask])
    bulge_mask=(data*updated_foreground_mask)>bulge_threshold #use all
    bulge_mask=find_mask_spot_closest_to_center(bulge_mask.astype(bool),approx_center)
    
    if debug:
        print("run_fit - 4 updated_bulge")
        view_fits(bulge_mask)
        
    #view_fits(bulge_mask)
    el_x, el_y, el_a, el_b, el_theta_ra = fit_ellipse_to_mask(bulge_mask)

    updated_forgiving_foreground_mask = np.zeros(data.shape)
    updated_forgiving_foreground_mask[cat_mask>=2] = 1.0
    updated_forgiving_foreground_mask = find_mask_spot_closest_to_center(updated_forgiving_foreground_mask.astype(bool),approx_center)
    
    near_bulge_threshold = np.mean(data[updated_forgiving_foreground_mask])#-np.std(data[updated_forgiving_foreground_mask])
    near_bulge_mask=(data*updated_forgiving_foreground_mask)>near_bulge_threshold #use all
    near_bulge_mask=find_mask_spot_closest_to_center(near_bulge_mask.astype(bool),approx_center)
    
    if debug:
        print("run_fit - 5 near_bulge")
        view_fits(near_bulge_mask)

    #view_fits(near_bulge_mask)
    nel_x, nel_y, nel_a, nel_b, nel_theta_ra = fit_ellipse_to_mask(near_bulge_mask)
    
    #data=fix_data_spots(data,bkg.globalback,bkg.globalrms)
    #Not actually fit a sersic
    #data[near_bulge_mask] = 0.0 #big bug - fitting everything away from bulge found 2/1/2023
    data[np.logical_not(near_bulge_mask)] = 0.0 #updated
    if debug:
        print("data to fit sersic on")
        view_fits(data)
    gal_r_eff = nel_a*2
    x=el_x
    y=el_y
    a=nel_a
    b=nel_b
    theta=nel_theta_ra
    sersic_model_final = fit_sersic(data,gal_r_eff,x, y, a, b, theta,
                                                     center_buffer=4,
                                                     theta_buffer=np.pi/16)#np.pi/16
    #o theta buffer look better for some like IC1199, but not all IC1151
    #x = getattr(sersic_model_final,'x').value #try this?
    #y = getattr(sersic_model_final,'y').value #try this?
    return [x, y, nel_a, nel_b, theta, to_mask]