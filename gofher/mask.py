import sep
import numpy as np
from astropy.stats import sigma_clipped_stats, sigma_clip
from skimage.morphology import remove_small_objects

from matrix import create_disk_angle_matrix

def create_valid_pixel_mask(data):
    """create a mask that only contains not infinite non nan pixels"""
    return np.logical_not(np.logical_or(np.isinf(data),np.isnan(data)))

def create_ellipse_mask(x,y,a,b,theta,r=1.0,shape=None,arr=None):
    """create a mask of pixels inside the ellipse"""
    if not isinstance(arr,np.ndarray):
        if shape is None:
            raise ValueError('If arr is None, shape can not be none.')
        arr = np.full(shape, False)

    sep.mask_ellipse(arr, x, y, a, b, theta, r)

    return arr

def create_ellipse_mask_from_gofher_params(the_params,the_shape,r=1.0):
    """using gofher_params specifying an ellipse create an ellipse mask"""
    return create_ellipse_mask(the_params.x,the_params.y,the_params.a,the_params.b,the_params.theta,r,the_shape)

def create_bisection_mask(x,y,theta,shape):
    """create two bisection masks"""
    disk_angle_matrix = create_disk_angle_matrix(x,y,theta,shape)
    pos_mask = (disk_angle_matrix<np.pi)
    neg_mask = (disk_angle_matrix>=np.pi)
    
    return (pos_mask, neg_mask)

def create_bisection_mask_from_gofher_params(the_params,shape):
    """using gofher_params create a bisection mask"""
    return create_bisection_mask(the_params.x,the_params.y,the_params.theta,shape)