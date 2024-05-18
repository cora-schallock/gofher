import sep
import numpy as np

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

def create_bisection_mask(x,y,theta,shape):
    """create two bisection masks"""
    disk_angle_matrix = create_disk_angle_matrix(x,y,theta,shape)
    pos_mask = (disk_angle_matrix<np.pi)
    neg_mask = (disk_angle_matrix>=np.pi)
    
    return (pos_mask, neg_mask)