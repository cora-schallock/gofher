import sep
import numpy as np
from astropy.stats import sigma_clipped_stats, sigma_clip
from skimage.morphology import remove_small_objects

from matrix import create_disk_angle_matrix

def create_valid_pixel_mask(data):
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

def clean_mask(mask,min_size=64,connectivity=1):
    """clean mask by removing small noise"""
    return remove_small_objects(mask.astype(bool),min_size,connectivity)

def create_foreground_mask(data,min_area_frac=0.0625,connectivity=9):
    """create a cleaned foreground mask"""
    #Step 1) calculate clipped stats:
    image_mean, image_median, image_stddev = sigma_clipped_stats(data, sigma=3) #estimate background with astropy

    #Step 2) select foreground based on clipped stats
    foreground = data >= image_mean+image_stddev
    
    #Step 3) clean foreground
    min_size = data.shape[0]*data.shape[1]*min_area_frac
    return clean_mask(foreground,min_size,connectivity)

def create_segmentation_masks(data):
    """cretae a foreground and background mask"""
    foreground_mask = create_foreground_mask(data)
    background_mask = np.logical_not(foreground_mask)

    return (foreground_mask,background_mask)