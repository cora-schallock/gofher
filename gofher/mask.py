import sep
import numpy as np

from matrix import create_disk_angle_matrix

def create_valid_pixel_mask(data: np.ndarray):
    """Create a mask of all valid pixels (pixels that are not nan's nor infs)
    
    Args:
        data: the data to mask against
    """
    return np.logical_not(np.logical_or(np.isinf(data),np.isnan(data)))

def create_ellipse_mask(x,y,a,b,theta,r=1.0,shape=None,arr=None):
    """Create an ellipse mask with center (x,y), angle of semi-major axis of theta, semi-major/semi-minor axis length of a/b, and scaling r

    Args:
        x: x-cordinate of center (h)
        y: y-cordinate of center (k)
        theta: angle of semi-major axis
            IMPORTANT: theta position angle is in radians counter clockwise from positive x axis to major axis, and lies in range [-pi/2, pi/2]
            For more information see: https://sep.readthedocs.io/en/stable/api/sep.ellipse_coeffs.html?highlight=theta
        a: semi-major axis length
        b: semi-minor axis length
        r: scaling factor of ellipse
            semi-major=r*a semi-minor=r*b
        shape: shape of ellipse mask
            Note: if none, must provide array arr for operation done inplace
        arr: arr to create ellipse_mask in
            Note: if none, must provide shape

    Returns:
        Ellipse Mask
    """
    if not isinstance(arr,np.ndarray):
        if shape is None:
            raise ValueError('If arr is None, shape can not be none.')
        arr = np.full(shape, False)

    sep.mask_ellipse(arr, x, y, a, b, theta, r)

    return arr

def create_bisection_mask(x,y,theta,shape):
    """Create 2 bisections mask (pos_mask, neg_mask)
    Note: When a disk_angle_matrix is created and values are [0,2*pi]
    pos_mask: the pixels in which value < pi
    neg_mask: the pixels in which value >= pi

    Args:
        x: x-cordinate of center (h)
        y: y-cordinate of center (k)
        theta: angle of semi-major axis
        shape: shape of bisection mask

    Returns:
        (pos_mask,neg_mask)
    """
    disk_angle_matrix = create_disk_angle_matrix(x,y,theta,shape)
    pos_mask = (disk_angle_matrix<np.pi)
    neg_mask = (disk_angle_matrix>=np.pi)
    
    return (pos_mask, neg_mask)