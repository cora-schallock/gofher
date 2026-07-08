"""Masks used by GOFHER

This module provides a collection of 2D binary masks 
that can be used on the data.
"""

from arrays import create_meshgrid, create_major_axis_angle_array, create_minor_axis_angle_array
from utils import is_float_int, is_2d_array_shape

import numpy as np

def create_ellipse_mask(h: float, k: float, 
                        a: float, b: float, 
                        theta: float, shape: tuple,
                        r: float = 1.0) -> np.ndarray:
    """Create a binary ellipse mask of a titled ellipse
    
    Each pixel is 1 if center of pixel is in ellipse or on border, 0 otherwise.
    
    h: x coordinate of center of ellipse
    k: y coordinate of center of ellipse
    a: semi-major axis length
    b: semi-major minor axis length
    theta: ang. in radians of major axis counter clockwise from positive x-axis
    shape: the shape of the array (assumes 2D array)
    r: scaling factor of ellipse (scales a and b by r)

    Returns:
        ellipse mask
    """
    # Validate input:
    if not is_float_int(h):
        raise ValueError("h must be float/int or numpy equivalent")
    
    if not is_float_int(k):
        raise ValueError("k must be float/int or numpy equivalent")
    
    if not is_float_int(a) or a <= 0:
        raise ValueError("a must be > 0 and float/int or numpy equivalent")
    
    if not is_float_int(b) or b <= 0:
        raise ValueError("b must be > 0 and float/int or numpy equivalent")
    
    if not is_float_int(theta):
        raise ValueError("theta must be float/int or numpy equivalent")
    
    if not is_2d_array_shape(shape):
        raise ValueError("shape must be tuple containing exactly 2 ints")
    
    if not is_float_int(r) or r <= 0:
        raise ValueError("r must be > 0 and float/int or numpy equivalent")
    
    # Calaulte distance from point (h,k) to all meshgrid elements:
    xx, yy = create_meshgrid(shape)
    xx_offset = xx-h
    yy_offset = yy-k

    # Calaulte terms in ellipse equation for titles ellipse:
    x_prime = xx_offset*np.cos(theta) + yy_offset*np.sin(theta)
    y_prime = -xx_offset*np.sin(theta) + yy_offset*np.cos(theta)

    ellipse_equation = (x_prime/(a*r))**2 + (y_prime/(b*r))**2
    return ellipse_equation <= 1

def create_near_major_axis_mask(sweep: float, h: float, k: float, 
                                theta: float, shape: tuple) -> np.ndarray:
    """Creates a binary mask indicating all points near ellipse major axis

    Given an ellipse with center (h,k) and a major axis angle theta,
    measured counter clockwise in radians from the positive x-axis,
    this function creates a mask indicating all points that are within
    a distance of sweep radians from the major axis.

    Sweep is the closest absolute radian distance from major axis.
    So allowed range is [0,pi/2] inclusive.

    Each pixel is 1 if center of pixel is within sweep radians of major axis, 
    0 otherwise.

    Note: This function includes pixels past boundary of ellipse.
    
    Args:
        sweep: distance from major axis in radians in inclusive range [0,pi/2]
        h: x coordinate of center of ellipse
        k: y coordinate of center of ellipse
        theta: ang. in radians of major axis counter clockwise from positive x-axis
        shape: the shape of the array (assumes 2D array)

    Returns:
        near major axis mask
    """
    # Validate input:
    if not is_float_int(sweep):
        raise ValueError("sweep must be and float/int or numpy equivalent")
    
    if not sweep >= 0 or not sweep <= np.pi/2:
        raise ValueError("sweep must be between 0 and pi/2")
    
    if not is_float_int(h):
        raise ValueError("h must be float/int or numpy equivalent")
    
    if not is_float_int(k):
        raise ValueError("k must be float/int or numpy equivalent")
    
    if not is_float_int(theta):
        raise ValueError("theta must be float/int or numpy equivalent")
    
    if not is_2d_array_shape(shape):
        raise ValueError("shape must be tuple containing exactly 2 ints")
    
    # Create mask (initally all False):
    near_major_axis_array = np.zeros((shape), bool)

    # Calculate angle from major axis:
    angle_from_maj_axis_array = create_major_axis_angle_array(h,k,theta,shape)

    # Set all points that are within sweep radians of major axis to true:
    near_major_axis_array[np.abs(angle_from_maj_axis_array) <= sweep] = True
    return near_major_axis_array

def create_near_minor_axis_mask(sweep: float, h: float, k: float, 
                                theta: float, shape: tuple) -> np.ndarray:
    """Creates a binary mask indicating all points near ellipse minor axis

    Given an ellipse with center (cx,cy) and a major axis angle theta,
    measured counter clockwise in radians from the positive x-axis,
    this function creates a mask indicating all points that are within
    a distance of sweep radians from the minor axis.

    Sweep is the closest absolute radian distance from minor axis.
    So allowed range is [0,pi/2] inclusive.

    Each pixel is 1 if center of pixel is within sweep radians of minor axis, 
    0 otherwise.

    Note: This function includes pixels past boundary of ellipse.
    
    Args:
        sweep: distance from minor axis in radians in inclusive range [0,pi/2]
        h: x coordinate of center of ellipse
        k: y coordinate of center of ellipse
        theta: ang. in radians of major axis counter clockwise from positive x-axis
        shape: the shape of the array (assumes 2D array)

    Returns:
        near major axis mask
    """
    # Validate input:
    if not is_float_int(sweep):
        raise ValueError("sweep must be and float/int or numpy equivalent")
    
    if not sweep >= 0 and sweep <= np.pi/2:
        raise ValueError("sweep must be between 0 and pi/2")
    
    if not is_float_int(h):
        raise ValueError("h must be float/int or numpy equivalent")
    
    if not is_float_int(k):
        raise ValueError("k must be float/int or numpy equivalent")
    
    if not is_float_int(theta):
        raise ValueError("theta must be float/int or numpy equivalent")
    
    if not is_2d_array_shape(shape):
        raise ValueError("shape must be tuple containing exactly 2 ints")
    
    # Create mask (initally all False):
    near_minor_axis_array = np.zeros((shape), bool)

    # Calculate angle from major axis:
    angle_from_min_axis_array = create_minor_axis_angle_array(h,k,theta,shape)

    # Set all points that are within sweep radians of major axis to true:
    near_minor_axis_array[np.abs(angle_from_min_axis_array) <= sweep] = True
    return near_minor_axis_array

"""
#TODO: show this follows an inverse tangent distro
import matplotlib.pyplot as plt

xs = []
ys = []
b = 256
s = 1000
for i in range(0,int(b/2)+1):
    ang = np.pi/b*i
    c = s/2 - 0.5
    print(c,s,ang)
    maj = create_near_major_axis_mask(ang,c,c,np.pi/21,(s,s))
    maj_top = np.sum(maj[0:int(s/2),:])
    maj_bottom = np.sum(maj[int(s/2):s,:])
    xs.append(maj_top)
    print(maj_top,maj_bottom)
    ys.append(i)
plt.scatter(xs,ys)
plt.show()
"""