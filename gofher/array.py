"""Arrays used by GOFHER to create image masks

This module provides a collection of predefined np.arrays to aid in
the creation of binary image masks GOFHER uses.
"""


import numpy as np

from utils import is_float, is_2d_array_shape

def create_meshgrid(shape: tuple) -> tuple[np.ndarray, np.ndarray]:
    """creates a standard meshgrid with given shape"""
    # Validate input:
    if not is_2d_array_shape(shape):
        raise ValueError("shape must be tuple containing exactly 2 ints")
    
    # Create meshgrid:
    x = np.arange(shape[0])
    y = np.arange(shape[1])
    return np.meshgrid(x, y)


def create_distance_array(cx: float, cy: float, shape: tuple) -> np.ndarray:
    """Creates an array where each element is euclidian distance from point (h,k)

    i.e. distance_array[10,50] is L2 euclidian distance from
    (50,10) and point point (cx,cy)
    
    Args:
        cx: x-axis value of point to calulcate distance from
        cy: y-axis value of point to calulcate distance from
        shape: the shape of the array (assumes 2D array)
        
    Returns:
        distance matrix
    """
    # Validate input:
    if not is_float(cx):
        raise ValueError("cx must be float/int or numpy equivalent")
    
    if not is_float(cy):
        raise ValueError("cy must be float/int or numpy equivalent")
    
    if not is_2d_array_shape(shape):
        raise ValueError("shape must be tuple containing exactly 2 ints")
    
    # Calaulte distance from point (cx,cy) to all meshgrid elements
    xx, yy = create_meshgrid(shape)
    return np.sqrt((xx - cx)**2 + (yy - cy)**2)

def create_angle_array(cx: float, cy: float, theta: float, shape: tuple) -> np.ndarray:
    """Creates an array of angles from line with slope theta through (cx,cy)

    Each entry contains the angle measured in radians between the line specified
    and the index taken as position in cartesian space. Angel used is smallest
    absolute value angle (i.e. 1.5*pi -> -0.5*pi)

    i.e. angle_array[0,10] is angle from point (10,0) and line with slope theta
    passing through the point (cx,cy)

    Important: range of value is [-pi ,pi]
    
    Args:
        cx: x coordinate of point that line passes through
        cy: y coordinate of point line line passes through
        theta: the ang. of line in rads. counter clockwise from positive x-axis
        shape: the shape of the array (assumes 2D array)
        
    Returns:
        angle array in radians
    """
    # Validate input:
    if not is_float(cx):
        raise ValueError("cx must be float/int or numpy equivalent")
    
    if not is_float(cy):
        raise ValueError("cy must be float/int or numpy equivalent")
    
    if not is_float(theta):
        raise ValueError("theta must be float/int or numpy equivalent")
    
    if not is_2d_array_shape(shape):
        raise ValueError("shape must be tuple containing exactly 2 ints")
    
    # Calculate ang. between line through (cx,cy)  to all meshgrid elements:
    xx, yy = create_meshgrid(shape)
    return np.arctan2(yy-cy, xx-cx) - theta

def create_major_axis_angle_array(cx: float, cy: float, theta: float, shape: tuple):
    """Creates an array of angles from major axis of ellipse

    Important: range of value is [-pi/2,pi/2]
    
    Args:
        cx: x coordinate of center of ellipse
        cy: y coordinate of center of ellipse
        theta: the ang. major axis counter clockwise from positive x-axis
        shape: the shape of the array (assumes 2D array)
        
    Returns:
        Angle from major axis
    """

    # Take angle array and +pi/2 to all values for subsequent transformation:
    angle_array_offset = create_angle_array(cx,cy,theta, shape)+np.pi/2

    # Mod angle_array_offset with pi to get:
    #    major axis = pi/2
    #    minor axis = 0, +/-pi
    # Then Take absolute value so both sides of y-axis are positive:
    #    major axis = pi/2
    #    minor axis = 0, pi
    # Finally subtract pi/2:
    #    major axis = 0
    #    minor axis = +/-pi/2

    return np.abs(np.mod(angle_array_offset,np.pi))-np.pi/2
