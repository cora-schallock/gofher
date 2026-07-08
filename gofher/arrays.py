"""Arrays used by GOFHER to create image masks

This module provides a collection of predefined np.arrays to aid in
the creation of binary image masks GOFHER uses.
"""


import numpy as np

from utils import (
    is_float_int, 
    is_2d_array_shape, 
    is_float_int_array, 
    is_2d_bool_array, 
    is_2d_same_shape_arrays, 
    is_finite_array
)

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
    if not is_float_int(cx):
        raise ValueError("cx must be float/int or numpy equivalent")
    
    if not is_float_int(cy):
        raise ValueError("cy must be float/int or numpy equivalent")
    
    if not is_2d_array_shape(shape):
        raise ValueError("shape must be tuple containing exactly 2 ints")
    
    # Calaulte distance from point (cx,cy) to all meshgrid elements
    xx, yy = create_meshgrid(shape)
    return np.sqrt((xx - cx)**2 + (yy - cy)**2)

def create_angle_array(cx: float, cy: float, theta: float, shape: tuple) -> np.ndarray:
    """Creates an array of angles from line with slope theta through (cx,cy)

    Each entry contains the angle measured in radians between the line specified
    and the index taken as position in cartesian space. Angle used is smallest
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
    if not is_float_int(cx):
        raise ValueError("cx must be float/int or numpy equivalent")
    
    if not is_float_int(cy):
        raise ValueError("cy must be float/int or numpy equivalent")
    
    if not is_float_int(theta):
        raise ValueError("theta must be float/int or numpy equivalent")
    
    if not is_2d_array_shape(shape):
        raise ValueError("shape must be tuple containing exactly 2 ints")
    
    # Calculate ang. between line through (cx,cy)  to all meshgrid elements:
    xx, yy = create_meshgrid(shape)
    angles = np.arctan2(yy-cy, xx-cx) - theta

    # Normalize angles so in range [-pi ,pi]
    return (angles + np.pi) % (2 * np.pi) - np.pi

def create_major_axis_angle_array(cx: float, cy: float, 
                                  theta: float, shape: tuple) -> np.ndarray:
    """Creates an array of angles in radians from major axis of ellipse

    Each element is closest angle to major axis, and sign indicates direction.

    Important: 
        range of value is [-pi/2,pi/2]
        positive elements are counter clockwise from closest point on minor axis
        negative elements are clockwise from closest point on minor axis
    
    Args:
        cx: x coordinate of center of ellipse
        cy: y coordinate of center of ellipse
        theta: the ang. major axis counter clockwise from positive x-axis
        shape: the shape of the array (assumes 2D array)
        
    Returns:
        Angle from major axis in radians
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

def create_minor_axis_angle_array(cx: float, cy: float, 
                                  theta: float, shape: tuple) -> np.ndarray:
    """Creates an array of angles in radians from minor axis of ellipse.

    Each element is closest angle to minor axis, and sign indicates direction.

    Important: 
        theta is meausring *MAJOR* axis, not minor axis
        range of value is [-pi/2,pi/2]
        positive elements are counter clockwise from closest point on minor axis
        negative elements are clockwise from closest point on minor axis
    
    Args:
        cx: x coordinate of center of ellipse
        cy: y coordinate of center of ellipse
        theta: the ang. major axis counter clockwise from positive x-axis
        shape: the shape of the array (assumes 2D array)
        
    Returns:
        Angle from minor axis in radians
    """

    # Take angle array:
    angle_array_offset = create_angle_array(cx,cy,theta,shape)

    # Mod angle_array_offset with pi to get:
    #    major axis = +/-pi/2
    #    minor axis = 0
    # Then Take absolute value so both sides of y-axis are positive:
    #    major axis = pi/2
    #    minor axis = 0, pi
    # Finally subtract pi/2:
    #    major axis = 0
    #    minor axis = +/-pi/2

    return np.abs(np.mod(angle_array_offset,np.pi))-np.pi/2

def normalize_array(array: np.ndarray, 
                    normalize_mask: np.ndarray | None = None)-> np.ndarray:
    """Normalize array so that all normalize_mask True values are between [0,1]
    (i.e. max array[normalize_mask] -> 1.0 & min array[normalize_mask] -> 0.0),
    and 0 elsewhere. 
    
    Note: If no normalize_mask is provided, normalizes all values
    
    Args:
        array: array to be normalized
        normalize_mask: boolean array specifying alements to be normalized
        theta: the ang. major axis counter clockwise from positive x-axis
        shape: the shape of the array (assumes 2D array)
        
    Returns:
        Angle from minor axis in radians
    """
    # If no normalize_mask is provided, normalize all values:
    if normalize_mask is None:
        normalize_mask = np.ones_like(array, dtype=bool)
    
    # Validate input:
    if not is_2d_bool_array(normalize_mask):
        raise ValueError("array must be 2D np.ndarray of bools")
    
    if not is_2d_same_shape_arrays(array, normalize_mask): 
        raise ValueError("array and normalize_mask must be same shape")
    
    if not is_float_int_array(array[normalize_mask]):
        raise ValueError("normalized array must be np.ndarray of int/floats")
    
    if not is_finite_array(array[normalize_mask]):
        raise ValueError("normalized array must be finite values (no NaN/inf)")
    
    # Create normalization array:
    normalized_array = np.zeros(array.shape)
    the_max = np.max(array[normalize_mask])
    the_min = np.min(array[normalize_mask])

    # If min ~= max, set all normed values to 1.0 (avoises division by 0):
    if np.isclose(the_max, the_min):
        normalized_array[normalize_mask] = 1.0
        return normalized_array

    normalized_array[normalize_mask] = (array[normalize_mask] - the_min) / (the_max - the_min)
    return normalized_array
