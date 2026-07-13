"""Misc. utility scripts utilized by various GOFHER scripts

"As it says on the tin!"
"""


import numpy as np

def is_int(value) -> bool:
    """Checks if the value is an int or numpy equivalent"""
    if isinstance(value, int):
        return True

    if isinstance(value, np.integer) and np.isscalar(value):
        return True

    return False

def is_float_int(value) -> bool:
    """Checks if the value is a float/int or numpy equivalent"""
    # Verify is python float or int:
    if isinstance(value, (float,int)):
        return True
    
    # Verify is numpy int/gloat and a single value (i.e. not array):
    if isinstance(value,(np.integer, np.floating)) and np.isscalar(value):
        return True
    
    return False
    
def is_2d_array_shape(value) -> bool:
    """Checks if the value is a valid 2D array shape"""
    # Verify is 2 element tuple. Note: 1st to avoid IndexError:
    if not isinstance(value, tuple) or len(value) != 2: 
        return False
    
    # Verify 1st element is stirctly positive int:
    if not isinstance(value[0], int) or value[0] <= 0: 
        return False

    # Verify 2nd tuple element is stirctly positive int:
    if not isinstance(value[1], int) or value[1] <= 0: 
        return False
    
    return True

def is_2d_array(value) -> bool:
    """Checks if value is 2D np.ndarray"""
    return isinstance(value,np.ndarray) and value.ndim == 2

def is_float_int_array(value) -> bool:
    """Checks if value is 2D np.ndarray where all values are floats/ints"""
    # Verify value is 2d np.ndarray: 
    if not isinstance(value,np.ndarray):
        return False

    # Verify value is np.ndarray of int/floats:
    is_float_array = np.issubdtype(value.dtype, np.floating)
    is_int_array = np.issubdtype(value.dtype, np.integer)
    return is_float_array or is_int_array

def is_2d_bool_array(value) -> bool:
    """Checks if value is 2D np.ndarray where all values are bools"""
    # Verify value is 2d np.ndarray: 
    if not is_2d_array(value):
        return False
    
    # Verify value is np.ndarray of bools:
    return value.dtype == bool

def is_2d_same_shape_arrays(value1, value2) -> bool:
    """Checks if value1, value2 2D are np.ndarray of same shape"""
    # Verify value1 & value2 are 2d np.ndarray: 
    if not is_2d_array(value1) or not is_2d_array(value2):
        return False

    # Verify value is 2 dimensional array:
    return value1.shape == value2.shape

def is_finite_array(value) -> bool:
    """Checks if value is np.ndarray where all elements in value are finite"""
    # Verify value is np.ndarray:
    if not isinstance(value,np.ndarray):
        return False
    
    # Verify value contains only finite values:
    return bool(np.isfinite(value).all())
