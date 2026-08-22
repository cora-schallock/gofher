"""Misc. utility scripts utilized by various GOFHER scripts

"As it says on the tin!"
"""

from itertools import combinations

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

def generate_band_pair_tuples(bluer_to_redder: list[str]) -> list[tuple]:
    """Given an ordered list of bluer to redder wavebands, generate all
    possible waveband pairs of (bluer_band,redder_band). Contains no duplicates.

    Important: This relies on the fact that itertools.combinations() returns
    is lexigraphical order according to provided input. See:
    https://docs.python.org/3/library/itertools.html#itertools.combinations
    
    Args:
        bluer_to_redder: the wavebands sorter in order bluest to reddest
        
    Returns:
        a list containing all the order band pairs tuples
    """

    if not isinstance(bluer_to_redder, list):
        raise TypeError("bluer_to_redder must be a list of strings")

    for each_band in bluer_to_redder:
        if not isinstance(each_band,str):
            raise TypeError("all elements of bluer_to_redder must be strings")

    if len(bluer_to_redder) < 2:
        raise ValueError("bluer_to_redder must contain at least 2 bands")

    all_tuples = []

    for each_pair in combinations(bluer_to_redder,2):
        all_tuples.append(each_pair)
    
    return all_tuples

def generate_all_band_pair_strings(bluer_to_redder: list[str]) -> list[str]:
    """Same as generate_band_pair_tuples() but generates strings in 
    format: '{bluer_band}-{redder_band}'
    Contains no duplicates.
    
    See: generate_band_pair_tuples()
        
    Args:
        bluer_to_redder: the wavebands sorter in order bluest to reddest
            
    Returns:
        a list containing all the order band pairs strings
    """

    # Validate input:
    if not isinstance(bluer_to_redder, list):
            raise TypeError("bluer_to_redder must be a list of strings")
    
    for each_band in bluer_to_redder:
        if not isinstance(each_band,str):
            raise TypeError("all elements of bluer_to_redder must be strings")

    if len(bluer_to_redder) < 2:
        raise ValueError("bluer_to_redder must contain at least 2 bands")

    # Generate all tuples, then make it into strings
    band_pair_tuples = generate_band_pair_tuples(bluer_to_redder)
    return list(map(lambda x: f"{x[0]}-{x[1]}",band_pair_tuples))