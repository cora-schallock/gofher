"""Misc. utility scripts utilized by various GOFHER scripts

"As it says on the tin!"
"""


import numpy as np

def is_float(value):
    """Checks if the value is a float/int or numpy equivalent"""
    # Verify is python float or int:
    if isinstance(value, (float,int)):
        return True
    
    # Verify is numpy int/gloat and a single value (i.e. not array):
    if isinstance(value,(np.integer, np.floating)) and np.isscalar(value):
        return True
    
    return False
    
    
    
def is_2d_array_shape(value):
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
