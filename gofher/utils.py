import numpy as np

def is_float(value):
    """Checks if the value is a float/int or numpy equivalent"""
    if isinstance(value, float):
        return True
    
    if isinstance(value, int):
        return True
    
    try:
        if np.issubdtype(value, np.integer):
            return True
        elif np.issubdtype(value, np.floating):
            return True
        else:
            return False
    except TypeError:
        return False
    except ValueError:
        return False
    
    
    
def is_2d_array_shape(value):
    """Checks if the value is a valid 2D array shape"""
    if not isinstance(value, tuple): 
        return False

    if len(value) != 2: 
        return False
    
    if not isinstance(value[0], int): 
        return False

    if not isinstance(value[1], int): 
        return False
    
    if value[0] <= 0: 
        return False

    if value[1] <= 0: 
        return False

    return True