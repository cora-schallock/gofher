
import os
import numpy as np
from astropy.io import fits

class InvalidFitsPath(Exception):
    """Exception for invalid fits path"""
    pass

#fits: i/o:
def write_fits(path: str, data: np.ndarray):
    """Write fits data to fits file
    
    Args: 
        path: file path of FITS to create
        data: numpy 2d array containing data to write
    """
    if not is_valid_fits_path_valid(path):
        raise InvalidFitsPath("write_fits: file at path {} invalid, must end in '.fits' extension")
    if len(data.shape) != 2:
        raise ValueError("write_fits: data must be 2-dimensional (got shape {})".format(data.shape))
    
    hdul = fits.PrimaryHDU(data)
    hdul.writeto(path)

def read_fits(path: str):
    """Read fits from path
    
    Args: 
        path: file path of FITS to read
    """
    if not os.path.isfile(path):
        raise FileExistsError("read_fits: Fits at file path not found {}".format(path))
    if not is_valid_fits_path_valid(path):
        raise InvalidFitsPath("read_fits: file at path {} invalid, must end in '.fits' extension")

    hdul = fits.open(path)
    data = hdul[0].data
    return data

#helper functions:
def is_valid_fits_path_valid(fits_path: str):
    """Check if extension for FITS file is correct
    
    Args: 
        fits_path: file path of FITS to check
    """
    return os.path.splitext(fits_path)[-1].lower() == ".fits" #https://stackoverflow.com/a/5900590/13544635