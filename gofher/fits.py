
import os
import numpy as np
from astropy.io import fits

class InvalidFitsPath(Exception):
    """Exception for invalid fits path"""
    pass

#fits: i/o:
def write_fits(path,data):
    """write fits file to path"""
    if not is_valid_fits_path_valid(path):
        raise InvalidFitsPath("write_fits: file at path {} invalid, must end in '.fits' extension")
    if not isinstance(data, np.ndarray):
        raise ValueError("write_fits: data must be of type np.ndarray (got type {})".format(type(data)))
    
    hdul = fits.PrimaryHDU(data)
    hdul.writeto(path)

def read_fits(path):
    """reads fits file specified by path"""
    if not os.path.isfile(path):
        raise FileExistsError("read_fits: Fits at file path not found {}".format(path))
    if not is_valid_fits_path_valid(path):
        raise InvalidFitsPath("read_fits: file at path {} invalid, must end in '.fits' extension")

    hdul = fits.open(path)
    data = hdul[0].data
    return data

#helper functions:
def is_valid_fits_path_valid(fits_path):
    """Check if fits_path is a valid fits path"""
    return os.path.splitext(fits_path)[-1].lower() == ".fits" #https://stackoverflow.com/a/5900590/13544635