
import os
import numpy as np
from astropy.io import fits

from mask import create_valid_pixel_mask

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

def bin_fits(data: np.ndarray, s: int) -> np.ndarray:
    """Bin fits data
    
    Args: 
        data: numpy 2d array containing data to bin
        s: bin size in x and y directions

    Notes:
        source: https://stackoverflow.com/a/36102436/13544635
        If a has the shape m, n, the reshape should have the form
        a.reshape(m_bins, m // m_bins, n_bins, n // n_bins)

        So:
        If a has the shape m, n, and the bin size is (s x s) the reshape should have the form
        a.reshape(m // s, s, n//s, s)


    Returns:
        binned numpy 2d array
    """
    to_bin = np.zeros(data.shape)
    valid_pixel_mask = create_valid_pixel_mask(data)

    to_bin[valid_pixel_mask] = data[valid_pixel_mask]
    binned = to_bin.reshape(data.shape[0]//s, s, data.shape[1]//s,s).sum(3).sum(1)
    valid_count = valid_pixel_mask.astype(int).reshape(data.shape[0]//s, s, data.shape[1]//s,s).sum(3).sum(1)

    #if all binned pixels in a region are invalid, set them to NaN
    to_mask_out = valid_count == 0
    valid_count[to_mask_out] = 1

    binned_fits = binned/valid_count
    binned_fits[to_mask_out] = np.NaN 

    return binned_fits

