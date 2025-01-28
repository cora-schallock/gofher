
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
    """Bin fits data with padding if necessary
    
    Args: 
        data: numpy 2d array containing data to bin
        s: bin size in x and y directions

    Notes:
        If the dimensions of the data array are not divisible by s, the array is padded with zeros.
    
    Returns:
        binned numpy 2d array
    """
    # Calculate padding needed
    pad_x = (s - data.shape[0] % s) % s
    pad_y = (s - data.shape[1] % s) % s

    # Pad the data with zeros if necessary
    padded_data = np.pad(data, ((0, pad_x), (0, pad_y)), mode='constant', constant_values=0)

    # Create a valid pixel mask (ignoring the padded zeros, and any infs or nans from original data)
    valid_pixel_mask = np.logical_and(padded_data != 0,create_valid_pixel_mask(padded_data)).astype(int)

    # Perform the binning process
    binned = padded_data.reshape(padded_data.shape[0] // s, s, padded_data.shape[1] // s, s).sum(axis=(3, 1))
    
    # Count valid pixels per bin
    valid_count = valid_pixel_mask.reshape(padded_data.shape[0] // s, s, padded_data.shape[1] // s, s).sum(axis=(3, 1))
    
    # Avoid division by zero
    to_mask_out = valid_count == 0 #Cache this for later, so we can replace invalid values with NaN
    valid_count[to_mask_out] = 1  # Set to 1 to avoid division by zero in empty bins

    # Final binned data
    binned_fits = binned / valid_count
    
    # Mask out the regions with no pixels
    binned_fits[to_mask_out] = np.NaN 

    return binned_fits


