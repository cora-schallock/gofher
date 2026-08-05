"""File helpers used by GOFHER

This mopdule contains a collection of functions for:
    * file I/O
    * file/directory bookeeping
"""

from pathlib import Path
import numpy as np
from astropy.io import fits


def read_fits(path: str) -> np.ndarray:
    """Read in a fits file
    
    Args:
        path: path to the fits file
    """

    if not isinstance(path, str):
        raise ValueError(f"path must be str; got {type(path)}")

    if not Path.is_file(path):
        raise FileNotFoundError(f"no fits file path {path} found.")
    
    if Path(path).suffix != ".fits":
        raise ValueError(f"fits path must be .fits file; got {path}")

    the_fits = None

    with fits.open(path) as hdul:
        the_fits = hdul[0].data

    return the_fits

def write_array_file(array: np.ndarray, path: str):
    """Save numpy array as numpy native binary format

     Args:
        array: the numpy array to be saved
        path: path to the array will be saved to
    """
    
    if not isinstance(array, np.ndarray):
        raise ValueError(f"array must be numpy array; got {type(array)}")
   
    if not isinstance(path,str):
        raise ValueError(f"path must be str; got {type(path)}")
    
    if Path(path).suffix != ".npy":
        raise ValueError(f"path must be .npy file got {path}")
   
    np.save(path, array)
       

def read_array_file(path: str) -> np.ndarray:
    """Read saved numpy array file
    
    Args:
        path: path to the array will be read from
    """

    if not isinstance(path,str):
        raise ValueError(f"array file path must be str; got {type(path)}")

    if Path(path).suffix != ".npy":
        raise ValueError(f"array file path must be .npy file got {path}")
    
    if not Path.is_file(path):
        raise FileNotFoundError(f"no array file path {path} found.")
    
    return np.load(path)

def assure_folder_exists(path: str):
    """Assure the folder exists, if not is created along with parents
    
    Args:
        path: path of the folder to be created if not already exists
    """

    if not isinstance(path,str):
        raise ValueError(f"folder {path} must be a str")

    if len(path) == 0:
        raise ValueError("folder can not be empty str")

    Path(path).mkdir(parents=True,exist_ok=True)
