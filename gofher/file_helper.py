"""File helpers used by GOFHER

This mopdule contains a collection of functions for:
    * file I/O
    * file/directory bookeeping
"""

from pathlib import Path
import numpy as np
from astropy.io.fits import getdata


def read_fits(fits_path: str | Path) -> np.ndarray:
    """Read in a fits file
    
    Args:
        path: path to the fits file

    Returns:
        array of fits data
    """

    if not isinstance(fits_path, (str, Path)):
        raise ValueError(f"fits_path must be str or Path; got {type(fits_path)}")

    if isinstance(fits_path, str):
        fits_path = Path(fits_path)

    if not fits_path.is_file():
        raise FileNotFoundError(f"no fits file path {fits_path} found.")
    
    if fits_path.suffix != ".fits":
        raise ValueError(f"fits path must be .fits file; got {fits_path}")

    the_fits = getdata(fits_path)

    return the_fits

def write_array_file(array: np.ndarray, array_path: str | Path):
    """Save numpy array as numpy native binary format

     Args:
        array: the numpy array to be saved
        array_path: path to the array will be saved to
    """
    
    if not isinstance(array, np.ndarray):
        raise ValueError(f"array must be numpy array; got {type(array)}")
   
    if not isinstance(array_path, (str, Path)):
        raise ValueError(f"array_path must be str; got {type(array_path)}")

    if isinstance(array_path, str):
        array_path = Path(array_path)
    
    if array_path.suffix != ".npy":
        raise ValueError(f"array_path must be .npy file got {array_path}")
   
    np.save(array_path, array)
       

def read_array_file(array_path: str | Path) -> np.ndarray:
    """Read saved numpy array file
    
    Args:
        path: path to the array will be read from
    """

    if not isinstance(array_path,(str,Path)):
        raise ValueError(f"array file array_path must be str or Path; got {type(array_path)}")

    if isinstance(array_path, str):
        array_path = Path(array_path)

    if array_path.suffix != ".npy":
        raise ValueError(f"array file array_path must be .npy file got {array_path}")
    
    if not array_path.is_file():
        raise FileNotFoundError(f"no array file path {array_path} found.")
    
    return np.load(str(array_path))

def assure_folder_exists(folder_path: str | Path):
    """Assure the folder exists, if not is created along with parents
    
    Args:
        path: path of the folder to be created if not already exists
    """

    if not isinstance(folder_path,(str, Path)):
        raise TypeError(f"folder {folder_path} must be a str")

    if isinstance(folder_path, str):
        if len(folder_path) == 0:
            raise ValueError("folder can not be empty str")

        folder_path = Path(folder_path)

    if folder_path.suffix != '':
        raise ValueError("folder_path must be folder not file")

    folder_path.mkdir(parents=True,exist_ok=True)
