"""Contains Galaxy class that determines the redder side of a galaxy"""

from gofher_parameters import GofherParameters
from file_helper import read_fits

class Galaxy:
    def __init__(self,
                 sparcfire_csv_path: str | None = None):
        raise NotImplementedError

    def construct_galaxy_band_from_fits(self, band: str, fits_path: str):
        raise NotImplementedError

    def run(self):
        # raise error if missing waveband
        # raise error if one waveband is different sizes
        # blue_to_red_band: list[str]
        # combine with valid pixel mask (below)
        # apply normalization for all bands
        # gofherParamaters update size
        # construct 
        # calculate diff image for all waveband pairs
        # split positive, negative side
        raise NotImplementedError
