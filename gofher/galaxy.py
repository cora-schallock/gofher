"""Contains Galaxy class that determines the redder side of a galaxy"""

import numpy as np

from gofher_parameters import GofherParameters
from file_helper import read_fits

class Galaxy:
    def __init__(self,
                 gofher_params: GofherParameters):
        raise NotImplementedError

    def construct_galaxy_band_from_fits(self, band: str, fits_path: str):
        raise NotImplementedError

    def run(self, 
            bluer_to_redder_bands: list[str],
            sparcfire_bulge_disk_f: float = 1.0,
            area_to_consider: np.ndarray | None = None,
            fail_silently_on_missing_band: bool = True):
        # raise error if missing waveband
        # raise error if one waveband is different sizes
        # blue_to_red_band: list[str]
        # area to consider is unioned with valid pixels and ellipse to make area_to_norm
        # combine with valid pixel mask (below)
        # apply normalization for all bands
        # gofherParamaters update size
        # pos, and negative
        # construct 
        # calculate diff image for all waveband pairs
        # split positive, negative side
        raise NotImplementedError
