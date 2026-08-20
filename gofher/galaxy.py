"""Contains Galaxy class that determines the redder side of a galaxy"""
from pathlib import Path

import numpy as np

from gofher_parameters import GofherParameters
from galaxy_band import GalaxyBand
from file_helper import read_fits
from utils import is_2d_bool_array

class Galaxy:
    def __init__(self,
                 gofher_params: GofherParameters):
        """Initalize Galaxy object"""

        # Validate arguments:
        if not isinstance(gofher_params, GofherParameters):
            raise TypeError(f"gofher_params {gofher_params} must be type GofherParameters")

        # Initialize parameters:
        self.gofher_params = gofher_params
        self._bands = []
        self._band_pairs = []

    def has_band(self, band: str) -> bool:
        """Check if Galaxy has specific band"""

        # Validate arguments:
        if not isinstance(band,str):
            raise TypeError(f"band {band} must be str")

        # Iterate through bands:
        for each_band in self._bands:
            # If not a GalaxyBand, skip it.
            # Programmer Note: This shouldn't happen but put this as guard statment
            if not isinstance(each_band, GalaxyBand):
                continue

            # If it finds the band, return True:
            if each_band.band == band:
                return True
        return False

    def get_band(self, band: str) -> GalaxyBand:
        """Get Galaxy band"""

        # Validate arguments:
        if not isinstance(band,str):
            raise TypeError(f"band {band} must be str")

        if not self.has_band(band):
            raise KeyError(f"no band {band}")

        # Iterate through bands:
        for each_band in self._bands:
            # If not a GalaxyBand, skip it.
            # Programmer Note: This shouldn't happen but put this as guard statment
            if not isinstance(each_band, GalaxyBand):
                continue

            #If it finds the band, return it:
            if each_band.band == band:
                return each_band

        return None
    
    def construct_galaxy_band_from_fits(self, band: str, fits_path: str):
        """Construct a galaxy_band from a fits file"""

        # Validate arguments:
        if not isinstance(band, str):
            raise TypeError(f"band {band} must be str")
        
        if len(band) == 0:
            raise ValueError("band can not be empty string")

        if band.count("-") > 0 or band.count("_") > 0:
            raise ValueError(f"band {band} can not have '-' or '_', reserved delimeters.")

        if not isinstance(fits_path, str):
            raise TypeError(f"fits_path {fits_path} must be str")

        if Path(fits_path).suffix != ".fits":
            raise ValueError(f"fits_path {fits_path} must be .fits file")

        if not Path(fits_path).is_file():
            raise ValueError(f"fits_path {fits_path} does not exist")

        # Create new galaxy band and add it to the bands:
        data = read_fits(fits_path)
        band = GalaxyBand(band,data)

        # Validate shape of data matches current shape:
        # If first band added, it sets shape of gofher_params
        # else checks it against gofher_params
        # Programmer Note: If you have errors later on about mismatched shape
        #   it likely indicates shape has changed since consturction time
        the_band_shape = band.get_shape()
        if self.gofher_params.shape == (-1,-1):
            self.gofher_params.shape = the_band_shape
        elif self.gofher_params.shape != the_band_shape:
            raise ValueError(f"data shape {data.shape} does not match current shape of {self.gofher_params.shape}")

        # Finally add the shape:
        self._bands.append(band)

    def run(self, 
            bluer_to_redder_bands: list[str],
            sparcfire_bulge_disk_f: float = 1.0,
            area_to_consider: np.ndarray | None = None,
            fail_silently_on_missing_band: bool = True):

        if not isinstance(bluer_to_redder_bands, list):
            raise TypeError(f"""bluer_to_redder_bands {bluer_to_redder_bands} 
                must be a list of string""")

        for band in bluer_to_redder_bands:
            if not isinstance(band,str):
                raise TypeError("bluer_to_redder_bands contains a non string")

        if not isinstance(sparcfire_bulge_disk_f,float):
            raise TypeError(f"""sparcfire_bulge_disk_f {sparcfire_bulge_disk_f} 
                must be float""")

        if sparcfire_bulge_disk_f < 0.0 or sparcfire_bulge_disk_f > 1.0:
            raise ValueError(f"""sparcfire_bulge_disk_f {sparcfire_bulge_disk_f} 
                must be in range [0,1]""")

        if area_to_consider is not None and is_2d_bool_array(area_to_consider):
            raise TypeError("area_to_norm must be numpy boolean 2D array or None")

        if not isinstance(fail_silently_on_missing_band, bool):
            raise TypeError(f"""fail_silently_on_missing_band {fail_silently_on_missing_band}
                must be bool""")

        # To avoid side effects when calling run() multiple times, clear band_pairs:
        self._band_pairs = []

        # Find the bands that this galaxy has:
        galaxy_has_bands: list[GalaxyBand] = []
        for band in bluer_to_redder_bands:
            if self.has_band(band):
                galaxy_has_bands.append(self.get_band(band))
            elif not fail_silently_on_missing_band:
                raise RuntimeError(f"missing {band} band")

        area_to_norm = np.ones(self.gofher_params.shape,bool)
        for band in galaxy_has_bands:
            the_band = self.get_band(band)
            the_band_valid_pixels = the_band.get_valid_pixel_mask()

            area_to_norm = np.logical_and(area_to_norm,the_band_valid_pixels)

        if area_to_consider is None:
            area_to_norm = np.logical_and(area_to_norm,area_to_consider)

        for band in galaxy_has_bands:
            the_band = self.get_band(band)
            the_band.apply_normalization(area_to_norm)

        #TODO:
        #construct band pairs
        #create diff image, bisect, classify

        #check galaxy has atleast 2 bands at start
        # if has_band less than 2, then raise error
        

        # raise error if missing waveband
        # blue_to_red_band: list[str]
        # area to consider is unioned with valid pixels and ellipse to make area_to_norm
        # combine with valid pixel mask (below)
        # apply normalization for all bands
        # gofherParamaters update size
        # pos, and negative
        # construct 
        # calculate diff image for all waveband pairs
        # split positive, negative side

        #TODO: how to handle issue with fail_silently_on_missing_band=True
        # When has no bands, one band, i.e. no waveband pair
        # enforce a single waveband pair perhaps? IDK
        raise NotImplementedError
