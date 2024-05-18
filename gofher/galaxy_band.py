import numpy as np
from mask import create_valid_pixel_mask
from gofher import normalize_array

class MissingGalaxyBand(Exception):
    """exception for missing band"""
    pass

class galaxy_band:
    """A signle galaxy wave band

    Attributes:
        band: the unique string that identifies the wave band by name
        data: the data taken from the fits file for the specific wave band
    """
    def __init__(self,band,data=None):
        self.band = band
        self.data = data
        self.valid_pixel_mask = None
        self.norm_data = None

        if not data is None: self.construct_valid_pixel_mask()

    def is_valid(self):
        """checks if data is valid"""
        return isinstance(self.data, np.ndarray) and isinstance(self.valid_pixel_mask, np.ndarray) and self.data.shape == self.valid_pixel_mask.shape
    
    def get_shape(self):
        """shape of the fits"""
        if not self.is_valid(): return (-1,-1)

        return self.data.shape
    
    def construct_valid_pixel_mask(self):
        """create a valid pixel mask for this specific wave band"""
        if not isinstance(self.data, np.ndarray): 
            raise ValueError("Can't construct valid_pixel_mask for band: {} - Data needs to be an np.npdarry".format(self.band))
        self.valid_pixel_mask = create_valid_pixel_mask(self.data)

    def normalize(self,area_to_norm=None):
        if not self.is_valid():
            raise ValueError("Can't normalize galaxy_band: {} - (make sure it has data and valid_pixel_mask)".format(self.band))
        to_diff_mask = np.logical_and(area_to_norm,self.valid_pixel_mask)
        self.norm_data = normalize_array(self.data,to_diff_mask)