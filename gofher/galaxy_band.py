import numpy as np

from matrix import normalize_matrix
from mask import create_valid_pixel_mask


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

    def normalize(self,area_to_norm: np.ndarray):
        """Normalize all pixels in area_to_norm so that values are in range [0,1], 0 if not in area_to_norm:
        pos_side_mean, pos_side_std, neg_side_mean, neg_side_std, ks_d_stat, ks_p_val, classification_label, score?
        
        Args: 
            area_to_norm: area mask to normalize values over
                Note: if you want to normalize all valid pixels can define area_to_norm = np.ones(galaxy_band.data.shape).astype(bool)
        """
        if not self.is_valid():
            raise ValueError("Can't normalize galaxy_band: {} - (make sure it has data and valid_pixel_mask)".format(self.band))
        to_diff_mask = np.logical_and(area_to_norm,self.valid_pixel_mask)
        self.norm_data = normalize_matrix(self.data,to_diff_mask)