"""Data container class for an single FITS waveband

When given a string representing the waveband, and a data array,
apply normalization of data using apply_normalization()
"""

import numpy as np

from utils import is_2d_bool_array, is_2d_array, is_2d_same_shape_arrays

class GalaxyBand:
    """Data container for a single FITS waveband, and allows normalization"""
    def __init__(self, band: str, data: np.ndarray):
        if not isinstance(band,str):
            raise TypeError(f"band must be str; got {type(band)}")

        if len(band) == 0:
            raise ValueError("band must be str of len > 0")

        if band.count("-") > 0 or band.count("_") > 0:
            raise ValueError("band must not have '-' or '_' in str")
        
        if not is_2d_array(data):
            raise TypeError("data must be 2D np.ndarray")

        if data.shape[0] == 0 or data.shape[1] == 0:
            raise ValueError(f"data shape must be > 0; got {data.shape}")

        self.band = band
        self.data = data
        self._normalized_data = None

    def get_shape(self) -> tuple[int]:
        """get shape of the data"""
        return self.data.shape
    
    def has_normalization(self) -> bool:
        """verify the data has been normalized"""
        return self._normalized_data is not None
    
    def get_normalization(self) -> np.ndarray:
        """get the normalized data
        
        Important: must call apply normalize prior
        """
        if not self.has_normalization():
            raise ValueError("must call apply_normalization first")
        
        return self._normalized_data
    
    def apply_normalization(self, area_to_norm: np.ndarray | None = None) -> np.ndarray:
        """Apply normalization to all pixels included in boolean mask

        Normalization occurs as follows:
        * If area_to_norm has no True values, return an array of all 0's
        * If max & min of data[area_to_norm] are same, returns array of all 0's
        * Otherwise, normalized such that:
            min -> 0.0
            max -> 1.0
            all others -> (value-min)/(max-min)
        
        Args:
           area_to_norm: the 2D boolean mask of pixels to normalize
                if None, normalizes all pixels
        
        Returns:
            the normalized data
        """
        if area_to_norm is None:
            area_to_norm = np.ones(self.data.shape,bool)

        if not is_2d_array(area_to_norm): #handle none
            raise TypeError("area_to_norm must be 2D np.ndarray")
        if not is_2d_same_shape_arrays(self.data, area_to_norm):
            raise ValueError("area_to_norm must have same shape as self.data")
        if not is_2d_bool_array(area_to_norm):
            raise TypeError(f"area_to_norm must be boolean np.ndarray {area_to_norm}")
        if not np.all(np.isfinite(self.data[area_to_norm])):
            raise ValueError("area_to_norm includes none finite values in self.data")

        # Create a new normalized data array, initally all 0.0's:
        self._normalized_data = np.zeros(self.data.shape,np.float32)

        # Verify the area_to_norm mask includes at least one entry:
        if np.sum(area_to_norm) == 0:
            return self._normalized_data
        
        # Get max/min and if same return all 0.0's:
        the_min = np.min(self.data[area_to_norm])
        the_max = np.max(self.data[area_to_norm])
        if the_max == the_min:
            return self._normalized_data
        
        # Calculate scale of data, and normalize data:
        scale = the_max - the_min
        self._normalized_data[area_to_norm] = (self.data[area_to_norm] - the_min)/scale
        return self._normalized_data
    
    