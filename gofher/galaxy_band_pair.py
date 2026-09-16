"""GalaxyBandPair is a single pair of bluer and redder galaxy wavebands

This is used to constuct the diff_image which is GOFHER's representation
of the "redder side". See GalaxyBandPair.calculate_diff_image() for more.
"""

import numpy as np

from gofher.galaxy_band import GalaxyBand
from gofher.gofher_parameters import INDETERMINANT_VOTE_LABEL
from gofher.utils import is_2d_same_shape_arrays, is_2d_bool_array, is_float_int

class GalaxyBandPair:
    """A single pair of bluer and redder galaxy wavebands"""
    def __init__(self, bluer_band: GalaxyBand, redder_band: GalaxyBand):
        """Construct the GalaxyBandPair
        
        Args:
            bluer_band: the GalaxyBand object of the bluer waveband
            redder_band: the GalaxyBand object of the redder waveband
        """

        if not isinstance(bluer_band, GalaxyBand):
            raise TypeError("bluer_band must be GalaxyBand")

        if not isinstance(redder_band, GalaxyBand):
            raise TypeError("bluer_band must be GalaxyBand")

        if bluer_band.band == redder_band.band:
            raise ValueError("bluer_band and redder_band must have different band strings")

        if bluer_band.get_shape() != redder_band.get_shape():
            raise ValueError("bluer_band and redder_band data must have same shape")

        self._bluer_band = bluer_band
        self._redder_band = redder_band

        self._diff_image = None

        self.pos_side_label = ""
        self.pos_side_values = None
        self.pos_side_mean = None

        self.neg_side_label = ""
        self.neg_side_values = None
        self.neg_side_mean = None

        self.mean_values = None
        self.std_values = None

        self.redder_side_label = INDETERMINANT_VOTE_LABEL

    def _clear_diff_and_classification(self):
        """Clear diff image, classification, and values from calculate_diff_image()
        
        Programmer Note: This is to avoid side effects from multiple runs
        """
        self._diff_image = None
        
        self.pos_side_label = ""
        self.pos_side_values = None
        self.pos_side_mean = None
        
        self.neg_side_label = ""
        self.neg_side_values = None
        self.neg_side_mean = None

        self.mean_values = None
        self.std_values = None

        self.redder_side_label = INDETERMINANT_VOTE_LABEL

    def get_histogram_range(self, 
                            std: float = 3) -> tuple[float, float]:
        """Calculate the range for histogram"""
        # Validate input:
        if not is_float_int(std):
            raise TypeError("mean/std must be float")

        if std <= 0:
            raise ValueError("std must be strictly positive")

        if not is_float_int(self.mean_values) or not is_float_int(self.std_values):
            raise RuntimeError("mean/std values must be float of int; ensure you called calculate_diff_image_prior")

        # Calculate bounds:
        lower_bound = self.mean_values - std * self.std_values
        upper_bound = self.mean_values + std * self.std_values
        return (lower_bound,upper_bound)


    def calculate_diff_image(self,
                             pos_mask: np.ndarray,
                             neg_mask: np.ndarray):
        """Calculate the diff_image for a band pair:
        
        The diff_image is the pixel by pixel subtraction of the normalized
        bluer waveband by the normalzied redder waveband.
        
        So pixels that are larger, represent areas of bluer light while pixels
        that are smaller represent areas of redder light.
        """

        # Clear values from previous runs to avoid side effects from multiple runs
        self._clear_diff_and_classification()

        # Validate input:
        if not is_2d_bool_array(pos_mask):
            raise ValueError("pos_mask must be 2D np.ndarray()")

        if not is_2d_bool_array(neg_mask):
            raise ValueError("neg_mask must be 2D np.ndarray()")

        # Validate that each bluer and redder GalaxyBand have been normalizaed:
        if not self._bluer_band.has_normalization():
            raise RuntimeError("must call bluer_band.apply_normalization(...) first")

        if not self._redder_band.has_normalization():
            raise RuntimeError("must call redder_band.apply_normalization(...) first")

        # Get the normalization:
        bluer_norm = self._bluer_band.get_normalization()
        redder_norm = self._redder_band.get_normalization()

        # Validate that they are the same size:
        # Programmer Note: this shouldn't occur, as at GalaxyBandPair initalization time
        #   the size was alreayd checked. So if this occurs, then the GalaxyBand(s)
        #   have been update since then likely indicating a bug somewhere.
        if not is_2d_same_shape_arrays(bluer_norm, redder_norm):
            raise RuntimeError("normalized bluer and redder band have different shape")

        # Validate mask is correct size
        # Programmer Note: here bluer_norm and redder_norm have same size
        if not is_2d_same_shape_arrays(pos_mask, bluer_norm):
            raise ValueError("pos_mask has shape {pos_mask.shape} but bands have shape {bluer_norm.shape}")

        if not is_2d_same_shape_arrays(neg_mask,bluer_norm):
            raise ValueError("neg_mask has shape {neg_mask.shape} but bands have shape {bluer_norm.shape}")

        self._diff_image = bluer_norm - redder_norm

        self.pos_side_values = self._diff_image[pos_mask]
        self.pos_side_mean = np.mean(self.pos_side_values)

        self.neg_side_values = self._diff_image[neg_mask]
        self.neg_side_mean = np.mean(self.neg_side_values)

        combined = np.concatenate((self.pos_side_values, self.neg_side_values))
        self.mean_values = np.mean(combined)
        self.std_values = np.std(combined, ddof=1) 

        return self._diff_image

    def get_diff_image(self):
        return self._diff_image

    def classify(self, pos_side: str, neg_side: str):
        """Classify which side is redder side and assign classification labels
        
        Args:
            pos_side: the human readable classification label of the pos_mask
            neg_side: the human readable classification label of the neg_mask
        """
        if not isinstance(pos_side, str):
            raise TypeError("pos_side must be str type")

        if not isinstance(neg_side, str):
            raise TypeError("neg_side must be str type")

        if neg_side == pos_side:
            raise ValueError("pos_side and neg_side can not be same")

        if len(pos_side) == 0:
            raise ValueError("pos_side can not be empty string")

        if len(neg_side) == 0:
            raise ValueError("neg_side can not be empty string")

        if self.pos_side_mean is None or self.neg_side_mean is None:
            raise RuntimeError("pos/neg_side_mean invalid, assure calculate_diff_image() has been called first")
    
        self.pos_side_label = pos_side
        self.neg_side_label = neg_side

        if self.pos_side_mean > self.neg_side_mean:
            self.redder_side_label = self.neg_side_label
        else:
            self.redder_side_label = self.pos_side_label

        return self.redder_side_label

    def __str__(self):
        return f"{self._bluer_band.band}-{self._redder_band.band}"

    def __repr__(self):
        return f"GalaxyBandPair({self._bluer_band.band}-{self._redder_band.band})"
