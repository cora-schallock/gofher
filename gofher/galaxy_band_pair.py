"""GalaxyBandPair is a single pair of bluer and redder galaxy wavebands

This is used to constuct the diff_image which is GOFHER's representation
of the "redder side". See GalaxyBandPair.calculate_diff_image() for more.
"""

from galaxy_band import GalaxyBand
from utils import is_2d_same_shape_arrays

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

    def calculate_diff_image(self):
        """Calculate the diff_image for a band pair:
        
        The diff_image is the pixel by pixel subtraction of the normalized
        bluer waveband by the normalzied redder waveband.
        
        So pixels that are larger, represent areas of bluer light while pixels
        that are smaller represent areas of redder light.
        """

        # Validate that each bluer and redder GalaxyBand have been normalizaed:
        if not self._bluer_band.has_normalization():
            raise RuntimeError("must call bluer_band.apply_normalization(...) first")

        if not self._redder_band.has_normalization():
            raise RuntimeError("must call redder_band.apply_normalization(...) first")

        # Get the normalization:
        bluer_norm = self._bluer_band.get_normalization()
        redder_norm = self._redder_band.get_normalization()

        # Validate that they are the same size:
        # Note: this shouldn't occur, as at GalaxyBandPair initalization time
        #   the size was alreayd checked. So if this occurs, then the GalaxyBand(s)
        #   have been update since then likely indicating a bug somewhere.
        if not is_2d_same_shape_arrays(bluer_norm, redder_norm):
            raise RuntimeError("normalized bluer and redder band have different shape")

        self._diff_image = bluer_norm - redder_norm

        return self._diff_image

    def __str__(self):
        return f"{self._bluer_band.band}-{self._redder_band.band}"

    def __repr__(self):
        return f"GalaxyBandPair({self._bluer_band.band}-{self._redder_band.band})"
