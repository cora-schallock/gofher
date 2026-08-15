from galaxy_band import GalaxyBand

class GalaxyBandPair:
    def __init__(self,bluer_band: GalaxyBand, redder_band: GalaxyBand):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError