from gofher_parameters import GofherParameters

class Galaxy:
    def __init__(self, blue_to_red_band: list[str],
                 sparcfire_csv_path: str | None = None):
        raise NotImplementedError

    def construct_band_from_fits(self, band: str, fits_path: str):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError