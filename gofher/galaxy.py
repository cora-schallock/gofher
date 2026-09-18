"""Contains Galaxy class that determines the redder side of a galaxy"""
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb

from gofher.gofher_parameters import GofherParameters, INDETERMINANT_VOTE_LABEL
from gofher.galaxy_band import GalaxyBand
from gofher.galaxy_band_pair import GalaxyBandPair
from gofher.file_helper import read_fits, write_array_file, assure_folder_exists
from gofher.utils import is_2d_bool_array, generate_band_pair_tuples
from gofher.plot import plot_mask, plot_diff_histogram, plot_diff_image

NAME_LABEL = "name"
POS_SIDE_LABEL = "pos_side"
NEG_SIDE_LABEL = "neg_side"
POS_SIDE_COUNT_LABEL = "pos_side_count"
NEG_SIDE_COUNT_LABEL = "neg_side_count"
MAJORITY_CLASSIFICATION_LABEL = "majority_classification"

BAND_REDDER_SIDE_LABEL = "redder_side"
BAND_POS_MEAN_LABEL = "pos_side_mean"
BAND_NEG_MEAN_LABEL = "neg_side_mean"

class Galaxy:
    def __init__(self,
                 gofher_params: GofherParameters):
        """Initalize Galaxy object"""

        # Validate arguments:
        if gofher_params is not None and not isinstance(gofher_params, GofherParameters):
            raise TypeError(f"gofher_params {gofher_params} must be type GofherParameters")

        # Initialize parameters:
        self.gofher_params = gofher_params
        self._bands = []
        self._band_pairs = []
        self._area_to_norm = None
        self.majority_classification_label = INDETERMINANT_VOTE_LABEL
        self.pos_label = INDETERMINANT_VOTE_LABEL
        self.neg_label = INDETERMINANT_VOTE_LABEL
        self.vote_count_pos = 0
        self.vote_count_neg = 0

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
    
    def construct_galaxy_band_from_fits(self, band: str, fits_path: str | Path) -> GalaxyBand:
        """Construct a galaxy_band from a fits file"""

        # Validate arguments:
        if not isinstance(band, str):
            raise TypeError(f"band {band} must be str")
        
        if len(band) == 0:
            raise ValueError("band can not be empty string")

        if band.count("-") > 0 or band.count("_") > 0:
            raise ValueError(f"band {band} can not have '-' or '_', reserved delimeters.")

        if not isinstance(fits_path, (str, Path)):
            raise TypeError(f"fits_path {fits_path} must be st or Path")

        if isinstance(fits_path, str):
            fits_path = Path(fits_path)

        if fits_path.suffix != ".fits":
            raise ValueError(f"fits_path {fits_path} must be .fits file")

        if not fits_path.is_file():
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

        # Finally add the shape and return the band:
        self._bands.append(band)

        return band

    def run(self, 
            bluer_to_redder_bands: list[str],
            sparcfire_bulge_disk_f: float = 1.0,
            area_to_consider: np.ndarray | None = None,
            fail_silently_on_missing_band: bool = True):
        """Given a list of wavebands in order from bluest to reddest:
            1) consturct all waveband pairs using ordered elements in provided list
            2) calculate the diff_image for each waveband pairs
            3) classify all waveband pairs
        
        Args:
            bluer_to_redder_bands: a list of waveband to run on
            sparcfire_bulge_disk_f: the bulge_disk_f when using sparcfire gofher parameters
            area_to_consider: a boolean mask of pixels to consider that is combined
                with the ellipse and pos/neg masks, if None considers all area in
                ellipse and pos/neg mask
            fail_silently_on_missing_band: if a waveband in bluer_to_redder_bands should it be
                skipped or should an exception be raised
        """

        # Validate arguments:
        if not isinstance(bluer_to_redder_bands, list):
            raise TypeError(f"""bluer_to_redder_bands {bluer_to_redder_bands} 
                must be a list of string""")

        if len(bluer_to_redder_bands) < 2:
            raise ValueError("bluer_to_redder_bands must have atleast 2 bands")

        for band in bluer_to_redder_bands:
            if not isinstance(band,str):
                raise TypeError("bluer_to_redder_bands contains a non string")

        if not isinstance(sparcfire_bulge_disk_f,float):
            raise TypeError(f"""sparcfire_bulge_disk_f {sparcfire_bulge_disk_f} 
                must be float""")

        if sparcfire_bulge_disk_f < 0.0 or sparcfire_bulge_disk_f > 1.0:
            raise ValueError(f"""sparcfire_bulge_disk_f {sparcfire_bulge_disk_f} 
                must be in range [0,1]""")

        if area_to_consider is not None and not is_2d_bool_array(area_to_consider):
            raise TypeError("area_to_norm must be numpy boolean 2D array or None")

        if not isinstance(fail_silently_on_missing_band, bool):
            raise TypeError(f"""fail_silently_on_missing_band {fail_silently_on_missing_band}
                must be bool""")

        if area_to_consider is not None and np.sum(area_to_consider) == 0:
            raise ValueError("""area_to_consider provided has no True elements.
            Verify area_to_consider is correct.""")

        # To avoid side effects when calling run() multiple times:
        #  * clear band_pairs
        #  * clear normalization:
        #  * clear any voting/ classification stats:
        self._band_pairs = []
        self._area_to_norm = None
        for band in self._bands:
            band.clear_normalization()

        self.majority_classification_label = INDETERMINANT_VOTE_LABEL
        self.pos_label = INDETERMINANT_VOTE_LABEL
        self.neg_label = INDETERMINANT_VOTE_LABEL
        self.vote_count_pos = 0
        self.vote_count_neg = 0
        
        # Find the bands that this galaxy has:
        galaxy_has_bands: list[GalaxyBand] = []
        for band in bluer_to_redder_bands:
            if self.has_band(band):
                galaxy_has_bands.append(band)
            elif not fail_silently_on_missing_band:
                raise RuntimeError(f"missing {band} band")

        # Assure there are atleast 2 wavebands:
        if len(galaxy_has_bands) < 2:
            raise RuntimeError(f"galaxy must have at least 2 valid bands from {bluer_to_redder_bands}, has {len(galaxy_has_bands)}")

        # Calaculate values for mask using sparcfire:
        self.gofher_params.calculate_from_sparcfire(sparcfire_bulge_disk_f)

        # Start with the ellipse:
        area_to_norm = self.gofher_params.create_ellipse_mask()

        # Combine all the valid pixel masks:
        for band in galaxy_has_bands:
            the_band = self.get_band(band)
            the_band_valid_pixels = the_band.get_valid_pixel_mask()

            area_to_norm = np.logical_and(area_to_norm,the_band_valid_pixels)

        # If area_to_norm is provided, cobine it as well:
        if area_to_consider is not None:
            area_to_norm = np.logical_and(area_to_norm,area_to_consider)

        if np.sum(area_to_norm) == 0:
            raise RuntimeError("""area_to_norm has no True pixels.
            Verify each band valid_pixel_mask is correct and the 
            intersection of them all have overlapping True pixels.""")

        # Cache area_to_norm for plotting later:
        self._area_to_norm = area_to_norm

        # Apply the same normalization to all the bands, and verify only finite values:
        for band in galaxy_has_bands:
            the_band = self.get_band(band)
            the_band.apply_normalization(area_to_norm)

        # Get the pos_mask and neg_mask which bisect the diff_image, and
        # combine with area_to_norm:
        (pos_mask, neg_mask) = self.gofher_params.create_bisection_masks()
        pos_mask = np.logical_and(area_to_norm,pos_mask)
        neg_mask = np.logical_and(area_to_norm,neg_mask)

        if np.sum(pos_mask) == 0:
            raise RuntimeError("""pos_mask has no True pixels.
            Verify area_to_norm and pos_mask are correct and have
            an intersection of at least one common True pixel.""")

        if np.sum(neg_mask) == 0:
            raise RuntimeError("""neg_mask has no True pixels.
            Verify area_to_norm and neg_mask are correct and have
            an intersection of at least one common True pixel.""")

        # Assign labels to pos and neg sides:
        pos, neg = self.gofher_params.get_pos_neg_labels()
        self.pos_label = pos
        self.neg_label = neg

        # Iterate through all waveband pairs:
        for (blue_band,red_band) in generate_band_pair_tuples(bluer_to_redder_bands):
            blue_band = self.get_band(blue_band)
            red_band = self.get_band(red_band)

            # Construct the waveband pair, calculate diff_image, then classify:
            the_band_pair = GalaxyBandPair(blue_band,red_band)
            the_band_pair.calculate_diff_image(pos_mask,neg_mask)
            the_band_pair.classify(self.pos_label,self.neg_label)

            # Add this waveband pair to band pairs:
            self._band_pairs.append(the_band_pair)

        self._get_classification_label_from_majority_voting()

    def _get_classification_label_from_majority_voting(self):
        """Conduct (unweighted) majority voting amongst band pairs to get classifcation"""
        if self.pos_label == INDETERMINANT_VOTE_LABEL or self.neg_label == INDETERMINANT_VOTE_LABEL:
            raise RuntimeError(f"pos and neg labels can not be same as INDETERMINANT_VOTE_LABEL {INDETERMINANT_VOTE_LABEL}")

        if len(self._band_pairs) == 0:
            raise RuntimeError("no bandpairs are present; assure Galaxy.run() called prior and has at least 1 band pair")

        # Iterate through each band_pair and check if it voted for pos or neg side as redder side:
        for bp in self._band_pairs:
            bp_label = bp.redder_side_label
            if bp_label == self.pos_label:
                self.vote_count_pos += 1
            elif bp_label == self.neg_label:
                self.vote_count_neg += 1
            else:
                raise ValueError(f"band pair {bp} voted '{bp_label}' which is neither pos label '{self.pos_label}' or neg label '{self.neg_label}'")

        # Assign classification label if pos/neg side has more votes otherwise indeterminant:
        if self.vote_count_pos > self.vote_count_neg:
            self.majority_classification_label = self.pos_label
        elif self.vote_count_pos < self.vote_count_neg:
            self.majority_classification_label = self.neg_label
        else:
            self.majority_classification_label = INDETERMINANT_VOTE_LABEL

    def make_lupton_rgb(self, 
                        r_band: str = "i", 
                        g_band: str = "r", 
                        b_band: str = "g") -> np.ndarray:
        """Construct a lupton color rgb image of a galaxy

        Programmer Note:
            For each missing color channel (r/g/b), if galaxy is missing the band
            it uses the reference color band. This way we still can have a rgb image
            for galaxies missing one of these wavebands. In this case, colors will be off.
        
        Args:
            r_band: the name of the galaxy waveband to use for red channel
            g_band: the name of the galaxy waveband to use for green channel
            b_band: the name of the galaxy waveband to use for blue channel
            
        Returns:
            a rgb color image
        """
        # Validate input:
        if not isinstance(r_band,str):
            raise TypeError("r_band '{r_band}' must be str")

        if not isinstance(g_band,str):
            raise TypeError("g_band '{g_band}' must be str")

        if not isinstance(b_band,str):
            raise TypeError("r_band '{b_band}' must be str")

        # Compile the bands in rgb order, and create a container for data:
        bands_in_rgb_order = [r_band,g_band,b_band]
        rgb_data = []

        # Get the reference band in the case of missing bands for color channels:
        ref_band = self.gofher_params.ref_band

        # Iterate through rgb color channels, if band present append to rgb_data,
        #   if missing use ref band. However if ref_band is missing raises exception.
        for band in bands_in_rgb_order:
            if self.has_band(band):
                data = self.get_band(band).data
            elif self.has_band(ref_band):
                data = self.get_band(ref_band).data
            else:
                raise RuntimeError(f"band '{band}' missing and missing ref_band '{ref_band}'")
            rgb_data.append(data)
        lupton_rgb = make_lupton_rgb(rgb_data[0],
                                     rgb_data[1],
                                     rgb_data[2],
                                     stretch=0.05,
                                     Q=5)
        return lupton_rgb

    def plot_figure(self, 
                    save_path: str | Path | None = None,
                    dpi: int = 300):
        """Plot the difference image and histograms for all band pairs
        
        Args:
            save_path: the path of where to save the image
                if None, will display it
            dpi: dpi for saved image
        """
        # Validate the parameters:
        if save_path is not None and not isinstance(save_path,(str, Path)):
            raise TypeError("save_path must be str if provided otherwise none")

        if isinstance(save_path, str):
            save_path = Path(save_path)

        if not isinstance(dpi,int):
            raise TypeError("dpi must be int")

        if dpi <= 0:
            raise ValueError("dpi must be strictly positive int")

        # Check if there has been a run by looking at the band pairs:
        if len(self._band_pairs) == 0:
            raise RuntimeError("""no band_pairs; 
            Assure Galaxy.run() was called prior to this and created at least one band pair""")

        # Check that area_to_norm exists:
        if not is_2d_bool_array(self._area_to_norm):
            raise RuntimeError("area_to_norm is not 2D bool numpy array")

        # Check the ref_band is present:
        if not self.has_band(self.gofher_params.ref_band):
            raise RuntimeError("missing ref_band")
        
        # Gather the following:
        # 1) pixel bounds - cropping used for diff image to not display background
        # 2) area_to_norm - a boolean np.ndarray that indicates if a pixel was normalized
        # 3) ref_band_data - the fits data from the ref_band
        pixel_bounds = self.gofher_params.get_ellipse_pixel_bounds(padding=10)
        area_to_norm = self._area_to_norm
        ref_band_data = self.get_band(self.gofher_params.ref_band).data

        # Find range for histograms to assure they all have same x-values:
        hrange = [np.inf,-np.inf]
        for bp in self._band_pairs:
            bp_range = bp.get_histogram_range()
            hrange[0] = min(hrange[0],bp_range[0])
            hrange[1] = max(hrange[1],bp_range[1])

        # Create a figure that contains the following:
        # top row: 
        #   left: a color image
        #   right: the bisection/ellipse masks used
        # each following row: one row per waveband
        #   left: on the diff image
        #   right: the histogram of diff values split between pos/neg sides
        gs_kw = dict(width_ratios=[2,3], height_ratios=[2] + [2]*len(self._band_pairs))
        fig, axs = plt.subplots(len(self._band_pairs)+1,2,figsize=(12.5,36),gridspec_kw=gs_kw)

        # Color image:
        color_image = self.make_lupton_rgb()
        axs[0][0].imshow(color_image,origin="lower")

        # Get title for color image in the format of:
        #   '{galaxy name}: {classifciation} ({vote count of classifciation} to {vote count of opposite ofclassifciation} votes)
        name = self.gofher_params.name
        vote = self.majority_classification_label
        if vote == self.neg_label:
            first_count = self.vote_count_neg
            second_count = self.vote_count_pos
        else:
            first_count = self.vote_count_pos
            second_count = self.vote_count_neg
        axs[0][0].set_title(f"{name}: {vote} ({first_count} to {second_count} votes)")

        # Plot the masks used for gofher:
        plot_mask(axs[0][1],
                  ref_band_data,
                  self.gofher_params,
                  self._area_to_norm,
                  pixel_bounds)

        # Iterate through band_pairs:
        for i, bp in enumerate(self._band_pairs):
            ax = axs[i+1]

            # Plot the difference image and add a color bar:
            diff,ticks = plot_diff_image(ax[0],bp,self.gofher_params,area_to_norm,pixel_bounds)

            fig.colorbar(diff, ax=ax[0], ticks=ticks)

            # Plot the histogram of the diff values:
            plot_diff_histogram(ax[1],bp,hrange)

        # Save figure if save_path is provided, otherwise display it:
        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show(fig)

    def save_normalizations(self, path_to_folder: str | Path):
        """Save the normalized data"""
    
        if not isinstance(path_to_folder, (str, Path)):
            raise TypeError(f"given path_to_folder '{path_to_folder}' is not a str or Path")
    
        if isinstance(path_to_folder, str):
            path_to_folder = Path(path_to_folder)

        if path_to_folder.suffix != '':
            raise ValueError("path_to_folder must be folder not file")

        if not path_to_folder.exists():
            raise FileNotFoundError(f"{path_to_folder} does not exist")

        if not is_2d_bool_array(self._area_to_norm):
            raise RuntimeError("""self._area_to_norm is not a 2D boolean numpy array
            Assure Galaxy.run() is called first and _area_to_norm is correct""")
            
        if len(self._bands) == 0:
            raise RuntimeError("""no bands""")
    
        if path_to_folder.is_file():
            raise ValueError(f"'{path_to_folder}' must be folder not file")

        if not path_to_folder.parent.exists():
            raise RuntimeError(f"folder '{path_to_folder.parent}' does not exist")

        write_array_file(self._area_to_norm, path_to_folder / "area_to_norm.npy")

        for gb in self._bands:
            if not gb.has_normalization():
                raise RuntimeError("""band {} is missing normalization
                Assure GalaxyBand.apply_normalization() has been called prior""")
                
            write_array_file(gb.get_normalization(), path_to_folder / f"{gb.band}_normalization.npy")

    def get_csv_dict(self) -> dict:
        if self.pos_label == INDETERMINANT_VOTE_LABEL or self.neg_label == INDETERMINANT_VOTE_LABEL:
            raise RuntimeError("""pos/neg label can not be INDETERMINANT_VOTE_LABEL
            Assure Galaxy.run() has been called prior.""")
        
        if self.vote_count_neg == 0 and self.vote_count_pos == 0:
            raise RuntimeError("""no votes for either pos or neg side
            Assure Galaxy.run() has been called prior.""")
        
        if len(self._band_pairs) == 0:
            raise RuntimeError("""missing band pairs""")

        # Collect gofher params data:
        data = self.gofher_params.get_csv_dict()

        # Collect data for each band pair:
        for bp in self._band_pairs:
            if bp.redder_side_label == INDETERMINANT_VOTE_LABEL:
                raise RuntimeError(f"""redder_side_label for {str(bp)} is INDETERMINANT_VOTE_LABEL
                Assure band_pair.classify() has been called prior""")

            data[f"{str(bp)}_{BAND_REDDER_SIDE_LABEL}"] = bp.redder_side_label
            data[f"{str(bp)}_{BAND_POS_MEAN_LABEL}"] = bp.pos_side_mean
            data[f"{str(bp)}_{BAND_NEG_MEAN_LABEL}"] = bp.neg_side_mean

        # Collect data for galaxy:
        data[POS_SIDE_LABEL] = self.pos_label
        data[NEG_SIDE_LABEL] = self.neg_label
        data[POS_SIDE_COUNT_LABEL] = self.vote_count_pos
        data[NEG_SIDE_COUNT_LABEL] = self.vote_count_neg
        data[MAJORITY_CLASSIFICATION_LABEL] = self.majority_classification_label

        return data

    def output_to_csv(self, csv_path: str | Path):
        """Write the gofher parameters to a csv file at csv_path."""
        if not isinstance(csv_path, (str, Path)):
            raise TypeError(f"given csv_path {csv_path} is not a str or Path")

        if isinstance(csv_path, str):
            csv_path = Path(csv_path)

        if csv_path.suffix != ".csv":
            raise ValueError(f"csv_path {csv_path} not be a .csv file")

        # Gather data:
        data = self.get_csv_dict()

        # Write to csv:
        df = pd.DataFrame([data])
        df.to_csv(csv_path, index=False, na_rep='')
