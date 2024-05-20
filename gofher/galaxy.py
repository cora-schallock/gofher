import copy
import itertools
import numpy as np

from fits import read_fits
from spin_parity import score_label
from ebm import EmpiricalBrownsMethod
from classify import pos_neg_label_from_theta
from gofher_parameters import gofher_parameters
from galaxy_band import galaxy_band, MissingGalaxyBand
from galaxy_band_pair import galaxy_band_pair, InvalidGalaxyBandPair, construct_galaxy_band_pair_key

class galaxy:
    """a galaxy that gofher is to be run on"""
    def __init__(self,name,paper_label=None):
        self.name = name
        self.bands = {}
        self.band_pairs = {}

        self.area_to_diff = None

        self.paper_label = paper_label

        self.pos_side_label = ''
        self.neg_side_label = ''

        self.ref_band = ""
        self.gofher_params = gofher_parameters()
        self.encountered_sersic_fit_error = False

        self.cumulative_classification_vote_count = 0
        self.cumulative_score = 0

        self.folder = ''

        self.disparate_sides_vote = None

    def has_band(self, band):
        """checks if galaxy has sepcific wave band given name"""
        return band in self.bands
    
    def has_valid_band(self, band):
        """checks if has wave band and if wave band is valid"""
        return band in self.bands and self[band].is_valid()
    
    def has_valid_ref_band(self):
        """checks if galaxy has reference band and is valid"""
        return self.has_valid_band(self.ref_band)
    
    def get_shape(self, band=None):
        """get the shape of the data for a given band (or of ref_band if none given)"""
        if band is None: band = self.ref_band
        if not self.has_band(band): MissingGalaxyBand('Band: {} is missing from the galaxy'.format(band))

        return self[band].get_shape()

    def construct_band(self,band,fits_path):
        """create a new galaxy_band given the band name and fits_path to the input data"""
        the_data = read_fits(fits_path) 
        self.bands[band] = galaxy_band(band,the_data)

        return self.bands[band]

    def create_bisection(self,the_params=None, shape=None):
        """create bisection of the galaxy using (x,y) center and theta" from the_params (or self.gofher_params if none given)"""
        if the_params is None: the_params = self.gofher_params
        if shape is None: shape = self.get_shape()

        return the_params.create_bisection_mask(shape)
    
    def create_ellipse(self,the_params=None, shape=None, r=1.0):
        """create ellipse mask of the galaxy using (x,y) center, theta, and (a,b) ellipse extents" from the_params (or self.gofher_params if none given)"""
        if the_params is None: the_params = self.gofher_params
        if shape is None: shape = self.get_shape()

        return the_params.create_ellipse_mask(shape,r)
    
    def can_construct_band_pair(self,blue_band_key: str,red_band_key: str):
        """checks if can construct a band_pair given a BLUER band blue_band and REDDER band red_band"""
        if not self.has_band(blue_band_key) or not self.has_band(red_band_key): return False #check has data
        if not self[blue_band_key].is_valid() or not self[red_band_key].is_valid(): return False #check has data/pixel mask and right type
        if self[blue_band_key].get_shape() != self[red_band_key].get_shape(): return False #check matching size
        
        return True
    
    def construct_band_pair(self,blue_band_key: str,red_band_key:str) -> galaxy_band_pair:
        """creates a new band_pair given a BLUER band blue_band and REDDER band red_band"""
        if not self.can_construct_band_pair(blue_band_key,red_band_key): raise InvalidGalaxyBandPair("Can't construct band pair with blue_band {} and red_band {}".format(blue_band_key,red_band_key))
        the_band_pair_key = construct_galaxy_band_pair_key(blue_band_key,red_band_key)

        band_pair = galaxy_band_pair(self.bands[blue_band_key],self.bands[red_band_key])
        self.band_pairs[the_band_pair_key] = band_pair

        band_pair.construct_diff_image()

        return band_pair
    
    def run_gofher(self,the_ordered_band_pairs: list):
        """Run gofher pipeline on all waveband pairs composed of bands in provided list

        Args:
            the_ordered_band_pairs: wavebands to use in order of bluest to reddest
                Note: Will only consider waveband pairs that have both the
                blue and red band containted in the_ordered_band_pairs.
        """

        #1) Construct ellipse and bisection mask
        el_mask = self.create_ellipse()
        pos_mask, neg_mask = self.create_bisection()

        #2) Figure out cardinal direction labels for pos and neg
        #   side of bisection based on theta
        self.pos_side_label, self.neg_side_label = pos_neg_label_from_theta(np.degrees(self.gofher_params.theta))

        #3) find area to norm (by looking at all valid pixel masks)
        #   and then normalize each galaxy_band
        to_norm = copy.deepcopy(el_mask)
        for band in self.bands:
            to_norm = np.logical_and(to_norm,copy.deepcopy(self.bands[band].valid_pixel_mask))
        self.area_to_diff = to_norm

        #4) normalize all wavebands
        for band in self.bands:
            self.bands[band].normalize(self.area_to_diff)

        #5) iterate through all wavband pairs composes of bands
        #   in the_ordered_band_pairs, create diff image and
        #   then classify based on results
        for (blue_band, red_band) in the_ordered_band_pairs:
            if not self.can_construct_band_pair(blue_band,red_band): continue

            the_band_pair = self.construct_band_pair(blue_band,red_band)

            the_band_pair.run(pos_mask,neg_mask,self.area_to_diff)
            the_band_pair.fit_norm()
            the_band_pair.classify(self.gofher_params.theta)

        #6) calcuylate cumulative score
        self.cumulative_score = int(np.sign(self.cumulative_classification_vote_count))

    def run_ebm(self, bands_in_order = []):
        """Calculate the most statistically signifcant red band

        Uses Emperical Brown's Method to combine ks test pvalues from waveband pairs

        Args:
            bands_in_order: wavebands to use in order of bluest to reddest
                Note: Will only consider waveband pairs that have both the
                blue and red band containted in bands_in_order.
        Returns:
            (label,pval): label determined by ebm and ebm pval
        """
        
        pos_pixels = []
        neg_pixels = []
        pos_pvals = []
        neg_pvals = []
        for (blue_band,red_band) in itertools.combinations(bands_in_order, 2):
            band_pair_key = construct_galaxy_band_pair_key(blue_band,red_band)
            band_pair = self.get_band_pair(band_pair_key)

            if int(band_pair.classification) == 1: #switched
                pos_pixels.append(band_pair.diff_image[self.area_to_diff])
                pos_pvals.append(band_pair.ks_p_value)
            elif int(band_pair.classification) == -1:
                neg_pixels.append(band_pair.diff_image[self.area_to_diff])
                neg_pvals.append(band_pair.ks_p_value)

        pos_ebm = 1.0
        neg_ebm = 1.0

        if len(pos_pvals) == 1:
            pos_ebm = pos_pvals[0]
        elif len(pos_pvals) > 1:
            pos_ebm = EmpiricalBrownsMethod(np.array(pos_pixels),pos_pvals)

        if len(neg_pvals) == 1:
            neg_ebm = neg_pvals[0]
        elif len(neg_pvals) > 1:
            neg_ebm = EmpiricalBrownsMethod(np.array(neg_pixels),neg_pvals)

        if neg_ebm < pos_ebm:
            return (self.pos_side_label, neg_ebm)
        else:
            return (self.neg_side_label, pos_ebm)

    
    def __getitem__(self, band):
        """getter funtion to get waveband of galaxy"""
        return self.bands[band]
    
    def get_band_pair(self,band_pair_key) -> galaxy_band_pair:
        """given a band_pair key, get specified band_pair"""
        return self.band_pairs[band_pair_key]
    
    def get_verbose_csv_header_and_row(self,bands_in_order=[],paper_label=''):
        """Generate galaxy's csv information

        Args:
            bands_in_order: wavebands to use in order of bluest to reddest
                Note: Will only consider waveband pairs that have both the
                blue and red band containted in bands_in_order.
            paper_label (optional): the dark_side label in the spin parity catalog.
                If provided includes scoring information.
        Returns:
            (header,row): galaxy's csv header and row
        """
        header = ["name"]
        row = [self.name]

        if paper_label != '':
            header.append("paper_label")
            row.append(paper_label)

        header.extend(["pos_label","neg_label"])
        row.extend([self.pos_side_label,self.neg_side_label])

        score = 0

        for (blue_band,red_band) in itertools.combinations(bands_in_order, 2):
            band_pair_key = construct_galaxy_band_pair_key(blue_band,red_band)
            band_pair = self.get_band_pair(band_pair_key)

            (bandpair_header,bandpair_row) = band_pair.get_verbose_csv_header_and_row(paper_label)
            header.extend(list(map(lambda x: "{}_{}".format(band_pair_key,x),bandpair_header)))
            row.extend(bandpair_row)
            if paper_label != '':
                score += bandpair_row[-1]
        
        if paper_label != '':
            header.extend(['total','score'])
            row.extend([score,int(np.sign(score))])
        return (header,row)
    
    def get_ebm_csv_header_and_row(self,bands_in_order=[],paper_label=''):
        """Generate galaxy's ebm csv information

        Args:
            bands_in_order: wavebands to use in order of bluest to reddest
                Note: Will only consider waveband pairs that have both the
                blue and red band containted in bands_in_order.
            paper_label (optional): the dark_side label in the spin parity catalog.
                If provided includes scoring information for considered waveband pairs.
        Returns:
            (header,row): galaxy's ebm csv header and row
        """
        header = ["name"]
        row = [self.name]

        (ebm_label, ebm_p) = self.run_ebm(bands_in_order)

        if paper_label != '':
             header.extend(["paper_label","ebm_label","ebm_pval","score"])
             row.extend([paper_label,ebm_label,"{:.3E}".format(ebm_p),str(score_label(ebm_label,paper_label))])
        else:
            header.extend(["ebm_label","ebm_pval"])
            row.extend([ebm_label,"{:.3E}".format(ebm_p)])

        return (header,row)   
        

