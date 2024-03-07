import numpy as np
from scipy import stats
from scipy.stats import ks_2samp
import copy

from fits import read_fits, is_fits_path_valid, InvalidFitsPath
from mask import create_valid_pixel_mask, create_bisection_mask_from_gofher_params, create_ellipse_mask_from_gofher_params
from gofher import create_diff_image, gofher_parameters

from spin_parity import pos_neg_label_from_theta

class MissingBand(Exception):
    """exception for missing band"""
    pass

class InvalidBandPair(Exception):
    """exception for invalid band pair"""
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
        if not isinstance(self.data, np.ndarray): raise ValueError("Can't construct valid_pixel_mask for band: {} - Data needs to be an np.npdarry".format(self.band))
        self.valid_pixel_mask = create_valid_pixel_mask(self.data)

class galaxy_band_pair:
    """A pair of wave bands from the same gaalxy

    Attributes:
        first_band: the unique string that identifies the BLUER wave band by name
        base_band: the unique string that identifies the REDDER wave band by name
        diff_image: the subtracted image created by first_band-base_band in area provided by diff_image_mask
        diff_image_mask: the area considered in the diff_image construction
    """
    def __init__(self, first_band, base_band, diff_image=None, diff_image_mask=None):
        self.first_band = first_band
        self.base_band = base_band

        self.diff_image = diff_image
        self.diff_image_mask = diff_image_mask

        self.pos_side = None
        self.neg_side = None

        self.pos_fit_norm_mean = None
        self.pos_fit_norm_std = None
        self.neg_fit_norm_mean = None
        self.neg_fit_norm_std = None

        self.d_stat = 0.0
        self.p_value = 0.0

        self.classification = 0
        self.classification_label = ""
        self.classification_score = 0

    def run(self,pos_mask,neg_mask):
        """split diff_image into pos/neg side"""
        pos_area = np.logical_and(pos_mask,self.diff_image_mask)
        neg_area = np.logical_and(neg_mask,self.diff_image_mask)

        self.pos_side = self.diff_image[pos_area]
        self.neg_side = self.diff_image[neg_area]

    def classify(self, theta, dark_label=""):
        """classify the band_pair and output a label
        
        Important: Must be done after run() and fit_norm()
        """
        pl, nl = pos_neg_label_from_theta(np.degrees(theta))
        mean_dif = np.sign(self.pos_fit_norm_mean-self.neg_fit_norm_mean)

        self.classification = np.sign(mean_dif)
        self.classification_label = nl if -np.sign(mean_dif) == -1.0 else pl

        if dark_label == "": return

        label = nl if -np.sign(mean_dif) == 1.0 else pl
        opposite = pl if -np.sign(mean_dif) == 1.0 else nl

        correct_label_letter_count = len(set([*label.lower()]).union([*dark_label.lower()]))
        incorrect_label_letter_count = len(set([*opposite.lower()]).union([*dark_label.lower()]))
        
        if correct_label_letter_count > incorrect_label_letter_count and correct_label_letter_count > 1:
            self.classification_score =  1
        elif incorrect_label_letter_count > correct_label_letter_count and incorrect_label_letter_count > 1:
            self.classification_score =  -1
        else:
            self.classification_score = 0


    def fit_norm(self):
        """fit 2 normal distrobutions (one to the pos and one to the neg side) to the bisected diff_image
        
        Important: Must be done after run()
        """
        self.pos_fit_norm_mean, self.pos_fit_norm_std = stats.norm.fit(self.pos_side)
        self.neg_fit_norm_mean, self.neg_fit_norm_std = stats.norm.fit(self.neg_side)

        ks_score = ks_2samp(self.pos_side,self.neg_side)
        self.d_stat = ks_score.statistic
        self.p_value = ks_score.pvalue

    def evaluate_fit_norm(self,samples=64):
        """evaluate the normal by getting pdf of the fit normal"""
        the_min = min(np.min(self.pos_side),np.min(self.neg_side))
        the_max = max(np.max(self.pos_side),np.max(self.neg_side))

        pos_x=np.linspace(np.min(self.pos_side),np.max(self.neg_side),samples)
        neg_x=np.linspace(np.min(self.pos_side),np.max(self.neg_side),samples)
        pos_pdf = stats.norm.pdf(pos_x, self.pos_fit_norm_mean, self.pos_fit_norm_std)
        neg_pdf = stats.norm.pdf(neg_x, self.neg_fit_norm_mean, self.neg_fit_norm_std)

        return pos_x, pos_pdf, neg_x, neg_pdf
    

class galaxy:
    """a galaxy that gofher is to be run on"""
    def __init__(self,name,dark_side=""):
        self.name = name
        self.bands = {}
        self.band_pairs = {}

        self.dark_side = dark_side

        self.ref_band = ""
        self.gofher_params = gofher_parameters()
        self.encountered_sersic_fit_error = False

        self.pos_side_label = ''
        self.neg_side_label = ''

        self.cumulative_classification_vote_count = 0
        self.cumulative_score = 0

        self.folder = ''

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
        if not self.has_band(band): MissingBand('Band: {} is missing from the galaxy'.format(band))

        return self[band].get_shape()

    def construct_band(self,band,fits_path):
        """create a new galaxy_band given the band name and fits_path to the input data"""
        if not is_fits_path_valid(fits_path): raise InvalidFitsPath('Band:{} with fits_path:{} is invalid'.format(band,fits_path))

        the_data = read_fits(fits_path) 
        self.bands[band] = galaxy_band(band,the_data)

        return self.bands[band]

    def create_bisection(self,the_params=None, the_shape=None):
        """create bisection of the galaxy using (x,y) center and theta" from the_params (or self.gofher_params if none given)"""
        if the_params is None: the_params = self.gofher_params
        if the_shape is None: the_shape = self.get_shape()

        return create_bisection_mask_from_gofher_params(the_params,the_shape)
    
    def create_ellipse(self,the_params=None, the_shape=None, r=1.0):
        """create ellipse mask of the galaxy using (x,y) center, theta, and (a,b) ellipse extents" from the_params (or self.gofher_params if none given)"""
        if the_params is None: the_params = self.gofher_params
        if the_shape is None: the_shape = self.get_shape()

        return create_ellipse_mask_from_gofher_params(the_params,the_shape,r)
    
    def can_construct_band_pair(self,first_band,base_band):
        """checks if can construct a band_pair given a BLUER band first_band and REDDER band base_band"""
        if not self.has_band(first_band) or not self.has_band(base_band): return False #check has data
        if not self[first_band].is_valid() or not self[base_band].is_valid(): return False #check has data/pixel mask and right type
        if self[first_band].get_shape() != self[base_band].get_shape(): return False #check matching size
        
        return True
    
    def construct_band_pair(self,first_band,base_band,area_to_diff) -> galaxy_band_pair:
        """creates a new band_pair given a BLUER band first_band and REDDER band base_band"""
        if not self.can_construct_band_pair(first_band,base_band): raise InvalidBandPair("Can't construct band pair with first_band {} and base_band {}".format(first_band,base_band))
        to_diff_mask = np.logical_and(area_to_diff,np.logical_and(self[first_band].valid_pixel_mask,self[base_band].valid_pixel_mask))

        the_band_pair_key = construct_band_pair_key(first_band, base_band)
        diff_image = create_diff_image(copy.deepcopy(self[first_band].data),copy.deepcopy(self[base_band].data),to_diff_mask)
        self.band_pairs[the_band_pair_key] = galaxy_band_pair(first_band,base_band,diff_image,to_diff_mask)

        return self.band_pairs[the_band_pair_key]
    
    def run_gofher(self,the_ordered_band_pairs):
        """run gofher on all valid band_apirs for the galaxy"""
        el_mask = self.create_ellipse()
        pos_mask, neg_mask = self.create_bisection()
        self.pos_side_label, self.neg_side_label = pos_neg_label_from_theta(np.degrees(self.gofher_params.theta))
        self.cumulative_classification_vote_count = 0
        self.cumulative_score = 0

        for (first_band, base_band) in the_ordered_band_pairs:
            if not self.can_construct_band_pair(first_band,base_band): continue

            area_to_diff = np.logical_and(el_mask, copy.deepcopy(self[first_band].valid_pixel_mask), copy.deepcopy(self[base_band].valid_pixel_mask))
            the_band_pair = self.construct_band_pair(first_band,base_band,area_to_diff)

            the_band_pair.run(pos_mask,neg_mask)
            the_band_pair.fit_norm()
            the_band_pair.classify(self.gofher_params.theta,self.dark_side)
            self.cumulative_classification_vote_count += the_band_pair.classification_score
        
        self.cumulative_score = int(np.sign(self.cumulative_classification_vote_count))
    
    def __getitem__(self, band):
        """getter funtion to get waveband of galaxy"""
        return self.bands[band]
    
    def get_band_pair(self,band_pair_key) -> galaxy_band_pair:
        """given a band_pair key, get specified band_pair"""
        return self.band_pairs[band_pair_key]


def construct_band_pair_key(first_band, base_band):
    """construct band_pair key given BLUER first_band and REDDER base_band"""
    return "{}-{}".format(first_band,base_band)