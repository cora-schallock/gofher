import numpy as np
from scipy import stats
from scipy.stats import ks_2samp

from galaxy_band import galaxy_band
from matrix import normalize_matrix
from spin_parity import score_label
from classify import pos_neg_label_from_theta

class InvalidGalaxyBandPair(Exception):
    """exception for invalid band pair"""
    pass

def construct_galaxy_band_pair_key(blue_band, red_band):
    """construct band_pair key given BLUER blue_band and REDDER red_band"""
    return "{}-{}".format(blue_band,red_band)

def split_galaxy_band_pair_key(galaxy_band_pair_key):
    """given the band_pair key splits BLUER blue_band and REDDER red_band"""
    return galaxy_band_pair.split("-")

def create_diff_image(blue_data,red_data,to_diff_mask):
    """first_data - fits data of bluer band
    base_data - fits data of redder band
    to_diff_mask - a single boolean mask indicating where to create diff (1's = included, 0's = ignored)"""
    first_norm = normalize_matrix(blue_data,to_diff_mask)
    base_norm = normalize_matrix(red_data,to_diff_mask)
    return first_norm-base_norm



class galaxy_band_pair:
    """A pair of wave bands from the same gaalxy

    Attributes:
        blue_band: the unique string that identifies the BLUER wave band by name
        red_band: the unique string that identifies the REDDER wave band by name
        diff_image: the subtracted image created by first_band-base_band in area provided by diff_image_mask
        diff_image_mask: the area considered in the diff_image construction
    """
    def __init__(self, blue_band: galaxy_band, red_band: galaxy_band):
        self.blue_band = blue_band
        self.red_band = red_band

        self.diff_image = None
        self.diff_image_mask = None

        self.pos_side = None
        self.neg_side = None

        self.pos_mean = None
        self.neg_mean = None
        self.mean_diff = None

        self.pos_fit_norm_std = None
        self.neg_fit_norm_std = None

        self.ks_d_stat = 0.0
        self.ks_p_value = 0.0

        self.classification = 0
        self.classification_label = ""

        self._has_run = False
        self._used_normed = False

    def construct_diff_image(self):
        """Perform a pixel-by-pixel subtraction of normed blue_band by the normed red_band"""
        self.diff_image = self.blue_band.norm_data-self.red_band.norm_data

    def run(self,pos_mask: np.ndarray, neg_mask: np.ndarray, area_to_compare: np.ndarray):
        """Split the difference image in half using bisection (only in area_to_compare) into pos side and neg side
        
        Args: 
            pos_mask: the pos side of the bisection mask
            neg_mask: the neg side of the bisection mask
            area_to_compare: which areas to include when comparing diff image on pos/neg sides
        """
        pos_area = np.logical_and(pos_mask,area_to_compare)
        neg_area = np.logical_and(neg_mask,area_to_compare)

        self.pos_side = self.diff_image[pos_area]
        self.neg_side = self.diff_image[neg_area]

        self._has_run = True

    def classify(self, theta: float, use_norm: bool = False):
        """Classifies the waveband pairs and sets the classification_label
        
        Args: 
            theta: angle of ellipse measured in degrees
                Important: This should match same angle as ellipse/bisection
        
        Important: Must be done after run()
        """
        if not self._has_run: raise ValueError("galaxy_band_pair.classify(): need to run method galaxy_band_pair.run() prior to classifying")

        if use_norm:
            self.pos_mean, self.pos_fit_norm_std = stats.norm.fit(self.pos_side)
            self.neg_mean, self.neg_fit_norm_std = stats.norm.fit(self.neg_side)
        else:
            self.pos_mean = np.mean(self.pos_side)
            self.neg_mean = np.mean(self.neg_side)
            
        self.mean_diff = self.pos_mean-self.neg_mean

        pl, nl = pos_neg_label_from_theta(np.degrees(theta))
        self.classification = np.sign(self.mean_diff)
        self.classification_label = nl if -np.sign(self.mean_diff) == -1.0 else pl

        ks_score = ks_2samp(self.pos_side,self.neg_side)
        self.ks_d_stat = ks_score.statistic
        self.ks_p_value = ks_score.pvalue

        self._used_normed = use_norm

    def evaluate_fit_norm(self, samples=64):
        """Classifies the waveband pairs and sets the classification_label
        
        Args: 
            samples: number of times each curve is sampled
        """
        pos_x=np.linspace(np.min(self.pos_side),np.max(self.pos_side),samples)
        neg_x=np.linspace(np.min(self.neg_side),np.max(self.neg_side),samples)
        pos_pdf = stats.norm.pdf(pos_x, self.pos_fit_norm_mean, self.pos_fit_norm_std)
        neg_pdf = stats.norm.pdf(neg_x, self.neg_fit_norm_mean, self.neg_fit_norm_std)

        return pos_x, pos_pdf, neg_x, neg_pdf
    
    def get_verbose_csv_header_and_row(self,paper_label=''):
        """Get csv information in the following order:
        pos_side_mean, pos_side_std, neg_side_mean, neg_side_std, ks_d_stat, ks_p_val, classification_label, score?
        
        Args: 
            paper_label: baseline label for comparison, if no baseline leave blank.
                Important: If provided score will be included, calcualted from score_label()
        """
        if self._used_normed:
            header = ["pos_mean","pos_std","neg_mean","neg_std","ks_stat","ks_pval","label"]
            row = [self.pos_mean,
                   self.pos_fit_norm_std,
                   self.neg_mean,
                   self.neg_fit_norm_std]
        else:
            header = ["pos_mean","neg_mean","mean_diff"]
            row = [self.pos_mean, self.neg_mean, self.mean_diff]
        if paper_label != '':
            header.extend(["ks_stat","ks_pval","label","score"])
            row.extend([self.ks_d_stat,self.ks_p_value,self.classification_label,score_label(self.classification_label,paper_label)])

        return (header,row)
