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

        self.pos_fit_norm_mean = None
        self.pos_fit_norm_std = None
        self.neg_fit_norm_mean = None
        self.neg_fit_norm_std = None

        self.ks_d_stat = 0.0
        self.ks_p_value = 0.0

        self.classification = 0
        self.classification_label = ""

    def construct_diff_image(self):
        self.diff_image = self.blue_band.norm_data-self.red_band.norm_data

    def run(self,pos_mask,neg_mask,area_to_compare):
        """split diff_image into pos/neg side"""
        pos_area = np.logical_and(pos_mask,area_to_compare)
        neg_area = np.logical_and(neg_mask,area_to_compare)

        self.pos_side = self.diff_image[pos_area]
        self.neg_side = self.diff_image[neg_area]

    def classify(self, theta):
        """classify the band_pair and output a label
        
        Important: Must be done after run() and fit_norm()
        """
        pl, nl = pos_neg_label_from_theta(np.degrees(theta))
        mean_dif = np.sign(self.pos_fit_norm_mean-self.neg_fit_norm_mean)

        self.classification = np.sign(mean_dif)
        self.classification_label = nl if -np.sign(mean_dif) == -1.0 else pl

    def fit_norm(self):
        """fit 2 normal distrobutions (one to the pos and one to the neg side) to the bisected diff_image
        
        Important: Must be done after run()
        """
        self.pos_fit_norm_mean, self.pos_fit_norm_std = stats.norm.fit(self.pos_side)
        self.neg_fit_norm_mean, self.neg_fit_norm_std = stats.norm.fit(self.neg_side)

        ks_score = ks_2samp(self.pos_side,self.neg_side)
        self.ks_d_stat = ks_score.statistic
        self.ks_p_value = ks_score.pvalue

    def evaluate_fit_norm(self,samples=64):
        """evaluate the normal by getting pdf of the fit normal"""
        the_min = min(np.min(self.pos_side),np.min(self.neg_side))
        the_max = max(np.max(self.pos_side),np.max(self.neg_side))

        pos_x=np.linspace(np.min(self.pos_side),np.max(self.neg_side),samples)
        neg_x=np.linspace(np.min(self.pos_side),np.max(self.neg_side),samples)
        pos_pdf = stats.norm.pdf(pos_x, self.pos_fit_norm_mean, self.pos_fit_norm_std)
        neg_pdf = stats.norm.pdf(neg_x, self.neg_fit_norm_mean, self.neg_fit_norm_std)

        return pos_x, pos_pdf, neg_x, neg_pdf
    
    def get_verbose_csv_header_and_row(self,paper_label=''):
        #pos_side_mean, pos_side_std, neg_side_mean, neg_side_std, ks_d_stat, ks_p_val, classification_label, self.classification_label (if paper_label provided)
        header = ["pos_mean","pos_std","neg_mean","neg_std","ks_stat","ks_pval","label"]
        row = [self.pos_fit_norm_mean,
                self.pos_fit_norm_std,
                self.neg_fit_norm_mean,
                self.neg_fit_norm_std,
                self.ks_d_stat,
                self.ks_p_value,
                self.classification_label]
        if paper_label != '':
            header.append("score")
            row.append(score_label(self.classification_label,paper_label))

        return (header,row)
