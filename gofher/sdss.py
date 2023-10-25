import copy
import numpy as np
import matplotlib.pyplot as plt

from gofher import normalize_array
from file_helper import write_csv
from visualize import create_visualize
from astropy.visualization import make_lupton_rgb
from galaxy import galaxy, galaxy_band_pair, construct_band_pair_key


SDSS_BANDS_IN_ORDER = ['u','g','r','i','z'] #SDSS
SDSS_REF_BANDS_IN_ORDER = ['r','i','z','g','u']

def create_sdss_csv(gals,the_band_pairs,csv_path):
    """create an csv containing the information from gofher of the given galaxies"""
    #Construct CSV header:
    csv_column_headers = ['name','dark_side_label','pos_side_label','neg_side_label','ref_band','encounted_sersic_error']
    per_band_column_headers = ['pos_side_mean','pos_side_std','neg_side_mean','neg_side_std','D','P','label','score']
    #per_band_column_headers = ['pos_side_mean','pos_side_std','neg_side_mean','neg_side_std','D','P','label','score','pos_side_data','neg_side_data']

    for band_pair in the_band_pairs:
        band_pair_key = construct_band_pair_key(band_pair[0],band_pair[1])
        csv_column_headers.extend(list(map(lambda x: "{}_{}".format(band_pair_key,x),per_band_column_headers)))

    csv_column_headers.extend(['vote_count','vote_score'])
    
    #Construct CSV rows:
    rows = []
    for gal in gals:
        if not isinstance(gal,galaxy): continue

        the_row = [gal.name,gal.dark_side,gal.pos_side_label,gal.neg_side_label,gal.ref_band,str(gal.encountered_sersic_fit_error)]
        for band_pair in the_band_pairs:
            band_pair_key = construct_band_pair_key(band_pair[0],band_pair[1])
            the_band_pair = gal.get_band_pair(band_pair_key)
            
            the_row.extend([the_band_pair.pos_fit_norm_mean,the_band_pair.pos_fit_norm_std,
                            the_band_pair.neg_fit_norm_mean,the_band_pair.neg_fit_norm_std,
                            the_band_pair.d_stat, the_band_pair.p_value,
                            the_band_pair.classification_label,
                            the_band_pair.classification_score])
            """
            pos_side_data_string = ';'.join(list(map(lambda x: str(x),the_band_pair.pos_side.flatten())))
            neg_side_data_string = ';'.join(list(map(lambda x: str(x),the_band_pair.neg_side.flatten())))
            the_row.extend([the_band_pair.pos_fit_norm_mean,the_band_pair.pos_fit_norm_std,
                            the_band_pair.neg_fit_norm_mean,the_band_pair.neg_fit_norm_std,
                            the_band_pair.d_stat, the_band_pair.p_value,
                            the_band_pair.classification_label,
                            the_band_pair.classification_score,
                            pos_side_data_string,neg_side_data_string])
            """
        
        the_row.extend([gal.cumulative_classification_vote_count,gal.cumulative_score])
        rows.append(the_row)
    
    write_csv(csv_path,csv_column_headers,rows)
        

def _get_normalized_wave_band(data,valid_pixel_mask,el_mask,sigma=3):
    """normalize waveband for color image construction"""
    normalized_wave_band = copy.deepcopy(data)
    area_of_interest = np.logical_and(copy.deepcopy(valid_pixel_mask),el_mask)
    
    m = np.mean(normalized_wave_band[area_of_interest])
    std = np.std(normalized_wave_band[area_of_interest])

    np.clip(normalized_wave_band,m-sigma*std,m+sigma*std)
    normalized_wave_band[np.logical_not(copy.deepcopy(valid_pixel_mask))] = 0.0
    return normalize_array(normalized_wave_band,valid_pixel_mask)


def consruct_color_image(the_gal,scale=10):
    """construct a color image for the given galaxy"""
    ones = np.ones(the_gal.get_shape(),dtype='bool')
    i = _get_normalized_wave_band(the_gal['i'].data,the_gal['i'].valid_pixel_mask,the_gal.create_ellipse())*scale
    r = _get_normalized_wave_band(the_gal['r'].data,the_gal['r'].valid_pixel_mask,the_gal.create_ellipse())*scale*0.8
    g = _get_normalized_wave_band(the_gal['g'].data,the_gal['g'].valid_pixel_mask,the_gal.create_ellipse())*scale*0.7

    return make_lupton_rgb(i, r, g, Q=10, stretch=0.3, minimum=0.0)

def visualize_sdss(the_gal: galaxy, save_path=''):
    """visualize the output for an sdss galaxy"""
    img = consruct_color_image(the_gal)
    fig, axd = create_visualize(the_gal,SDSS_BANDS_IN_ORDER)

    if the_gal.has_valid_band('i') and the_gal.has_valid_band('r') and the_gal.has_valid_band('g'):
        img = consruct_color_image(the_gal)
        axd['color'].imshow(img, interpolation='nearest',origin='lower')
        axd['color'].set_title("{}\n paper label={}".format(the_gal.name,the_gal.dark_side))

    if save_path != "":
        fig.savefig(save_path, dpi = 300, bbox_inches='tight')
        fig.clear()
        plt.close(fig)
    else:
        plt.show()
    
    