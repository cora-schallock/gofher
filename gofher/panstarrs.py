import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from gofher import normalize_array
from file_helper import write_csv
from visualize import create_visualize, create_visualize_of_diff
from astropy.visualization import make_lupton_rgb
from galaxy import galaxy, galaxy_band_pair, construct_band_pair_key


#PANSTARRS_BANDS_IN_ORDER = ['g','r','i','z','y'] #PANSTARRS
#PANSTARRS_REF_BANDS_IN_ORDER = ['i','z','y','r','g']

PANSTARRS_BANDS_IN_ORDER = ['r','i','z','y'] #PANSTARRS
PANSTARRS_REF_BANDS_IN_ORDER = ['i','z','y','r']

def create_panstarrs_csv(gals,the_band_pairs,csv_path):
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


def visualize_panstarrs(the_gal: galaxy, save_path='',color_img_path=''):
    """visualize the output for an sdss galaxy"""
    fig, axd = create_visualize(the_gal,PANSTARRS_BANDS_IN_ORDER)

    color = mpimg.imread(color_img_path)
    axd['color'].imshow(color)
    axd['color'].set_title("{}\n paper label={}".format(the_gal.name,the_gal.dark_side))

    if save_path != "":
        fig.savefig(save_path, dpi = 300, bbox_inches='tight')
        fig.clear()
        plt.close(fig)
    else:
        plt.show()
        
def visualize_panstarrs_diff(the_gal: galaxy, save_path='',color_img_path=''):
    fig, axd = create_visualize_of_diff(the_gal,PANSTARRS_BANDS_IN_ORDER)

    color = mpimg.imread(color_img_path)
    axd['color'].imshow(color)
    axd['color'].set_title("{}\n paper label={}".format(the_gal.name,the_gal.dark_side))

    if save_path != "":
        fig.savefig(save_path, dpi = 300, bbox_inches='tight')
        fig.clear()
        plt.close(fig)
    else:
        plt.show()
    
    
    