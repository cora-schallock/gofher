import copy
import math
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from galaxy import galaxy, galaxy_band, construct_band_pair_key

DEFAULT_POSITIVE_RGB_VECTOR = [60/255,179/255,113/255] #mediumseagreen
DEFAULT_NEGATIVE_RGB_VECTOR = [240/255,128/255,125/255] #lightcoral
DEFAULT_BAD_PIXEL_RGB_VECTOR = [169/255,169/255,169/255] #lightgrey

def create_color_map_class(pos,neg,el):
    """create a color map that sepearte pos/neg side inside ellipse from rest of data"""
    cmap_class = np.ones((pos.shape[0],pos.shape[1],3))*DEFAULT_BAD_PIXEL_RGB_VECTOR
    cmap_class[np.logical_and(pos,el)] = DEFAULT_POSITIVE_RGB_VECTOR
    cmap_class[np.logical_and(neg,el)] = DEFAULT_NEGATIVE_RGB_VECTOR
    
    return cmap_class

def get_subplot_mosaic_strtings(bands_in_order):
    """get the strings for the band pairs used in the mosaic plt plot"""
    band_keys = []
    for (first_band,base_band) in itertools.combinations(bands_in_order, 2):
        band_keys.append(construct_band_pair_key(first_band,base_band))
    return np.array(band_keys).reshape(5,2).tolist()


def create_visualize(the_gal: galaxy, bands_in_order = []):
    """create and return a visualize of the galaxy"""
    mo_labels = [['color','ref_band']]
    mo_labels.extend(get_subplot_mosaic_strtings(bands_in_order))
    
    gs_kw = dict(width_ratios=[1,1], height_ratios=[2,1,1,1,1,1])
    fig, axd = plt.subplot_mosaic(mo_labels,
                                  gridspec_kw=gs_kw, figsize = (24,30),
                                  constrained_layout=True,num=1, clear=True) #num=1, clear=True #https://stackoverflow.com/a/65910539/13544635
    fig.patch.set_facecolor('white')

    for band_pair_key in the_gal.band_pairs:
        pl = ''; ml = '' #temp
        band_pair = the_gal.get_band_pair(band_pair_key)

        both_sides = np.concatenate((band_pair.pos_side,band_pair.neg_side))
        mean = np.mean(both_sides)
        std = np.std(both_sides)
        lower_bound = mean-3*std; upper_bound = mean+2*std

        axd[band_pair_key].hist(band_pair.pos_side,bins=30,color='green',alpha=0.5, weights=np.ones_like(band_pair.pos_side) / len(band_pair.pos_side))
        axd[band_pair_key].axvline(band_pair.pos_fit_norm_mean,color='green',label="{} mean = {:.5f}".format(pl,band_pair.pos_fit_norm_mean))

        axd[band_pair_key].hist(band_pair.neg_side,bins=30,color='red',alpha=0.5, weights=np.ones_like(band_pair.neg_side) / len(band_pair.neg_side))
        axd[band_pair_key].axvline(band_pair.neg_fit_norm_mean,color='red',label="{} mean = {:.5f}".format(pl,band_pair.neg_fit_norm_mean))
        
        pos_x, pos_pdf, neg_x, neg_pdf = band_pair.evaluate_fit_norm()
        axd[band_pair_key].plot(pos_x,pos_pdf/pos_pdf.sum(),c='green',alpha=0.5 ,linestyle='dashed')
        axd[band_pair_key].plot(neg_x,neg_pdf/neg_pdf.sum(),c='red',alpha=0.5 ,linestyle='dashed')

        axd[band_pair_key].set_xlabel('Diff')
        axd[band_pair_key].set_ylabel('Freq.')

        axd[band_pair_key].set_title("{}: {}".format(band_pair_key,band_pair.classification_label))
        axd[band_pair_key].legend()

    data = copy.deepcopy(the_gal[the_gal.ref_band].data)
    el_mask = the_gal.create_ellipse()
    pos_mask, neg_mask = the_gal.create_bisection()
    pos_mask = np.logical_and(pos_mask,el_mask)
    neg_mask = np.logical_and(neg_mask,el_mask)
    
    the_mask = the_gal[the_gal.ref_band].valid_pixel_mask
    m, s = np.mean(data[the_mask]), np.std(data[the_mask])
    
    cmap = create_color_map_class(pos_mask,neg_mask,np.logical_and(el_mask,the_mask))
    data[np.logical_not(the_mask)] = 0

    axd['ref_band'].imshow(data, interpolation='nearest', cmap='gray', vmin=m-3*s, vmax=m+3*s, origin='lower') #, cmap='gray'
    axd['ref_band'].imshow(cmap, origin= 'lower',alpha=0.4)
    axd['ref_band'].set_title('ref band: {}\nVote:{}'.format(the_gal.ref_band,the_gal.cumulative_classification_vote_count))

    return fig, axd