import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from fits import write_fits

PANSTARRS_COLOR_DICT = {'g':'green',
                        'r':'red',
                        'i':'indigo',
                        'z':'blue',
                        'y':'orange'}

DEFAULT_POSITIVE_RGB_VECTOR = [60/255,179/255,113/255] #mediumseagreen
DEFAULT_NEGATIVE_RGB_VECTOR = [240/255,128/255,125/255] #lightcoral
DEFAULT_BAD_PIXEL_RGB_VECTOR = [169/255,169/255,169/255] #lightgrey

def create_color_map_class(pos,neg,el):
    cmap_class = np.ones((pos.shape[0],pos.shape[1],3))*DEFAULT_BAD_PIXEL_RGB_VECTOR
    cmap_class[np.logical_and(pos,el)] = DEFAULT_POSITIVE_RGB_VECTOR
    cmap_class[np.logical_and(neg,el)] = DEFAULT_NEGATIVE_RGB_VECTOR
    
    return cmap_class

def get_key(first,base):
    return "{}-{}".format(first,base)

def get_subplot_mosaic_strtings(bands_in_order):
    band_keys = []
    for (first_band,base_band) in itertools.combinations(bands_in_order, 2):
        band_keys.append(get_key(first_band,base_band))
    return np.array(band_keys).reshape(5,2).tolist()

def visualize_hist(the_gal, el_mask, pos_mask, neg_mask, pl, nl, pos_side_diff_dict, neg_side_diff_dict, mean_diff_dict, the_label_dict, the_score_dict, bands_in_order, dark_side_label, color_image_path, save_path=''):
    mo_labels = [['color','ref_band']]
    mo_labels.extend(get_subplot_mosaic_strtings(bands_in_order))
    
    gs_kw = dict(width_ratios=[1,1], height_ratios=[2,1,1,1,1,1])
    fig, axd = plt.subplot_mosaic(mo_labels,
                                  gridspec_kw=gs_kw, figsize = (24,32),
                                  constrained_layout=True,num=1, clear=True) #num=1, clear=True #https://stackoverflow.com/a/65910539/13544635
    fig.patch.set_facecolor('white')
    
    for band_pair in mean_diff_dict:
        pos_side = pos_side_diff_dict[band_pair]
        neg_side = neg_side_diff_dict[band_pair]
        pos_mean = np.mean(pos_side)
        neg_mean = np.mean(neg_side)

        axd[band_pair].hist(pos_side,bins=30,color='green',alpha=0.5, weights=np.ones_like(pos_side) / len(pos_side))
        axd[band_pair].axvline(pos_mean,color='green',label="{} mean = {:.5f}".format(pl,pos_mean))

        axd[band_pair].hist(neg_side ,bins=30,color='red',alpha=0.5, weights=np.ones_like(neg_side) / len(neg_side))
        axd[band_pair].axvline(neg_mean,color='red',label="{} mean = {:.5f}".format(nl, neg_mean))
        
        axd[band_pair].set_xlabel('Diff')
        axd[band_pair].set_ylabel('Freq.')
        
        axd[band_pair].set_title("{}: {:0.3f} = {}".format(band_pair,mean_diff_dict[band_pair],the_label_dict[band_pair]))
        axd[band_pair].legend()
    
    #color:
    color = mpimg.imread(color_image_path(the_gal.name))
    axd['color'].imshow(color)
    axd['color'].set_title("{}: dark side {}".format(the_gal.name,dark_side_label))
    
    data = the_gal.data[the_gal.ref_band]
    
    the_mask = the_gal.valid_pixel_mask[the_gal.ref_band]
    m, s = np.mean(data[the_mask]), np.std(data[the_mask])
    cmap = create_color_map_class(pos_mask,neg_mask,np.logical_and(el_mask,the_mask))
    data[np.logical_not(the_mask)] = 0
    #write_fits('C:\\Users\\school\\Desktop\\check_it.fits', data)
    #write_fits('C:\\Users\\school\\Desktop\\check_it2.fits', el_mask)
    #print(m,s)
    axd['ref_band'].imshow(data, interpolation='nearest', cmap='gray', vmin=m-3*s, vmax=m+3*s, origin='lower') #, cmap='gray'
    axd['ref_band'].imshow(cmap, origin= 'lower',alpha=0.4)
    axd['ref_band'].set_title('ref band: {}'.format(the_gal.ref_band))
    
    if save_path != "":
        fig.savefig(save_path, dpi = 300, bbox_inches='tight')
    else:
        plt.show()
    fig.clear()
    plt.close(fig)
    #plt.close('all') #possible memory leak