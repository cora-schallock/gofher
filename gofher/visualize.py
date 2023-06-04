import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from fits import write_fits

#for lecture:
from gofher import normalize_array

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

#for lecture:
def visualize_map(data,pos_mask,neg_mask,el_mask,save_path=""):
    cmap = create_color_map_class(pos_mask,neg_mask,el_mask)
    #data[np.logical_not(the_mask)] = 0
    m, s = np.mean(data[el_mask]), np.std(data[el_mask])
    plt.imshow(data, interpolation='nearest', cmap='gray', vmin=m-3*s, vmax=m+3*s, origin='lower') #, cmap='gray'
    plt.imshow(cmap, origin= 'lower',alpha=0.4)

    if save_path != "":
        plt.savefig(save_path, dpi = 300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def visualize_single_hist(data,pos_mask,neg_mask,el_mask,diff):
    pos_mask = np.logical_and(pos_mask,el_mask)
    neg_mask = np.logical_and(neg_mask,el_mask)
    pos_mean = np.mean(data[pos_mask])
    neg_mean = np.mean(data[neg_mask])

    plt.hist(data[pos_mask],bins=30,color='green',alpha=0.5, weights=np.ones_like(data[pos_mask]) / len(data[pos_mask]))
    plt.axvline(pos_mean,color='green',label="mean = {:.5f}".format(pos_mean))
    plt.legend()
    plt.savefig("C:\\Users\\school\\Desktop\\adv_lecture\\ngc3368_pos.png")
    plt.clf()
    plt.hist(data[neg_mask],bins=30,color='red',alpha=0.5, weights=np.ones_like(data[neg_mask]) / len(data[neg_mask]))
    plt.axvline(neg_mean,color='red',label="mean = {:.5f}".format(neg_mean))
    plt.legend()
    plt.savefig("C:\\Users\\school\\Desktop\\adv_lecture\\ngc3368_neg.png")
    plt.clf()

    data = normalize_array(data,el_mask)
    pos_mean = np.mean(data[pos_mask])
    neg_mean = np.mean(data[neg_mask])
    plt.hist(data[pos_mask],bins=30,color='green',alpha=0.5, weights=np.ones_like(data[pos_mask]) / len(data[pos_mask]))
    plt.axvline(pos_mean,color='green',label="mean = {:.5f}".format(pos_mean))
    plt.legend()
    plt.savefig("C:\\Users\\school\\Desktop\\adv_lecture\\ngc3368_pos_norm.png")
    plt.clf()
    plt.hist(data[neg_mask],bins=30,color='red',alpha=0.5, weights=np.ones_like(data[neg_mask]) / len(data[neg_mask]))
    plt.axvline(neg_mean,color='red',label="mean = {:.5f}".format(neg_mean))
    plt.legend()
    plt.savefig("C:\\Users\\school\\Desktop\\adv_lecture\\ngc3368_neg_norm.png")
    plt.clf()

    pos_mean = np.mean(diff[pos_mask])
    neg_mean = np.mean(diff[neg_mask])
    plt.hist(diff[pos_mask],bins=30,color='green',alpha=0.5, weights=np.ones_like(diff[pos_mask]) / len(diff[pos_mask]))
    plt.axvline(pos_mean,color='green',label="mean = {:.5f}".format(pos_mean))
    plt.hist(diff[neg_mask],bins=30,color='red',alpha=0.5, weights=np.ones_like(diff[neg_mask]) / len(diff[neg_mask]))
    plt.axvline(neg_mean,color='red',label="mean = {:.5f}".format(neg_mean))
    plt.legend()
    plt.savefig("C:\\Users\\school\\Desktop\\adv_lecture\\ngc3368_diff.png")
    plt.clf()

def get_key(first,base):
    return "{}-{}".format(first,base)

def get_subplot_mosaic_strtings(bands_in_order):
    band_keys = []
    for (first_band,base_band) in itertools.combinations(bands_in_order, 2):
        band_keys.append(get_key(first_band,base_band))
    return np.array(band_keys).reshape(5,2).tolist()

def visualize_hist(the_gal, el_mask, pos_mask, neg_mask, pl, nl, pos_side_diff_dict, neg_side_diff_dict, mean_diff_dict, the_label_dict, the_score_dict, pos_x_dict, neg_x_dict, pos_pdf_point_dict, neg_pdf_point_dict, bands_in_order, dark_side_label, color_image_path, save_path=''):
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
        #axd[band_pair].hist(pos_side,bins=30,color='green',alpha=0.5,density=True,stacked=True)
        axd[band_pair].axvline(pos_mean,color='green',label="{} mean = {:.5f}".format(pl,pos_mean))

        axd[band_pair].hist(neg_side ,bins=30,color='red',alpha=0.5, weights=np.ones_like(neg_side) / len(neg_side))
        #axd[band_pair].hist(neg_side ,bins=30,color='red',alpha=0.5,density=True,stacked=True)
        axd[band_pair].axvline(neg_mean,color='red',label="{} mean = {:.5f}".format(nl, neg_mean))
        #axd[band_pair].set_ylim([0,0.2])

        if band_pair in pos_x_dict:
            axd[band_pair].plot(pos_x_dict[band_pair],pos_pdf_point_dict[band_pair]/pos_pdf_point_dict[band_pair].sum(),c='green',alpha=0.5 ,linestyle='dashed')
            axd[band_pair].plot(neg_x_dict[band_pair],neg_pdf_point_dict[band_pair]/neg_pdf_point_dict[band_pair].sum(),c='red',alpha=0.5 ,linestyle='dashed')

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