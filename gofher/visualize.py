USE_BACKED_MATPLOT = False #Important: If saving many visualizations at once, set this to True to avoid slow down due to memeory leak

if USE_BACKED_MATPLOT:
    import matplotlib #https://matplotlib.org/stable/users/explain/figure/backends.html
    matplotlib.use('Agg') #for memory leak in plt backend: https://stackoverflow.com/a/73698657/13544635
  
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
import itertools
import numpy as np

from galaxy import galaxy
from galaxy_band_pair import construct_galaxy_band_pair_key
from spin_parity import score_label

DEFAULT_POSITIVE_RGB_VECTOR = [60/255,179/255,113/255] #mediumseagreen
DEFAULT_NEGATIVE_RGB_VECTOR = [240/255,128/255,125/255] #lightcoral
DEFAULT_BAD_PIXEL_RGB_VECTOR = [169/255,169/255,169/255] #lightgrey

def create_color_map_class(pos,neg,el):
    """Generates a colormap for visualization purposes

    Wavebandpairs (x,y) are ordered left-to-right top-to-bottom (2 per row)
        x bluest to reddest band choosen first
        y bluest to reddest band choosen second

    Note: If c(n,2) where n=len(bands_in_order) is odd, adds extra
        mosaic string in last row
    
    Args:
        pos: one side of the bisection mask (called the pos_mask)
        neg: the other side of the bisection mask (called the neg_mask)
        el: the ellipse mask the gofher uses
    """
    cmap_class = np.ones((pos.shape[0],pos.shape[1],3))*DEFAULT_BAD_PIXEL_RGB_VECTOR
    cmap_class[np.logical_and(pos,el)] = DEFAULT_POSITIVE_RGB_VECTOR
    cmap_class[np.logical_and(neg,el)] = DEFAULT_NEGATIVE_RGB_VECTOR
    
    return cmap_class

def get_subplot_mosaic_strtings(bands_in_order):
    """Generates waveband pair keys to display in visualize

    Wavebandpairs (x,y) are ordered left-to-right top-to-bottom (2 per row)
        x bluest to reddest band choosen first
        y bluest to reddest band choosen second

    Note: If c(n,2) where n=len(bands_in_order) is odd, adds extra
        mosaic string in last row
    
    Args:
        bands_in_order: wavebands to use in order of bluest to reddest
    """
    band_keys = []
    for (blue_band,red_band) in itertools.combinations(bands_in_order, 2):
        band_keys.append(construct_galaxy_band_pair_key(blue_band,red_band))
    if len(band_keys)%2 != 0: band_keys.append('')
    return np.array(band_keys).reshape(int(len(band_keys)/2),2).tolist()

def visualize(the_gal: galaxy, color_image: np.ndarray, bands_in_order = [], paper_label='', save_path=''):
    """Visualize the classification process gofher uses for determining label

    Displays ellipse mask, bisection mask on refernce image and histograms
    for all included waveband pairs.
    
    If paper_label is provided, includes if it agrees/disagrees.
    
    Note: Only considers waveband pairs composed of bands in bands_in_order
    

    Args:
        color_image: the color refercne image that is displayed as thumbnail
        bands_in_order: wavebands to use in order of bluest to reddest
        save_path: if given path saves visualize image, if not displays it
    """
    mo_labels = [['color','ref_band']]
    mo_labels.extend(get_subplot_mosaic_strtings(bands_in_order))
    height_ratios = [2] + [1] * int(len(mo_labels)-1) #only works if band_pair numbers is even
    gs_kw = dict(width_ratios=[1,1], height_ratios=height_ratios)
    fig, axd = plt.subplot_mosaic(mo_labels,
                                  gridspec_kw=gs_kw, figsize = (24,30),
                                  constrained_layout=True,num=1, clear=True) #num=1, clear=True #https://stackoverflow.com/a/65910539/13544635
    fig.patch.set_facecolor('white')

    sum_of = 0.0
    sum_of_squares = 0.0
    n = 0

    for (blue_band,red_band) in itertools.combinations(bands_in_order, 2):
        band_pair_key = construct_galaxy_band_pair_key(blue_band,red_band)
        band_pair = the_gal.get_band_pair(band_pair_key)
        elements = band_pair.diff_image[the_gal.area_to_diff]

        sum_of += np.sum(elements)
        sum_of_squares += np.sum(elements**2)
        n += len(elements)

    mean = sum_of/n
    std = ((sum_of_squares-(sum_of/n))/n)**0.5

    hist_range_to_plot = [mean-3*std,mean+3*std] #[min,max] value of range of histograms
    
    votes = []
    vote_outcome = "No vote"
    majority_vote = ''
    for (blue_band,red_band) in itertools.combinations(bands_in_order, 2):
        band_pair_key = construct_galaxy_band_pair_key(blue_band,red_band)
        pl = ''
        band_pair = the_gal.get_band_pair(band_pair_key)
        votes.append(band_pair.classification_label)

        axd[band_pair_key].hist(band_pair.pos_side,bins=30,color='green',alpha=0.5, weights=np.ones_like(band_pair.pos_side) / len(band_pair.pos_side))
        axd[band_pair_key].axvline(band_pair.pos_fit_norm_mean,color='green',label="{} mean = {:.5f}".format(the_gal.pos_side_label,band_pair.pos_fit_norm_mean))

        axd[band_pair_key].hist(band_pair.neg_side,bins=30,color='red',alpha=0.5, weights=np.ones_like(band_pair.neg_side) / len(band_pair.neg_side))
        axd[band_pair_key].axvline(band_pair.neg_fit_norm_mean,color='red',label="{} mean = {:.5f}".format(the_gal.neg_side_label,band_pair.neg_fit_norm_mean))
        
        pos_x, pos_pdf, neg_x, neg_pdf = band_pair.evaluate_fit_norm()
        axd[band_pair_key].plot(pos_x,pos_pdf/pos_pdf.sum(),c='green',alpha=0.5 ,linestyle='dashed')
        axd[band_pair_key].plot(neg_x,neg_pdf/neg_pdf.sum(),c='red',alpha=0.5 ,linestyle='dashed')
        axd[band_pair_key].set_xlim(hist_range_to_plot[0],hist_range_to_plot[1])
        axd[band_pair_key].set_title("{}: {} (ks pval={:.3E})".format(band_pair_key,band_pair.classification_label,band_pair.ks_p_value))
        axd[band_pair_key].legend()
        
    if len(set(votes)) == 1:
        majority_vote = votes[0]
    elif votes.count(list(set(votes))[0]) != votes.count(list(set(votes))[1]):
        majority_vote = max(set(votes), key=votes.count)

    if majority_vote != '' and paper_label != '':
        result = score_label(majority_vote,paper_label)
        if result == 1:
            vote_outcome = 'Agree'
        elif result == -1:
            vote_outcome = 'Disagree'

    if mo_labels[-1][-1] == '':
        axd[''].axis('off')

    data = the_gal[the_gal.ref_band].data
    el_mask = the_gal.create_ellipse()
    pos_mask, neg_mask = the_gal.create_bisection()
    the_mask = the_gal[the_gal.ref_band].valid_pixel_mask
    cmap = create_color_map_class(pos_mask,neg_mask,np.logical_and(el_mask,the_mask))

    m, s = np.mean(data[the_mask]), np.std(data[the_mask])
    axd['ref_band'].imshow(data, interpolation='nearest', cmap='gray', vmin=m-3*s, vmax=m+3*s, origin='lower') #, cmap='gray'
    axd['ref_band'].imshow(cmap, origin= 'lower',alpha=0.4)
    

    axd['color'].imshow(color_image)
    if paper_label != '':
        axd['color'].set_title("{}\n paper label={}".format(the_gal.name,paper_label))
        axd['ref_band'].set_title('ref band: {}\ngofher label = {} ({})'.format(the_gal.ref_band,majority_vote,vote_outcome))
    else:
        axd['color'].set_title(the_gal.name)
        axd['ref_band'].set_title('ref band: {} gofher label = {}'.format(the_gal.ref_band,majority_vote))

    if save_path != "":
        fig.savefig(save_path, dpi = 300, bbox_inches='tight')
        fig.clear()
        plt.close(fig)
    else:
        plt.show()
