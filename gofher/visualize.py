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
from compare_histograms import compute_laplace_smoothed_kld, compute_wasserstein_distance

#DEFAULT_POSITIVE_RGB_VECTOR = [60/255,179/255,113/255] #mediumseagreen
DEFAULT_POSITIVE_RGB_VECTOR = [190/255,67/255,159/255] #BE439F
#DEFAULT_NEGATIVE_RGB_VECTOR = [240/255,128/255,125/255] #lightcoral
DEFAULT_NEGATIVE_RGB_VECTOR = [222/255,158/255,54/255] #DE9E36
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

def visualize(the_gal: galaxy, color_image: np.ndarray, bands_in_order = [], paper_label='', save_path='',color_flip=False,show_stats=True,visual_string="") -> dict:
    """Visualize the classification process gofher uses for determining label

    Displays ellipse mask, bisection mask on refernce image and histograms
    for all included waveband pairs.
    
    If paper_label is provided, includes if it agrees/disagrees.
    
    Note: Only considers waveband pairs composed of bands in bands_in_order
    

    Args:
        color_image: the color refercne image that is displayed as thumbnail
        bands_in_order: wavebands to use in order of bluest to reddest
        save_path: if given path saves visualize image, if not displays it

    Returns:
        A dictionary containning the data from each hisogram
            key: the band pair key
            value: a list as follows: [list_of_bin_edges, bin_width, pos_counts, neg_counts]
                Note: len(list_of_bin_edges) == len(pos_counts) == len(neg_counts)
    """


    import matplotlib.pyplot as plt
    #import matplotlib
    #matplotlib.rcParams['nbagg.transparent'] = False
    
    # in points - start with the body text size and play around
    
    SMALL_SIZE = int(20*1.5)
    MEDIUM_SIZE = int(24*1.5)
    BIGGER_SIZE = int(28*1.5)

    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)
    
    plt.rc('font', family='Nimbus Roman No9 L')

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

    #make equal bin width:
    binwidth = (hist_range_to_plot[1]-hist_range_to_plot[0])/50
    bins = np.arange(min(hist_range_to_plot), max(hist_range_to_plot) + binwidth, binwidth)

    hist_dict = dict()
    votes = []
    vote_outcome = "No vote"
    majority_vote = ''
    for (blue_band,red_band) in itertools.combinations(bands_in_order, 2):
        band_pair_key = construct_galaxy_band_pair_key(blue_band,red_band)
        band_pair = the_gal.get_band_pair(band_pair_key)
        votes.append(band_pair.classification_label)

        #see: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
        pos_counts, pos_bin_edges, _ = axd[band_pair_key].hist(band_pair.pos_side,bins=bins,color='#BE439F',alpha=0.5, weights=np.ones_like(band_pair.pos_side) / len(band_pair.pos_side))
        axd[band_pair_key].axvline(band_pair.pos_mean,color='#BE439F',label="{} μ = {:.3f}".format(the_gal.pos_side_label,band_pair.pos_mean))

        neg_counts, neg_bin_edges, _ = axd[band_pair_key].hist(band_pair.neg_side,bins=bins,color='#DE9E36',alpha=0.5, weights=np.ones_like(band_pair.neg_side) / len(band_pair.neg_side))
        axd[band_pair_key].axvline(band_pair.neg_mean,color='#B47613',label="{} μ = {:.3f}".format(the_gal.neg_side_label,band_pair.neg_mean))

        #hist_dict[band_pair_key] = [bin_edges, binwidth, pos_counts, neg_counts]
        #from scipy.special import rel_entr
        #print(pos_counts)
        #print(neg_counts)
        #print(len(band_pair.pos_side)*pos_counts)
        #print(rel_entr(pos_counts, neg_counts))
        #print(compute_chisquare(pos_counts, neg_counts))
        #print(compute_kld(pos_counts, neg_counts))
        #print()
        
        
        band_pair.wasserstein_distance = compute_wasserstein_distance(pos_bin_edges, neg_bin_edges, pos_counts, neg_counts)
        
        pos_n = len(band_pair.pos_side)
        neg_n = len(band_pair.neg_side)
        band_pair.laplace_smoothed_kld = compute_laplace_smoothed_kld(pos_counts, neg_counts, pos_n,neg_n)
        
        
        if band_pair._used_normed:
            pos_x, pos_pdf, neg_x, neg_pdf = band_pair.evaluate_fit_norm()
            axd[band_pair_key].plot(pos_x,pos_pdf/pos_pdf.sum(),c='#BE439F',alpha=0.5 ,linestyle='dashed')
            axd[band_pair_key].plot(neg_x,neg_pdf/neg_pdf.sum(),c='#DE9E36',alpha=0.5 ,linestyle='dashed')
            axd[band_pair_key].set_xlim(hist_range_to_plot[0],hist_range_to_plot[1])
        if show_stats:
            axd[band_pair_key].set_title("{}: {} (pval={:.2E})".format(band_pair_key,band_pair.classification_label,band_pair.mannwhitneyu_p_value))
        else:
            axd[band_pair_key].set_title("{}: {}".format(band_pair_key,band_pair.classification_label))
        axd[band_pair_key].legend()

        x_min, x_max = axd[band_pair_key].get_ylim()
        axd[band_pair_key].set_ylim(x_min,x_max*1.5)
        
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
    
    if color_flip:
        axd['color'].imshow(color_image, origin='lower')
    else:
        axd['color'].imshow(color_image)
    if paper_label != '':
        axd['color'].set_title("{} {}\n paper label={}".format(the_gal.name,visual_string,paper_label))
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

    return hist_dict
