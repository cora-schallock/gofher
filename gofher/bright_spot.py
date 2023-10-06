import scipy
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from fits import view_fits
#from run_fit import calculate_dist

def the_avg(x):
    return np.sum(x)/len(x)

def calculate_dist(cm,center):
    return np.linalg.norm(np.array(cm)-np.array(center))

import scipy
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from fits import view_fits

def the_max(x):
    return np.max(x)

def the_min(x):
    return np.min(x)

def create_bright_spot_mask(data):
    inner_dim = 3
    outer_dim = 25
    t=int((outer_dim-inner_dim)*0.5)

    inner_square = np.ones((inner_dim,inner_dim))
    inner = np.pad(inner_square, ((t,t),(t,t)), 'constant')
    outer = 1-inner

    inner_avg = scipy.ndimage.generic_filter(data,the_avg,footprint=inner)
    outer_avg = scipy.ndimage.generic_filter(data,the_max,footprint=outer)
    
    bright_diff=inner_avg-outer_avg
    
    spot_mask = np.zeros(data.shape)
    spot_mask[bright_diff>0] = 1.0
    
    spot_mask = scipy.ndimage.generic_filter(spot_mask,the_max,footprint=inner)
    spot_mask = scipy.ndimage.gaussian_filter(spot_mask,3)
    spot_mask[spot_mask>0.5] = 1.0

    return spot_mask

def combine_spot_and_star_mask(spot_mask,bulge_mask,star_mask):
    to_mask = np.logical_or(bulge_mask,spot_mask)
    
    all_labels = measure.label(to_mask)
    blobs_labels = measure.label(to_mask, background=0)
    
    for label in np.unique(blobs_labels):
        label_area = np.zeros(spot_mask.shape)
        label_area[blobs_labels==label] = 1.0
        bulge_overlap_count=np.sum(np.logical_and(label_area,bulge_mask))
        
        if bulge_overlap_count > 0:
            to_mask[blobs_labels==label] = 0
            
    #to_mask = np.logical_or(to_mask,star_mask)
    return to_mask

def find_mask_spot_closest_to_center(the_mask,approx_center):
    shape=the_mask.shape
    all_labels = measure.label(the_mask) #https://scipy-lectures.org/packages/scikit-image/auto_examples/plot_labels.html
    blobs_labels = measure.label(all_labels, background=0)

    unique = np.unique(blobs_labels.flatten()) #https://stackoverflow.com/a/28663910/13544635

    center_of_masses = scipy.ndimage.center_of_mass(np.ones(shape),blobs_labels, index=unique)
    dist_to_center = list(map(lambda i: np.inf if i == 0 else calculate_dist(center_of_masses[i],approx_center),range(len(center_of_masses))))

    return blobs_labels==unique[np.argmin(dist_to_center)]