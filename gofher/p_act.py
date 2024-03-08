import copy
import itertools
import scipy
import numpy as np
from scipy.stats import ttest_ind
from scipy.stats import norm, mvn

# Define constants
version = 1.2
level1 = 25000
cutoff2 = 0.05
level2 = 250000
cutoff3 = 0.01
level3 = 2500000
cutoff4 = 0.00001
level4 = 25000000

def normalize_array(the_array):
    the_min = np.min(the_array)
    the_max = np.max(the_array)
    return (the_array-the_min)/(the_max-the_min)

def evaluate_p_act(the_gal, ordered_bands):
    el_mask = the_gal.create_ellipse()
    valid_pixel_mask = np.ones(el_mask.shape,dtype=bool)

    for band in the_gal.bands:
        valid_pixel_mask = np.logical_and(valid_pixel_mask,copy.deepcopy(the_gal[band].valid_pixel_mask))
    
    area_to_consider = np.logical_and(el_mask,valid_pixel_mask)

    band_data_dict = dict()
    for band in the_gal.bands:
        #band_data_dict[band] = normalize_array(copy.deepcopy(the_gal.bands[band].data)[area_to_consider])
        band_data_dict[band] = copy.deepcopy(the_gal.bands[band].data)[area_to_consider]
        print(band_data_dict[band].shape)

    bands_for_gal_in_order = list(filter(lambda x: x in band_data_dict.keys(),ordered_bands))

    dim = len(band_data_dict.keys())
    pearson_matrix = np.zeros((dim,dim))

    pvals = []

    for (first_band,base_band) in list(itertools.combinations(bands_for_gal_in_order, 2)):
        i = bands_for_gal_in_order.index(first_band)
        j = bands_for_gal_in_order.index(base_band)

        the_stat = scipy.stats.pearsonr(band_data_dict[first_band],band_data_dict[base_band]).statistic
        pearson_matrix[i][j] = the_stat
        pearson_matrix[j][i] = the_stat

        t_stat, p_val = ttest_ind(band_data_dict[first_band],band_data_dict[base_band])
        pvals.append(p_val)

    pvals.sort()

    v=pearson_matrix
    
    minp = pvals[0]
    if minp == 0:
        p_ACT = 0
    elif minp >= 0.5:
        p_ACT = 1
    else:
        L = len(pvals)
        lower = np.repeat(norm.ppf(minp/2), L)
        upper = np.repeat(norm.ppf(1-minp/2), L)
        if L > 3:
            lower = np.repeat(-np.inf, L)
            upper = np.repeat(np.inf, L)
            lower = norm.ppf(minp/2)
            upper = norm.ppf(1-minp/2)

        print(lower)
        print(upper)

        p_ACT = 1 - mvn.mvnun(lower, upper, np.zeros(v.shape[0]), v, maxpts=level1, abseps=1e-13)[0]
        if p_ACT < cutoff2:
            p_ACT = 1 - mvn.mvnun(lower, upper, np.zeros(v.shape[0]), v, maxpts=level2, abseps=1e-13)[0]
            if p_ACT < cutoff3:
                p_ACT = 1 - mvn.mvnun(lower, upper, np.zeros(v.shape[0]), v, maxpts=level3, abseps=1e-13)[0]
                if p_ACT < cutoff4:
                    p_ACT = 1 - mvn.mvnun(lower, upper, np.zeros(v.shape[0]), v, maxpts=level4, abseps=1e-13)[0]
        print(p_ACT)