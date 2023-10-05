import numpy as np

def normalize_array(data,to_diff_mask):
    """data - fits file (after read fits)
    to_diff_mask - a single boolean mask indicating where to create diff (1's = included, 0's = ignored)"""
    normalized = np.zeros(data.shape)
    the_min = np.min(data[to_diff_mask]); the_max = np.max(data[to_diff_mask])
    normalized[to_diff_mask] = (data[to_diff_mask]- the_min)/(the_max-the_min)
    return normalized

def create_diff_image(first_data,base_data,to_diff_mask):
    """first_data - fits data of bluer band
    base_data - fits data of redder band
    to_diff_mask - a single boolean mask indicating where to create diff (1's = included, 0's = ignored)"""
    first_norm = normalize_array(first_data,to_diff_mask)
    base_norm = normalize_array(base_data,to_diff_mask)
    return first_norm-base_norm

def extract_data_for_histogram(diff,pos_mask,neg_mask,to_diff_mask):
    """diff - the diff image
    pos_mask - a signle boolean mask (1's in pos. area, 0's not in pos. area)
    neg_mask - a signle boolean mask (1's in neg. area, 0's not in neg. area)
    to_diff_mask - a single boolean mask indicating where to create diff (1's = included, 0's = ignored)"""
    pos_area = np.logical_and(pos_mask,to_diff_mask)
    neg_area = np.logical_and(neg_mask,to_diff_mask)

    pos_side = diff[pos_area]
    neg_side = diff[neg_area]

    return pos_side, neg_side