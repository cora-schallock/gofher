import itertools
from visualize import get_key
import csv
import numpy as np
import os

def write_csv(csv_path,csv_col,csv_rows):
    to_write = [csv_col]
    to_write.extend(csv_rows)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(to_write) 

from scipy.stats import gmean

def prepare_csv_row(the_gal,dark_side_label,the_band_pairs, mean_dict, p_value_dict, score_dict, pl, nl, the_label_dict):
    the_row = [the_gal.name, dark_side_label, pl, nl]
    
    mean_row = []
    avg_mean_plus = 0
    avg_mean_min = 0
    p_row = []
    p_mean_plus = 0
    p_mean_min = 0
    label_row = []
    score_row = []
    vote_count = 0
    vote_outcome = 0
    
    for bp in the_band_pairs:
        mean_row.append(mean_dict[bp])
        p_row.append(p_value_dict[bp])
        label_row.append(the_label_dict[bp])
        score_row.append(score_dict[bp])

    scores = np.array(score_row)
    if len(scores[scores==1]) > 0:
        avg_mean_plus = np.mean(np.array(mean_row)[scores==1])
        #p_mean_plus = np.mean(np.array(p_row)[scores==1]) #arithmetic mean
        p_mean_plus = gmean(np.array(p_row)[scores==1]) #geometric mean
    else:
        p_mean_plus = ""
    if len(scores[scores==-1]) > 0:
        avg_mean_min = np.mean(np.array(mean_row)[scores==-1])
        #p_mean_min = np.mean(np.array(p_row)[scores==-1]) #arithmetic mean
        p_mean_min = gmean(np.array(p_row)[scores==-1]) #geometric mean
    else:
        p_mean_min = ""
    vote_count = len(scores[scores==1]) - len(scores[scores==-1])
    vote_outcome = np.sign(vote_count)

    the_row.extend(mean_row)
    the_row.extend([avg_mean_plus,avg_mean_min])
    the_row.extend(p_row)
    the_row.extend([p_mean_plus,p_mean_min])
    the_row.extend(label_row)
    the_row.extend(score_row)
    the_row.extend([vote_count,vote_outcome])
    
    return the_row

def get_csv_cols(bands_in_order):
    csv_cols = ['name','dark','pos','neg']
    the_band_pairs = []
    for (first_band,base_band) in itertools.combinations(bands_in_order, 2):
        the_band_pairs.append(get_key(first_band,base_band))
    csv_cols.extend(list(map(lambda x: x+" mean diff",the_band_pairs)))
    csv_cols.extend(['avg. mean diff score +1','avg. mean diff score -1'])
    csv_cols.extend(list(map(lambda x: x+" ks p-value",the_band_pairs)))
    csv_cols.extend(['avg. ks p-value score +1','avg. ks p-value score -1'])
    csv_cols.extend(list(map(lambda x: x+" label",the_band_pairs)))
    csv_cols.extend(list(map(lambda x: x+" score",the_band_pairs)))
    csv_cols.extend(['vote count', 'vote score'])
    return the_band_pairs, csv_cols


#folder helper
def check_if_folder_exists_and_create(path):
    '''check if folder exists and if not, create it'''
    if not os.path.exists(path):
        os.makedirs(path)