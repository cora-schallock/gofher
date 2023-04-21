import numpy as np

import pandas as pd

# NOTE: this should match csv label data
NAME = "name"
SZ = "sz"
DARK = "dark"
APPR = "appr"
TL = "tl"
REFERENCE = "ref"

FOUND = "dis"

def read_spin_parity_galaxies_label_from_csv(csv_path):
    the_csv = pd.read_csv(csv_path)

    galaxy_name_to_dark_side_dict = dict()

    for i in range(len(the_csv[NAME])):
        name = the_csv[NAME][i]
        dark = the_csv[DARK][i]

        galaxy_name_to_dark_side_dict[name] = dark

    return galaxy_name_to_dark_side_dict

def pos_neg_label_from_theta(theta):
    new_theta = theta % 360
    pos = '-'
    neg = '-'
    if new_theta <= 22.5:
        pos = 'n'
        neg = 's'
    elif new_theta > 22.5 and new_theta < 67.5:
        pos = 'ne'
        neg = 'sw'
    elif new_theta >= 67.5 and new_theta <= 112.5:
        pos = 'e'
        neg = 'w'
    elif new_theta > 112.5 and new_theta < 157.5:
        pos = 'se'
        neg = 'nw'
    elif new_theta >= 157.5 and new_theta <= 202.5:
        pos = 's'
        neg = 'n'
    elif new_theta > 202.5 and new_theta < 247.5:
        pos = 'sw'
        neg = 'ne'
    elif new_theta >= 247.5 and new_theta <= 292.5:
        pos = 'w'
        neg = 'e'
    elif new_theta > 292.5 and new_theta < 337.5:
        pos = 'nw'
        neg = 'se'
    else:
        pos = 'n'
        neg = 's'
    return pos.upper(),neg.upper()

def score(dark,pl,nl,mean_dif):
    label = nl if -np.sign(mean_dif) == 1.0 else pl
    opposite = pl if -np.sign(mean_dif) == 1.0 else nl

    correct_label_letter_count = len(set([*label.lower()]).union([*dark.lower()]))
    incorrect_label_letter_count = len(set([*opposite.lower()]).union([*dark.lower()]))
        
    if correct_label_letter_count > incorrect_label_letter_count and correct_label_letter_count > 1:
        return 1
    elif incorrect_label_letter_count > correct_label_letter_count and incorrect_label_letter_count > 1:
        return -1
    else:
        return 0

def classify_spin_parity(the_gal,dark_side_label,pos_diff_dict,neg_diff_dict):
    pl, nl = pos_neg_label_from_theta(np.degrees(the_gal.theta))
    mean_diff_dict = dict()
    the_score_dict = dict()
    the_label_dict = dict()
    
    for band_pair in pos_diff_dict.keys():
        pos_mean = np.mean(pos_diff_dict[band_pair])
        neg_mean = np.mean(neg_diff_dict[band_pair])
        
        mean_dif = pos_mean-neg_mean
        mean_diff_dict[band_pair] = mean_dif
        
        the_label = nl if -np.sign(mean_dif) == -1.0 else pl
        the_label_dict[band_pair] = the_label
        
        the_score = score(dark_side_label,pl,nl,mean_dif)
        the_score_dict[band_pair] = the_score
    return mean_diff_dict, the_label_dict, the_score_dict, pl, nl