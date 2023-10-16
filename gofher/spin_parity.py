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
    """read in the spin parity csv"""
    the_csv = pd.read_csv(csv_path)

    galaxy_name_to_dark_side_dict = dict()

    for i in range(len(the_csv[NAME])):
        name = the_csv[NAME][i]
        dark = the_csv[DARK][i]

        galaxy_name_to_dark_side_dict[name] = dark

    return galaxy_name_to_dark_side_dict

def read_spin_parity_galaxies_label_from_csv_sdss(csv_path,cross_path):
    """read in the spin parity with cross ids"""
    cross_id = dict()
    with open(cross_path,'r') as f:
        first = True
        for lines in f.readlines():
            if first:
                first = False
                continue
            to_parse = lines.split("\t")
            cross_id[to_parse[1]] = to_parse[0]
    
    return read_spin_parity_galaxies_label_from_csv(csv_path), cross_id


def pos_neg_label_from_theta(theta):
    """get pos/neg label from theta"""
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