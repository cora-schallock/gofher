import itertools
from visualize import get_key
import csv

def write_csv(csv_path,csv_col,csv_rows):
    to_write = [csv_col]
    to_write.extend(csv_rows)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(to_write) 

def prepare_csv_row(the_gal,dark_side_label,the_band_pairs, mean_dict, score_dict, pl, nl, the_label_dict):
    the_row = [the_gal.name, dark_side_label, pl, nl]
    
    mean_row = []
    label_row = []
    score_row = []
    
    for bp in the_band_pairs:
        mean_row.append(mean_dict[bp])
        label_row.append(the_label_dict[bp])
        score_row.append(score_dict[bp])
    the_row.extend(mean_row)
    the_row.extend(label_row)
    the_row.extend(score_row)
    
    return the_row

def get_csv_cols(bands_in_order):
    csv_cols = ['name','dark','pos','neg']
    the_band_pairs = []
    for (first_band,base_band) in itertools.combinations(bands_in_order, 2):
        the_band_pairs.append(get_key(first_band,base_band))
    csv_cols.extend(list(map(lambda x: x+" mean diff",the_band_pairs)))
    csv_cols.extend(list(map(lambda x: x+" label",the_band_pairs)))
    csv_cols.extend(list(map(lambda x: x+" score",the_band_pairs)))
    return the_band_pairs, csv_cols