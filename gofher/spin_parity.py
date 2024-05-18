from classify import get_opposite_label

def score_label(label,paper_label):
    opposite = get_opposite_label(label)

    correct_label_letter_count = len(set([*label.lower()]).union([*paper_label.lower()]))
    incorrect_label_letter_count = len(set([*opposite.lower()]).union([*paper_label.lower()]))

    if correct_label_letter_count > incorrect_label_letter_count and correct_label_letter_count > 1:
        return -1
    elif incorrect_label_letter_count > correct_label_letter_count and incorrect_label_letter_count > 1:
        return 1
    return 0

import numpy as np
import pandas as pd

def read_spin_parity_galaxies_label_from_csv(csv_path):
    """read in the spin parity csv"""
    the_csv = pd.read_csv(csv_path)

    galaxy_name_to_dark_side_dict = dict()

    for i in range(len(the_csv["name"])):
        name = the_csv["name"][i]
        dark = the_csv["dark"][i]

        galaxy_name_to_dark_side_dict[name] = dark

    return galaxy_name_to_dark_side_dict