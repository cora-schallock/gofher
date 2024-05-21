import numpy as np
import pandas as pd

from classify import get_opposite_label

def standardize_galaxy_name(name: str):
    """Normalize galaxy name by removing spaces and *'s
    Important: Note the name's in catalog may have *'s when galaxy name refers to alternate name
    
    Args:
        name: galaxy name to normalize
    """
    return name.strip().replace("*","").replace(" ","")

def score_label(label: str, paper_label: str):
    """Calculates score of label compared to paper_label
    score = 1 if label & paper label agree
    score = -1 if label & paper label disagree
    score = 0 if label & paper neither agree or disagree

    Important: When scoring labels, allows rounds cardinal directions by up to 45degrees to score agree/disagree cases
    
    If label=E
        Agree if paper_label: E, SE, or NE
        Disagree if paper_label: W, SW, or NW
        Neither Agree nor Disagree if paper_label: S or N
    """
    opposite = get_opposite_label(label)

    correct_label_letter_count = len(set([*label.lower()]).union([*paper_label.lower()]))
    incorrect_label_letter_count = len(set([*opposite.lower()]).union([*paper_label.lower()]))

    if correct_label_letter_count > incorrect_label_letter_count and correct_label_letter_count > 1:
        return -1
    elif incorrect_label_letter_count > correct_label_letter_count and incorrect_label_letter_count > 1:
        return 1
    return 0

def read_spin_parity_galaxies_label_from_csv(csv_path: str):
    """Parse csv of catalog info to get dictionary of names and dark dusty side labels
    
    Args:
        csv_path: the file path of catalog csv
    """
    the_csv = pd.read_csv(csv_path)

    galaxy_name_to_dark_side_dict = dict()

    for i in range(len(the_csv["name"])):
        name = standardize_galaxy_name(the_csv["name"][i])
        dark = the_csv["dark"][i]

        galaxy_name_to_dark_side_dict[name] = dark

    return galaxy_name_to_dark_side_dict