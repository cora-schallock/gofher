import csv
import os

def write_csv(csv_path: str ,csv_col: list,csv_rows: list):
    """Write output csv
    
    Args: 
        csv_path: file path of csv to be created
        csv_col: a list of strings to be the csv's header
        csv_rows: a list of lists of strings, where each inner list represents a row of the csv
    """
    to_write = [csv_col]
    to_write.extend(csv_rows)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(to_write)

def construct_csv_dict(csv_path: str, the_key: str, the_value: str):
    with open(csv_path, mode='r', newline='') as file:
        to_return = dict()
        csv_dict_reader = csv.DictReader(file)
        for row in csv_dict_reader:
            to_return[row[the_key]] = row[the_value]
        return to_return


def check_if_folder_exists_and_create(path: str):
    """Addure folder exists
    
    Args: 
        path: path of folder to check if exists, and if not create
    """
    if not os.path.exists(path):
        os.makedirs(path)