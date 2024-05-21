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


def check_if_folder_exists_and_create(path: str):
    """Addure folder exists
    
    Args: 
        path: path of folder to check if exists, and if not create
    """
    if not os.path.exists(path):
        os.makedirs(path)