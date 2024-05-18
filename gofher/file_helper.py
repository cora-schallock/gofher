import csv
import os

def write_csv(csv_path,csv_col,csv_rows):
    '''write oute csv'''
    to_write = [csv_col]
    to_write.extend(csv_rows)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(to_write)


def check_if_folder_exists_and_create(path):
    '''check if folder exists and if not, create it'''
    if not os.path.exists(path):
        os.makedirs(path)