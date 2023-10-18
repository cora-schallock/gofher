import os
import itertools

from spin_parity import read_spin_parity_galaxies_label_from_csv
from run_gofher_on_a_galaxy import run_sdss, run_panstarrs
from sdss import create_sdss_csv, SDSS_BANDS_IN_ORDER
from panstarrs import create_panstarrs_csv, PANSTARRS_BANDS_IN_ORDER


path_to_input = "C:\\Users\\school\\Desktop\\github\\spin-parity-catalog\\original\\galaxies\\" #PANSTARRS
#path_to_input = "C:\\Users\\school\\Desktop\\cross_id\\sdss_mosaic_construction" #SDSS
folder_name = "table5"

path_to_output = "C:\\Users\\school\\Desktop\\gofher_output_refactor"
#output_folder_name = "gofher_SDSS_mosaic_bounds_test"
output_folder_name = 'gofher_panstarrs_bounds_test'

visualize_gals=False
make_csv=True

def get_fits_path(name,band):
    """the file path of where existing fits files can be found"""
    return os.path.join(path_to_input,folder_name,name,"{}_{}.fits".format(name,band))

def get_galaxy_list():
    """the list of galaxy names to run on"""
    return os.listdir(os.path.join(path_to_input,folder_name))

def get_save_vis_path(name):
    """the file path specifying where to save the visualizitaion"""
    return os.path.join(path_to_output,output_folder_name,folder_name,"{}.png".format(name))

def get_csv_out_path():
    """the file path specifying where to save the ouput csv"""
    return os.path.join(path_to_output,output_folder_name,"{}.csv".format(folder_name))

def get_color_image_path(name):
    return os.path.join(path_to_input,folder_name,name,"{}_color.jfif".format(name))

def get_dark_side_csv_path():
    csv_path = "C:\\Users\\school\\Desktop\\github\\spin-parity-catalog\\table_info\\csv_format_of_table\\"
    return os.path.join(csv_path,"table_{}.csv".format(folder_name.strip()[-1]))


if __name__ == "__main__":
    dark_side_labels = read_spin_parity_galaxies_label_from_csv(get_dark_side_csv_path())
    if folder_name == "table2" and "IC 2101" in dark_side_labels:
        dark_side_labels["IC2101"] = dark_side_labels["IC 2101"]

    the_gals = []
    i = 1
    for name in get_galaxy_list():
        print(i,name)
        i += 1
        try:
            save_vis_path = ''
            if visualize_gals: 
                save_vis_path=get_save_vis_path(name)

            paper_dark_side_label = dark_side_labels[name]
            the_gal = run_panstarrs(name, get_fits_path, save_vis_path=save_vis_path,dark_side_label=paper_dark_side_label,color_image_path=get_color_image_path(name))#PANSTARRS
            #the_gal = run_sdss(name, get_fits_path, save_vis_path=save_vis_path, dark_side_label='') #SDSS
            the_gals.append(the_gal)
        except:
            print("Error when running on",name)

    if make_csv:
        the_band_pairs = list(itertools.combinations(PANSTARRS_BANDS_IN_ORDER, 2)) #PANSTARRS
        #the_band_pairs = list(itertools.combinations(SDSS_BANDS_IN_ORDER, 2)) #SDSS
        create_panstarrs_csv(the_gals,the_band_pairs,get_csv_out_path())