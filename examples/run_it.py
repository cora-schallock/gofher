import os
import sys
sys.path.insert(0, '../gofher')

import itertools

from galaxy import galaxy
from spin_parity import read_spin_parity_galaxies_label_from_csv
from sparcfire import read_sparcfire_galaxy_csv, get_ref_band_and_gofher_params
from panstarrs import visualize_panstarrs, create_panstarrs_csv, PANSTARRS_REF_BANDS_IN_ORDER, PANSTARRS_BANDS_IN_ORDER
from sdss import visualize_sdss, create_sdss_csv, SDSS_REF_BANDS_IN_ORDER, SDSS_BANDS_IN_ORDER
#from run_gofher_on_a_galaxy import run_panstarrs_with_sparcfire_center_only, run_panstarrs_on_galaxy_with_sparcfire_center_inital_guess
#from run_gofher_on_a_galaxy import run_gofher_on_galaxy_with_sparcfire_center_inital_guess
from file_helper import check_if_folder_exists_and_create

use_panstarrs = True

visualize_gals=False
make_csv=False

use_fixed_center = False

if use_panstarrs:
    #path_to_input = "C:\\Users\\school\\Desktop\\github\\spin-parity-catalog\\original\\galaxies\\" #PANSTARRS
    path_to_input = "/Users/cora-at-work/Desktop/github/spin-parity-catalog/original/galaxies"
else:
    path_to_input = "C:\\Users\\school\\Desktop\\cross_id\\sdss_mosaic_construction" #SDSS

if use_panstarrs:
    if use_fixed_center:
        output_folder_name = 'tgofher_panstarrs_sparcfire_center_only_test' #PANSTARRS
    else:
        output_folder_name = 'tgofher_panstarrs_sparcfire_inital_guess' #PANSTARRS
else:
    if use_fixed_center:
        output_folder_name = 'tgofher_sdss_sparcfire_center_only_test' #SDSS
    else:
        output_folder_name = 'tgofher_sdss_sparcfire_inital_guess' #SDSS

folder_name = "table2"
#path_to_output = "C:\\Users\\school\\Desktop\\gofher_output_refactor"
path_to_output = "/Users/cora-at-work/Desktop"

def get_fits_path2(name,band):
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
    #csv_path = "C:\\Users\\school\\Desktop\\github\\spin-parity-catalog\\table_info\\csv_format_of_table\\"
    csv_path = "/Users/cora-at-work/Desktop/github/spin-parity-catalog/table_info/csv_format_of_table"
    return os.path.join(csv_path,"table_{}.csv".format(folder_name.strip()[-1]))

def setup_output_directories():
    output_path = os.path.join(path_to_output, output_folder_name)
    check_if_folder_exists_and_create(output_path)

    output_folder_path = os.path.join(path_to_output, output_folder_name, folder_name)
    check_if_folder_exists_and_create(output_folder_path)

def run_with_sparcfire_center_only(name, fits_path, sparcfire_bands, save_vis_path='', dark_side_label='', color_image_path=''):
    """run gofher on a single sdss galaxy"""
    the_gal = galaxy(name,dark_side_label)

    bands_in_oder = PANSTARRS_BANDS_IN_ORDER if use_panstarrs else SDSS_BANDS_IN_ORDER
    ref_bands_in_order = PANSTARRS_REF_BANDS_IN_ORDER if use_panstarrs else SDSS_REF_BANDS_IN_ORDER

    for band in bands_in_oder:
        the_gal.construct_band(band,fits_path(name,band))

    the_ref_band, the_sparcfire_derived_params = get_ref_band_and_gofher_params(sparcfire_bands,ref_bands_in_order )
    if the_ref_band == None or the_sparcfire_derived_params == None: return

    the_gal.ref_band = the_ref_band
    the_band_pairs = list(itertools.combinations(bands_in_oder, 2))

    the_gal = run_gofher_on_galaxy_with_fixed_center_only(the_gal,the_band_pairs,the_sparcfire_derived_params)
    if save_vis_path != '':
        if use_panstarrs:
            visualize_panstarrs(the_gal,save_vis_path,color_image_path)
        else:
            visualize_sdss(the_gal,save_vis_path)
    return the_gal

def run_with_sparcfire_center_inital_guess(name, fits_path, sparcfire_bands, save_vis_path='', dark_side_label='', color_image_path=''):
    """run gofher on a single sdss galaxy"""
    the_gal = galaxy(name,dark_side_label)

    bands_in_oder = PANSTARRS_BANDS_IN_ORDER if use_panstarrs else SDSS_BANDS_IN_ORDER
    ref_bands_in_order = PANSTARRS_REF_BANDS_IN_ORDER if use_panstarrs else SDSS_REF_BANDS_IN_ORDER

    for band in bands_in_oder:
        the_gal.construct_band(band,fits_path(name,band))

    the_ref_band, the_sparcfire_derived_params = get_ref_band_and_gofher_params(sparcfire_bands,ref_bands_in_order )
    if the_ref_band == None or the_sparcfire_derived_params == None: return

    the_gal.ref_band = the_ref_band
    the_band_pairs = list(itertools.combinations(bands_in_oder, 2))

    the_gal = run_gofher_on_galaxy_with_sparcfire_center_inital_guess(the_gal,the_band_pairs,the_sparcfire_derived_params)
    if save_vis_path != '':
        if use_panstarrs:
            visualize_panstarrs(the_gal,save_vis_path,color_image_path)
        else:
            visualize_sdss(the_gal,save_vis_path)
    return the_gal

if use_panstarrs:
    #path_to_sparcfire_csv = "C:\\Users\\school\\Desktop\\github\\spin-parity-catalog\\original\\output_from_running\\SpArcFiRe\\{}\\galaxy.csv".format(folder_name) #PANSTARRS
    path_to_sparcfire_csv = "/Users/cora-at-work/Desktop/github/spin-parity-catalog/original/output_from_running/SpArcFiRe/{}/galaxy.csv".format(folder_name) #PANSTARRS
else:
    path_to_sparcfire_csv = "C:\\Users\\school\\Desktop\\cross_id\\sdss_mosaic_construction\\SpArcFiRe_output\\{}\\G.out\\galaxy.csv".format(folder_name) #SDSS
sparcfire_galaxy_csv = read_sparcfire_galaxy_csv(path_to_sparcfire_csv)

dark_side_labels = read_spin_parity_galaxies_label_from_csv(get_dark_side_csv_path())
if folder_name == "table2" and "IC 2101" in dark_side_labels:
    dark_side_labels["IC2101"] = dark_side_labels["IC 2101"]

setup_output_directories()

the_gals = []
i = 1
for name in sparcfire_galaxy_csv:
    print(i,name)
    i += 1
    if True:
        sparcfire_bands = sparcfire_galaxy_csv[name]

        save_vis_path = ''
        if visualize_gals:
            save_vis_path=get_save_vis_path(name)

        paper_dark_side_label = dark_side_labels[name]

        if use_fixed_center:
            the_gal = run_with_sparcfire_center_only(name, get_fits_path2, sparcfire_bands, save_vis_path=save_vis_path,dark_side_label=paper_dark_side_label,color_image_path=get_color_image_path(name))
            break
        else:
            the_gal = run_with_sparcfire_center_inital_guess(name, get_fits_path2, sparcfire_bands, save_vis_path=save_vis_path,dark_side_label=paper_dark_side_label,color_image_path=get_color_image_path(name))
        the_gals.append(the_gal)
        break
    else:
        print("Error when running on",name)
    break

if make_csv:
    if use_panstarrs:
        the_band_pairs = list(itertools.combinations(PANSTARRS_BANDS_IN_ORDER, 2)) #PANSTARRS
        create_panstarrs_csv(the_gals,the_band_pairs,get_csv_out_path())
    else:
        the_band_pairs = list(itertools.combinations(SDSS_BANDS_IN_ORDER, 2)) #SDSS
        create_sdss_csv(the_gals,the_band_pairs,get_csv_out_path())