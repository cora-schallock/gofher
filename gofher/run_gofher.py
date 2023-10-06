import itertools
import os

from galaxy import galaxy
from gofher import run_gofher_on_galaxy
from spin_parity import read_spin_parity_galaxies_label_from_csv, read_spin_parity_galaxies_label_from_csv_sdss
   
def new_run_gofher_on_galaxy(name, fits_path, bands_in_order, ref_bands_in_order, dark_side_label, save_path):
    the_gal = galaxy(name,dark_side_label)

    for band in bands_in_order:
        the_gal.construct_band(band,fits_path(name,band))

    for ref_band in ref_bands_in_order:
        if the_gal.has_valid_band(ref_band):
            the_gal.ref_band = ref_band 
            break

    if the_gal.ref_band == "":
        print("error: no valid ref band")
        return
    
    the_band_pairs = itertools.combinations(bands_in_order, 2)

    the_gal = run_gofher_on_galaxy(the_gal,the_band_pairs)
    for each_band_pair in the_gal.band_pairs:
        print(the_gal.band_pairs[each_band_pair].classification_score)
    

#for testing:
#path_to_input = "C:\\Users\\school\\Desktop\\github\\spin-parity-catalog\\original\\galaxies\\" #PANSTARRS
path_to_input = "C:\\Users\\school\\Desktop\\cross_id\\sdss_mosaic_construction" #SDSS
csv_path = "C:\\Users\\school\\Desktop\\github\\spin-parity-catalog\\table_info\\csv_format_of_table\\"
folder_name = "table2"
output_folder = "gofher_SDSS_mosaic_run_1"

#bands_in_order = ['g','r','i','z','y'] #PANSTARRS
#ref_bands_in_order = ['i','z','y','r','g']  #PANSTARRS

bands_in_order = ['u','g','r','i','z'] #SDSS
ref_bands_in_order = ['r','i','g','z','u'] #SDSS

def fits_path(name,band):
    return os.path.join(path_to_input,folder_name,name,"{}_{}.fits".format(name,band))

def get_galaxy_list():
    return os.listdir(os.path.join(path_to_input,folder_name))

def color_image_path(name):
    #return os.path.join(path_to_input,folder_name,name,"{}_color.jfif".format(name))
    return os.path.join(path_to_input,folder_name,name,"{}_color.png".format(name))

def get_csv_path():
    return os.path.join(csv_path,"table_{}.csv".format(folder_name.strip()[-1]))

def get_cross_path():
    return os.path.join("C:\\Users\\school\\Desktop\\cross_id","{}.txt".format(folder_name)) #SDSS

def get_save_hist_path(name):
    return "C:\\Users\\school\\Desktop\\gofher_output\\{}\\{}\\{}.png".format(output_folder,folder_name,name)

def get_csv_out_path():
    return "C:\\Users\\school\\Desktop\\gofher_output\\{}\\{}.csv".format(output_folder,folder_name)

if __name__ == "__main__":
    dark_side_labels, cross_id = read_spin_parity_galaxies_label_from_csv_sdss(get_csv_path(),get_cross_path())
    dark_side_labels = read_spin_parity_galaxies_label_from_csv(get_csv_path())
    dark_side_labels['IC2101'] = dark_side_labels['IC 2101']
    #the_band_pairs, the_csv_cols = get_csv_cols(bands_in_order)
    the_csv_rows = []
    i = 1
    for name in get_galaxy_list():
    #print(cross_id)
    #for name in ['1237668623014952982']:
        #try:
        if True:
            print(i,name)
            #if name not in cross_id: continue
            dark_label_for_gal = dark_side_labels[name] #SDSS #dark_side_labels[name]
            #print(cross_id[name])
            #csv_row = run_gofher_on_galaxy(name, fits_path, bands_in_order, ref_bands_in_order, dark_label_for_gal, save_path=get_save_hist_path(name))
            #try:
            if True:
                new_run_gofher_on_galaxy(name, fits_path, bands_in_order, ref_bands_in_order, dark_label_for_gal, save_path=get_save_hist_path(name))
                break
            #except:
            #     pass
            #csv_row = run_gofher_on_galaxy(name, fits_path, bands_in_order, ref_bands_in_order, dark_label_for_gal, save_path=get_save_hist_path(cross_id[name]))
            #csv_row[0] = cross_id[name]
            ##the_csv_rows.append(csv_row)
            i += 1
            #break
        #except Exception as e:
            #print(e)
            #break
    #if True:
    #    write_csv(get_csv_out_path(),the_csv_cols,the_csv_rows)