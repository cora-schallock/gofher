import os
import scipy
import numpy as np
import itertools
from skimage import measure
import matplotlib.pyplot as plt
from scipy.stats import expon, norm

from galaxy import galaxy
#from run_sersic import run_custom_sersic
from sep_helper import run_sep
from gofher import extract_data_for_histogram
from spin_parity import classify_spin_parity
from visualize import visualize_hist, get_key
from spin_parity import read_spin_parity_galaxies_label_from_csv
from run_fit import calculate_dist
from run_sersic import fit_sersic, evaluate_sersic_model
from mask import create_ellipse_mask
from fits import view_fits
from file_helper import prepare_csv_row, get_csv_cols, write_csv

def find_mask_spot_closest_to_center(the_mask,approx_center):
    shape=the_mask.shape
    all_labels = measure.label(the_mask) #https://scipy-lectures.org/packages/scikit-image/auto_examples/plot_labels.html
    blobs_labels = measure.label(all_labels, background=0)
    
    
    
    unique = np.unique(blobs_labels.flatten()) #https://stackoverflow.com/a/28663910/13544635
    #print(unique)
    center_of_masses = scipy.ndimage.center_of_mass(np.ones(shape),blobs_labels, index=unique)
    dist_to_center = list(map(lambda i: np.inf if i == 0 else calculate_dist(center_of_masses[i],approx_center),range(len(center_of_masses))))

    #plt.imshow(blobs_labels,origin='lower',vmin=0,vmax=4)
    #plt.colorbar()
    #plt.show()

    #print(dist_to_center)
    
    return blobs_labels==unique[np.argmin(dist_to_center)]

def run_gofher_on_galaxy(name, fits_path, bands_in_order, ref_bands_in_order, dark_side_label, save_path):
    the_gal = galaxy(name)

    for band in bands_in_order:
        the_gal.load_data(band,fits_path(name,band))

    for ref_band in ref_bands_in_order:
        if the_gal.is_ref_band_valid(ref_band):
            the_gal.ref_band = ref_band 
            break

    if the_gal.ref_band == "":
        print("error: no valid ref band")
        return
    
    data = the_gal.data[ref_band]
    shape = the_gal.data[ref_band].shape
    (cm_x, cm_y) = (shape[1]*0.5, shape[0]*0.5)
    the_el_sep, mu_bkg = run_sep(data, cm_x, cm_y)
    
    el_mask = create_ellipse_mask(the_el_sep['x'],the_el_sep['y'],the_el_sep['a'],the_el_sep['b'],the_el_sep['theta'],r=1.0,shape=shape)
    #view_fits(el_mask)

    inside_ellipse = data[np.logical_and(el_mask,the_gal.valid_pixel_mask[the_gal.ref_band])].flatten()
    loc, scale = expon.fit(inside_ellipse) #https://stackoverflow.com/questions/25085200/scipy-stats-expon-fit-with-no-location-parameter
    pdf_in = expon.pdf(data, loc=loc, scale=scale)
    #view_fits(pdf_in)

    outside_ellipse = data[np.logical_and(np.logical_not(el_mask),the_gal.valid_pixel_mask[the_gal.ref_band])].flatten()
    loc, scale = expon.fit(outside_ellipse) #https://stackoverflow.com/questions/25085200/scipy-stats-expon-fit-with-no-location-parameter
    pdf_out = expon.pdf(data, loc=loc, scale=scale)
    #view_fits(pdf_out)

    the_mask = pdf_out < pdf_in
    center_mask = find_mask_spot_closest_to_center(the_mask,(cm_x, cm_y))
    bright_spot_mask = np.logical_and(the_mask,np.logical_not(center_mask))

    center_mask = np.logical_and(center_mask,the_gal.valid_pixel_mask[the_gal.ref_band])

    sersic_model = fit_sersic(data, the_el_sep['b']*0.5, the_el_sep['x'],the_el_sep['y'],the_el_sep['a'],the_el_sep['b'],the_el_sep['theta'], center_mask,center_buffer=8,theta_buffer=np.pi/16)
    #print(the_el_sep['x'],the_el_sep['y'],the_el_sep['a'],the_el_sep['b'],the_el_sep['theta'])
    #print(sersic_model)
    #view_fits(evaluate_sersic_model(sersic_model,shape))
    eval_fit = data-evaluate_sersic_model(sersic_model,shape)
    eval_fit[bright_spot_mask] = 0
    #view_fits(eval_fit,std_range=3)


    #set galaxy parameters:
    the_gal.x = getattr(sersic_model,'x_0').value
    the_gal.y = getattr(sersic_model,'y_0').value
    the_gal.theta = getattr(sersic_model,'theta').value
    the_gal.a = the_el_sep['a']
    the_gal.b = the_el_sep['b']

    el_mask = the_gal.create_ellipse(r=1.0)
    pos_mask, neg_mask = the_gal.create_bisection()

    pos_side_diff_dict = dict()
    neg_side_diff_dict = dict()
    the_band_pairs = []
    for (first_band,base_band) in itertools.combinations(bands_in_order, 2):
        if the_gal.is_band_pair_valid(first_band,base_band):
            band_pair_key = get_key(first_band,base_band)
            the_band_pairs.append(band_pair_key)

            diff_image, mask = the_gal.create_diff_image(first_band,base_band,el_mask)

            #view_fits(diff_image)
            #view_fits(pos_mask)
            #view_fits(neg_mask)
            #view_fits(el_mask)
            pos_side_diffs, neg_side_diffs = extract_data_for_histogram(diff_image,pos_mask,neg_mask,el_mask)
            pos_side_diff_dict[band_pair_key] = pos_side_diffs
            neg_side_diff_dict[band_pair_key] = neg_side_diffs

    mean_diff_dict, the_label_dict, the_score_dict, pl, nl = classify_spin_parity(the_gal,dark_side_label,pos_side_diff_dict,neg_side_diff_dict)
    #view_fits(el_mask)
    visualize_hist(the_gal, el_mask, pos_mask, neg_mask, pl, nl,
                   pos_side_diff_dict, neg_side_diff_dict, 
                   mean_diff_dict, the_label_dict, the_score_dict, 
                   bands_in_order, dark_side_label, color_image_path, save_path=save_path)
    
    return prepare_csv_row(the_gal,dark_side_label,the_band_pairs, mean_diff_dict, the_score_dict, pl, nl, the_label_dict)
   


#for testing:
path_to_input = "C:\\Users\\school\\Desktop\\github\\spin-parity-catalog\\original\\galaxies\\"
csv_path = "C:\\Users\\school\\Desktop\\github\\spin-parity-catalog\\table_info\\csv_format_of_table\\"
folder_name = "table3"

bands_in_order = ['g','r','i','z','y']
ref_bands_in_order = ['i','z','y','r','g']

def fits_path(name,band):
    return os.path.join(path_to_input,folder_name,name,"{}_{}.fits".format(name,band))

def get_galaxy_list():
    return os.listdir(os.path.join(path_to_input,folder_name))

def color_image_path(name):
    return os.path.join(path_to_input,folder_name,name,"{}_color.jfif".format(name))

def get_csv_path():
    return os.path.join(csv_path,"table_{}.csv".format(folder_name.strip()[-1]))

def get_save_hist_path(name):
    return "C:\\Users\\school\\Desktop\\gofher_test_0\\{}\\{}.png".format(folder_name,name)

def get_csv_out_path():
    return "C:\\Users\\school\\Desktop\\gofher_test_0\\{}.csv".format(folder_name)


if __name__ == "__main__":
    dark_side_labels = read_spin_parity_galaxies_label_from_csv(get_csv_path())
    the_band_pairs, the_csv_cols = get_csv_cols(bands_in_order)
    the_csv_rows = []
    i = 1 
    for name in get_galaxy_list():
        try:
            print(i,name)
            csv_row = run_gofher_on_galaxy(name, fits_path, bands_in_order, ref_bands_in_order, dark_side_labels[name], save_path=get_save_hist_path(name))
            the_csv_rows.append(csv_row)
            i += 1
        except Exception as e:
            print(e)

    if True:
        write_csv(get_csv_out_path(),the_csv_cols,the_csv_rows)