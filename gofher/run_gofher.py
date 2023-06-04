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
from fits import view_fits, read_fits
from file_helper import prepare_csv_row, get_csv_cols, write_csv

#for lecture:
from visualize import visualize_map, visualize_single_hist

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

from scipy import stats

def run_gamma_fit(pos_dict,neg_dict):
    gamma_dict = dict()
    for key in pos_dict:
        #print(pos_dict[key])
        #shape_pos, loc_pos, scale_pos = stats.gamma.fit(pos_dict[key])
        #shape_neg, loc_neg, scale_neg = stats.gamma.fit(neg_dict[key])
        #gamma_dict[key] = [shape_pos, loc_pos, scale_pos, shape_neg, loc_neg, scale_neg]

        pos_mean, pos_std = stats.norm.fit(pos_dict[key])
        neg_mean, neg_std = stats.norm.fit(neg_dict[key])
        gamma_dict[key] = [pos_mean, pos_std, neg_mean, neg_std]

    return gamma_dict

def run_gamma_pdf(gamma_dict,pos_dict,neg_dict,samples=64):
    pos_x_dict = dict()
    neg_x_dict = dict()
    pos_pdf_point_dict = dict()
    neg_pdf_point_dict = dict()
    for key in gamma_dict:
        if len(gamma_dict[key]) == 4:
            #[shape_pos, loc_pos, scale_pos, shape_neg, loc_neg, scale_neg] = gamma_dict[key]
            [pos_mean, pos_std, neg_mean, neg_std] = gamma_dict[key]

            the_min = min(np.min(pos_dict[key]),np.min(neg_dict[key]))
            the_max = max(np.max(pos_dict[key]),np.max(neg_dict[key]))

            pos_x=np.linspace(np.min(pos_dict[key]),np.max(pos_dict[key]),samples)
            neg_x=np.linspace(np.min(neg_dict[key]),np.max(neg_dict[key]),samples)
            pos_pdf = stats.norm.pdf(pos_x, pos_mean, pos_std)
            neg_pdf = stats.norm.pdf(neg_x, neg_mean, neg_std)
            #plt.plot(pos_x,pos_pdf/pos_pdf.sum())
            #plt.plot(neg_x,neg_pdf/neg_pdf.sum())
            #plt.show()
            #pos_pdf = stats.gamma.pdf(x=pos_x, a=shape_pos, loc=loc_pos, scale=scale_pos)
            #neg_pdf = stats.gamma.pdf(x=neg_x, a=shape_neg, loc=loc_neg, scale=scale_neg)

            pos_x_dict[key] = pos_x
            neg_x_dict[key] = neg_x
            pos_pdf_point_dict[key] = pos_pdf
            neg_pdf_point_dict[key] = neg_pdf

    return pos_x_dict, neg_x_dict, pos_pdf_point_dict, neg_pdf_point_dict


def run_ks_stats(pos_dict,neg_dict):
    p_value_dict = dict()
    for key in pos_dict:
        if key in neg_dict:
            stat, pvalue = stats.kstest(pos_dict[key], neg_dict[key])
            p_value_dict[key] = pvalue
    return p_value_dict

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
    #view_fits(data,std_range=3,save_path="C:\\Users\\school\\Desktop\\adv_lecture\\ngc3368.png")
    
    el_mask = create_ellipse_mask(the_el_sep['x'],the_el_sep['y'],the_el_sep['a'],the_el_sep['b'],the_el_sep['theta'],r=1.0,shape=shape)#r=1.0
    #view_fits(el_mask,save_path="C:\\Users\\school\\Desktop\\adv_lecture\\ngc3368_inital_foreground.png")
    #view_fits(np.logical_not(el_mask),save_path="C:\\Users\\school\\Desktop\\adv_lecture\\ngc3368_inital_background.png")

    inside_ellipse = data[np.logical_and(el_mask,the_gal.valid_pixel_mask[the_gal.ref_band])].flatten()
    loc, scale = expon.fit(inside_ellipse) #https://stackoverflow.com/questions/25085200/scipy-stats-expon-fit-with-no-location-parameter
    pdf_in = expon.pdf(data, loc=loc, scale=scale)
    #view_fits(pdf_in,save_path="C:\\Users\\school\\Desktop\\adv_lecture\\ngc3368_pdf_in.png")
    #view_fits(pdf_out)

    outside_ellipse = data[np.logical_and(np.logical_not(el_mask),the_gal.valid_pixel_mask[the_gal.ref_band])].flatten()
    loc, scale = expon.fit(outside_ellipse) #https://stackoverflow.com/questions/25085200/scipy-stats-expon-fit-with-no-location-parameter
    pdf_out = expon.pdf(data, loc=loc, scale=scale)
    #view_fits(pdf_out)

    the_mask = pdf_out < pdf_in
    #view_fits(pdf_in)
    #view_fits(pdf_out,save_path="C:\\Users\\school\\Desktop\\adv_lecture\\ngc3368_pdf_out.png")
    #view_fits(the_mask,save_path="C:\\Users\\school\\Desktop\\adv_lecture\\ngc3368_the_mask.png")
    center_mask = find_mask_spot_closest_to_center(the_mask,(cm_x, cm_y))
    bright_spot_mask = np.logical_and(the_mask,np.logical_not(center_mask))

    center_mask = np.logical_and(center_mask,the_gal.valid_pixel_mask[the_gal.ref_band])
    #view_fits(center_mask,save_path="C:\\Users\\school\\Desktop\\adv_lecture\\ngc3368_center_mask.png")

    sersic_model = fit_sersic(data, the_el_sep['b']*0.5, the_el_sep['x'],the_el_sep['y'],the_el_sep['a'],the_el_sep['b'],the_el_sep['theta'], center_mask,center_buffer=8,theta_buffer=np.pi/16)
    #print(the_el_sep['x'],the_el_sep['y'],the_el_sep['a'],the_el_sep['b'],the_el_sep['theta'])
    #print(sersic_model)
    view_fits(evaluate_sersic_model(sersic_model,shape),save_path="C:\\Users\\school\\Desktop\\adv_lecture\\ngc1_sersic.png")
    eval_fit = data-evaluate_sersic_model(sersic_model,shape)
    eval_fit[bright_spot_mask] = 0
    view_fits(eval_fit,std_range=3,save_path="C:\\Users\\school\\Desktop\\adv_lecture\\ngc1_sersic_res.png")


    #set galaxy parameters:
    the_gal.x = getattr(sersic_model,'x_0').value
    the_gal.y = getattr(sersic_model,'y_0').value
    the_gal.theta = getattr(sersic_model,'theta').value
    the_gal.a = the_el_sep['a']
    the_gal.b = the_el_sep['b']

    el_mask = the_gal.create_ellipse(r=1.0) #r=1.0
    pos_mask, neg_mask = the_gal.create_bisection()

    #view_fits(eval_fit,std_range=3,save_path="C:\\Users\\school\\Desktop\\adv_lecture\\ngc3368_sersic_res.png")
    #visualize_map(data=data,pos_mask=pos_mask,neg_mask=neg_mask,el_mask=el_mask,save_path="C:\\Users\\school\\Desktop\\adv_lecture\\ngc3368_masks.png")

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

            if first_band == "g" and base_band == "y":
                #view_fits(diff_image,std_range=3,save_path="C:\\Users\\school\\Desktop\\adv_lecture\\ngc3368_diff_fits.png")
                #visualize_single_hist(data,pos_mask,neg_mask,el_mask,diff_image)
                pass



    #fit gamma distro:
    gamma_dict = run_gamma_fit(pos_side_diff_dict,neg_side_diff_dict)
    pos_x_dict, neg_x_dict, pos_pdf_point_dict, neg_pdf_point_dict = run_gamma_pdf(gamma_dict,pos_side_diff_dict,neg_side_diff_dict)

    #run p-value:
    p_value_dict = run_ks_stats(pos_side_diff_dict,neg_side_diff_dict)

    #score:
    mean_diff_dict, the_label_dict, the_score_dict, pl, nl = classify_spin_parity(the_gal,dark_side_label,pos_side_diff_dict,neg_side_diff_dict)

    """
    use_max_p_value = False
    if use_max_p_value:
        min_p_value = 0.05
        for band_pair in p_value_dict:
            if p_value_dict[band_pair] > min_p_value:
                the_score_dict[band_pair] = 0
    """
    

    """
    visualize_hist(the_gal, el_mask, pos_mask, neg_mask, pl, nl,
                   pos_side_diff_dict, neg_side_diff_dict, 
                   mean_diff_dict, the_label_dict, the_score_dict,
                   pos_x_dict, neg_x_dict, pos_pdf_point_dict, neg_pdf_point_dict, 
                   bands_in_order, dark_side_label, color_image_path, save_path=save_path)
    """
    
    return prepare_csv_row(the_gal,dark_side_label,the_band_pairs, mean_diff_dict, p_value_dict, the_score_dict, pl, nl, the_label_dict)

   


#for testing:
path_to_input = "C:\\Users\\school\\Desktop\\github\\spin-parity-catalog\\original\\galaxies\\"
csv_path = "C:\\Users\\school\\Desktop\\github\\spin-parity-catalog\\table_info\\csv_format_of_table\\"
folder_name = "table3"
output_folder = "gofher_test_0_stats_gmean"

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
    return "C:\\Users\\school\\Desktop\\gofher_output\\{}\\{}\\{}.png".format(output_folder,folder_name,name)

def get_csv_out_path():
    return "C:\\Users\\school\\Desktop\\gofher_output\\{}\\{}.csv".format(output_folder,folder_name)

#print(read_fits("C:\\Users\\school\\Downloads\\NGC0001.CALIFA.V1200.stekin.fits"))

"""
from astropy.io import fits
hdu_list = fits.open("C:\\Users\\school\\Downloads\\NGC0001.CALIFA.V1200.stekin.fits")
print(hdu_list.info())
image_data = hdu_list[0].data
print(image_data)
print(image_data[0])
print(image_data.shape)
plt.imshow(image_data,origin='lower')
"""

if __name__ == "__main__":
    dark_side_labels = read_spin_parity_galaxies_label_from_csv(get_csv_path())
    the_band_pairs, the_csv_cols = get_csv_cols(bands_in_order)
    the_csv_rows = []
    i = 1
    #for name in get_galaxy_list():
    for name in ['NGC1']: #NGC3368
        #try:
        if True:
            print(i,name)
            csv_row = run_gofher_on_galaxy(name, fits_path, bands_in_order, ref_bands_in_order, dark_side_labels[name], save_path=get_save_hist_path(name))
            the_csv_rows.append(csv_row)
            i += 1
            #if i > 3:
            #    break
        #except Exception as e:
        #    print(e)

    if True:
        #write_csv(get_csv_out_path(),the_csv_cols,the_csv_rows)
        pass