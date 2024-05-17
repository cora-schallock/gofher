import os
import itertools

from spin_parity import read_spin_parity_galaxies_label_from_csv
from run_gofher_on_a_galaxy import run_sdss, run_panstarrs
from sdss import create_sdss_csv, SDSS_BANDS_IN_ORDER
from panstarrs import create_panstarrs_csv, PANSTARRS_BANDS_IN_ORDER
from file_helper import check_if_folder_exists_and_create, write_tsv, write_csv
from disparate_sides import run_most_disparate_side_script_on_galaxy, run_ebm

use_panstarrs = True

visualize_gals=False
make_csv=False
get_disparate_sides = True

if use_panstarrs:
    ##path_to_input = "C:\\Users\\school\\Desktop\\github\\spin-parity-catalog\\original\\galaxies\\" #PANSTARRS
    path_to_input = "C:\\Users\\school\\Desktop\\github\\spin-parity-catalog-data\\panstarrs"
else:
    path_to_input = "C:\\Users\\school\\Desktop\\cross_id\\sdss_mosaic_construction" #SDSS

if use_panstarrs:
    #output_folder_name = "gofher_panstarrs_bounds_test" #PANSTARRS
    #output_folder_name = "gofher_panstarrs_sans_g" #PANSTARRS
    output_folder_name = "source_extraction"
else:
    #output_folder_name = "gofher_SDSS_mosaic_bounds_test" #SDSS
    output_folder_name = "gofher_sdss_sans_u" #SDSS

#folder_name = "table3"
folder_name = "figure8"
#path_to_output = "C:\\Users\\school\\Desktop\\gofher_output_refactor"
path_to_output = "E:\\grad_school\\research\\spin_parity_panstarrs"

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
    pa = "2" #folder_name.strip()[-1]
    return os.path.join(csv_path,"table_{}.csv".format(pa))

def setup_output_directories():
    output_path = os.path.join(path_to_output, output_folder_name)
    check_if_folder_exists_and_create(output_path)

    output_folder_path = os.path.join(path_to_output, output_folder_name, folder_name)
    check_if_folder_exists_and_create(output_folder_path)

def output_normed_pixels_table_for_ebm(the_gal, dir):
    import numpy as np
    from gofher import normalize_array
    import copy
    import matplotlib.pyplot as plt
    from file_helper import write_csv


    pos_mask, neg_mask = the_gal.create_bisection()
    el_mask = the_gal.create_ellipse()
    normed_dict = dict()

    for band in the_gal.bands:
        #pos_mask = np.logical_and(the_gal[band].valid_pixel_mask,pos_mask)
        el_mask = np.logical_and(el_mask, copy.deepcopy(the_gal[band].valid_pixel_mask))
        #neg_mask = np.logical_and(neg_mask, copy.deepcopy(the_gal[band].valid_pixel_mask))
        #neg_mask = np.logical_and(the_gal[band].valid_pixel_mask,neg_mask)
    for band in the_gal.bands:
        normed_dict[band] = normalize_array(copy.deepcopy(the_gal[band].data),np.logical_and(el_mask,the_gal[band].valid_pixel_mask))

    #print(normed_dict)
    sa = np.Inf
    rows_to_write = []
    for i in range(el_mask.shape[0]):
        for j in range(el_mask.shape[1]):
            if el_mask[i,j] <= 0:
                continue

            side = None
            if pos_mask[i,j] > 0:
                side = 1
            elif neg_mask[i,j] > 0:
                side = 0
            else:
                break
            #if normed_dict['r'][i,j] > 0:
            #    print(normed_dict['r'][i,j])

            if side == None:
                continue
            #print(normed_dict['r'][i,j])

            the_row = []
            for band in PANSTARRS_BANDS_IN_ORDER:
                if band not in normed_dict:
                    the_row.append("None")
                else:
                    the_row.append(normed_dict[band][i,j])
            the_row.append(side)

            if min(the_row[:-1]) <= 0.0:
                continue

            sa = np.Inf
            for i in range(len(the_row)):
                for j in range(i+1,len(the_row)):
                    di = the_row[i]-the_row[j]
                    sa = min(sa,abs(di))
            
            rows_to_write.append(the_row)
    print(sa)

    pa = "E:\\grad_school\\research\\spin_parity_panstarrs\\paper_writing\\normed_fits_output\\{}\\{}.csv".format(dir,the_gal.name)
    write_csv(pa,PANSTARRS_BANDS_IN_ORDER+['side'],rows_to_write)

    pa = "E:\\grad_school\\research\\spin_parity_panstarrs\\paper_writing\\normed_fits_output\\{}\\{}.tsv".format(dir,the_gal.name)
    write_tsv(pa,PANSTARRS_BANDS_IN_ORDER+['side'],rows_to_write)

def make_diff_image_example_for_paper(the_gal):
    #https://walmsley.dev/posts/typesetting-mnras-figures
    #https://support.microsoft.com/en-us/office/add-a-font-b7c5f17c-4426-4b53-967f-455339c564c1
    #error - font not appearing: https://stackoverflow.com/a/26106170
    ## delet file at: /mnt/c/Users/school/.matplotlib

    import matplotlib.pyplot as plt
    from gofher import normalize_array
    from visualize import create_color_map_class
    import numpy as np

    SMALL_SIZE = 9
    MEDIUM_SIZE = 9
    BIGGER_SIZE = 9

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)
    plt.rc('font', family='Nimbus Roman No9 L')

    fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(10/3, 4))  # height can be anything
    diff = the_gal.get_band_pair('r-y').diff_image

    data = the_gal['r'].data - the_gal['y'].data
    to_diff_mask = the_gal.create_ellipse()
    mask_out_it = np.logical_not(to_diff_mask)
    
    ##bad_norm_diff = data * to_diff_mask##normalize_array(data,to_diff_mask)
    bad_norm_diff = normalize_array(data,to_diff_mask)

    diff[mask_out_it] = -np.Inf #to make background same color
    bad_norm_diff[mask_out_it] = -np.Inf #to make background same color
    #diff = bad_norm_diff

    shape = diff.shape
    cent = int(shape[0]/2),int(shape[1]/2)
    size = int(shape[0]/5),int(shape[1]/5)
    top_left = cent[0] - size[0]
    bottom_right = cent[0] + size[0]

    the_mask = the_gal[the_gal.ref_band].valid_pixel_mask
    pos_mask,neg_mask = the_gal.create_bisection()
    #m, s = np.mean(data[the_mask]), np.std(data[the_mask])
    
    cmap = create_color_map_class(pos_mask,neg_mask,np.logical_and(to_diff_mask,the_mask))

    import matplotlib.image as mpimg
    #pa_color = "C:\\Users\\school\\Desktop\\github\\spin-parity-catalog\\galaxies\\panstarrs\\{}\\{}\\{}_color.jfif".format(folder_name,the_gal.name,the_gal.name)
    pa_color = "C:\\Users\\school\\Desktop\\github\\spin-parity-catalog-data\\panstarrs\\figure9\\NGC3367\\NGC3367_color.jfif"
    image = mpimg.imread(pa_color)
    
    #ax[0].set_title('(a) color:')
    #ax[0].imshow(image)
    #ax[0].set_axis_off()

    ax[0,0].set_title('(a) color:')
    ax[0,0].imshow(image)
    ax[0,0].set_axis_off()
    
    ax[0,1].set_title('(b) reference band:')
    ref = the_gal['i'].data
    m, s = np.mean(ref[to_diff_mask]), np.std(ref[to_diff_mask])

    ax[0,1].imshow(ref[top_left:bottom_right,top_left:bottom_right],origin='lower',cmap='plasma')
    ax[0,1].imshow(cmap[top_left:bottom_right,top_left:bottom_right], origin= 'lower',alpha=0.5)
    ax[0,1].set_axis_off()

    
    ax[1,0].set_title('(c) difference:')
    im2 = ax[1,0].imshow(diff[top_left:bottom_right,top_left:bottom_right],origin='lower',cmap='plasma')
    ax[1,0].set_axis_off()

    ax[1,1].set_title('(d) r-y:')
    data[mask_out_it] = -np.Inf #to make background same color
    im2 = ax[1,1].imshow(data[top_left:bottom_right,top_left:bottom_right],origin='lower',cmap='plasma')
    ax[1,1].set_axis_off()
    fig.tight_layout()

    
    pa = "E:\\grad_school\\research\\spin_parity_panstarrs\\paper_writing\\figure_maker\\{}_alt_norm_diff.png".format(the_gal.name)
    fig.savefig(pa, dpi=300)

    
    pa = "E:\\grad_school\\research\\spin_parity_panstarrs\\paper_writing\\figure_maker\\{}_histogram.png".format(the_gal.name)
    fig.savefig(pa, dpi=300)

def disparate_csv(the_gal,disparte_info,dark_side_label,output_csv_path):
    from collections import defaultdict

    count_dict = defaultdict(int)

    csv_col = ["name","dark_side","majority_label","ebm_label","ebm_p_val"]
    csv_rows = [the_gal.name,dark_side_label, '', disparte_info[0],disparte_info[2][disparte_info[0]]]
    for band_pair in disparte_info[-1]:
        csv_col.extend([band_pair,"{}_p_val".format(band_pair)])
        csv_rows.extend(disparte_info[-1][band_pair])
        count_dict[disparte_info[-1][band_pair][0]] += 1

    count_dict = dict(count_dict)

    if len(count_dict) == 1:
        csv_rows[2] = list(count_dict.keys())[0]
    elif count_dict[list(count_dict.keys())[0]] > count_dict[list(count_dict.keys())[1]]:
        csv_rows[2] = list(count_dict.keys())[0]
    elif count_dict[list(count_dict.keys())[0]] < count_dict[list(count_dict.keys())[1]]:
        csv_rows[2] = list(count_dict.keys())[1]
    
    write_csv(output_csv_path,csv_col,[csv_rows])


if __name__ == "__main__":
    dark_side_labels = read_spin_parity_galaxies_label_from_csv(get_dark_side_csv_path())
    if folder_name == "figure8" and "IC2101" in dark_side_labels:
        dark_side_labels["IC 2101"] = dark_side_labels["IC2101"]

    setup_output_directories()

    #diff_gals = ['NGC1','NGC278', 'NGC2207', 'NGC3887', 'NGC4536', 'NGC5135', 'UGC12274', 'NGC450']
    #test_gals = ['NGC4527'] #1/11
    #test_gals = ['IC1199'] 1/5

    # NGC598: 6000 #too large
    new_figure_8_gals = """NGC5236: 2000"""
    new_figure_9_gals = """NGC1232: 1200
    NGC2553: 800
    NGC3344: 800
    NGC3346: 800
    NGC3351: 800
    NGC3359: 1000
    NGC3367: 600
    NGC3381: 600
    NGC3395: 400
    NGC3423: 800
    NGC3445: 400
    PGC39728: 200
    PGC46767: 400
    PGC49906: 200"""

    #the_gals = ["NGC3367"]

    #test_gals = list(map(lambda x: x.strip().split(':')[0],new_figure_9_gals.strip().split('\n')))

    the_gals = []
    i = 1
    for name in get_galaxy_list():
        print(i,name)
        i += 1
        try:
            save_vis_path = ''
            if visualize_gals: 
                save_vis_path=get_save_vis_path(name)

            if name in dark_side_labels:
                paper_dark_side_label = dark_side_labels[name]
            else:
                continue
            
            #ds = "C:\\Users\\school\\Desktop\\disparate_side"
            #output_csv_path = os.path.join(ds,folder_name,"{}.csv".format(name))
            #if os.path.isfile(output_csv_path): continue

            #else:
            #    paper_dark_side_label = "placeholder"
            if use_panstarrs:
                the_gal = run_panstarrs(name, get_fits_path, save_vis_path=save_vis_path,dark_side_label=paper_dark_side_label,color_image_path=get_color_image_path(name)) #PANSTARRS

                if get_disparate_sides:
                    positive_one_ebm, negative_one_ebm = run_ebm(the_gal)

                    if positive_one_ebm == min(positive_one_ebm,negative_one_ebm):
                        print(name,the_gal.pos_side_label,positive_one_ebm)
                    else:
                        print(name,the_gal.neg_side_label,negative_one_ebm)
                    
                    #print(the_gal.pos_side_label,positive_one_ebm,the_gal.neg_side_label,negative_one_ebm)
                    #the_gal.disparate_sides_vote = run_most_disparate_side_script_on_galaxy(the_gal)
                    """
                    if the_gal.disparate_sides_vote == None: continue
                    
                    ds = "C:\\Users\\school\\Desktop\\disparate_side"
                    output_csv_path = os.path.join(ds,folder_name,"{}.csv".format(the_gal.name))

                    if not os.path.isfile(output_csv_path):
                        run_ebm(the_gal)
                        dsv_info = the_gal.disparate_sides_vote.get_info(the_gal.pos_side_label,the_gal.neg_side_label)
                        print(dsv_info)
                        disparate_csv(the_gal,dsv_info,paper_dark_side_label,output_csv_path)
                        break
                    """
                ###make_diff_image_example_for_paper(the_gal) #for paper
                ###output_normed_pixels_table_for_ebm(the_gal, folder_name) #for outputting normed pixels
                #break
        

                """
                for each_band_pair in the_gal.band_pairs:
                    import matplotlib.pyplot as plt
                    the_diff_image = the_gal.band_pairs[each_band_pair].diff_image

                    if name not in diff_gals: continue
                    plt.imshow(the_diff_image, origin = 'lower')
                    diff_image_for_paper_path = "E:\\grad_school\\research\\spin_parity_panstarrs\\diff_images_for_questionable_cases\\{}_{}_diff.png".format(the_gal.name, each_band_pair)
                    plt.savefig(diff_image_for_paper_path)
                """
            else:
                the_gal = run_sdss(name, get_fits_path, save_vis_path=save_vis_path, dark_side_label=paper_dark_side_label) #SDSS
            the_gals.append(the_gal)
            #break
        except Exception as e:
            print("Error when running on",name,e)
            #break
            #break
        #if i > 1:
        #    break
        #break

    if make_csv:
        if use_panstarrs:
            the_band_pairs = list(itertools.combinations(PANSTARRS_BANDS_IN_ORDER, 2)) #PANSTARRS
            create_panstarrs_csv(the_gals,the_band_pairs,get_csv_out_path())
        else:
            the_band_pairs = list(itertools.combinations(SDSS_BANDS_IN_ORDER, 2)) #SDSS
            create_sdss_csv(the_gals,the_band_pairs,get_csv_out_path())