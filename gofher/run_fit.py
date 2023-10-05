
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import color, morphology
from skimage import measure
from skimage.restoration import denoise_nl_means, estimate_sigma


from fits import view_fits
from mask import create_segmentation_masks
from gofher import normalize_array

def calculate_dist(cm,center):
    return np.linalg.norm(np.array(cm)-np.array(center))

def run_cutom_segmentation(data,use_denoise=True,debug=False):
    if debug:
        print("data")
        view_fits(data)
    #Step 1) Read data
    data = data.byteswap(True).newbyteorder()
    shape = data.shape
    
    #Step 2) Denoise data:
    if use_denoise:
        sigma_est = np.mean(estimate_sigma(data))
        patch_kw = dict(patch_size=5, patch_distance=6)
        data = denoise_nl_means(data, h=0.6 * sigma_est, sigma=sigma_est, fast_mode=True, **patch_kw)
    
    #Step 3) Segment Foreground/ Background:
    (foreground_mask,background_mask) = create_segmentation_masks(data)
    if debug:
        print("run_segmentation - 1: inital foreground/ background")
        view_fits(foreground_mask)
        view_fits(background_mask)
        
    
    #Step 4) Run sobel:
    im = data.astype('int32')
    dx = ndimage.sobel(im, 0)  # horizontal derivative
    dy = ndimage.sobel(im, 1)  # vertical derivative
    mag = np.hypot(dx, dy)  # magnitude
    
    normed_mag = normalize_array(mag,foreground_mask)
    if debug: 
        print("run_segmentation - 2: run sobel")
        view_fits(normed_mag)
    
    #Step 5) Look for areas of sobel that are of interest:
    m = np.mean(normed_mag[foreground_mask])
    areas_of_interest= normed_mag>m
    if debug: 
        print("run_segmentation - 3: areas of interest")
        view_fits(areas_of_interest)
    
    #Step 6) Remove small noise
    footprint = morphology.disk(1)
    small_noise = morphology.white_tophat(areas_of_interest, footprint)
    areas_of_interest = np.logical_and(areas_of_interest,np.logical_not(small_noise)).astype('float32') #as type for cv2 operations
    if debug:
        print("run_segmentation - 4: remove small noise from areas of interest")
        view_fits(areas_of_interest)
    
    #Step 7) Clean areas of interest:
    kernel = np.ones((3, 3), np.uint8)
    areas_of_interest = cv2.morphologyEx(areas_of_interest,cv2.MORPH_ERODE,kernel)
    areas_of_interest = cv2.morphologyEx(areas_of_interest, cv2.MORPH_CLOSE, kernel)
    if debug: 
        print("run_segmentation - 5: clean areas of interest")
        view_fits(areas_of_interest)
    
    #Step 8) Label areas of interest:
    all_labels = measure.label(areas_of_interest) #https://scipy-lectures.org/packages/scikit-image/auto_examples/plot_labels.html
    blobs_labels = measure.label(all_labels, background=0)
    if debug: 
        print("run_segmentation - 6: label areas of interest")
        colored_by_label = color.label2rgb(blobs_labels)
        fig, ax = plt.subplots() #https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D
        im = ax.imshow(colored_by_label, interpolation='nearest', origin='lower')
        plt.show()
    
    #Step 9) Locate expected center of bulge and 
    approx_center = (shape[0]/2,shape[1]/2)
    center_max_gap = shape[0]*0.125#*0.0625 #don't make too big or too small
    
    #Step 10) Find unique labels and number of times they occur
    unique, counts = np.unique(blobs_labels.flatten(), return_counts=True) #https://stackoverflow.com/a/28663910/13544635
    
    #Step 11) Locate Center of Mass for all labels, then only consider center_of_masses near approx_center
    #Note: over np.ones, should be equivalent to center of label 
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.center_of_mass.html
    center_of_masses = ndimage.center_of_mass(np.ones(shape),blobs_labels, index=unique)
    dist_to_center = list(map(lambda x: calculate_dist(x,approx_center),center_of_masses))
    near_center = list(filter(lambda i: dist_to_center[i] < center_max_gap, range(len(dist_to_center))))
    
    #Step 12) Locate label of bulge cluster (Must be in near_center):
    #Note: largest (by count) should be background, find buldge by traversing counts starting at 2nd biggest. 
    background_index = np.argsort(counts)[-1] #kinda from here: https://stackoverflow.com/a/54495793/13544635
    bulge_index = -1
    
    for size_index in list(reversed(np.argsort(counts)[0:-1])):
        if size_index in near_center:
            bulge_index = size_index
            break
            
    if bulge_index == -1: raise ValueError("Bulge cluster not found")
    
    #Step 13) Take labels and where the bulge is so seperate bulge from stars:
    bulge_label = unique[bulge_index]
    bulge_mask = (blobs_labels == bulge_label)
    star_mask = np.logical_and(np.clip(blobs_labels,0,1).astype(bool),np.logical_not(bulge_mask))
    
    if debug: 
        print("run_segmentation - 7: pick label")
        view_fits(bulge_mask)
        view_fits(star_mask)
    #if debug: print("star_mask",star_mask.shape)
    
    #Step 14) Fill holes in bulge mask and expand star mask (slightly)
    bulge_mask = ndimage.binary_fill_holes(bulge_mask)
    star_mask = ndimage.binary_dilation(star_mask, iterations=1)
    
    if debug: 
        print("run_segmentation - 8: cleaned picked label")
        view_fits(bulge_mask)
        view_fits(star_mask)
    
    return data, foreground_mask, background_mask, bulge_mask, star_mask