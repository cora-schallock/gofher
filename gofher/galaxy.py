import numpy as np

from fits import read_fits, is_fits_path_valid
from mask import create_valid_pixel_mask, create_bisection_mask, create_ellipse_mask
from gofher import create_diff_image


class galaxy:
    def __init__(self,name):
        self.name = name
        self.data = {}
        self.valid_pixel_mask = {}

        self.ref_band = ""
        self.x = 0
        self.y = 0
        self.a = 0
        self.b = 0
        self.theta = 0

    def load_data(self,band,fits_path):
        if is_fits_path_valid(fits_path):
            self.data[band] = read_fits(fits_path)
            self.valid_pixel_mask[band]= create_valid_pixel_mask(self.data[band])

    def create_bisection(self,bisection_theta=None):
        if bisection_theta is None:
            bisection_theta = self.theta
        return create_bisection_mask(self.x,self.y,bisection_theta,self.data[self.ref_band].shape)
    
    def create_ellipse(self,ellipse_theta=None, r=1.0):
        if ellipse_theta is None:
            ellipse_theta = self.theta
        return create_ellipse_mask(self.x,self.y,self.a,self.b,ellipse_theta,r=r,shape=self.data[self.ref_band].shape)
    
    def create_diff_image(self,first_band,base_band,area_to_diff):
        ##print(self.data[self.ref_band].shape,self.valid_pixel_mask[first_band].shape,self.valid_pixel_mask[base_band].shape,area_to_diff.shape)
        to_diff_mask = np.logical_and(area_to_diff,np.logical_and(self.valid_pixel_mask[first_band],self.valid_pixel_mask[base_band]))
        diff_image = create_diff_image(self.data[first_band],self.data[base_band],to_diff_mask)
        return diff_image, to_diff_mask

    def is_band_pair_valid(self,first_band,base_band):
        if first_band not in self.data or  base_band not in self.data: return False #check has data
        if not isinstance(self.data[first_band], np.ndarray) or not isinstance(self.data[base_band], np.ndarray): return False #check data has right type
        if self.data[first_band].shape != self.data[base_band].shape: return False #check data has right size

        if first_band not in self.valid_pixel_mask or base_band not in self.valid_pixel_mask: return False #check has valid pixel mask
        if not isinstance(self.valid_pixel_mask[first_band], np.ndarray) or not isinstance(self.valid_pixel_mask[base_band], np.ndarray): return False #check valid pixel mask has right type
        if self.valid_pixel_mask[first_band].shape != self.valid_pixel_mask[base_band].shape: return False #check valid pixel mask has right size
        return True
    
    def is_ref_band_valid(self,ref_band):
        return ref_band in self.data and ref_band in self.valid_pixel_mask
    
    