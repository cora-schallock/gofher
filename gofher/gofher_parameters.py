import numpy as np

from utils import is_float_int, is_2d_array_shape

from mask import create_ellipse_mask, create_bisection_mask, create_near_major_axis_mask, create_near_minor_axis_mask

class GofherParameters:
    """Contains gofher ellipse parameters"""

    def __init__(self):
        self.name = ""
        self.ref_band = ""

        self.h = np.nan
        self.k = np.nan
        self.a = np.nan
        self.b = np.nan
        self.theta = np.nan

        self.sparcfire_input_c = np.nan
        self.sparcfire_input_r = np.nan
        self.sparcfire_maj_axis_len = np.nan
        self.sparcfire_min_axis_len = np.nan
        self.sparcfire_disk_maj_axs_angle = np.nan
        self.sparcfire_bulge_maj_axis_len = np.nan
        self.sparcfire_bulge_min_axis_len = np.nan

        self.sparcfire_bulge_disk_f = np.nan

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return f"GofherParameters('{self.name}')"

    def calculate_from_sparcfire(self, bulge_disk_f: float = 1.0):
        """Calculate gofher paramteres from sparcfire data
        
        Args:
            bulge_disk_f: a float in range [0,1] that specifies the scale
                of the semi major axis a. bulge_disk_f of 1.0 uses the
                disk's semi-major axis, bulge_disk_0 use the bulge's semi
                major axis. Fractions are calulated as follow:

                    a = bulge_disk_f * (disk-bulge) + bulge

                Important: b is chosen so that a/b = disk major/disk minor
                I.E. bulge_disk_f scales b as well but axis ratio is fixed.
        """

        if not is_float_int(self.sparcfire_input_c):
            raise ValueError("self.sparcfire_input_c is invalid")
        
        if not is_float_int(self.sparcfire_input_r):        
            raise ValueError("self.sparcfire_input_r is invalid")
        
        if not is_float_int(self.sparcfire_disk_maj_axs_angle):
            raise ValueError("self.sparcfire_disk_maj_axs_angle is invalid")

        if bulge_disk_f != 0.0 and not is_float_int(self.sparcfire_maj_axis_len):
            raise ValueError("self.sparcfire_maj_axis_len is invalid")
        
        if bulge_disk_f != 0.0 and not is_float_int(self.sparcfire_min_axis_len):
            raise ValueError("self.sparcfire_min_axis_len is invalid")
        
        if bulge_disk_f != 1.0 and not is_float_int(self.sparcfire_bulge_maj_axis_len):
            raise ValueError("self.sparcfire_bulge_maj_axis_len is invalid")
        
        if bulge_disk_f != 1.0 and not is_float_int(self.sparcfire_bulge_min_axis_len):
            raise ValueError("self.sparcfire_bulge_min_axis_len is invalid")
        
        if not is_float_int(bulge_disk_f) or bulge_disk_f < 0.0 or bulge_disk_f > 1.0:
            raise ValueError("bulge_disk_f must be float in range [0,1]")

        self.sparcfire_bulge_disk_f = bulge_disk_f

        self.h = self.sparcfire_input_c - 1.5
        self.k = self.sparcfire_input_r - 1.5
        self.theta = self.sparcfire_disk_maj_axs_angle * -1.0   

        diff = self.sparcfire_maj_axis_len - self.sparcfire_bulge_maj_axis_len
    
        self.a = self.sparcfire_bulge_maj_axis_len + diff*self.sparcfire_bulge_disk_f
        self.b = self.sparcfire_min_axis_len * (self.a/self.sparcfire_maj_axis_len)

        self.a *= 0.5
        self.b *= 0.5

    def create_ellipse_mask(self, shape: tuple[int], 
                            r: float = 1.0) -> np.ndarray:
        """Using the gofher parameters, create an ellipse mask.
        See: create_ellipse_mask in mask.py

        IMPORTANT: All gofher parameters must be set prior to calling
                   this function. If using sparcfire data, also call
                   calculate_from_sparcfire() first. 
        
        Args:
            shape: the shape of the array 
                (assumes 2D array)
            r: scaling factor of ellipse 
                (scales self.a and self.b by r)

        Returns:
            A boolean ellipse mask for all pixels in ellipse.
        """
        if not is_2d_array_shape(shape):
            raise ValueError("shape must be tuple containing exactly 2 ints & > 0")
        
        if not is_float_int(r) or r < 0.0:
            raise ValueError("r must be > 0 and float/int or numpy equivalent")
        
        if not is_float_int(self.h):
            raise ValueError("""self.h must be float/int or numpy equivalent
                 Assure all gofher parameters have been set.
                 If using sparcifre, self.calculate_from_sparcfire() 
                 must be called first.""")
    
        if not is_float_int(self.k):
            raise ValueError("""self.k must be float/int or numpy equivalent
                 Assure all gofher parameters have been set.
                 If using sparcifre, self.calculate_from_sparcfire() 
                 must be called first.""")
        
        if not is_float_int(self.a) or self.a <0:
            raise ValueError(""""self.a must be float/int or numpy equivalent
                 Assure all gofher parameters have been set.
                 If using sparcifre, self.calculate_from_sparcfire() 
                 must be called first.""")
    
        if not is_float_int(self.b) or self.b < 0:
            raise ValueError(""""self.b must be float/int or numpy equivalent
                 Assure all gofher parameters have been set.
                 If using sparcifre, self.calculate_from_sparcfire() 
                 must be called first.""")
    
        if not is_float_int(self.theta):
            raise ValueError(""""self.theta must be float/int or 
                numpy equivalent. Assure all gofher parameters have 
                been set. If using sparcifre, self.calculate_from_sparcfire() 
                must be called first.""")
        
        return create_ellipse_mask(self.h,self.k,self.a,self.b,self.theta,shape,r)


    def create_bisection_masks(self, shape: tuple[int]) -> tuple[np.ndarray]:
        """Using the gofher parameters, create the bisection masks.
        See: create_bisection_masks in mask.py

        IMPORTANT: All gofher parameters must be set prior to calling
                   this function. If using sparcfire data, also call
                   calculate_from_sparcfire() first. 
        
        Args:
            shape: the shape of the array 
                (assumes 2D array)

        Returns:
            (pos_mask, neg_mask) boolean masks
        """
        if not is_2d_array_shape(shape):
            raise ValueError("shape must be tuple containing exactly 2 ints & > 0")
        
        if not is_float_int(self.h):
            raise ValueError("""self.h must be float/int or numpy equivalent
                 Assure all gofher parameters have been set.
                 If using sparcifre, self.calculate_from_sparcfire() 
                 must be called first.""")
    
        if not is_float_int(self.k):
            raise ValueError("""self.k must be float/int or numpy equivalent
                 Assure all gofher parameters have been set.
                 If using sparcifre, self.calculate_from_sparcfire() 
                 must be called first.""")
        
        if not is_float_int(self.theta):
            raise ValueError(""""self.theta must be float/int or 
                numpy equivalent. Assure all gofher parameters have 
                been set. If using sparcifre, self.calculate_from_sparcfire() 
                must be called first.""")
        
        return create_bisection_mask(self.h,self.k,self.theta,shape)
    
    def create_near_major_axis_mask(self, sweep: float, 
                                     shape: tuple[int]) -> np.ndarray:
        """Using the gofher parameters, create the near major axis mask.
        See: create_near_major_axis_mask in mask.py

        IMPORTANT: All gofher parameters must be set prior to calling
                   this function. If using sparcfire data, also call
                   calculate_from_sparcfire() first. 
        
        Args:
            sweep: distance from minor axis in radians
                inclusive range [0,pi/2]
            shape: the shape of the array 
                assumes 2D array

        Returns:
            A boolean mask indicating all pixels with in 
                sweep angle of major axis
        """
        if not is_float_int(sweep):
            raise ValueError("sweep must be and float/int or numpy equivalent")
    
        if not sweep >= 0 and sweep <= np.pi/2:
            raise ValueError("sweep must be between 0 and pi/2")
        
        if not is_2d_array_shape(shape):
            raise ValueError("shape must be tuple containing exactly 2 ints")
        
        if not is_float_int(self.h):
            raise ValueError("""self.h must be float/int or numpy equivalent
                 Assure all gofher parameters have been set.
                 If using sparcifre, self.calculate_from_sparcfire() 
                 must be called first.""")
    
        if not is_float_int(self.k):
            raise ValueError("""self.k must be float/int or numpy equivalent
                 Assure all gofher parameters have been set.
                 If using sparcifre, self.calculate_from_sparcfire() 
                 must be called first.""")
        
        if not is_float_int(self.theta):
            raise ValueError(""""self.theta must be float/int or 
                numpy equivalent. Assure all gofher parameters have 
                been set. If using sparcifre, self.calculate_from_sparcfire() 
                must be called first.""")
        
        return create_near_major_axis_mask(sweep,self.h,self.k,self.theta,shape)
    
    def create_near_minor_axis_mask(self, sweep: float, 
                                     shape: tuple[int]) -> np.ndarray:
        """Using the gofher parameters, create the near minor axis mask.
        See: create_near_minor_axis_mask in mask.py

        IMPORTANT: All gofher parameters must be set prior to calling
                   this function. If using sparcfire data, also call
                   calculate_from_sparcfire() first. 
        
        Args:
            sweep: distance from minor axis in radians
                inclusive range [0,pi/2]
            shape: the shape of the array 
                assumes 2D array

        Returns:
            A boolean mask indicating all pixels with in 
                sweep angle of major axis
        """
        if not is_float_int(sweep):
            raise ValueError("sweep must be and float/int or numpy equivalent")
    
        if not sweep >= 0 and sweep <= np.pi/2:
            raise ValueError("sweep must be between 0 and pi/2")
        
        if not is_2d_array_shape(shape):
            raise ValueError("shape must be tuple containing exactly 2 ints")
        
        if not is_float_int(self.h):
            raise ValueError("""self.h must be float/int or numpy equivalent
                 Assure all gofher parameters have been set.
                 If using sparcifre, self.calculate_from_sparcfire() 
                 must be called first.""")
    
        if not is_float_int(self.k):
            raise ValueError("""self.k must be float/int or numpy equivalent
                 Assure all gofher parameters have been set.
                 If using sparcifre, self.calculate_from_sparcfire() 
                 must be called first.""")
        
        if not is_float_int(self.theta):
            raise ValueError(""""self.theta must be float/int or 
                numpy equivalent. Assure all gofher parameters have 
                been set. If using sparcifre, self.calculate_from_sparcfire() 
                must be called first.""")
        
        return create_near_minor_axis_mask(sweep,self.h,self.k,self.theta,shape)
    
#TODO: CSV output for parameters - export two lists? header and values
#   indicate if sparcfire to be included, include scaling and f?

#TODO: load sparcfire parameters from csv row - likely place in sparcfire file? IDK...

#LATER TODO: Add sep parameters and binning?
        
