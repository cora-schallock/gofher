"""A class to contain the gofher parameters to be used by GOFHER

It features:
    * parsing of a SpArcFiRe CSV to create parameters
    * creating masks using the parameters
"""

from pathlib import Path

import numpy as np
import pandas as pd

from utils import is_float_int, is_2d_array_shape, is_int

from mask import (
    create_ellipse_mask, 
    create_bisection_mask, 
    create_near_major_axis_mask, 
    create_near_minor_axis_mask
)


NAME_KEY = "name"
REF_BAND_KEY = "ref_band"
SHAPE_ROW_KEY = "shape_row"
SHAPE_COL_KEY = "shape_col"

H_KEY = "h"
K_KEY = "k"
A_KEY = "a"
B_KEY = "b"
THETA_KEY = "theta"

SPARCFIRE_INPUT_C_KEY = "sparcfire_input_c"
SPARCFIRE_INPUT_R_KEY = "sparcfire_input_r"
SPARCFIRE_DISK_MAJ_AXIS_LEN_KEY = "sparcfire_disk_maj_axis_len"
SPARCFIRE_DISK_MIN_AXIS_LEN_KEY = "sparcfire_disk_min_axis_len"
SPARCFIRE_DISK_MAJ_AXIS_ANGLE_KEY = "sparcfire_disk_maj_axis_angle"
SPARCFIRE_BULGE_MAJ_AXIS_LEN_KEY = "sparcfire_bulge_axis_len"
SPARCFIRE_BULGE_AXIS_RATIO_KEY = "sparcfire_bulge_axis_ratio"
SPARCFIRE_BULGE_AXIS_ANGLE_KEY = "sparcfire_bulge_axis_angle"

SPARCFIRE_BULGE_DISK_F_KEY = "sparcfire_bulge_disk_f"

ELLIPSE_DATA_COLUMNS = [
    H_KEY,
    K_KEY,
    A_KEY,
    B_KEY,
    THETA_KEY]

SPARCFIRE_DATA_COLUMNS = [
    SPARCFIRE_INPUT_C_KEY,
    SPARCFIRE_INPUT_R_KEY,
    SPARCFIRE_DISK_MAJ_AXIS_LEN_KEY,
    SPARCFIRE_DISK_MIN_AXIS_LEN_KEY,
    SPARCFIRE_DISK_MAJ_AXIS_ANGLE_KEY,
    SPARCFIRE_BULGE_MAJ_AXIS_LEN_KEY,
    SPARCFIRE_BULGE_AXIS_RATIO_KEY,
    SPARCFIRE_BULGE_AXIS_ANGLE_KEY,
    SPARCFIRE_BULGE_DISK_F_KEY
]

class GofherParameters:
    """Contains gofher ellipse parameters"""

    def __init__(self):
        self.name = ""
        self.ref_band = ""
        self.shape = (-1,-1)

        self.h = np.nan
        self.k = np.nan
        self.a = np.nan
        self.b = np.nan
        self.theta = np.nan

        self.sparcfire_input_c = np.nan
        self.sparcfire_input_r = np.nan
        self.sparcfire_disk_maj_axis_len = np.nan
        self.sparcfire_disk_min_axis_len = np.nan
        self.sparcfire_disk_maj_axis_angle = np.nan
        self.sparcfire_bulge_maj_axis_len = np.nan
        self.sparcfire_bulge_axis_ratio = np.nan
        self.sparcfire_bulge_axis_angle = np.nan

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
            raise TypeError("self.sparcfire_input_c is invalid")
        
        if not is_float_int(self.sparcfire_input_r):        
            raise TypeError("self.sparcfire_input_r is invalid")
        
        if not is_float_int(self.sparcfire_disk_maj_axis_angle):
            raise TypeError("self.sparcfire_disk_maj_axs_angle is invalid")

        if bulge_disk_f != 0.0 and not is_float_int(self.sparcfire_disk_maj_axis_len):
            raise TypeError("self.sparcfire_maj_axis_len is invalid")
        
        if bulge_disk_f != 0.0 and not is_float_int(self.sparcfire_disk_min_axis_len):
            raise TypeError("self.sparcfire_min_axis_len is invalid")
        
        if bulge_disk_f != 1.0 and not is_float_int(self.sparcfire_bulge_maj_axis_len):
            raise TypeError("self.sparcfire_bulge_maj_axis_len is invalid")

        if bulge_disk_f not in [0.0,1.0] and \
            self.sparcfire_bulge_maj_axis_len > self.sparcfire_disk_maj_axis_len:
            raise ValueError("bulge maj. axis len can not be larger or same as disk")
        
        if not is_float_int(bulge_disk_f):
            raise TypeError("bulge_disk_f must be float")

        if bulge_disk_f < 0.0 or bulge_disk_f > 1.0:
            raise ValueError("bulge_disk_f must be in range [0,1]")

        self.sparcfire_bulge_disk_f = bulge_disk_f

        self.h = self.sparcfire_input_c - 1.5
        self.k = self.sparcfire_input_r - 1.5
        self.theta = self.sparcfire_disk_maj_axis_angle * -1.0   

        diff = self.sparcfire_disk_maj_axis_len - self.sparcfire_bulge_maj_axis_len
        self.a = self.sparcfire_bulge_maj_axis_len + diff*bulge_disk_f
        self.b = self.sparcfire_disk_min_axis_len * (self.a/self.sparcfire_disk_maj_axis_len) #TODO: fix this!

        self.a *= 0.5
        self.b *= 0.5

    def create_ellipse_mask(self,
                            r: float = 1.0) -> np.ndarray:
        """Using the gofher parameters, create an ellipse mask.
        See: create_ellipse_mask in mask.py

        IMPORTANT: All gofher parameters must be set prior to calling
                   this function. If using sparcfire data, also call
                   calculate_from_sparcfire() first. 
        
        Args:
            r: scaling factor of ellipse (self.a & self.b)

        Returns:
            A boolean ellipse mask for all pixels in ellipse.
        """

        # Validate Input:
        if not is_float_int(r) or r < 0.0:
            raise ValueError("r must be > 0 and float/int or numpy equivalent")
        
        # Validate Object Parameters:
        if not is_2d_array_shape(self.shape):
            raise ValueError("self.shape must be tuple containing exactly 2 ints & > 0")
        
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
        
        # Create Ellipse Mask:
        return create_ellipse_mask(self.h,self.k,self.a,self.b,self.theta,self.shape,r)


    def create_bisection_masks(self) -> tuple[np.ndarray]:
        """Using the gofher parameters, create the bisection masks.
        See: create_bisection_masks in mask.py

        IMPORTANT: All gofher parameters must be set prior to calling
                   this function. If using sparcfire data, also call
                   calculate_from_sparcfire() first. 

        Returns:
            (pos_mask, neg_mask) boolean masks
        """

        # Validate Object Parameters:
        if not is_2d_array_shape(self.shape):
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
        
        # Create Bisection Mask:
        return create_bisection_mask(self.h,self.k,self.theta,self.shape)
    
    def create_near_major_axis_mask(self, sweep: float) -> np.ndarray:
        """Using the gofher parameters, create the near major axis mask.
        See: create_near_major_axis_mask in mask.py

        IMPORTANT: All gofher parameters must be set prior to calling
                   this function. If using sparcfire data, also call
                   calculate_from_sparcfire() first. 
        
        Args:
            sweep: distance from minor axis in radians
                inclusive range [0,pi/2]

        Returns:
            A boolean mask indicating all pixels with in 
                sweep angle of major axis
        """

        # Validate Input:
        if not is_float_int(sweep):
            raise ValueError("sweep must be and float/int or numpy equivalent")
    
        if not sweep >= 0 and sweep <= np.pi/2:
            raise ValueError("sweep must be between 0 and pi/2")
        
        # Validate Object Parameters:
        if not is_2d_array_shape(self.shape):
            raise ValueError("self.shape must be tuple containing exactly 2 ints")
        
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
        
        # Create near major axis mask:
        return create_near_major_axis_mask(sweep,self.h,self.k,self.theta,self.shape)
    
    def create_near_minor_axis_mask(self, sweep: float) -> np.ndarray:
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

        # Validate Input:
        if not is_float_int(sweep):
            raise ValueError("sweep must be and float/int or numpy equivalent")
    
        if not sweep >= 0 and sweep <= np.pi/2:
            raise ValueError("sweep must be between 0 and pi/2 (inclusive)")
        
        # Validate Object Parameters:
        if not is_2d_array_shape(self.shape):
            raise ValueError("self.shape must be tuple containing exactly 2 ints")
        
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
        
        # Create near minor axis mask:
        return create_near_minor_axis_mask(sweep,self.h,self.k,self.theta,self.shape)
    
    def output_to_csv(self, csv_path: str):
        """Write the gofher parameters to a csv file at csv_path."""

        if not isinstance(csv_path, str):
            raise TypeError(f"given csv_path {csv_path} is not a str")

        if Path(csv_path).suffix != ".csv":
            raise ValueError(f"csv_path {csv_path} not be a .csv file")

        # Collect data to write:
        data = {
            NAME_KEY: self.name,
            REF_BAND_KEY: self.ref_band,
            SHAPE_ROW_KEY: self.shape[0],
            SHAPE_COL_KEY: self.shape[1],
            H_KEY: self.h,
            K_KEY: self.k,
            A_KEY: self.a,
            B_KEY: self.b,
            THETA_KEY: self.theta,
            SPARCFIRE_INPUT_C_KEY: self.sparcfire_input_c,
            SPARCFIRE_INPUT_R_KEY: self.sparcfire_input_r,
            SPARCFIRE_DISK_MAJ_AXIS_LEN_KEY: self.sparcfire_disk_maj_axis_len,
            SPARCFIRE_DISK_MIN_AXIS_LEN_KEY: self.sparcfire_disk_min_axis_len,
            SPARCFIRE_DISK_MAJ_AXIS_ANGLE_KEY: self.sparcfire_disk_maj_axis_angle,
            SPARCFIRE_BULGE_MAJ_AXIS_LEN_KEY: self.sparcfire_bulge_maj_axis_len,
            SPARCFIRE_BULGE_AXIS_RATIO_KEY: self.sparcfire_bulge_axis_ratio,
            SPARCFIRE_BULGE_AXIS_ANGLE_KEY: self.sparcfire_bulge_axis_angle,
            SPARCFIRE_BULGE_DISK_F_KEY: self.sparcfire_bulge_disk_f
        }

        # Write to csv:
        df = pd.DataFrame([data])
        df.to_csv(csv_path, index=False, na_rep='')
    

def read_gofher_parameters_from_csv(csv_path: str
                                    ) -> GofherParameters:
    """Given a GofherParaemeters csv create a GofherParaemeters 
    
    Args:
        csv_path: the path to the csv
        
    Returns:
        GofherParameters with values from the csv
    """

    if not isinstance(csv_path, str):
        raise TypeError(f"csv_path must be str, given {csv_path}")

    if Path(csv_path).suffix != ".csv":
        raise ValueError(f"csv_path must be .csv file, given {csv_path}")

    if not Path.is_file(csv_path):
        raise ValueError(f"given csv_path {csv_path} does not exist")

    df = pd.read_csv(csv_path, na_values=[""])
    the_galaxy = df.iloc[0]
    has_columns = the_galaxy.index.tolist()

    the_gofher_params = GofherParameters()

    # Validate name and ref_band:
    for col in [NAME_KEY,REF_BAND_KEY]:
        if not col in has_columns:
            raise ValueError(f"Missing required column {col}")
        
        if not isinstance(the_galaxy[col],str):
            raise ValueError(f"Column {col} must be a string")
        
    # Set name and ref_band:
    the_gofher_params.name = the_galaxy[NAME_KEY]
    the_gofher_params.ref_band = the_galaxy[REF_BAND_KEY]

    # Validate shape: 
    for col in [SHAPE_ROW_KEY,SHAPE_COL_KEY]:
        if not col in has_columns:
            raise ValueError(f"Missing required column {col}")
        
        if not is_int(the_galaxy[col]) or the_galaxy[col] < 0:
            raise ValueError(f"Column {col} must be a int > 0")
        
    # Set shape:
    the_gofher_params.shape = (the_galaxy[SHAPE_ROW_KEY],the_galaxy[SHAPE_COL_KEY])

    # Validate ellipse parameters (a,b,h,k,theta): 
    for col in ELLIPSE_DATA_COLUMNS:
        if not col in has_columns:
            raise ValueError(f"Missing required column {col}")
        
        if not is_float_int(the_galaxy[col]):
            raise ValueError(f"Column {col} must be a float/int or numpy equivalent")
        
    # Set ellipse parameters (a,b,h,k,theta): 
    the_gofher_params.a = the_galaxy[A_KEY]
    the_gofher_params.b = the_galaxy[B_KEY]
    the_gofher_params.h = the_galaxy[H_KEY]
    the_gofher_params.k = the_galaxy[K_KEY]
    the_gofher_params.theta = the_galaxy[THETA_KEY]

    #TODO: if not using sparcfire, skip part below:
    
    # Validate sparcfire parameters:
    for col in SPARCFIRE_DATA_COLUMNS:
        if not col in has_columns:
            raise ValueError(f"Missing required sparcfire column {col}")
            
        if not is_float_int(the_galaxy[col]):
            raise ValueError(f"Column {col} must be a float/int or numpy equivalent")
        
    # Set sparcfire parameters:
    the_gofher_params.sparcfire_input_c = the_galaxy[SPARCFIRE_INPUT_C_KEY]
    the_gofher_params.sparcfire_input_r = the_galaxy[SPARCFIRE_INPUT_R_KEY]
    the_gofher_params.sparcfire_disk_maj_axis_len = the_galaxy[SPARCFIRE_DISK_MAJ_AXIS_LEN_KEY]
    the_gofher_params.sparcfire_disk_min_axis_len = the_galaxy[SPARCFIRE_DISK_MIN_AXIS_LEN_KEY]
    the_gofher_params.sparcfire_disk_maj_axis_angle = the_galaxy[SPARCFIRE_DISK_MAJ_AXIS_ANGLE_KEY]
    the_gofher_params.sparcfire_bulge_maj_axis_len = the_galaxy[SPARCFIRE_BULGE_MAJ_AXIS_LEN_KEY]
    the_gofher_params.sparcfire_bulge_axis_ratio = the_galaxy[SPARCFIRE_BULGE_AXIS_RATIO_KEY]
    the_gofher_params.sparcfire_bulge_axis_angle = the_galaxy[SPARCFIRE_BULGE_AXIS_ANGLE_KEY]
    the_gofher_params.sparcfire_bulge_disk_f = the_galaxy[SPARCFIRE_BULGE_DISK_F_KEY]
        
    return the_gofher_params

#LATER TODO: Add sep parameters and binning?
#TODO: make sparcfire read/write optional for csv
        
