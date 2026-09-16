"""A class to contain the gofher parameters to be used by GOFHER

It features:
    * parsing of a SpArcFiRe CSV to create parameters
    * creating masks using the parameters
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd

from gofher.utils import is_float_int, is_2d_array_shape, is_int

from gofher.mask import (
    create_ellipse_mask, 
    create_bisection_mask, 
    create_near_major_axis_mask, 
    create_near_minor_axis_mask
)

INDETERMINANT_VOTE_LABEL = "-"

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

    def get_pos_neg_labels(self) -> tuple[str]:
        """Get the (pos,neg) labels using the gofher parameter theta
        
        IMPORTANT: Must have valid theta prior to calling
        """

        if np.isnan(self.theta):
            raise RuntimeError("theta is nan, make sure theta is set correctly first")

        if not np.isfinite(self.theta):
            raise ValueError("theta must be finite value")

        if not is_float_int(self.theta):
            raise TypeError("theta must be float/int or Numpy equivalent")

        theta = self.theta%(2*np.pi)
        if theta < 0.0 or theta > 2*np.pi:
            raise RuntimeError("converted theta error: theta modulo 2pi expected in range [0,2pi]")

        labels = [("N","S"), # in range [0,1pi/8]
                  ("NE","SW"), # in range (1pi/8,3pi/8)
                  ("E","W"), # in range [3pi/8,5pi/8]
                  ("SE","NW"), # in range (5pi/8,7pi/8)
                  ("S","N"), # in range [7pi/8,9pi/8]
                  ("SW","NE"), # in range (9pi/8,11pi/8)
                  ("W","E"), # in range [11pi/8,13pi/8]
                  ("NW","SE"), # in range (13pi/8,15pi/8)
        ]

        for i, label in enumerate(labels):
            upper_bounds = (2*i+1)*np.pi/8
            include_upper_bounds = (i%2) == 0
            if theta < upper_bounds or (include_upper_bounds and theta == upper_bounds):
                return label

        return labels[0] #in range [15pi/8,16pi/8]

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

    def get_ellipse_pixel_bounds(self,
                                 r: float = 1.0,
                                 padding: int = 0) -> tuple[int]:
        """Using the ellipse from gofher parameters, find the pixel bounding box

        Important:

            First, this uses Uses pixel bounds so xmin, ymin use floor and
                xmax, xmin us ceil to assure all ellipse is contained
                in bounds

            Second, it applies additional padding

            Third it assures the following bounds: 
                xmin in [0,shape[1]]
                xmax in [0,shape[1]]
                ymin in [0,shape[0]]
                ymax in [0,shape[0]]

                If bounds exceeded, rounds to closets bounds

        Args:
            r: scaling factor of ellipse (self.a & self.b)
            padding: additional pixel padding added to sides

        Returns:
            (xmin,xmax,ymin,ymax) of pixel bounding box of ellipse + additional padding
        """

        # Validate Input:
        if not is_float_int(r):
            raise TypeError("r must be float/int or numpy equivalent")

        if r <= 0:
            raise ValueError("r must be > 0")
                
        # Validate Object Parameters:
        if not is_2d_array_shape(self.shape):
            raise TypeError("self.shape must be tuple containing exactly 2 ints & > 0")
                
        if not is_float_int(self.h):
            raise TypeError("""self.h must be float/int or numpy equivalent
                Assure all gofher parameters have been set.
                If using sparcifre, self.calculate_from_sparcfire() 
                must be called first.""")

        if self.h <= 0:
            raise ValueError("self.h must be > 0") 
                   
        if not is_float_int(self.k):
            raise TypeError("""self.k must be float/int or numpy equivalent
                Assure all gofher parameters have been set.
                If using sparcifre, self.calculate_from_sparcfire() 
                must be called first.""")

        if self.k <= 0:
            raise ValueError("self.k must be > 0") 
                
        if not is_float_int(self.a):
            raise TypeError(""""self.a must be float/int or numpy equivalent
                Assure all gofher parameters have been set.
                If using sparcifre, self.calculate_from_sparcfire() 
                must be called first.""")

        if self.a <= 0:
            raise ValueError("""self.a must be > 0""")
            
        if not is_float_int(self.b):
            raise TypeError(""""self.b must be float/int or numpy equivalent
                Assure all gofher parameters have been set.
                If using sparcifre, self.calculate_from_sparcfire() 
                must be called first.""")

        if self.b <= 0:
            raise ValueError("""self.b must be > 0""")
            
        if not is_float_int(self.theta):
            raise TypeError(""""self.theta must be float/int or 
                numpy equivalent. Assure all gofher parameters have 
                been set. If using sparcifre, self.calculate_from_sparcfire() 
                must be called first.""")

        if not isinstance(padding,int):
            raise TypeError("padding must be int")

        if padding < 0:
            raise ValueError("padding must can not be negative")

        # Scale a and b by factor r:
        a = self.a*r
        b = self.b*r

        # Calculate the x and y scale from extreme bounds to center of ellipse:
        x_half = np.sqrt(a**2 * np.cos(self.theta)**2 + b**2 * np.sin(self.theta)**2)
        y_half = np.sqrt(a**2 * np.sin(self.theta)**2 + b**2 * np.cos(self.theta)**2)

        # Calculate the min and max bounds with additional padding:
        xmin = math.floor(self.h-x_half) - padding
        xmax = math.ceil(self.h+x_half) + padding
        ymin = math.floor(self.k-y_half) - padding
        ymax = math.ceil(self.k+y_half) + padding

        # Assure values are in proper range:
        xmin = np.clip(xmin,0,self.shape[1])
        xmax = np.clip(xmax,0,self.shape[1])
        ymin = np.clip(ymin,0,self.shape[0])
        ymax = np.clip(ymax,0,self.shape[0])

        return [xmin,xmax,ymin,ymax]
    
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

    def get_csv_dict(self) -> dict:
        # Collect data:
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

        return data
    
    def output_to_csv(self, csv_path: str | Path):
        """Write the gofher parameters to a csv file at csv_path."""

        if not isinstance(csv_path, (str,Path)):
            raise TypeError(f"given csv_path {csv_path} is not a str or Path")

        if isinstance(csv_path,Path):
            csv_path = Path(csv_path)

        if Path(csv_path).suffix != ".csv":
            raise ValueError(f"csv_path {csv_path} not be a .csv file")

        if not Path(csv_path).parent.exists():
            raise RuntimeError(f"folder '{Path(csv_path).parent}' does not exist")

        # Gather data to write:
        data = self.get_csv_dict()

        # Write to csv:
        df = pd.DataFrame([data])
        df.to_csv(csv_path, index=False, na_rep='')

def get_gofher_parameters_from_dict(the_dict: dict) -> GofherParameters:
    """Create a GofherParameters object using data from dictionary"""

    # Validate input:
    if not isinstance(the_dict, dict):
        raise TypeError("the_dict must be dict")

    if len(the_dict) == 0:
        raise ValueError("the_dict can not be empty dict")

    the_gofher_params = GofherParameters()
    
    # Validate name and ref_band:
    for col in [NAME_KEY,REF_BAND_KEY]:
        if not col in the_dict:
            raise KeyError(f"Missing required column {col}")
            
        if not isinstance(the_dict[col],str):
            raise ValueError(f"Column {col} must be a string")
            
    # Set name and ref_band:
    the_gofher_params.name = the_dict[NAME_KEY]
    the_gofher_params.ref_band = the_dict[REF_BAND_KEY]
    
    # Validate shape: 
    for col in [SHAPE_ROW_KEY,SHAPE_COL_KEY]:
        if not col in the_dict:
                raise KeyError(f"Missing required column {col}")
            
        if not is_int(the_dict[col]) or the_dict[col] < 0:
            raise ValueError(f"Column {col} must be a int > 0")
            
    # Set shape:
    the_gofher_params.shape = (int(the_dict[SHAPE_ROW_KEY]),int(the_dict[SHAPE_COL_KEY]))
    
    # Validate ellipse parameters (a,b,h,k,theta): 
    for col in ELLIPSE_DATA_COLUMNS:
        if not col in the_dict:
            raise KeyError(f"Missing required column {col}")

        if not is_float_int(the_dict[col]):
            raise ValueError(f"Column {col} must be a float/int or numpy equivalent")
            
    # Set ellipse parameters (a,b,h,k,theta): 
    the_gofher_params.a = float(the_dict[A_KEY])
    the_gofher_params.b = float(the_dict[B_KEY])
    the_gofher_params.h = float(the_dict[H_KEY])
    the_gofher_params.k = float(the_dict[K_KEY])
    the_gofher_params.theta = float(the_dict[THETA_KEY])
    
    #TODO: if not using sparcfire, skip part below:
        
    # Validate sparcfire parameters:
    for col in SPARCFIRE_DATA_COLUMNS:
        if not col in the_dict:
            raise KeyError(f"Missing required sparcfire column {col}")
                
        if not is_float_int(the_dict[col]):
            raise ValueError(f"Column {col} must be a float/int or numpy equivalent")
            
    # Set sparcfire parameters:
    the_gofher_params.sparcfire_input_c = float(the_dict[SPARCFIRE_INPUT_C_KEY])
    the_gofher_params.sparcfire_input_r = float(the_dict[SPARCFIRE_INPUT_R_KEY])
    the_gofher_params.sparcfire_disk_maj_axis_len = float(the_dict[SPARCFIRE_DISK_MAJ_AXIS_LEN_KEY])
    the_gofher_params.sparcfire_disk_min_axis_len = float(the_dict[SPARCFIRE_DISK_MIN_AXIS_LEN_KEY])
    the_gofher_params.sparcfire_disk_maj_axis_angle = float(the_dict[SPARCFIRE_DISK_MAJ_AXIS_ANGLE_KEY])
    the_gofher_params.sparcfire_bulge_maj_axis_len = float(the_dict[SPARCFIRE_BULGE_MAJ_AXIS_LEN_KEY])
    the_gofher_params.sparcfire_bulge_axis_ratio = float(the_dict[SPARCFIRE_BULGE_AXIS_RATIO_KEY])
    the_gofher_params.sparcfire_bulge_axis_angle = float(the_dict[SPARCFIRE_BULGE_AXIS_ANGLE_KEY])
    the_gofher_params.sparcfire_bulge_disk_f = float(the_dict[SPARCFIRE_BULGE_DISK_F_KEY])

    return the_gofher_params

def read_gofher_parameters_from_csv(csv_path: str | Path
                                    ) -> GofherParameters:
    """Given a GofherParaemeters csv create a GofherParaemeters 
    
    Args:
        csv_path: the path to the csv
        
    Returns:
        GofherParameters with values from the csv
    """

    if not isinstance(csv_path, (str, Path)):
        raise TypeError(f"csv_path must be str or Path, given {csv_path}")

    if isinstance(csv_path, str):
        csv_path = Path(csv_path)

    if csv_path.suffix != ".csv":
        raise ValueError(f"csv_path must be .csv file, given {csv_path}")

    if not csv_path.is_file():
        raise ValueError(f"given csv_path {csv_path} does not exist")

    df = pd.read_csv(csv_path, na_values=[""])
    the_dict = df.iloc[0].to_dict()
    the_gofher_params = get_gofher_parameters_from_dict(the_dict)
        
    return the_gofher_params

#LATER TODO: Add sep parameters and binning?
#TODO: make sparcfire read/write optional for csv
        
