import os
import inspect
from collections.abc import Callable

import numpy as np
import pandas as pd

from gofher_parameters import GofherParameters
from utils import is_float_int

#CSV KEYS - DO NOT EDIT - Must be same as SpArcFiRe galaxy.csv columns:
NAME_KEY = "name"

DISK_MAJ_ANGLE_KEY = 'diskMajAxsAngleRadians'
DISK_MIN_AXS_LEN_KEY = 'diskMinAxsLen'
DISK_MAJ_AXS_LEN_KEY = 'diskMajAxsLen'

INPUT_CENTER_C_KEY = 'inputCenterC'
INPUT_CENTER_R_KEY ='inputCenterR'

BULGE_MAJ_AXS_LEN_KEY = "bulgeMajAxsLen"
BULGE_AXS_RATIO_KEY = "bulgeAxisRatio"
BULGE_MAJ_AXS_ANGLE_KEY = "bulgeMajAxsAngle"

DATA_COLUMNS = [
    DISK_MAJ_ANGLE_KEY,
    DISK_MIN_AXS_LEN_KEY,
    DISK_MAJ_AXS_LEN_KEY,
    INPUT_CENTER_C_KEY,
    INPUT_CENTER_R_KEY,
    BULGE_MAJ_AXS_LEN_KEY,
    BULGE_AXS_RATIO_KEY,
    BULGE_MAJ_AXS_ANGLE_KEY]

def standard_normalize_name(name: str) -> str:
    """Given a name string in format of 'name_refband', returns 'name'"""
    return name.strip().rsplit("_")[0]

def standard_ref_band_from_name(name: str) -> str:
    """Given a name string in format of 'name_refband', returns 'refband'"""
    return name.strip().rsplit("_")[-1]

def gofher_params_from_sparcfire_row(row,
        normalize_name: Callable[[str],str] | None = standard_normalize_name,
        get_ref_band: Callable[[str],str] | None = standard_ref_band_from_name
    ) -> GofherParameters:
    """TODO: write this"""
    #print(row)
    the_gofher_params = GofherParameters()
    has_columns = row.index.tolist()

    # Verify row[NAME_KEY] exists and is valid:
    if not NAME_KEY in has_columns:
        raise ValueError("Missing required column {NAME_KEY}")
    
    if not isinstance(row[NAME_KEY],str):
        raise ValueError(f"Column {NAME_KEY} must be a string")
    
    # Verify all required columns exist and are valid floats/int:
    for column in DATA_COLUMNS:
        if not column in has_columns:
            raise ValueError(f"Missing required column {column}")
        
        if not is_float_int(row[column]):
            raise ValueError(f"Column {column} must be a float/int or numpy equivalent")
        
    # Verify normalize_name is a function or row[NAME_KEY] is not empty:
    if not normalize_name is None:
        # Verify normalize_name is a function:
        if not isinstance(normalize_name, Callable):
            raise ValueError("normalize_name must be a function")
        
        # Verify normalize_name has exactly 1 parameter:
        sig = inspect.signature(normalize_name)
        if len(sig.parameters) != 1: 
            raise ValueError("normalize_name must have exactly 1 parameter")
    elif len(row[NAME_KEY]) == 0:
        raise ValueError(f"Column {NAME_KEY} must have len > 0")
    
        
    # Verify ref_band_from_normalized_name is a function or None:
    if not get_ref_band is None:
        # Verify normalize_name is a function:
        if not isinstance(get_ref_band, Callable):
            raise ValueError("ref_band_from_name must be a function")
        
        # Verify normalize_name has exactly 1 parameter:
        sig = inspect.signature(get_ref_band)
        if len(sig.parameters) != 1: 
            raise ValueError("ref_band_from_name must have exactly 1 parameter")
        
    # Normalize Name if applicable, if not just use the raw name:
    if normalize_name is None:
        the_gofher_params.name = row[NAME_KEY]
    else:
        # Get normalized name and verify it is a string of len > 0:
        normalized_name = normalize_name(row[NAME_KEY])
        if not isinstance(normalized_name,str) or len(normalized_name) == 0:
            raise ValueError("normalize_name must return a string of len > 0")

        the_gofher_params.name = normalized_name

    # Get ref band if applicable, if not just skip for now:
    if not get_ref_band is None:
        # Get ref band and verify it is a string of len > 0:
        the_ref_band = get_ref_band(row[NAME_KEY])
        if not isinstance(the_ref_band,str) or len(the_ref_band) == 0:
            raise ValueError("normalize_name must return a string of len > 0")

        the_gofher_params.ref_band = the_ref_band
    
    # Set all data:
    the_gofher_params.disk_maj_angle = row[DISK_MAJ_ANGLE_KEY]
    the_gofher_params.disk_min_axs_len = row[DISK_MIN_AXS_LEN_KEY]
    the_gofher_params.disk_maj_axs_len = row[DISK_MAJ_AXS_LEN_KEY]
    the_gofher_params.input_center_c = row[INPUT_CENTER_C_KEY]
    the_gofher_params.input_center_r = row[INPUT_CENTER_R_KEY]
    the_gofher_params.bulge_maj_axs_len = row[BULGE_MAJ_AXS_LEN_KEY]
    the_gofher_params.bulge_axis_ratio = row[BULGE_AXS_RATIO_KEY]
    the_gofher_params.bulge_maj_axs_angle = row[BULGE_MAJ_AXS_ANGLE_KEY]

    return the_gofher_params

def construct_gofher_params_from_sparcfire_csv(
        csv_path: str, 
        normalize_name: Callable[[str],str] | None = standard_normalize_name,
        get_ref_band: Callable[[str],str] | None = standard_ref_band_from_name,
        fail_silently_on_row_error: bool = False) -> list[GofherParameters]:
    """TODO: write this"""
    # Validate csv_path:
    if not os.path.exists(csv_path):
        raise ValueError(f"No file found at csf_path={csv_path}")
    
    # Verify normalize_name is a function or None:
    if not normalize_name is None:
        # Verify normalize_name is a function:
        if not isinstance(normalize_name, Callable):
            raise ValueError("normalize_name must be a function")
        
        # Verify normalize_name has exactly 1 parameter:
        sig = inspect.signature(normalize_name)
        if len(sig.parameters) != 1: 
            raise ValueError("normalize_name must have exactly 1 parameter")
        
    # Verify ref_band_from_normalized_name is a function or None:
    if not get_ref_band is None:
        # Verify normalize_name is a function:
        if not isinstance(get_ref_band, Callable):
            raise ValueError("ref_band_from_name must be a function")
        
        # Verify normalize_name has exactly 1 parameter:
        sig = inspect.signature(get_ref_band)
        if len(sig.parameters) != 1: 
            raise ValueError("ref_band_from_name must have exactly 1 parameter")
    
    all_gofher_params = []

    # Read CSV:
    df = pd.read_csv(csv_path,encoding = 'ascii' , on_bad_lines='skip')#'ISO-8859-1'

    # IMPORTANT - Strip whitespace from column names:
    df.columns = df.columns.str.strip()

    # Iterate through rows and construct gofher params:
    for _, row in df.iterrows():
        try:
            all_gofher_params.append(gofher_params_from_sparcfire_row(row))
        except ValueError as e:
            if not fail_silently_on_row_error:
                raise e
            else:
                print(e)
                continue

    return all_gofher_params

construct_gofher_params_from_sparcfire_csv("C:\\Users\\school\\Desktop\\github\\gofher-refactor\\gofher\\tests\\data\\NGC2347_SDSS_psf4_background_256\\sparcfire_r_band_output\\NGC2347_r.csv")
#TODO: make path relative and work on mas, windows, linux
