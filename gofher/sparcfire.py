import numpy as np
import pandas as pd
import scipy.interpolate

from gofher import gofher_parameters

#CSV KEYS - DO NOT EDIT - Must be same as SpArcFiRe galaxy.csv columns:
NAME_KEY = "name"
BAND_KEY = "band"

DISK_MAJ_ANGLE_KEY = 'diskMajAxsAngleRadians'
DISK_MIN_AXS_LEN_KEY = 'diskMinAxsLen'
DISK_MAJ_AXS_LEN_KEY = 'diskMajAxsLen'
INPUT_CENTER_C_KEY = 'inputCenterC'
INPUT_CENTER_R_KEY ='inputCenterR'

BULGE_MIN_AXS_LEN_KEY = "bulgeMinAxsLen"
BULGE_MAJ_AXS_LEN_KEY = "bulgeMajAxsLen"
BULGE_AXS_RATIO_KEY = "bulgeAxisRatio"
BULGE_MAJ_AXS_ANGLE_KEY = "bulgeMajAxsAngle"

#Keys required in BAND DICT
BAND_DICT_KEYS = [DISK_MAJ_ANGLE_KEY,
                 DISK_MIN_AXS_LEN_KEY,
                 DISK_MAJ_AXS_LEN_KEY,
                 INPUT_CENTER_C_KEY,
                 INPUT_CENTER_R_KEY,
                 BULGE_AXS_RATIO_KEY,
                 BULGE_MAJ_AXS_LEN_KEY,
                 BULGE_MAJ_AXS_ANGLE_KEY]

def can_float(to_check: str) -> bool:
    """Checks if a string can be floated

        Args:
            to_check: the string
        Returns:
            a bool representing if float(to_check) does *not* result in an exception
    """
    try:
        float(to_check)
        return True
    except Exception:
        return False

def convert_disk_angle_to_bisection_angle(disk_maj_axs_angle: float) -> float:
    """Converts SpArcFiRe angle to gofher bisection angle

        Args:
            diskMajAxsAngle: the SpArcFiRe angle as a float
        Returns:
            a float representing the gofher bisection (ellipse) angle
    """
    return disk_maj_axs_angle * -1.0

def convert_sparcfire_to_gofher_center(input_center_c: float, input_center_r: float) -> tuple:
    """Converts matlab (colum,row) to gofher (colum,row)

    Note: The 1.5 difference is from:
        MATLAB using 1 based indexing while python (which gofher uses) using 0 based indexing
        plus 0.5 difference between center of pixel between gofher and SpArcFiRe
            SpArcFiRe use (x.5,y.5) as center of pixel (x,y) (see Davis & Hayes 2014... although this detail maybe in Darren Davis' dissertation...)
            gofher use (x.0, y.0) as center of pixel (x,y) (see: https://sep.readthedocs.io/en/v1.1.x/reference.html?highlight=Center%20of%20ellipse#reference-api)
                From sep documentation:
                The coordinate convention in SEP is that (0, 0) corresponds to the center of the first element of the data array. This agrees with the 0-based indexing in Python and C. 
                However, note that this differs from the FITS convention where the center of the first element is at coordinates (1, 1). As Source Extractor deals with FITS files, its 
                outputs follow the FITS convention. Thus, the coordinates from SEP will be offset from Source Extractor coordinates by -1 in x and y.

        Args:
            input_center_c: SpArcFiRe center of disk coordinate column
            input_center_r: SpArcFiRe center of disk coordinate row
        Returns:
            a tuple containing the gofher (colum,row)
    """
    cx = input_center_c - 1.5
    cy = input_center_r - 1.5
    return (cx,cy)

def extract_params_from_sparcfire_params(input_center_c: float, input_center_r: float, disk_maj_axs_len: float,
                                         disk_min_axs_len: float, disk_maj_axs_angle_radians: float, bulge_maj_axs_len: float,
                                         bulge_disk_r = 1.0) -> gofher_parameters:
    """When given information from SpArcFiRe galaxy csv amd bulge_disk_r creates a gofher parameter object 

    Note: The 1.5 difference is from:
        MATLAB using 1 based indexing while python (which gofher uses) using 0 based indexing
        plus 0.5 difference between center of pixel between gofher and SpArcFiRe
            SpArcFiRe use (x.5,y.5) as center of pixel (x,y) (see Davis & Hayes 2014... although this detail maybe in Darren Davis' dissertation...)
            gofher use (x.0, y.0) as center of pixel (x,y) (see: https://sep.readthedocs.io/en/v1.1.x/reference.html?highlight=Center%20of%20ellipse#reference-api)
                From sep documentation:
                The coordinate convention in SEP is that (0, 0) corresponds to the center of the first element of the data array. This agrees with the 0-based indexing in Python and C. 
                However, note that this differs from the FITS convention where the center of the first element is at coordinates (1, 1). As Source Extractor deals with FITS files, its 
                outputs follow the FITS convention. Thus, the coordinates from SEP will be offset from Source Extractor coordinates by -1 in x and y.

        Args:
            input_center_c: SpArcFiRe center of disk coordinate column (see Davis & Hayes 2014)
            input_center_r: SpArcFiRe center of disk coordinate row (see Davis & Hayes 2014)
            disk_maj_axs_len: SpArcFiRe's major axis length (see Davis & Hayes 2014)
            disk_min_axs_len: SpArcFiRe's minor axis length (see Davis & Hayes 2014)
            disk_maj_axs_angle_radians: SpArcFiRe's angle of disk (see Davis & Hayes 2014)
            bulge_maj_axs_len: SpArcFiRe's bulge maj axis (see Davis & Hayes 2014)
            bulge_disk_r: scaling of ellipse
                When bulge_disk_r=0.0 -> a=bulge_maj_axs_len*0.5
                When bulge_disk_r=1.0 -> a=disk_maj_axs_len*0.5
                Else: Weighting
                IMPORTANT: scales ellipse with intact ratio semi-major/semi-minor axis ratio (i.e. b is found after finding a as shown above, and scaling it)

        Returns:
            gofher_parameters that specify the ellipse given by SpArcFiRe params and bulge_disk_r
    """
    bulge_disk_r = max(0,min(bulge_disk_r,1.0))
    diff = disk_maj_axs_len - bulge_maj_axs_len
    
    maj_axis_len = bulge_maj_axs_len + diff*bulge_disk_r
    minor_axis_len = disk_min_axs_len * (maj_axis_len/disk_maj_axs_len)

    the_params = gofher_parameters()
    (h, k) = convert_sparcfire_to_gofher_center(input_center_c, input_center_r)

    the_params.x = h
    the_params.y = k
    the_params.a = maj_axis_len * 0.5
    the_params.b = minor_axis_len * 0.5
    the_params.theta = convert_disk_angle_to_bisection_angle(disk_maj_axs_angle_radians)
    return the_params

def load_sparcfire_dict(band_dict: dict, bulge_disk_r=1.0)  -> gofher_parameters:
    """When given dict of SpArcFiRe's galaxy.csv output for single galaxy and creates a gofher parameter object 

        Args:
            band_dict: the dictionary of the single band of a single galaxy from SpArcFiRe's galaxy.csv output
            bulge_disk_r: scaling of ellipse
                When bulge_disk_r=0.0 -> a=bulge_maj_axs_len*0.5
                When bulge_disk_r=1.0 -> a=disk_maj_axs_len*0.5
                Else: Weighting
                IMPORTANT: scales ellipse with intact ratio semi-major/semi-minor axis ratio (i.e. b is found after finding a as shown above, and scaling it)

        Returns:
            gofher_parameters that specify the ellipse given by SpArcFiRe params and bulge_disk_r
    """
    inputCenterC = 0;inputCenterR = 0
    diskMajAxsLen = 0; diskMinAxsLen = 0; diskMajAxsAngleRadians = 0
    for each_key in band_dict:
        if can_float(band_dict[each_key]):
            if each_key == DISK_MAJ_ANGLE_KEY:
                diskMajAxsAngleRadians = float(band_dict[each_key])
            elif each_key == DISK_MIN_AXS_LEN_KEY:
                diskMinAxsLen = float(band_dict[each_key])
            elif each_key == DISK_MAJ_AXS_LEN_KEY:
                diskMajAxsLen = float(band_dict[each_key])
            elif each_key == INPUT_CENTER_C_KEY:
                inputCenterC = float(band_dict[each_key])
            elif each_key == INPUT_CENTER_R_KEY:
                inputCenterR = float(band_dict[each_key])
            elif each_key == BULGE_AXS_RATIO_KEY:
                bulgeAxisRatio = float(band_dict[each_key])
            elif each_key == BULGE_MAJ_AXS_LEN_KEY:
                bulgeMajAxsLen = float(band_dict[each_key])
            elif each_key == BULGE_MAJ_AXS_ANGLE_KEY:
                bulgeMajAxsAngle = float(band_dict[each_key])
    return extract_params_from_sparcfire_params(inputCenterC,inputCenterR,diskMajAxsLen,diskMinAxsLen,diskMajAxsAngleRadians,bulgeMajAxsLen,bulge_disk_r)

def get_ref_band_and_gofher_params(sparcfire_bands: dict, ref_bands_in_order: list, bulge_disk_r=1.0) -> tuple:
    """When given the sparcfire_bands object for a single galaxy and bulge_disk_r, selects ref band, and returns tuple of (ref_band,gofher_params)

        Args:
            sparcfire_bands: a dict where the keys are the bands of the SpArcFiRe runs for the galaxy and the values are the SpArcFiRe dict of galaxy.csv output
            ref_bands_in_order: the refrence band selection in order
            bulge_disk_r: scaling of ellipse
                When bulge_disk_r=0.0 -> a=bulge_maj_axs_len*0.5
                When bulge_disk_r=1.0 -> a=disk_maj_axs_len*0.5
                Else: Weighting
                IMPORTANT: scales ellipse with intact ratio semi-major/semi-minor axis ratio (i.e. b is found after finding a as shown above, and scaling it)

        Returns:
            tuple of (ref_band,gofher_params) where ref_band is selected ref_band and gofher params for selected ref band (according to specifications of load_sparcfire_dict())
    """
    for band in ref_bands_in_order:
        if band not in sparcfire_bands: continue
        
        the_params = load_sparcfire_dict(sparcfire_bands[band],bulge_disk_r)
        if not isinstance(the_params,gofher_parameters): continue

        return band, the_params
    return None, None

def get_gofher_params_for_fixed_ref_band(sparcfire_bands: dict, the_ref_band: str, bulge_disk_r=1.0) -> gofher_parameters:
    """When given the sparcfire_bands object for a single galaxy, if has band the_ref_band uses that as ref band, and returns gofher_params

        Args:
            sparcfire_bands: a dict where the keys are the bands of the SpArcFiRe runs for the galaxy and the values are the SpArcFiRe dict of galaxy.csv output
            the_ref_band: the specific refernce band to use
            bulge_disk_r: scaling of ellipse
                When bulge_disk_r=0.0 -> a=bulge_maj_axs_len*0.5
                When bulge_disk_r=1.0 -> a=disk_maj_axs_len*0.5
                Else: Weighting
                IMPORTANT: scales ellipse with intact ratio semi-major/semi-minor axis ratio (i.e. b is found after finding a as shown above, and scaling it)

        Returns:
            gofher_params using  the_ref_band and bulge_disk_r
    """
    if the_ref_band not in sparcfire_bands: return None

    the_params = load_sparcfire_dict(sparcfire_bands[the_ref_band],bulge_disk_r)
    if not isinstance(the_params,gofher_parameters): return None

    return the_params

def normalize_row_keys(the_row: dict) -> dict:
    """Returns dictionary of normalized keys (removed whitespace at start and end)

        Args:
            the_row: to normalize

        Returns:
            a dict where keys are normalized and values are original keys
    """

    to_return = dict()
    for key in the_row.keys():
        to_return[key.strip()] = key
    return to_return

def row_to_band_dict(the_row: dict) -> tuple:
    """Take a dictionary representing a single row in SpArcFiRe's galaxy.csv output and returns a tuple of (gal_name, gal_band, gal_dict)

        Args:
            the_row: the single row in SpArcFiRe's galaxy.csv output

        Returns:
            returns a tuple of (gal_name, gal_band, gal_dict)
            IMPORTANT: Assumes format of NAME in SpArcFiRe is name_band
                ex: NGC1_g would have gal_name=NGC1 and gal_band=g
                * Names *should* be able to contain '-''s but last will be used to delinate name and band 
    """
    gal_name = ''
    gal_band = ''
    gal_dict = {}
    normalized_rows = normalize_row_keys(the_row)
    #if row is missing, will be nan
    
    if 'name' in the_row:
        [gal_name,gal_band] = the_row['name'].strip().rsplit("_",1)
    elif 'name' in normalized_rows and normalized_rows['name'] != 'name':
        [gal_name,gal_band] = the_row[normalized_rows['name']].strip().rsplit("_",1)
    
    for key in BAND_DICT_KEYS:
        if key in the_row and not np.isnan(the_row[key]):
            gal_dict[key] = the_row[key]
        elif key in normalized_rows and normalized_rows[key] != key:
            gal_dict[key] = the_row[normalized_rows[key]]

    return (gal_name, gal_band, gal_dict)

def read_sparcfire_galaxy_csv(csv_path: str) -> dict:
    """read galaxy data from SpArcFiRe's galaxy.csv and returnes nested dict of {name:{band:galaxy_dict}}

        Args:
            csv_path: path to sparcfire csv

        Returns:
            returns  nested dict of {name:{band:galaxy_dict}}
    """
    if not os.path.exists(csv_path):
        raise ValueError("The path to the SpArcFiRe csv is not found {} - make sure you update csv_path whereever read_sparcfire_galaxy_csv(csv_path) is called".format(csv_path))

    csv_dict = dict()
    df = pd.read_csv(csv_path,encoding = 'ascii' , on_bad_lines='skip')#'ISO-8859-1'
    
    for index, row in df.iterrows():
        (gal_name, gal_band, gal_dict) = row_to_band_dict(row)
        if gal_name != "" and gal_band != "" and len(gal_dict) == len(BAND_DICT_KEYS):
            if gal_name not in csv_dict:
                csv_dict[gal_name] = dict()
                
            csv_dict[gal_name][gal_band] = gal_dict
    return csv_dict