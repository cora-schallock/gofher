import numpy as np
import pandas as pd
import scipy.interpolate


from gofher import gofher_parameters

#CSV KEYS:
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

def can_float(to_check):
    try:
        float(to_check)
        return True
    except Exception:
        return False

def convert_disk_angle_to_bisection_angle(diskMajAxsAngle):
    return diskMajAxsAngle * -1.0

def convert_matlab_to_cartessian(inputCenterC,inputCenterR):
    cx = inputCenterC - 1.5
    cy = inputCenterR - 1.5
    return (cx,cy)

def extract_params_from_sparcfire_params(inputCenterC,inputCenterR,diskMajAxsLen,diskMinAxsLen,diskMajAxsAngleRadians,bulgeMajAxsLen,bulge_disk_r):
    #calculate maj_axis:
    ##  When bulge_disk_r=0.0 -> bulgeMajAxsLen
    ##  When bulge_disk_r=1.0 -> diskMajAxsLen
    ## Else: Weighting
    bulge_disk_r = max(0,min(bulge_disk_r,1.0))
    diff = diskMajAxsLen - bulgeMajAxsLen
    
    maj_axis_len = bulgeMajAxsLen + diff*bulge_disk_r
    minor_axis_len = diskMinAxsLen * (maj_axis_len/diskMajAxsLen)

    the_params = gofher_parameters()
    (h, k) = convert_matlab_to_cartessian(inputCenterC, inputCenterR)

    the_params.x = h
    the_params.y = k
    #the_params.a = diskMajAxsLen * 0.5
    #the_params.b = diskMinAxsLen * 0.5
    the_params.a = maj_axis_len * 0.5
    the_params.b = minor_axis_len * 0.5
    the_params.theta = convert_disk_angle_to_bisection_angle(diskMajAxsAngleRadians)
    return the_params

def load_sparcfire_dict(band_dict,bulge_disk_r=1.0):
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

def get_ref_band_and_gofher_params(sparcfire_bands,ref_bands_in_order,bulge_disk_r=1.0):
    for band in ref_bands_in_order:
        if band not in sparcfire_bands: continue
        
        the_params = load_sparcfire_dict(sparcfire_bands[band],bulge_disk_r)
        if not isinstance(the_params,gofher_parameters): continue

        return band, the_params
    return None, None

def get_gofher_params_for_fixed_ref_band(sparcfire_bands, the_ref_band, bulge_disk_r=1.0):
    if the_ref_band not in sparcfire_bands: return None

    the_params = load_sparcfire_dict(sparcfire_bands[the_ref_band],bulge_disk_r)
    if not isinstance(the_params,gofher_parameters): return None

    return the_params

def normalize_row_keys(the_row):
    to_return = dict()
    for key in the_row.keys():
        to_return[key.strip()] = key
    return to_return

def row_to_band_dict(the_row):
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

    return gal_name, gal_band, gal_dict

def read_sparcfire_galaxy_csv(csv_path):
    '''read galaxy data from csv'''
    csv_dict = dict()
    df = pd.read_csv(csv_path,encoding = 'ascii' , on_bad_lines='skip')#'ISO-8859-1'
    
    for index, row in df.iterrows():
        gal_name, gal_band, gal_dict = row_to_band_dict(row)
        #print(row)
        #print(gal_dict)
        if gal_name != "" and gal_band != "" and len(gal_dict) == len(BAND_DICT_KEYS):
            if gal_name not in csv_dict:
                csv_dict[gal_name] = dict()
                
            csv_dict[gal_name][gal_band] = gal_dict
    return csv_dict