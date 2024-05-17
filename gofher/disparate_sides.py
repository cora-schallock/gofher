import os
import subprocess
import sys
import pathlib
import csv
import copy
import numpy as np
from scipy.stats import ks_2samp

from gofher import normalize_array
from galaxy import galaxy
from file_helper import write_csv
from ebm import EmpiricalBrownsMethod

DISPARATE_SIDES_DIR = "disparate_sides"
#DISPARATE_SIDES_DIR = "./most-disparate-side.sh"
#pa = "NGC2841n.tsv"
#pa = "IC1151n.tsv"
#pa = "NGC1056.tsv"
pa = "NGC3367.tsv"

# TODO: constuct normed tsv (see IC1151n.tsv for details)
# allow run_disparate_sides() take a variable tsv
# parse disparate_sides_vote and output final csv
# implement methods to check for no vote
# make callable from python scripts outside
# clean up this file


def run_ebm(the_gal: galaxy):
    to_diff_mask = the_gal.create_ellipse()
    normed_pixels = dict()

    pos_mask,neg_mask = the_gal.create_bisection()

    for band in the_gal.bands:
        the_band = the_gal[band]
        normed_pixels[band] = the_band.data

        if the_band.valid_pixel_mask is None: the_band.construct_valid_pixel_mask()
        to_diff_mask = np.logical_and(to_diff_mask,the_band.valid_pixel_mask)

    bands = list(normed_pixels.keys())
    pos_diff_mask = np.logical_and(to_diff_mask,pos_mask)
    neg_diff_mask = np.logical_and(to_diff_mask,neg_mask)

    positive_normed_array = np.zeros((len(normed_pixels),np.count_nonzero(pos_diff_mask)))
    negative_normed_array = np.zeros((len(normed_pixels),np.count_nonzero(neg_diff_mask)))

    for band in normed_pixels:
        normed_pixels[band] = normalize_array(normed_pixels[band],to_diff_mask)

    for i in range(len(bands)):
        positive_normed_array[i] = normed_pixels[bands[i]][pos_diff_mask]
        negative_normed_array[i] = normed_pixels[bands[i]][neg_diff_mask]
    
    negative_one_votes = []
    positive_one_votes = []
    for band_pair_key in the_gal.band_pairs:
        vote = the_gal.get_band_pair(band_pair_key).classification
        if int(vote) == 1:
            negative_one_votes.append(band_pair_key) #classifcation flipped
        elif int(vote) == -1:
            positive_one_votes.append(band_pair_key)
            
    positive_one_ebm = 1.0
    negative_one_ebm = 1.0
    
    if len(positive_one_votes) != 0:
        positive_ones_matrix = np.zeros((len(positive_one_votes),np.count_nonzero(pos_diff_mask)))
        positive_ones_pval = []
        for i in range(len(positive_one_votes)):
            band_pair = positive_one_votes[i]
            [first_band,base_band] = band_pair.split("-")
            j = bands.index(first_band)
            k = bands.index(base_band)

            positive_ones_matrix[i] = positive_normed_array[j]-positive_normed_array[k]
            p_val = ks_2samp(positive_normed_array[j],positive_normed_array[k]).pvalue
            positive_ones_pval.append(p_val)

        positive_one_ebm = positive_ones_pval[0]
        if len(positive_one_votes) > 1:
            positive_one_ebm = EmpiricalBrownsMethod(positive_ones_matrix, positive_ones_pval)
        #print('pos ebm:', positive_one_ebm)

    if len(negative_one_votes) != 0:
        #print(len(negative_one_votes))
        negative_ones_matrix = np.zeros((len(negative_one_votes),np.count_nonzero(neg_diff_mask)))
        negative_ones_pval = []
        for i in range(len(negative_one_votes)):
            band_pair = negative_one_votes[i]
            [first_band,base_band] = band_pair.split("-")
            j = bands.index(first_band)
            k = bands.index(base_band)

            negative_ones_matrix[i] = negative_normed_array[j]-negative_normed_array[k]
            p_val = ks_2samp(negative_normed_array[j],negative_normed_array[k]).pvalue
            negative_ones_pval.append(p_val)

        negative_one_ebm = negative_ones_pval[0]
        if len(negative_one_votes) > 1:
            negative_one_ebm = EmpiricalBrownsMethod(negative_ones_matrix, negative_ones_pval)
        #print('neg ebm:', negative_one_ebm)

    return positive_one_ebm, negative_one_ebm

    """
    normed_pixels = dict()

    for band in the_gal.bands:
        the_band = the_gal[band]
        normed_pixels[band] = the_band.data

        if the_band.valid_pixel_mask is None: the_band.construct_valid_pixel_mask()
        valid_pixel_mask = copy.deepcopy(the_band.valid_pixel_mask)
        to_diff_mask = np.logical_and(to_diff_mask,valid_pixel_mask)

    pos_mask,neg_mask = the_gal.create_bisection()
    
    csv_bands = list(normed_pixels.keys())
    #csv_enteries = np.zeros((len(normed_pixels)+1,np.count_nonzero(to_diff_mask)))
    csv_enteries = np.zeros((len(normed_pixels),np.count_nonzero(to_diff_mask)))
    for i in range(len(csv_bands)):
        csv_enteries[i] = normalize_array(normed_pixels[csv_bands[i]],to_diff_mask)[to_diff_mask]
    #csv_enteries[-1] = pos_mask[to_diff_mask]
    """

class disparate_sides_vote:
    def __init__(self,band_pair_lines, mv_winner_line, ebm_lines):
        self.band_pairs = dict()
        self.mv_winner_label = None

        self.side_0_ebm_p_val = None
        self.side_1_ebm_p_val = None

        self.side_0_naive_p_val = None
        self.side_1_naive_p_val = None

        for each_pair in band_pair_lines:
            try:
                pair_dict = dict()
                pair_dict["label"] = int(each_pair[1])
                pair_dict["gap"] = float(each_pair[3])
                pair_dict["ci"] = float(each_pair[5])
                pair_dict["is_success"] = (each_pair[6].lower() == "success")
                pair_dict["p_val"] = float(each_pair[7])
                self.band_pairs[each_pair[0]] = pair_dict
            except Exception as e:
                print(e) 
                pass
        
        if len(mv_winner_line) > 2:
            try:
                self.mv_winner_label = int(mv_winner_line[1][-1])
            except: pass

        for line in ebm_lines:
            #print(line)
            if len(line) >= 6:
                try:
                    label = int(line[1])
                    ebm_p_val = float(line[4])
                    p_val = float(line[10])

                    if label == 1:
                        self.side_1_ebm_p_val = ebm_p_val
                        self.side_1_naive_p_val = p_val
                    elif label == 0:
                        self.side_0_ebm_p_val = ebm_p_val
                        self.side_0_naive_p_val = p_val
                except: pass

        #print(self.band_pairs)
        #print(self.mv_winner_label)
        #print(self.side_0_ebm_p_val)
        #print(self.side_1_ebm_p_val)
        #print(self.side_0_naive_p_val)
        #print(self.side_1_naive_p_val)

    def get_info(self,pos_label,neg_label):
        best_side_label = ''
        worst_side_label = ''
        ebm_pvalue_dict = dict()
        naive_pvalue_dict = dict()
        band_pair_dict = dict()

        if self.side_0_ebm_p_val != None and self.side_0_naive_p_val != None:
            ebm_pvalue_dict[pos_label] = self.side_0_ebm_p_val
            naive_pvalue_dict[pos_label] = self.side_0_naive_p_val
        else:
            ebm_pvalue_dict[pos_label] = 1.0
            naive_pvalue_dict[pos_label] = 1.0

        if self.side_1_ebm_p_val != None and self.side_1_naive_p_val != None:
            ebm_pvalue_dict[neg_label] = self.side_1_ebm_p_val
            naive_pvalue_dict[neg_label] = self.side_1_naive_p_val
        else:
            ebm_pvalue_dict[neg_label] = 1.0
            naive_pvalue_dict[neg_label] = 1.0

        best_side_label = min(ebm_pvalue_dict, key = ebm_pvalue_dict.get)
        worst_side_label = max(ebm_pvalue_dict, key = ebm_pvalue_dict.get)

        for band_pair_key in self.band_pairs:
            pair_dict = self.band_pairs[band_pair_key]

            vote_label = ''
            p_val = 1

            if pair_dict["label"] == 1:
                vote_label = neg_label
            elif pair_dict["label"] == 0:
                vote_label = pos_label

            p_val = pair_dict["p_val"]

            band_pair_dict[band_pair_key] = [vote_label,p_val]

        return [best_side_label,worst_side_label,ebm_pvalue_dict,naive_pvalue_dict,band_pair_dict]

        




def is_windows():
    return os.name == "nt"

def is_linux():
    return os.name == "posix"

def parse_most_disparate_side_output(output):
    parsed_output = output.decode("utf-8").strip().split("\n")
    parsed_output = list(map(lambda x: x.replace('\t', ' ').replace('(', '').replace(')', '').replace(',', '').replace(':','').strip().split(' '), parsed_output)) #remove tabs, parenthesis, commas, and colons
    parsed_output = list(map(lambda x: list(filter(lambda y: len(y) != 0, x)), parsed_output)) #remove empty entries in list
    return parsed_output


def run_disparate_sides_on_tsv(tsv_path):
    #old_working_dir = os.getcwd()
    #new_working_dir = os.path.dirname(os.path.realpath(__file__))
    #os.chdir(new_working_dir)
    #./most-disparate-side.sh -ebm ~/SpArcFiRe/regression-tests/disparate-sides/NGC2841n.tsv
    #os.chdir('gofher/disparate_side')

    parsed_output = []

    if is_windows():
        proc = subprocess.Popen(
                ['cmd', '/c', 'ubuntu2204', 'run', 'export PATH=$(pwd):$PATH', ';', DISPARATE_SIDES_DIR, "-ebm",  pa], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ##['cmd', '/c', 'ubuntu2204', 'run', DISPARATE_SIDES_DIR, pa], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #['cmd', '/c', 'ubuntu2204', 'run', 'ls'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = proc.communicate()
        rc = proc.returncode
        print(str(output))
        parsed_output = output.decode("utf-8").strip().split("\n") #decode bytes object to string using utf-8 encoding; strip spaces from ends, split on new line char
        parsed_output = list(map(lambda x: x.replace('\t', ' ').replace('(', '').replace(')', '').replace(',', '').replace(':','').strip().split(' '), parsed_output)) #remove tabs, parenthesis, commas, and colons
        parsed_output = list(map(lambda x: list(filter(lambda y: len(y) != 0, x)), parsed_output)) #remove empty entries in list
        print(output.decode("utf-8"))
            #^ see post for more info: https://stackoverflow.com/questions/57693460/using-wsl-bash-from-within-python
    elif is_linux():
        proc = subprocess.Popen(
            [DISPARATE_SIDES_DIR, "-ebm", pa], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = proc.communicate()
        rc = proc.returncode
        print(output.decode("utf-8"))
        parsed_output = output.decode("utf-8").strip().split("\n") #decode bytes object to string using utf-8 encoding; strip spaces from ends, split on new line char
        parsed_output = list(map(lambda x: x.replace('\t', ' ').replace('(', '').replace(')', '').replace(',', '').replace(':','').strip().split(' '), parsed_output)) #remove tabs, parenthesis, commas, and colons
        parsed_output = list(map(lambda x: list(filter(lambda y: len(y) != 0, x)), parsed_output)) #remove empty entries in list

    if len(parsed_output) < 4:
        return None
    else:
        print(parsed_output)
        #for each_line in parsed_output[:-2]:
        #    print(each_line)
 
        dsv = disparate_sides_vote(parsed_output[:-3],parsed_output[-3],parsed_output[-2:])

def run_most_disparate_side_script_on_galaxy(the_gal: galaxy):
    to_diff_mask = the_gal.create_ellipse()
    normed_pixels = dict()

    for band in the_gal.bands:
        the_band = the_gal[band]
        normed_pixels[band] = the_band.data

        if the_band.valid_pixel_mask is None: the_band.construct_valid_pixel_mask()
        valid_pixel_mask = copy.deepcopy(the_band.valid_pixel_mask)
        to_diff_mask = np.logical_and(to_diff_mask,valid_pixel_mask)

    pos_mask,neg_mask = the_gal.create_bisection()
    
    csv_bands = list(normed_pixels.keys())
    csv_enteries = np.zeros((len(normed_pixels)+1,np.count_nonzero(to_diff_mask)))
    for i in range(len(csv_bands)):
        csv_enteries[i] = normalize_array(normed_pixels[csv_bands[i]],to_diff_mask)[to_diff_mask]
    csv_enteries[-1] = pos_mask[to_diff_mask]
    csv_bands.append('side')

    old_working_dir = os.getcwd()
    new_working_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),DISPARATE_SIDES_DIR)

    csv_file_name = "{}.csv".format(the_gal.name.replace(" ",""))
    tsv_file_name = "{}.tsv".format(the_gal.name.replace(" ",""))

    
    csv_path = os.path.join(new_working_dir,csv_file_name)
    if os.path.isfile(csv_path): os.remove(csv_path)

    tsv_path = os.path.join(new_working_dir,tsv_file_name)
    if os.path.isfile(tsv_path): os.remove(tsv_path)

    to_write = csv_enteries.T.tolist()
    for i in range(len(to_write)):
        to_write[i][-1] = int(to_write[i][-1])

    write_csv(csv_path,csv_bands,to_write)

    os.chdir(new_working_dir)
    
    parsed_output = None


    if is_windows():
        proc = subprocess.Popen(
            ['cmd', '/c', 'ubuntu2204', 'run', 'export PATH=$(pwd):$PATH', ';', "dos2unix", csv_file_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = proc.communicate()
        rc = proc.returncode
        if rc != 0:
            print("Error running dos2unix:", error.decode("utf-8"))
            os.chdir(old_working_dir)
            os.remove(csv_path)
            return None

        proc = subprocess.Popen(
            ['cmd', '/c', 'ubuntu2204', 'run', 'export PATH=$(pwd):$PATH', ';', "csv2tsv", csv_file_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = proc.communicate()
        rc = proc.returncode
        if rc != 0:
            print("Error running csv2tsv:", error.decode("utf-8"))
            os.chdir(old_working_dir)
            os.remove(csv_path)
            return None
        
        #if os.path.isfile(tsv_path):
        #    print("tsv {} does not exist in {}".format(tsv_file_name,new_working_dir))
        #    os.chdir(old_working_dir)
        #    return None
        
        proc = subprocess.Popen(
            ['cmd', '/c', 'ubuntu2204', 'run', 'export PATH=$(pwd):$PATH', ';', "./most-disparate-side.sh", "-ebm",  tsv_file_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = proc.communicate()
        rc = proc.returncode
        if rc != 0:
            print("Error running ./most-disparate-side.sh:", error.decode("utf-8"))
            os.chdir(old_working_dir)
            os.remove(csv_path)
            os.remove(tsv_path)
            return None

        parsed_output = parse_most_disparate_side_output(output)
    elif is_linux():
        proc = subprocess.Popen(
            ["csv2tsv", csv_file_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        output, error = proc.communicate()
        rc = proc.returncode
        if rc != 0:
            print("Error running csv2tsv:", error.decode("utf-8"))
            os.chdir(old_working_dir)
            os.remove(csv_path)
            return None
        
        #if os.path.isfile(tsv_path):
        #    print("tsv {} does not exist in {}".format(tsv_file_name,new_working_dir))
        #    os.chdir(old_working_dir)
        #    return None
        
        proc = subprocess.Popen(
            ["./most-disparate-side.sh", "-ebm", tsv_file_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = proc.communicate()
        rc = proc.returncode
        if rc != 0:
            print("Error running ./most-disparate-side.sh:", error.decode("utf-8"))
            os.chdir(old_working_dir)
            os.remove(csv_path)
            os.remove(tsv_path)
            return None
    else:
        print("Error must be running on Windows (via WSL) or Linux natively")
        return None
    
    os.remove(csv_path)
    os.remove(tsv_path)
    
    return disparate_sides_vote(parsed_output[:-3],parsed_output[-3], parsed_output[-2:])

    #dsv = disparate_sides_vote(parsed_output[:-3],parsed_output[-3],parsed_output[-2:])
    #dsv_info = dsv.get_info(the_gal.pos_side_label,the_gal.neg_side_label)
    #print(dsv_info)

    



"""
def convert_normed_csv_to_normed_tsv(csv_path,tsv_path):
    all_rows=[]
    is_header = True
    with open(csv_path,'r') as f:
        for line in f.readlines():
            to_write = []
            if is_header:
                to_write = line.strip().split(',')
                all_rows.append(to_write)
                is_header = False
            else:
                line = line.strip().split(',')
                for each in line[:-1]:
                    to_write.append("{:.8f}".format(float(each)))
                to_write.append(line[-1])
                all_rows.append(to_write)
    with open(tsv_path, 'w', newline='') as f_output:
        tsv_output = csv.writer(f_output, delimiter='\t')
        tsv_output.writerows(all_rows)
"""

#csv_path = 'C:\\Users\\school\\Desktop\\diff_output\\files_for_meeting\\NGC157.csv'
#tsv_path = 'C:\\Users\\school\\Desktop\\diff_output\\files_for_meeting\\NGC157.tsv'    
#convert_normed_csv_to_normed_tsv(csv_path,tsv_path)
"""
def run_sextractor(fits_path,output_path='output.txt'):

    #path issue with source exctator:
    original_working_dir = os.getcwd()
    #print(original_working_dir)
    new_working_dir = pathlib.Path(__file__).parent.absolute()
    os.chdir(new_working_dir)
    #print(new_working_dir)

    res = 0
    try:
        # Add any other sextractor parameters here
        # Any other output paramers you want you need to put in the star_rm.param
        # A complete list of parameters can be seen by running ./sex -dp
        # default.param

        #1) Create subprocess (if on windows, run on WSL, or if on linux use bash)
        if is_windows():
            proc = subprocess.Popen(
                ['cmd', '/c', 'ubuntu1804', 'run', SEX_PATH, fits_path, '-c', CONFIGURATION_FILE_PATH, '-CATALOG_NAME',
                 output_path], stderr=subprocess.PIPE) #run with wsl on windows
            #^ see post for more info: https://stackoverflow.com/questions/57693460/using-wsl-bash-from-within-python
            
            #This was for a test regarding spaces in name:
            #cmd = ['cmd', '/c', 'ubuntu1804', 'run', SEX_PATH, fits_path.replace(' ', '\ '), '-c', CONFIGURATION_FILE_PATH, '-CATALOG_NAME', output_path.replace(' ', '\ ')]
            #fits_path = '"{}"'.format(fits_path) if ' ' in fits_path else fits_path
            #output_path = '"{}"'.format(output_path) if ' ' in output_path else fits_path
            #cmd = ['cmd', '/c', 'ubuntu1804', 'run', SEX_PATH, fits_path, '-c', CONFIGURATION_FILE_PATH, '-CATALOG_NAME', output_path]
            #print(cmd)
            #proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,stdin=subprocess.PIPE)
            
            
        elif is_linux():
            proc = subprocess.Popen([SEX_PATH, fits_path, '-c', CONFIGURATION_FILE_PATH, '-CATALOG_NAME', output_path],
                                    stderr=subprocess.PIPE)
        else:
            raise OSError("Not recognized OS, must be Windows or Linux")
"""

##run_disparate_sides()
