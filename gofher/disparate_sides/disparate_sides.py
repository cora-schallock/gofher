import os
import subprocess
import sys
import pathlib

DISPARATE_SIDES_DIR = "./most-disparate-side.sh"
pa = "NGC2841n.tsv"
#pa = "IC1151n.tsv"

# TODO: constuct normed tsv (see IC1151n.tsv for details)
# allow run_disparate_sides() take a variable tsv
# parse disparate_sides_vote and output final csv
# implement methods to check for no vote
# make callable from python scripts outside
# clean up this file

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
            except: pass
        
        if len(mv_winner_line) > 2:
            try:
                self.mv_winner_label = int(mv_winner_line[1][-1])
            except: pass

        for line in ebm_lines:
            print(line)
            if len(line) >= 6:
                try:
                    label = int(line[1])
                    ebm_p_val = float(line[2])
                    p_val = float(line[4])

                    if label == 1:
                        self.side_1_ebm_p_val = ebm_p_val
                        self.side_1_naive_p_val = p_val
                    elif label == 0:
                        self.side_0_ebm_p_val = ebm_p_val
                        self.side_0_naive_p_val = p_val
                except: pass

        print(self.band_pairs)
        print(self.mv_winner_label)
        print(self.side_0_ebm_p_val)
        print(self.side_1_ebm_p_val)
        print(self.side_0_naive_p_val)
        print(self.side_1_naive_p_val)




def is_windows():
    return os.name == "nt"

def is_linux():
    return os.name == "posix"

def run_disparate_sides():
    old_working_dir = os.getcwd()
    new_working_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(new_working_dir)
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
        #print(str(output))
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
        #for each_line in parsed_output[:-2]:
        #    print(each_line)
 
        dsv = disparate_sides_vote(parsed_output[:-3],parsed_output[-3],parsed_output[-2:])

        

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

run_disparate_sides()