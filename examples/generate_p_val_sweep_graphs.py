import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from scipy.stats.mstats import gmean

#folders = ["figure8","figure10","figure11"]
#title = "Observed Galaxies: Geomean Welch p-value vs offset angle"
#title = "Observed Galaxies: Normed Welch p-value vs offset angle"
folders = ["figure9"]
#title = "Welch geomean - Inferred p-value sweep"
#title = "Inferred Galaxies: Geomean Welch p-value vs offset angle"
title = "Inferred Galaxies: Normed Welch p-value vs offset angle"

def get_csv_path(ang: float, folder: str) -> str:
    return "C:\\Users\\school\\Desktop\\welch_p_val_sweep_angle_test\\{}\\{}_ebm.csv".format(int(ang),folder)


def parse_csv(filename: str) -> dict:
    to_return = dict()
    df = pd.read_csv(filename)

    for index, row in df.iterrows():
        to_return[row["name"]] = row["ebm_pval_winning"]

    return to_return

def generate_p_val_sweep_graphs():
    all_data = dict()
    for folder in folders:
        for ang in np.linspace(0, 175, 36):
            the_path = get_csv_path(ang,folder)
            current_dict = parse_csv(the_path)
            for name in current_dict:
                if name not in all_data:
                    all_data[name] = dict()
                all_data[name][ang] = current_dict[name]
    to_plot(all_data)


def to_plot(data: dict):
    average = np.zeros(36)
    for name in data:
        the_list = np.array(list(data[name].values()))
        nonzeros = np.nonzero(the_list)
        the_list[nonzeros] = -np.log(the_list[nonzeros])
        #the_value = -np.log(list(data[name].values()))
        #the_value = list(data[name].values())
        the_value = the_list
        the_max = np.max(the_value)
        the_value = the_value / the_max
        average += the_value
        plt.plot(list(data[name].keys()), the_value, color = 'blue', alpha = 0.2)
    plt.plot(list(data[name].keys()), average/len(data), color = 'red')
    plt.title(title)
    plt.xlabel("theta")
    plt.ylabel("|-log(p-value)|")
    #plt.show()
    output_path = "C:\\Users\\school\\Desktop\\welch_p_val_sweep_angle_test\\normed_inf.png"
    plt.savefig(output_path, dpi = 300, bbox_inches='tight')

"""
def to_plot(data: dict):
    plotting = np.zeros((len(data), 36))
    keys = sorted(data[list(data.keys())[0]].keys())
    for i in range(len(data)):
        name = list(data.keys())[i]
        for j in range(36):
            plotting[i, j] = data[name][keys[j]]

    y_vals = np.zeros(36)
    for i in range(36):
        every = plotting[:, i]
        nonzeros = np.nonzero(every)
        y_vals[i] = gmean(every[nonzeros])

    the_min = np.min(y_vals)
    the_max = np.max(y_vals)

    y_vals = np.log(y_vals)

    full_title = "{}\n min:{:.2e} max:{:.2e}".format(title, the_min, the_max)

    plt.plot(keys, y_vals)
    plt.title(full_title)
    plt.xlabel("theta")
    plt.ylabel("log p-value")
    #plt.show()
    output_path = "C:\\Users\\school\\Desktop\\welch_p_val_sweep_angle_test\\inf.png"
    plt.savefig(output_path, dpi = 300, bbox_inches='tight')
"""

    
if __name__ == "__main__":
    #for formatting mnras paper: source: https://walmsley.dev/posts/typesetting-mnras-figures

    SMALL_SIZE = 9
    MEDIUM_SIZE = 9
    BIGGER_SIZE = 9

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)
    plt.rc('font', family='Nimbus Roman No9 L') #uncomment this to use same font as MNRAS - requires font Nimbus Roman No9 L (can be downloaded from https://www.fontsquirrel.com/fonts/nimbus-roman-no9-l)

    generate_p_val_sweep_graphs()