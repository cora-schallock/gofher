import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

folders = ["figure8", "figure10","figure11"]
#folders = ['figure9']
title = "Observed: Axis Ratio vs p-values"

def get_ebm_path(ang: float, folder: str) -> str:
    return "C:\\Users\\school\\Desktop\\p_val_sweep_angle_test\\{}\\{}_ebm.csv".format(int(ang),folder)

def get_csv_path(ang: float, folder: str) -> str:
    return "C:\\Users\\school\\Desktop\\p_val_sweep_angle_test\\{}\\{}_params.csv".format(int(ang),folder)


def parse_ebm(filename: str) -> dict:
    to_return = dict()

    df = pd.read_csv(filename)

    for index, row in df.iterrows():
        to_return[row["name"]] = row["ebm_pval_winning"]

    return to_return

def parse_csv(filename: str) -> dict:
    to_return = dict()

    df = pd.read_csv(filename)

    for index, row in df.iterrows():
        a = row["a"]
        b = row["b"]
        to_return[row["name"]] = b/a #(a-b)/a 

    return to_return

def run():
    ang = 0

    ebm = dict()
    e = dict()

    for folder in folders:
        the_ebm = parse_ebm(get_ebm_path(ang,folder))
        for name in the_ebm:
            ebm[name] = the_ebm[name]
        the_e = parse_csv(get_csv_path(ang,folder))
        for name in the_e:
            e[name] = the_e[name]

    x = []
    y = []

    for name in ebm:
        if name not in e: continue
        x.append(e[name])
        y.append(ebm[name])

    the_list = np.array(list(y))
    nonzeros = np.nonzero(the_list)
    the_list[nonzeros] = -np.log(the_list[nonzeros])

    x = np.array(x)

    z = np.polyfit(x, the_list, 1)
    p = np.poly1d(z)
    print(p)

    print(np.average(x),np.std(x))
    
    #add trendline to plot
    plt.plot(x, p(x),"r--",alpha=0.5,label="trendling: {}".format(p))

    plt.scatter(x,the_list)
    plt.title(title)
    plt.xlabel("Axis Ratio")
    plt.ylabel("-log(p-value)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run()



